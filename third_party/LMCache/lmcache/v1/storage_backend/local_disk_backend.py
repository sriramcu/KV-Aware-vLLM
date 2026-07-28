# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import OpType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.batched_message_sender import BatchedMessageSender
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.path_sharder import PathSharder

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


def _kvio_trace_enabled() -> bool:
    return os.environ.get("SRIRAM_KV_IO_TRACE", "0") == "1"


def _kvio_proc_io_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    if not _kvio_trace_enabled():
        return values
    try:
        with open("/proc/self/io", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                values[key.strip()] = int(value.strip())
    except Exception as exc:
        logger.warning("[KVIO_PROC_IO_ERROR] pid=%d error=%r", os.getpid(), exc)
    return values


def _kvio_io_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in set(before) | set(after)
    }


# TODO(Jiayi): handle cases where cache is repetitvely prefetched.
class LocalDiskWorker:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        # TODO(Jiayi): make executor and its parameters configurable
        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=4)
        self.loop = loop
        self._closed = False
        # Observability-only counters. They do not gate or reorder work.
        self._kvio_state_lock = threading.Lock()
        self._kvio_active = {"prefetch": 0, "put": 0, "delete": 0}

        # P0-B: bound disk puts before LocalDiskBackend takes an extra
        # MemoryObj ref and before work enters the executor's unbounded queue.
        # A value of 0 disables P0-B.
        self._p0_put_limit = int(
            os.environ.get(
                "LMCACHE_P0_DISK_PUT_MAX_PENDING",
                "8",
            )
        )
        if self._p0_put_limit < 0:
            raise ValueError(
                "LMCACHE_P0_DISK_PUT_MAX_PENDING must be >= 0, "
                f"got {self._p0_put_limit}"
            )
        self._p0_put_admission = (
            threading.BoundedSemaphore(self._p0_put_limit)
            if self._p0_put_limit > 0
            else None
        )
        self._p0_put_state_lock = threading.Lock()
        self._p0_put_inflight = 0
        self._p0_put_waiters = 0
        self._p0_put_peak = 0

        logger.info(
            "P0 disk put admission: limit=%d (0 disables P0-B)",
            self._p0_put_limit,
        )

    def acquire_put_admission(self, key: CacheEngineKey) -> bool:
        """Block before ref_count_up until one bounded put slot is free.

        A disabled gate returns True immediately so callers continue through
        the original LMCache write path without admission throttling.
        """
        semaphore = self._p0_put_admission
        if semaphore is None:
            return True

        queued_at = time.monotonic()
        trace_enabled = _kvio_trace_enabled()
        with self._p0_put_state_lock:
            self._p0_put_waiters += 1
            waiters = self._p0_put_waiters
            inflight = self._p0_put_inflight

        if trace_enabled:
            logger.warning(
                "[P0_PUT_ADMISSION_WAIT] pid=%d key_hash=%s "
                "limit=%d inflight=%d waiters=%d queue_depth=%d "
                "thread=%s",
                os.getpid(),
                key.chunk_hash,
                self._p0_put_limit,
                inflight,
                waiters,
                self.executor._queue.qsize(),
                threading.current_thread().name,
            )

        # Blocking the storage-manager event-loop thread would prevent queued
        # puts from completing and releasing slots. That path is not expected
        # for normal vLLM stores, so reject rather than deadlock if it occurs.
        on_event_loop = threading.get_ident() == getattr(
            self.loop,
            "_thread_id",
            None,
        )
        if on_event_loop:
            acquired = semaphore.acquire(blocking=False)
            if not acquired:
                with self._p0_put_state_lock:
                    self._p0_put_waiters -= 1
                    waiters = self._p0_put_waiters
                    inflight = self._p0_put_inflight
                logger.error(
                    "[P0_PUT_ADMISSION_REJECT_LOOP_THREAD] "
                    "pid=%d key_hash=%s waited=%.6f limit=%d "
                    "inflight=%d waiters=%d queue_depth=%d",
                    os.getpid(),
                    key.chunk_hash,
                    time.monotonic() - queued_at,
                    self._p0_put_limit,
                    inflight,
                    waiters,
                    self.executor._queue.qsize(),
                )
                return False
        else:
            while not semaphore.acquire(timeout=5.0):
                logger.warning(
                    "[P0_PUT_ADMISSION_STALLED] pid=%d key_hash=%s "
                    "waited=%.6f limit=%d queue_depth=%d "
                    "put_tasks=%d thread=%s",
                    os.getpid(),
                    key.chunk_hash,
                    time.monotonic() - queued_at,
                    self._p0_put_limit,
                    self.executor._queue.qsize(),
                    len(self.put_tasks),
                    threading.current_thread().name,
                )

        with self._p0_put_state_lock:
            self._p0_put_waiters -= 1
            self._p0_put_inflight += 1
            self._p0_put_peak = max(
                self._p0_put_peak,
                self._p0_put_inflight,
            )
            waiters = self._p0_put_waiters
            inflight = self._p0_put_inflight
            peak = self._p0_put_peak

        if trace_enabled:
            logger.warning(
                "[P0_PUT_ADMISSION_ACQUIRE] pid=%d key_hash=%s "
                "waited=%.6f limit=%d inflight=%d waiters=%d "
                "peak=%d queue_depth=%d put_tasks=%d thread=%s",
                os.getpid(),
                key.chunk_hash,
                time.monotonic() - queued_at,
                self._p0_put_limit,
                inflight,
                waiters,
                peak,
                self.executor._queue.qsize(),
                len(self.put_tasks),
                threading.current_thread().name,
            )
        return True

    def release_put_admission(
        self,
        key: CacheEngineKey,
        reason: str,
        error: Optional[BaseException] = None,
    ) -> None:
        """Release a put slot after completion or failed submission."""
        semaphore = self._p0_put_admission
        if semaphore is None:
            return

        with self._p0_put_state_lock:
            self._p0_put_inflight -= 1
            if self._p0_put_inflight < 0:
                self._p0_put_inflight = 0
                raise RuntimeError(
                    f"P0 put admission underflow for key={key}"
                )
            inflight = self._p0_put_inflight
            waiters = self._p0_put_waiters
            peak = self._p0_put_peak

        semaphore.release()
        if _kvio_trace_enabled() or error is not None:
            logger.warning(
                "[P0_PUT_ADMISSION_RELEASE] pid=%d key_hash=%s "
                "reason=%s error=%r limit=%d inflight=%d "
                "waiters=%d peak=%d queue_depth=%d put_tasks=%d "
                "thread=%s",
                os.getpid(),
                key.chunk_hash,
                reason,
                error,
                self._p0_put_limit,
                inflight,
                waiters,
                peak,
                self.executor._queue.qsize(),
                len(self.put_tasks),
                threading.current_thread().name,
            )

    async def submit_task(
        self,
        task_type: str,
        task: Callable,
        *args,
        **kwargs,
    ) -> Any:
        if task_type == "prefetch":
            priority = 0
            # self.insert_prefetch_task(kwargs["key"], None)
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        kvio_id = str(kwargs.pop("_kvio_id", "unknown"))
        enqueued_at = time.monotonic()
        queue_depth_before = self.executor._queue.qsize()
        if _kvio_trace_enabled():
            with self._kvio_state_lock:
                active_snapshot = dict(self._kvio_active)
            logger.warning(
                "[KVIO_DISKQ_ENQUEUE] pid=%d id=%s task_type=%s "
                "task=%s priority=%d queue_depth_before=%d active=%s "
                "put_tasks=%d mono=%.6f",
                os.getpid(),
                kvio_id,
                task_type,
                getattr(task, "__name__", repr(task)),
                priority,
                queue_depth_before,
                active_snapshot,
                len(self.put_tasks),
                enqueued_at,
            )

        def traced_task(*task_args, **task_kwargs):
            started_at = time.monotonic()
            with self._kvio_state_lock:
                self._kvio_active[task_type] += 1
                active_snapshot = dict(self._kvio_active)
            if _kvio_trace_enabled():
                logger.warning(
                    "[KVIO_DISKQ_START] pid=%d id=%s task_type=%s "
                    "task=%s queue_wait=%.6f queue_depth_now=%d "
                    "active=%s put_tasks=%d thread=%s",
                    os.getpid(),
                    kvio_id,
                    task_type,
                    getattr(task, "__name__", repr(task)),
                    started_at - enqueued_at,
                    self.executor._queue.qsize(),
                    active_snapshot,
                    len(self.put_tasks),
                    threading.current_thread().name,
                )
            try:
                return task(*task_args, **task_kwargs)
            finally:
                finished_at = time.monotonic()
                with self._kvio_state_lock:
                    self._kvio_active[task_type] -= 1
                    active_after = dict(self._kvio_active)
                if _kvio_trace_enabled():
                    logger.warning(
                        "[KVIO_DISKQ_DONE] pid=%d id=%s task_type=%s "
                        "task=%s service=%.6f total=%.6f "
                        "queue_depth_now=%d active=%s put_tasks=%d thread=%s",
                        os.getpid(),
                        kvio_id,
                        task_type,
                        getattr(task, "__name__", repr(task)),
                        finished_at - started_at,
                        finished_at - enqueued_at,
                        self.executor._queue.qsize(),
                        active_after,
                        len(self.put_tasks),
                        threading.current_thread().name,
                    )

        return await self.executor.submit_job(
            traced_task,
            *args,
            priority=priority,
            **kwargs,
        )

    def remove_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            if key in self.put_tasks:
                self.put_tasks.remove(key)
                if _kvio_trace_enabled():
                    logger.warning(
                        "[KVIO_PUT_TASK_REMOVE] pid=%d key_hash=%s "
                        "remaining_put_tasks=%d",
                        os.getpid(),
                        key.chunk_hash,
                        len(self.put_tasks),
                    )
            else:
                logger.warning(f"Key {key} not found in put tasks.")

    def insert_put_task(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            if key in self.put_tasks:
                return False
            self.put_tasks.append(key)
            if _kvio_trace_enabled():
                logger.warning(
                    "[KVIO_PUT_TASK_INSERT] pid=%d key_hash=%s put_tasks=%d",
                    os.getpid(),
                    key.chunk_hash,
                    len(self.put_tasks),
                )
            return True

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def close(self):
        # Gracefully shut down the executor
        if self._closed:
            return
        if _kvio_trace_enabled():
            with self._kvio_state_lock:
                active_snapshot = dict(self._kvio_active)
            logger.warning(
                "[KVIO_DISKQ_CLOSE_ENTER] pid=%d queue_depth=%d "
                "active=%s put_tasks=%d",
                os.getpid(),
                self.executor._queue.qsize(),
                active_snapshot,
                len(self.put_tasks),
            )
        self._closed = True
        self.executor.shutdown(wait=True)
        if _kvio_trace_enabled():
            logger.warning(
                "[KVIO_DISKQ_CLOSE_DONE] pid=%d queue_depth=%d put_tasks=%d",
                os.getpid(),
                self.executor._queue.qsize(),
                len(self.put_tasks),
            )


class LocalDiskBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = torch_device_type,
        lmcache_worker: Optional["LMCacheWorker"] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ):
        if torch_dev.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.dst_device = dst_device

        self.local_cpu_backend = local_cpu_backend

        self.disk_lock = threading.Lock()

        assert config.local_disk is not None

        sharder = PathSharder(
            raw_csv=config.local_disk,
            strategy=config.local_disk_path_sharding,
            dst_device=dst_device,
            create_dirs=True,
        )
        self.path: str = sharder.selected

        logger.info(
            "Local disk cache path: %s (device %s, %d path(s) configured)",
            self.path,
            dst_device,
            len(sharder.all_paths),
        )

        self.loop = loop

        self.use_local_cpu = config.local_cpu

        # Block size (for file system I/O)
        stat = os.statvfs(self.path)
        self.os_disk_bs = stat.f_bsize
        self.use_odirect = False

        if config.extra_config is not None:
            self.use_odirect = config.extra_config.get("use_odirect", False)
        logger.info("Using O_DIRECT for disk I/O: %s", self.use_odirect)

        self.disk_worker = LocalDiskWorker(loop)

        # TODO(Jiayi): We need a disk space allocator to avoid fragmentation
        # and hide the following details away from the backend.
        self.max_cache_size = int(config.max_local_disk_size * 1024**3)
        self.current_cache_size = 0.0

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

        # Batched message sender for controller communication
        self.batched_msg_sender: Optional[BatchedMessageSender] = None

        # Initialize batched message sender
        if lmcache_worker and metadata is not None:
            self.batched_msg_sender = BatchedMessageSender(
                metadata=metadata,
                config=config,
                location=str(self),
                lmcache_worker=lmcache_worker,
            )
        else:
            logger.warning("Controller message sender is not initialized")

    def __str__(self) -> str:
        return "LocalDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.disk_worker.exists_in_put_tasks(key)

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            else:
                return False

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            else:
                return False

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        if force:
            self.disk_lock.acquire()

        if not (meta := self.dict.pop(key, None)):
            if force:
                self.disk_lock.release()
            return False

        path = meta.path
        size = meta.size
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # NOTE: The following code will cause deadlock
        # res = asyncio.run_coroutine_threadsafe(
        #     self.disk_worker.submit_task("delete", os.remove, path),
        #     self.loop,
        # )
        # res.result()

        os.remove(path)

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.disk_lock.release()

        # Push kv evict msg with batching
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.EVICT,
                key=key.chunk_hash,
            )

        return True

    def insert_key(
        self,
        key: CacheEngineKey,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        cached_positions: Optional[torch.Tensor] = None,
    ) -> None:
        path = self._key_to_path(key)

        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                # Update cache recency
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(
                    path, size, shape, dtype, cached_positions, fmt, 0
                )

        # Push kv admit msg with batching
        if self.batched_msg_sender is not None and not has_stored:
            self.batched_msg_sender.add_kv_op(
                op_type=OpType.ADMIT,
                key=key.chunk_hash,
            )

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ):
        """
        Submit a single put task to store KV cache to disk asynchronously.

        P0-B acquires a bounded admission slot before incrementing the
        MemoryObj refcount or placing the write in the executor queue.

        :param key: The cache key for this KV chunk.
        :param memory_obj: The memory object containing the KV data.
        :param on_complete_callback: Optional callback invoked once per key
            after the disk write completes. Callback exceptions are caught
            and logged.
        """
        assert memory_obj.tensor is not None

        # Fast duplicate check avoids waiting for a slot for work that is
        # already pending. insert_put_task() repeats the check atomically after
        # admission to close the race between concurrent submitters.
        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        if not self.disk_worker.acquire_put_admission(key):
            logger.warning(
                "Skipping disk put for %s because bounded admission rejected it.",
                key,
            )
            return None

        if not self.disk_worker.insert_put_task(key):
            self.disk_worker.release_put_admission(
                key,
                reason="duplicate_after_wait",
            )
            return None

        # TODO(Jiayi): Fragmentation is not considered here.
        required_size = memory_obj.get_physical_size()
        all_evict_keys = []
        evict_success = True
        with self.disk_lock:
            while self.current_cache_size + required_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. Disk space under pressure."
                    )
                    evict_success = False
                    break

                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size

                self.batched_remove(evict_keys, force=False)

                all_evict_keys.extend(evict_keys)
            if evict_success:
                self.current_cache_size += required_size
                self.cache_policy.update_on_put(key)

        if not evict_success:
            self.disk_worker.remove_put_task(key)
            self.disk_worker.release_put_admission(
                key,
                reason="disk_capacity_rejected",
            )
            return None

        # This extra ref is now taken only after bounded admission succeeds.
        memory_obj.ref_count_up()

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.disk_worker.submit_task(
                    "put",
                    self.async_save_bytes_to_disk,
                    key=key,
                    memory_obj=memory_obj,
                    on_complete_callback=on_complete_callback,
                    _kvio_id=f"put:{key.chunk_hash}",
                ),
                self.loop,
            )
        except BaseException as exc:
            memory_obj.ref_count_down()
            self.disk_worker.remove_put_task(key)
            self.disk_worker.release_put_admission(
                key,
                reason="submit_failed",
                error=exc,
            )
            raise

        def release_admission(done_future: Future) -> None:
            error: Optional[BaseException] = None
            reason = "completed"
            if done_future.cancelled():
                reason = "cancelled"
            else:
                try:
                    error = done_future.exception()
                except BaseException as exc:
                    error = exc
                if error is not None:
                    reason = "failed"
            self.disk_worker.release_put_admission(
                key,
                reason=reason,
                error=error,
            )

        future.add_done_callback(release_admission)
        return future

    # TODO(Jiayi): enable real batching
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to store KV caches to disk asynchronously.

        :param keys: The cache keys for the KV chunks.
        :param memory_objs: The memory objects containing the KV data.
        :param transfer_spec: Optional transfer specification (unused).
        :param on_complete_callback: Optional callback invoked once per key
            after that key's disk write completes (not once per batch).
            Callback exceptions are caught and logged.
        """
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(
                key, memory_obj, on_complete_callback=on_complete_callback
            )

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Load a cached KV chunk from disk synchronously.

        The cache policy is updated only after a successful load so that a
        failed load (``load_bytes_from_disk`` returning ``None``) does not
        record a phantom cache hit and skew future eviction decisions.

        :param key: The cache key identifying the KV chunk.
        :returns: A ``MemoryObj`` containing the loaded KV data, or ``None``
            if the key is not present or the load fails.
        """
        with self.disk_lock:
            if key not in self.dict:
                return None

            disk_meta = self.dict[key]
            path = disk_meta.path
            dtype = disk_meta.dtype
            shape = disk_meta.shape
            fmt = disk_meta.fmt
            assert dtype is not None
            assert shape is not None

        # Load is performed outside the lock: it can block for a non-trivial
        # amount of time (CPU staging pool allocation + memcpy from disk) and
        # must not hold disk_lock while waiting, or concurrent insert/evict
        # operations would deadlock.
        memory_obj = self.load_bytes_from_disk(
            key, path, dtype=dtype, shape=shape, fmt=fmt
        )

        if memory_obj is not None:
            # Re-acquire the lock to update the eviction policy.  The key
            # membership check guards against the entry being evicted between
            # the two lock regions — in that case the policy state is already
            # consistent and no update is needed.
            with self.disk_lock:
                if key in self.dict:
                    self.cache_policy.update_on_hit(key, self.dict)

        return memory_obj

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        mem_objs: list[MemoryObj] = []
        paths: list[str] = []
        started = time.monotonic()
        debug_enabled = os.environ.get("SRIRAM_KV_MEM_DEBUG", "0") == "1"

        if debug_enabled:
            logger.warning(
                "[KVDBG_DISK_LOAD_START] "
                "pid=%d lookup_id=%s requested_keys=%d thread=%s",
                os.getpid(),
                lookup_id,
                len(keys),
                threading.current_thread().name,
            )

        logger.debug(
            "lookup_id: %s; Prefetching %d keys from disk.",
            lookup_id,
            len(keys),
        )

        for key_index, key in enumerate(keys):
            # Use a context manager so every exit path releases disk_lock.
            # The previous code returned on allocation failure while still
            # holding this lock, freezing subsequent disk operations.
            with self.disk_lock:
                assert key in self.dict, (
                    f"Key {key} not found in disk cache after pinning"
                )

                path = self.dict[key].path
                dtype = self.dict[key].dtype
                shape = self.dict[key].shape
                fmt = self.dict[key].fmt

                assert dtype is not None
                assert shape is not None

                # busy_loop=False prevents spinning on the event-loop thread.
                allocation_started = time.monotonic()
                memory_obj = self.local_cpu_backend.allocate(
                    shape,
                    dtype,
                    fmt,
                    busy_loop=False,
                )
                allocation_elapsed = time.monotonic() - allocation_started
                if _kvio_trace_enabled():
                    logger.warning(
                        "[KVIO_STAGE_ALLOC] pid=%d lookup_id=%s "
                        "key_index=%d/%d key_hash=%s shape=%s dtype=%s fmt=%s "
                        "success=%s elapsed=%.6f",
                        os.getpid(),
                        lookup_id,
                        key_index,
                        len(keys),
                        key.chunk_hash,
                        shape,
                        dtype,
                        fmt,
                        memory_obj is not None,
                        allocation_elapsed,
                    )

                if memory_obj is None:
                    logger.error(
                        "[KVDBG_DISK_STAGE_FAIL] "
                        "pid=%d lookup_id=%s key_index=%d/%d "
                        "allocated_so_far=%d elapsed=%.3fs thread=%s. "
                        "Memory allocation failed during async disk load "
                        "for key %s. CPU staging pool may be exhausted. "
                        "Returning partial results.",
                        os.getpid(),
                        lookup_id,
                        key_index,
                        len(keys),
                        len(mem_objs),
                        time.monotonic() - started,
                        threading.current_thread().name,
                        key,
                    )
                    return mem_objs

                # Extra disk pin for the physical read. The lookup pin is
                # released separately by lookup_unpin().
                self.dict[key].pin()

                # NOTE(Jiayi): Currently, we consider prefetch as cache hit.
                self.cache_policy.update_on_hit(key, self.dict)

            logger.debug("Prefetching %s from disk.", key)
            memory_obj.pin()
            mem_objs.append(memory_obj)
            paths.append(path)

        if _kvio_trace_enabled():
            logger.warning(
                "[KVIO_STAGE_BATCH_DONE] pid=%d lookup_id=%s "
                "requested_keys=%d staged=%d elapsed=%.6f",
                os.getpid(),
                lookup_id,
                len(keys),
                len(mem_objs),
                time.monotonic() - started,
            )

        try:
            results = await self.disk_worker.submit_task(
                "prefetch",
                self.batched_async_load_bytes_from_disk,
                paths=paths,
                keys=keys,
                memory_objs=mem_objs,
                lookup_id=lookup_id,
                _kvio_id=lookup_id,
            )
        except Exception:
            logger.exception(
                "[KVDBG_DISK_LOAD_FAILED] "
                "pid=%d lookup_id=%s requested_keys=%d "
                "allocated=%d elapsed=%.3fs thread=%s",
                os.getpid(),
                lookup_id,
                len(keys),
                len(mem_objs),
                time.monotonic() - started,
                threading.current_thread().name,
            )
            raise

        if debug_enabled or time.monotonic() - started >= 1.0:
            logger.warning(
                "[KVDBG_DISK_LOAD_DONE] "
                "pid=%d lookup_id=%s requested_keys=%d returned=%d "
                "elapsed=%.3fs thread=%s",
                os.getpid(),
                lookup_id,
                len(keys),
                len(results),
                time.monotonic() - started,
                threading.current_thread().name,
            )

        return results

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        with self.disk_lock:
            for key in keys:
                if key not in self.dict:
                    return num_hit_counts
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        return num_hit_counts

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.

        :param on_complete_callback: Optional callback invoked after the disk
            write completes for this key. Callback exceptions are caught and
            logged.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        buffer = memory_obj.byte_array
        path = self._key_to_path(key)

        size = len(buffer)
        self.usage += size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # TODO(Jiayi): need to add ref count in disk memory object
        self.write_file(buffer, path, key=key)

        # ref count down here because there's a ref_count_up in
        # `submit_put_task` above.
        # Ref count down better be before `insert_key` for testing
        # purposes (e.g., testing mem_leak).
        # TODO(Jiayi): This could be problematic if the
        # freed memory object is immediately reused.
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        cached_positions = memory_obj.metadata.cached_positions
        memory_obj.ref_count_down()

        self.insert_key(key, size, shape, dtype, fmt, cached_positions=cached_positions)

        self.disk_worker.remove_put_task(key)

        # Call the completion callback if provided
        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning(f"on_complete_callback failed for key {key}: {e}")

    def batched_async_load_bytes_from_disk(
        self,
        paths: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        write_back: bool = False,
        lookup_id: str = "unknown",
    ) -> list[MemoryObj]:
        """
        Async load bytearray from disk.
        """

        logger.debug("Executing `async_load_bytes` from disk.")
        batch_started = time.monotonic()
        if _kvio_trace_enabled():
            logger.warning(
                "[KVIO_READ_BATCH_START] pid=%d lookup_id=%s files=%d "
                "bytes=%d thread=%s mono=%.6f",
                os.getpid(),
                lookup_id,
                len(paths),
                sum(len(mem_obj.byte_array) for mem_obj in memory_objs),
                threading.current_thread().name,
                batch_started,
            )
        # TODO (Jiayi): handle the case where loading fails.
        for file_index, (path, key, mem_obj) in enumerate(
            zip(paths, keys, memory_objs, strict=False)
        ):
            buffer = mem_obj.byte_array
            self.read_file(
                key,
                buffer,
                path,
                lookup_id=lookup_id,
                file_index=file_index,
                total_files=len(paths),
            )

            # TODO(Jiayi): Please recover the metadata in a more
            # elegant way in the future.
            cached_positions = self.dict[key].cached_positions
            mem_obj.metadata.cached_positions = cached_positions

            self.disk_lock.acquire()
            self.dict[key].unpin()
            self.disk_lock.release()

        if _kvio_trace_enabled():
            logger.warning(
                "[KVIO_READ_BATCH_DONE] pid=%d lookup_id=%s files=%d "
                "elapsed=%.6f thread=%s",
                os.getpid(),
                lookup_id,
                len(paths),
                time.monotonic() - batch_started,
                threading.current_thread().name,
            )
        return memory_objs

    def load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """

        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None, "Memory allocation failed during disk load."

        buffer = memory_obj.byte_array
        self.read_file(key, buffer, path)

        # TODO(Jiayi): Please recover the metadata in a more
        # elegant way in the future.
        cached_positions = self.dict[key].cached_positions
        memory_obj.metadata.cached_positions = cached_positions

        return memory_obj

    def write_file(self, buffer, path, key: Optional[CacheEngineKey] = None):
        total_started = time.monotonic()
        size = len(buffer)
        io_before = _kvio_proc_io_snapshot()
        open_elapsed = 0.0
        write_elapsed = 0.0
        bytes_written = 0

        if size % self.os_disk_bs != 0 or not self.use_odirect:
            open_started = time.monotonic()
            with open(path, "wb") as f:
                open_elapsed = time.monotonic() - open_started
                write_started = time.monotonic()
                bytes_written = f.write(buffer)
                write_elapsed = time.monotonic() - write_started
        else:
            open_started = time.monotonic()
            fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
            open_elapsed = time.monotonic() - open_started
            try:
                write_started = time.monotonic()
                bytes_written = os.write(fd, buffer)
                write_elapsed = time.monotonic() - write_started
            finally:
                os.close(fd)

        total_elapsed = time.monotonic() - total_started
        io_after = _kvio_proc_io_snapshot()
        logger.debug(
            f"Disk write size: {size} bytes, "
            f"Bandwidth: {size / max(total_elapsed, 1e-9) / 1e6:.2f} MB/s"
        )
        if _kvio_trace_enabled():
            logger.warning(
                "[KVIO_FILE_WRITE] pid=%d key_hash=%s bytes_expected=%d "
                "bytes_written=%d open=%.6f write=%.6f total=%.6f "
                "odirect=%s proc_io_delta=%s thread=%s path=%s",
                os.getpid(),
                getattr(key, "chunk_hash", "unknown"),
                size,
                bytes_written,
                open_elapsed,
                write_elapsed,
                total_elapsed,
                self.use_odirect,
                _kvio_io_delta(io_before, io_after),
                threading.current_thread().name,
                path,
            )

    def read_file(
        self,
        key,
        buffer,
        path,
        lookup_id: str = "unknown",
        file_index: int = -1,
        total_files: int = -1,
    ):
        total_started = time.monotonic()
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0
        if not fblock_aligned and self.use_odirect:
            logger.warning(
                "Cannot use O_DIRECT for this file, "
                "size is not aligned to disk block size."
            )

        io_before = _kvio_proc_io_snapshot()
        open_elapsed = 0.0
        read_elapsed = 0.0
        bytes_read = 0
        effective_odirect = fblock_aligned and self.use_odirect

        try:
            if not effective_odirect:
                open_started = time.monotonic()
                with open(path, "rb") as f:
                    open_elapsed = time.monotonic() - open_started
                    read_started = time.monotonic()
                    bytes_read = f.readinto(buffer)
                    read_elapsed = time.monotonic() - read_started
            else:
                open_started = time.monotonic()
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                open_elapsed = time.monotonic() - open_started
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    read_started = time.monotonic()
                    bytes_read = fdo.readinto(buffer)
                    read_elapsed = time.monotonic() - read_started
        except FileNotFoundError:
            logger.warning(f"File not found on disk: {path}")
            if self.dict.get(key, None):
                self.dict.pop(key)
            if _kvio_trace_enabled():
                logger.warning(
                    "[KVIO_FILE_READ_MISSING] pid=%d lookup_id=%s "
                    "file_index=%d/%d key_hash=%s elapsed=%.6f path=%s",
                    os.getpid(),
                    lookup_id,
                    file_index,
                    total_files,
                    getattr(key, "chunk_hash", "unknown"),
                    time.monotonic() - total_started,
                    path,
                )
            return

        total_elapsed = time.monotonic() - total_started
        io_after = _kvio_proc_io_snapshot()
        logger.debug(
            f"Disk read size: {size} bytes, "
            f"Bandwidth: {size / max(total_elapsed, 1e-9) / 1e6:.2f} MB/s"
        )
        if _kvio_trace_enabled():
            logger.warning(
                "[KVIO_FILE_READ] pid=%d lookup_id=%s file_index=%d/%d "
                "key_hash=%s bytes_expected=%d bytes_read=%d "
                "open=%.6f read=%.6f total=%.6f bandwidth_MBps=%.3f "
                "odirect=%s proc_io_delta=%s thread=%s path=%s",
                os.getpid(),
                lookup_id,
                file_index,
                total_files,
                getattr(key, "chunk_hash", "unknown"),
                size,
                bytes_read,
                open_elapsed,
                read_elapsed,
                total_elapsed,
                bytes_read / max(read_elapsed, 1e-9) / 1e6,
                effective_odirect,
                _kvio_io_delta(io_before, io_after),
                threading.current_thread().name,
                path,
            )

    def get_allocator_backend(self) -> LocalCPUBackend:
        return self.local_cpu_backend

    def close(self) -> None:
        if self.batched_msg_sender is not None:
            self.batched_msg_sender.close()
        self.disk_worker.close()
