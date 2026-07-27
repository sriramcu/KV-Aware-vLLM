# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import os
import threading
import time

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.async_lookup_message import (
    LookupCleanupMsg,
    LookupRequestMsg,
    LookupResponseMsg,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc_utils import (
    get_zmq_context,
    get_zmq_rpc_path_lmcache,
    get_zmq_socket,
)

logger = init_logger(__name__)


def _kvio_trace_enabled() -> bool:
    return os.environ.get("SRIRAM_KV_IO_TRACE", "0") == "1"


# NOTE(Jiayi): Prefetch could load extra redundant cache if multiple
# workers has different hit tokens.
class LMCacheAsyncLookupClient(LookupClientInterface):
    """
    ZMQ-based lookup client that communicates with a lookup server.

    Related extra_config:
    - lookup_server_worker_ids:
        is a config to control create lookup server on some workers.
        if mla is not enabled, default is [];
        if mla is enabled, default is [0];
        - if lookup_server_worker_ids is [], start lookup server on all workers
        - if lookup_server_worker_ids is [0], start lookup server on worker0
        - if lookup_server_worker_ids is [0, 3, 6], start lookup server on
          worker0, worker3 and worker6
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        # lookup_id -> first lookup time
        # this helps us support timeout semantics
        self.first_lookup_time: dict[str, float] = {}
        # Observability-only timestamps/counters. They do not participate in
        # timeout, cleanup, or scheduling decisions.
        self._kvio_lookup_start_mono: dict[str, float] = {}
        self._kvio_poll_count: dict[str, int] = {}
        self.config = config

        self.ctx = get_zmq_context(use_asyncio=False)
        kv_connector_extra_config = metadata.kv_connector_extra_config or {}
        rpc_port = kv_connector_extra_config.get("lmcache_rpc_port", 0)
        engine_id = metadata.engine_id
        assert engine_id is not None, "engine_id is required for RPC communication"
        self.world_size = metadata.world_size
        self.lookup_server_worker_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )

        self.push_sockets = []
        if len(self.lookup_server_worker_ids) > 0:
            ranks = self.lookup_server_worker_ids
            self.world_size = len(self.lookup_server_worker_ids)
        else:
            ranks = [i for i in range(self.world_size)]

        for rank in ranks:
            worker_socket_path = get_zmq_rpc_path_lmcache(
                engine_id, "lookup_worker", rpc_port, rank
            )
            logger.info(
                "lmcache lookup client connect to rank %s with worker socket path %s",
                rank,
                worker_socket_path,
            )

            push_socket = get_zmq_socket(
                self.ctx,
                worker_socket_path,
                "ipc",
                zmq.PUSH,  # type: ignore[attr-defined]
                "connect",
            )

            self.push_sockets.append(push_socket)

        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            engine_id, "lookup_scheduler", rpc_port, 0
        )
        self.pull_socket = get_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            "ipc",
            zmq.PULL,  # type: ignore[attr-defined]
            "bind",
        )
        logger.info(
            "lmcache lookup client connect to scheduler with socket path %s",
            scheduler_socket_path,
        )

        # First Party
        from lmcache.v1.token_database import (
            ChunkedTokenDatabase,
            SegmentTokenDatabase,
            TokenDatabase,
        )

        self.token_database: TokenDatabase
        if config.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)

        # A lock is needed since we need another thread to pull
        # responses from the lookup_and_prefetch server
        # (e.g., worker process).
        self.lock = threading.Lock()

        # map from lookup_id (i.e., req_id) to req's status.
        # None indicates ongoing.
        # int indicates number of hit tokens.
        self.reqs_status: dict[str, Optional[int]] = {}

        # map from lookup_id (i.e., req_id) to number of hit tokens for each worker
        self.res_for_each_worker: dict[str, list[int]] = {}

        # The two parts are [lookup_id (i.e., req_id), num_hit_tokens]
        self.num_parts = 2

        # Track lookup_ids that have been aborted for cleanup
        self.aborted_lookups: set[str] = set()

        self.running = True

        self.thread = threading.Thread(
            target=self.process_responses_from_workers,
            daemon=True,
            name="async-lookup-client-thread",
        )
        self.thread.start()

        # default backoff time
        self.lookup_backoff_time = 0.01
        if config.extra_config is not None:
            self.lookup_backoff_time = float(
                config.extra_config.get("lookup_backoff_time", self.lookup_backoff_time)
            )

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        -1 means not found;
        None means ongoing;
        int >= 0 means number of hit tokens
        """
        # Check if any aborted lookups are finished, send cleanup messages
        self._cleanup_finished_aborted_lookups()

        with self.lock:
            if (req_status := self.reqs_status.get(lookup_id, -1)) == -1:
                self.reqs_status[lookup_id] = None
                self.first_lookup_time[lookup_id] = time.time()
                self._kvio_lookup_start_mono.setdefault(lookup_id, time.monotonic())
                self._kvio_poll_count[lookup_id] = 1
                if _kvio_trace_enabled():
                    logger.warning(
                        "[KVIO_LOOKUP_STATUS_CREATE] lookup_id=%s "
                        "mono=%.6f pending=%d aborted=%d",
                        lookup_id,
                        self._kvio_lookup_start_mono[lookup_id],
                        sum(v is None for v in self.reqs_status.values()),
                        len(self.aborted_lookups),
                    )
            elif req_status is None:
                self._kvio_poll_count[lookup_id] = (
                    self._kvio_poll_count.get(lookup_id, 0) + 1
                )
                time.sleep(self.lookup_backoff_time)
                if (
                    time.time() - self.first_lookup_time[lookup_id]
                ) * 1000 > self.config.lookup_timeout_ms:
                    age = time.monotonic() - self._kvio_lookup_start_mono.get(
                        lookup_id, time.monotonic()
                    )
                    logger.warning(
                        (
                            "Request %s is still waiting for async lookup "
                            "after %d seconds, returning 0 lmcache cached tokens "
                            "so vllm can recompute"
                        ),
                        lookup_id,
                        self.config.lookup_timeout_ms // 1000,
                    )
                    if _kvio_trace_enabled():
                        logger.warning(
                            "[KVIO_LOOKUP_TIMEOUT] lookup_id=%s age=%.6f "
                            "polls=%d pending=%d aborted_before=%d "
                            "partial_worker_responses=%d/%d",
                            lookup_id,
                            age,
                            self._kvio_poll_count.get(lookup_id, 0),
                            sum(v is None for v in self.reqs_status.values()),
                            len(self.aborted_lookups),
                            len(self.res_for_each_worker.get(lookup_id, [])),
                            self.world_size,
                        )
                    self.cancel_lookup(lookup_id)
                    self.first_lookup_time.pop(lookup_id, None)
                    return 0

            return req_status

    # TODO(Jiayi): Consider batching here
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        hashes: list[int] = []
        offsets = []
        for start, end, hash_val in self.token_database.process_tokens(
            token_ids, make_key=False
        ):
            hashes.append(hash_val)  # type: ignore[arg-type]
            offsets.append(end - start)

        # Create structured message
        msg = LookupRequestMsg(
            lookup_id=lookup_id,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        )

        # Serialize message using msgspec
        msg_buf = msgspec.msgpack.encode(msg)

        self._kvio_lookup_start_mono.setdefault(lookup_id, time.monotonic())
        if _kvio_trace_enabled():
            logger.warning(
                "[KVIO_LOOKUP_REQUEST_SEND] lookup_id=%s hashes=%d "
                "offset_tokens=%d workers=%d msg_bytes=%d mono=%.6f",
                lookup_id,
                len(hashes),
                sum(offsets),
                self.world_size,
                len(msg_buf),
                self._kvio_lookup_start_mono[lookup_id],
            )

        for i in range(self.world_size):
            self.push_sockets[i].send(msg_buf, copy=False)
        time.sleep(self.lookup_backoff_time)
        return None

    def process_responses_from_workers(self):
        while self.running:
            try:
                msg_buf = self.pull_socket.recv(copy=False)
                # Deserialize message using msgspec
                msg = msgspec.msgpack.decode(msg_buf, type=LookupResponseMsg)
                lookup_id = msg.lookup_id
                res = msg.num_hit_tokens

                with self.lock:
                    if lookup_id not in self.res_for_each_worker:
                        self.res_for_each_worker[lookup_id] = [res]
                    else:
                        self.res_for_each_worker[lookup_id].append(res)
                    all_res = self.res_for_each_worker[lookup_id]

                    if _kvio_trace_enabled():
                        logger.warning(
                            "[KVIO_LOOKUP_WORKER_RESPONSE] lookup_id=%s "
                            "response_index=%d/%d tokens=%d age=%.6f aborted=%s",
                            lookup_id,
                            len(all_res),
                            self.world_size,
                            res,
                            time.monotonic() - self._kvio_lookup_start_mono.get(
                                lookup_id, time.monotonic()
                            ),
                            lookup_id in self.aborted_lookups,
                        )

                    if len(all_res) == self.world_size:
                        self.res_for_each_worker.pop(lookup_id)

                        # NOTE: it is possible that the number of hit
                        # tokens is different across (TP and PP) ranks, so we
                        # can use the minimum value as the number of
                        # hit tokens.
                        self.reqs_status[lookup_id] = min(all_res)
                        if _kvio_trace_enabled():
                            logger.warning(
                                "[KVIO_LOOKUP_ALL_RESPONSES] lookup_id=%s "
                                "worker_tokens=%s min_tokens=%d age=%.6f "
                                "aborted=%s",
                                lookup_id,
                                all_res,
                                min(all_res),
                                time.monotonic()
                                - self._kvio_lookup_start_mono.get(
                                    lookup_id, time.monotonic()
                                ),
                                lookup_id in self.aborted_lookups,
                            )

            except Exception as e:
                logger.error("Error processing response from worker: %s", e)

    def clear_lookup_status(self, lookup_id: str) -> None:
        with self.lock:
            if _kvio_trace_enabled():
                logger.warning(
                    "[KVIO_LOOKUP_STATUS_CLEAR] lookup_id=%s age=%.6f "
                    "status=%s aborted=%s",
                    lookup_id,
                    time.monotonic()
                    - self._kvio_lookup_start_mono.get(
                        lookup_id, time.monotonic()
                    ),
                    self.reqs_status.get(lookup_id),
                    lookup_id in self.aborted_lookups,
                )
            self.reqs_status.pop(lookup_id, None)
            self.first_lookup_time.pop(lookup_id, None)
            self._kvio_lookup_start_mono.pop(lookup_id, None)
            self._kvio_poll_count.pop(lookup_id, None)

    def cancel_lookup(self, lookup_id: str) -> None:
        """Mark lookup as aborted. Cleanup will happen after task finishes."""
        if _kvio_trace_enabled():
            logger.warning(
                "[KVIO_LOOKUP_MARK_ABORTED] lookup_id=%s age=%.6f "
                "aborted_before=%d",
                lookup_id,
                time.monotonic()
                - self._kvio_lookup_start_mono.get(
                    lookup_id, time.monotonic()
                ),
                len(self.aborted_lookups),
            )
        self.aborted_lookups.add(lookup_id)

    def _cleanup_finished_aborted_lookups(self) -> None:
        """Check for finished aborted lookups and send cleanup messages to workers."""
        # A lookup whose status is None is still loading.
        # We wait for it to finish before cleanup.
        finished_lookups = [
            lookup_id
            for lookup_id in self.aborted_lookups
            if self.reqs_status.get(lookup_id) is not None
        ]
        if finished_lookups:
            if _kvio_trace_enabled():
                logger.warning(
                    "[KVIO_LOOKUP_CLEANUP_READY] lookup_ids=%s "
                    "remaining_aborted_before=%d",
                    finished_lookups,
                    len(self.aborted_lookups),
                )
            self.aborted_lookups.difference_update(finished_lookups)

        # Tell the server to free the reserved memory buffers for each aborted lookup.
        for lookup_id in finished_lookups:
            self._send_cleanup_message(lookup_id)
            self.clear_lookup_status(lookup_id)

    def _send_cleanup_message(self, lookup_id: str) -> None:
        """Send cleanup message to workers to release memory objects."""
        msg = LookupCleanupMsg(lookup_id=lookup_id)
        msg_buf = msgspec.msgpack.encode(msg)

        for i in range(self.world_size):
            self.push_sockets[i].send(msg_buf, copy=False)
        logger.debug("Sent cleanup message for lookup_id=%s", lookup_id)
        if _kvio_trace_enabled():
            logger.warning(
                "[KVIO_LOOKUP_CLEANUP_SEND] lookup_id=%s workers=%d age=%.6f",
                lookup_id,
                self.world_size,
                time.monotonic()
                - self._kvio_lookup_start_mono.get(
                    lookup_id, time.monotonic()
                ),
            )

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports producer kvcache reuse"""
        return True

    def close(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            for s in self.push_sockets:
                s.close(linger=0)  # type: ignore[arg-type]
            self.pull_socket.close(linger=0)  # type: ignore[arg-type]
            self.ctx.term()
        except Exception as e:
            logger.warning("Failed to join thread during close: %s", e)


class LMCacheAsyncLookupServer:
    """ZMQ-based async lookup server that handles lookup and prefetch
    requests using LMCacheEngine."""

    def __init__(
        self,
        lmcache_engine: LMCacheEngine,
        metadata: LMCacheMetadata,
    ):
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        kv_connector_extra_config = metadata.kv_connector_extra_config or {}
        rpc_port = kv_connector_extra_config.get("lmcache_rpc_port", 0)
        assert metadata.engine_id is not None, (
            "engine_id is required for RPC communication"
        )
        worker_socket_path = get_zmq_rpc_path_lmcache(
            metadata.engine_id, "lookup_worker", rpc_port, metadata.worker_id
        )
        scheduler_socket_path = get_zmq_rpc_path_lmcache(
            metadata.engine_id, "lookup_scheduler", rpc_port, 0
        )
        self.push_socket = get_zmq_socket(
            self.ctx,
            scheduler_socket_path,
            "ipc",
            zmq.PUSH,  # type: ignore[attr-defined]
            "connect",
        )
        self.pull_socket = get_zmq_socket(
            self.ctx,
            worker_socket_path,
            "ipc",
            zmq.PULL,  # type: ignore[attr-defined]
            "bind",
        )

        self.lmcache_engine = lmcache_engine
        self.running = True

        logger.info(
            "lmcache lookup server start with"
            " scheduler socket path %s, "
            "worker socket path %s",
            scheduler_socket_path,
            worker_socket_path,
        )
        self.thread = threading.Thread(
            target=self.process_requests_from_scheduler,
            daemon=True,
            name="async-lookup-server-thread",
        )
        self.thread.start()

    def process_requests_from_scheduler(self):
        while self.running:
            try:
                msg_buf = self.pull_socket.recv(copy=False)
                # rely on msgspec to automatically discriminate
                # between LookupRequestMsg and LookupCleanupMsg
                msg = msgspec.msgpack.decode(
                    msg_buf,
                    type=Union[LookupRequestMsg, LookupCleanupMsg],
                )

                if isinstance(msg, LookupRequestMsg):
                    # Handle lookup request
                    self.lmcache_engine.async_lookup_and_prefetch(
                        lookup_id=msg.lookup_id,
                        hashes=msg.hashes,
                        offsets=msg.offsets,
                        pin=True,
                        request_configs=msg.request_configs,
                    )

                elif isinstance(msg, LookupCleanupMsg):
                    # Handle cleanup request - release memory objects for aborted lookup
                    self.lmcache_engine.cleanup_memory_objs(msg.lookup_id)

                else:
                    logger.warning("Unknown message type: %s", type(msg))

            except Exception as e:
                logger.error("Error processing request from scheduler: %s", e)

    def send_response_to_scheduler(self, lookup_id: str, num_hit_tokens: int):
        # Create structured response message
        msg = LookupResponseMsg(
            lookup_id=lookup_id,
            num_hit_tokens=num_hit_tokens,
        )

        # Serialize message using msgspec
        msg_buf = msgspec.msgpack.encode(msg)
        self.push_socket.send(msg_buf, copy=False)

    def close(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            self.push_socket.close(linger=0)  # type: ignore[arg-type]
            self.pull_socket.close(linger=0)  # type: ignore[arg-type]
            self.ctx.term()
        except Exception as e:
            logger.warning("Failed to join thread during close: %s", e)
