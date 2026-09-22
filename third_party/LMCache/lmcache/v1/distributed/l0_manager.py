# SPDX-License-Identifier: Apache-2.0
"""Persistent GPU-resident L0 object manager for multiprocess LMCache.

L0 is deliberately local to the MP storage server.  It owns GPU memory that is
separate from vLLM's execution/prefix-cache pool; hits are copied into vLLM
owned KV blocks before attention executes.

The v1 implementation is intentionally small:
* one fixed-size paged arena per KV rank/device;
* chunk-granularity LRU eviction (all object-group/rank pieces of a logical
  chunk are evicted together when possible);
* L1-like read/write lifetime locks;
* temporary state is represented, but defaults to disabled.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from dataclasses import dataclass
import os
import threading

# Third Party
import torch
from opentelemetry import metrics

# First Party
from lmcache.lmcache_native import TTLLock
from lmcache.logging import init_logger
from lmcache.utils import get_size_bytes
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L0ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.mp_observability.otel_init import register_gauge

logger = init_logger(__name__)

_MAX_READ_LOCK_COUNT = 128


@dataclass
class _L0ObjectState:
    memory_obj: MemoryObj
    write_lock: TTLLock
    read_lock: TTLLock
    is_temporary: bool
    published: bool = False

    def readable(self) -> bool:
        return self.published and not self.write_lock.is_locked()

    def evictable(self) -> bool:
        return not self.write_lock.is_locked() and not self.read_lock.is_locked()


@dataclass
class _L0Arena:
    kv_rank: int
    device: torch.device
    allocator: GPUMemoryAllocator
    page_bytes: int
    configured_bytes: int
    usable_bytes: int


# LRU is keyed at logical LMCache-chunk granularity. Object group / TP rank are
# intentionally omitted so one victim means one logical cached chunk.
_ChunkIdentity = tuple[bytes, str, str]


def _chunk_identity(key: ObjectKey) -> _ChunkIdentity:
    return (key.chunk_hash, key.model_name, key.cache_salt)


def _clamp_read_locks(read_locks: int) -> int:
    if read_locks < 1:
        return 1
    return min(read_locks, _MAX_READ_LOCK_COUNT)


class L0Manager:
    """Fixed-capacity persistent GPU tier owned by LMCache.

    The manager is safe to construct while disabled. GPU arenas are allocated
    lazily, only after a placement path explicitly prepares a KV rank for L0.
    Applying the L0 patch therefore has no GPU-memory effect by itself.
    """

    _gauge_registered = False
    _gauge_target: "L0Manager | None" = None

    def __init__(self, config: L0ManagerConfig):
        self._config = config
        self._lock = threading.RLock()
        self._objects: dict[ObjectKey, _L0ObjectState] = {}
        self._arenas: dict[int, _L0Arena] = {}
        self._lru: OrderedDict[_ChunkIdentity, None] = OrderedDict()
        self._temp_objects = 0
        self._temp_bytes_by_rank: dict[int, int] = {}

        meter = metrics.get_meter("lmcache.l0_persistent")
        self._store_counter = meter.create_counter(
            "lmcache_mp.l0_stores_total",
            description="L0 objects published after a completed GPU store",
        )
        self._hit_counter = meter.create_counter(
            "lmcache_mp.l0_read_hits_total",
            description="L0 object read reservations that succeeded",
        )
        self._miss_counter = meter.create_counter(
            "lmcache_mp.l0_read_misses_total",
            description="L0 object read reservations that missed or were unreadable",
        )
        self._eviction_counter = meter.create_counter(
            "lmcache_mp.l0_evictions_total",
            description="Logical L0 chunks evicted by the local LRU",
        )
        self._admission_failure_counter = meter.create_counter(
            "lmcache_mp.l0_admission_failures_total",
            description="L0 object admissions that failed for capacity/layout reasons",
        )
        self._temp_counter = meter.create_counter(
            "lmcache_mp.l0_temp_allocations_total",
            description="Temporary L0 allocations (expected to remain zero in v1)",
        )

        L0Manager._gauge_target = self
        if not L0Manager._gauge_registered:
            L0Manager._gauge_registered = True
            register_gauge(
                "lmcache.l0_persistent",
                "lmcache_mp.l0_memory_usage_bytes",
                "Bytes physically occupied by persistent LMCache L0 pages",
                lambda: (
                    L0Manager._gauge_target.get_memory_usage()[0]
                    if L0Manager._gauge_target is not None
                    else 0
                ),
            )
            register_gauge(
                "lmcache.l0_persistent",
                "lmcache_mp.l0_capacity_bytes",
                "Usable bytes in all initialized LMCache L0 rank-local arenas",
                lambda: (
                    L0Manager._gauge_target.get_memory_usage()[1]
                    if L0Manager._gauge_target is not None
                    else 0
                ),
            )
            register_gauge(
                "lmcache.l0_persistent",
                "lmcache_mp.l0_temp_objects",
                "Current temporary L0 objects (expected zero in v1)",
                lambda: (
                    L0Manager._gauge_target.temp_object_count
                    if L0Manager._gauge_target is not None
                    else 0
                ),
            )

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self._config.capacity_bytes > 0

    @property
    def temp_object_count(self) -> int:
        with self._lock:
            return self._temp_objects

    def prepare_rank_arena(
        self,
        kv_rank: int,
        device: torch.device | str,
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> bool:
        """Create the fixed Q arena for one KV rank if it does not exist.

        The page size is the largest object-group layout registered for the
        model. Smaller groups occupy one page and leave the tail unused. This
        keeps one allocator/arena per rank instead of multiplying Q by the
        number of object groups.
        """
        if not self.enabled:
            return False
        if not group_layout_descs:
            return False

        with self._lock:
            dev = torch.device(device)
            existing = self._arenas.get(kv_rank)
            max_desc = max(
                group_layout_descs.values(),
                key=lambda d: get_size_bytes(d.shapes, d.dtypes),
            )
            page_bytes = get_size_bytes(max_desc.shapes, max_desc.dtypes)
            if page_bytes <= 0:
                return False

            if existing is not None:
                if existing.device != dev:
                    raise ValueError(
                        f"L0 kv_rank={kv_rank} already bound to {existing.device}, "
                        f"cannot rebind to {dev}"
                    )
                if page_bytes > existing.page_bytes:
                    logger.error(
                        "L0 rank %d was initialized with page_bytes=%d but a "
                        "larger layout (%d bytes) was later registered; L0 "
                        "admission for that layout will fail closed",
                        kv_rank,
                        existing.page_bytes,
                        page_bytes,
                    )
                return True

            usable = (self._config.capacity_bytes // page_bytes) * page_bytes
            if usable <= 0:
                logger.warning(
                    "L0 rank %d disabled effectively: configured Q=%d bytes is "
                    "smaller than one %d-byte LMCache page",
                    kv_rank,
                    self._config.capacity_bytes,
                    page_bytes,
                )
                return False

            allocator = GPUMemoryAllocator(
                size=usable,
                device=dev,
                use_paging=True,
                shapes=max_desc.shapes,
                dtypes=max_desc.dtypes,
                fmt=MemoryFormat.KV_2LTD,
            )
            self._arenas[kv_rank] = _L0Arena(
                kv_rank=kv_rank,
                device=dev,
                allocator=allocator,
                page_bytes=page_bytes,
                configured_bytes=self._config.capacity_bytes,
                usable_bytes=usable,
            )
            logger.info(
                "Initialized LMCache L0 arena rank=%d device=%s Q=%d usable=%d "
                "page_bytes=%d pages=%d",
                kv_rank,
                dev,
                self._config.capacity_bytes,
                usable,
                page_bytes,
                usable // page_bytes,
            )
            return True

    def _touch_locked(self, key: ObjectKey) -> None:
        cid = _chunk_identity(key)
        if cid in self._lru:
            self._lru.move_to_end(cid)
        else:
            self._lru[cid] = None

    def _free_entry_locked(self, key: ObjectKey, entry: _L0ObjectState) -> None:
        arena = self._arenas.get(key.kv_rank)
        if arena is None:
            return
        if entry.is_temporary:
            self._temp_objects = max(0, self._temp_objects - 1)
            self._temp_bytes_by_rank[key.kv_rank] = max(
                0,
                self._temp_bytes_by_rank.get(key.kv_rank, 0)
                - entry.memory_obj.get_physical_size(),
            )
        arena.allocator.free(entry.memory_obj)

    def _evict_one_chunk_locked(self, target_rank: int) -> bool:
        """Evict the oldest fully-unlocked logical chunk touching target_rank."""
        for cid in list(self._lru.keys()):
            chunk_items = [
                (key, entry)
                for key, entry in self._objects.items()
                if _chunk_identity(key) == cid
            ]
            if not any(key.kv_rank == target_rank for key, _ in chunk_items):
                continue
            if not chunk_items or any(
                not entry.evictable() for _, entry in chunk_items
            ):
                continue

            for key, entry in chunk_items:
                self._free_entry_locked(key, entry)
                del self._objects[key]
            self._lru.pop(cid, None)
            self._eviction_counter.add(1)
            if os.getenv("LMCACHE_L0_SMOKE_TRACE_D2D", "0") == "1":
                logger.info(
                    "[L0_SMOKE_EVICT] target_rank=%d chunk=%s pieces=%d",
                    target_rank,
                    cid[0].hex()[:16],
                    len(chunk_items),
                )
            return True
        return False

    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        *,
        is_temporary: bool = False,
    ) -> dict[ObjectKey, tuple[L1Error, MemoryObj | None]]:
        """Reserve L0 pages for a write; capacity pressure uses local LRU."""
        with self._lock:
            ret: dict[ObjectKey, tuple[L1Error, MemoryObj | None]] = {}
            requested_bytes = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)

            for key in keys:
                entry = self._objects.get(key)
                # L0 v1 is first-assignment-wins: existing objects are never
                # updated/promoted in place. A later placement decision can only
                # establish a new L0 owner after the old object is genuinely gone.
                if entry is not None:
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue

                arena = self._arenas.get(key.kv_rank)
                if arena is None or requested_bytes > arena.page_bytes:
                    self._admission_failure_counter.add(1)
                    ret[key] = (L1Error.OUT_OF_MEMORY, None)
                    continue

                if is_temporary:
                    next_temp_bytes = (
                        self._temp_bytes_by_rank.get(key.kv_rank, 0) + arena.page_bytes
                    )
                    if (
                        self._config.temp_max_in_flight <= 0
                        or self._config.temp_capacity_bytes <= 0
                        or self._temp_objects >= self._config.temp_max_in_flight
                        or next_temp_bytes > self._config.temp_capacity_bytes
                    ):
                        self._admission_failure_counter.add(1)
                        ret[key] = (L1Error.OUT_OF_MEMORY, None)
                        continue

                mem_obj = arena.allocator.allocate(
                    layout_desc.shapes,
                    layout_desc.dtypes,
                    MemoryFormat.KV_2LTD,
                )
                while mem_obj is None and self._evict_one_chunk_locked(key.kv_rank):
                    mem_obj = arena.allocator.allocate(
                        layout_desc.shapes,
                        layout_desc.dtypes,
                        MemoryFormat.KV_2LTD,
                    )

                if mem_obj is None:
                    self._admission_failure_counter.add(1)
                    ret[key] = (L1Error.OUT_OF_MEMORY, None)
                    continue

                state = _L0ObjectState(
                    memory_obj=mem_obj,
                    write_lock=TTLLock(self._config.write_ttl_seconds),
                    read_lock=TTLLock(self._config.read_ttl_seconds),
                    is_temporary=is_temporary,
                    published=False,
                )
                state.write_lock.lock()
                self._objects[key] = state
                self._touch_locked(key)
                if is_temporary:
                    self._temp_objects += 1
                    self._temp_bytes_by_rank[key.kv_rank] = (
                        self._temp_bytes_by_rank.get(key.kv_rank, 0) + arena.page_bytes
                    )
                    self._temp_counter.add(1)
                ret[key] = (L1Error.SUCCESS, mem_obj)
            return ret

    def finish_write(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Publish completed L0 writes. Call only after the D2D stream completes."""
        with self._lock:
            ret: dict[ObjectKey, L1Error] = {}
            for key in keys:
                entry = self._objects.get(key)
                if entry is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                if not entry.write_lock.is_locked() or entry.read_lock.is_locked():
                    ret[key] = L1Error.KEY_IN_WRONG_STATE
                    continue
                entry.write_lock.unlock()
                entry.published = True
                self._touch_locked(key)
                self._store_counter.add(1)
                ret[key] = L1Error.SUCCESS
            return ret

    def abort_write(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Drop unpublished write reservations after a failed/cancelled store."""
        with self._lock:
            ret: dict[ObjectKey, L1Error] = {}
            for key in keys:
                entry = self._objects.get(key)
                if entry is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                if entry.published or not entry.write_lock.is_locked():
                    ret[key] = L1Error.KEY_IN_WRONG_STATE
                    continue
                self._free_entry_locked(key, entry)
                del self._objects[key]
                cid = _chunk_identity(key)
                if not any(_chunk_identity(k) == cid for k in self._objects):
                    self._lru.pop(cid, None)
                ret[key] = L1Error.SUCCESS
            return ret

    def reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, tuple[L1Error, MemoryObj | None]]:
        """Reserve readable L0 objects; unpublished writes never hit."""
        total = _clamp_read_locks(read_locks)
        with self._lock:
            ret: dict[ObjectKey, tuple[L1Error, MemoryObj | None]] = {}
            for key in keys:
                entry = self._objects.get(key)
                if entry is None:
                    self._miss_counter.add(1)
                    ret[key] = (L1Error.KEY_NOT_EXIST, None)
                    continue
                if not entry.readable():
                    self._miss_counter.add(1)
                    ret[key] = (L1Error.KEY_NOT_READABLE, None)
                    continue
                for _ in range(total):
                    entry.read_lock.lock()
                self._touch_locked(key)
                self._hit_counter.add(1)
                ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            return ret

    def unsafe_read(
        self, keys: list[ObjectKey]
    ) -> dict[ObjectKey, tuple[L1Error, MemoryObj | None]]:
        """Read already-reserved L0 objects without acquiring another lock."""
        with self._lock:
            ret: dict[ObjectKey, tuple[L1Error, MemoryObj | None]] = {}
            for key in keys:
                entry = self._objects.get(key)
                if entry is None:
                    ret[key] = (L1Error.KEY_NOT_EXIST, None)
                elif not entry.published or not entry.read_lock.is_locked():
                    ret[key] = (L1Error.KEY_NOT_READABLE, None)
                else:
                    ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            return ret

    def finish_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1Error]:
        total = _clamp_read_locks(read_locks)
        with self._lock:
            ret: dict[ObjectKey, L1Error] = {}
            to_delete: list[tuple[ObjectKey, _L0ObjectState]] = []
            for key in keys:
                entry = self._objects.get(key)
                if entry is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                if entry.write_lock.is_locked() or not entry.read_lock.is_locked():
                    ret[key] = L1Error.KEY_IN_WRONG_STATE
                    continue
                for _ in range(total):
                    if entry.read_lock.is_locked():
                        entry.read_lock.unlock()
                if entry.is_temporary and not entry.read_lock.is_locked():
                    to_delete.append((key, entry))
                ret[key] = L1Error.SUCCESS

            for key, entry in to_delete:
                self._free_entry_locked(key, entry)
                del self._objects[key]
                cid = _chunk_identity(key)
                if not any(_chunk_identity(k) == cid for k in self._objects):
                    self._lru.pop(cid, None)
            return ret

    def has_read_lock(self, key: ObjectKey) -> bool:
        with self._lock:
            entry = self._objects.get(key)
            return bool(entry is not None and entry.read_lock.is_locked())

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        with self._lock:
            for key in keys:
                if key in self._objects:
                    self._touch_locked(key)

    def clear(self, force: bool = False) -> None:
        with self._lock:
            victims = [
                (key, entry)
                for key, entry in self._objects.items()
                if force or entry.evictable()
            ]
            for key, entry in victims:
                self._free_entry_locked(key, entry)
                del self._objects[key]
            self._lru = OrderedDict(
                (cid, None)
                for cid in self._lru
                if any(_chunk_identity(key) == cid for key in self._objects)
            )

    def get_memory_usage(self) -> tuple[int, int]:
        with self._lock:
            used = sum(
                entry.memory_obj.get_physical_size() for entry in self._objects.values()
            )
            total = sum(arena.usable_bytes for arena in self._arenas.values())
            return used, total

    def report_status(self) -> dict:
        with self._lock:
            used, total = self.get_memory_usage()
            return {
                "is_healthy": True,
                "enabled": self.enabled,
                "configured_capacity_bytes_per_rank": self._config.capacity_bytes,
                "used_bytes": used,
                "initialized_capacity_bytes": total,
                "objects": len(self._objects),
                "logical_chunks": len(self._lru),
                "temp_objects": self._temp_objects,
                "temp_capacity_bytes": self._config.temp_capacity_bytes,
                "temp_max_in_flight": self._config.temp_max_in_flight,
                "ranks": {
                    rank: {
                        "device": str(arena.device),
                        "page_bytes": arena.page_bytes,
                        "configured_bytes": arena.configured_bytes,
                        "usable_bytes": arena.usable_bytes,
                    }
                    for rank, arena in self._arenas.items()
                },
            }

    def memcheck(self) -> bool:
        with self._lock:
            return all(arena.allocator.memcheck() for arena in self._arenas.values())

    def close(self) -> None:
        with self._lock:
            self.clear(force=True)
            self._arenas.clear()
