# SPDX-License-Identifier: Apache-2.0

# Standard
import threading

# Third Party
import pytest
import torch

pytest.importorskip("lmcache.lmcache_native")

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.bitmap_ops import fold_unfold_ranked
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_manager import StorageManager


class _EventBus:
    def publish(self, _event) -> None:
        pass


class _FakeLocalManager:
    def __init__(self, readable=()):
        self.readable = set(readable)
        self.locked: set[ObjectKey] = set()
        self.finished: list[ObjectKey] = []
        self.touched: list[ObjectKey] = []

    @property
    def enabled(self) -> bool:
        return True

    def reserve_read(self, keys, read_locks=1):
        del read_locks
        out = {}
        for key in keys:
            if key in self.readable:
                self.locked.add(key)
                out[key] = (L1Error.SUCCESS, object())
            else:
                out[key] = (L1Error.KEY_NOT_EXIST, None)
        return out

    def unsafe_read(self, keys):
        return {
            key: (
                (L1Error.SUCCESS, object())
                if key in self.locked
                else (L1Error.KEY_NOT_READABLE, None)
            )
            for key in keys
        }

    def has_read_lock(self, key):
        return key in self.locked

    def finish_read(self, keys, read_locks=1):
        del read_locks
        out = {}
        for key in keys:
            if key in self.locked:
                self.locked.remove(key)
                self.finished.append(key)
                out[key] = L1Error.SUCCESS
            else:
                out[key] = L1Error.KEY_IN_WRONG_STATE
        return out

    def touch_keys(self, keys) -> None:
        self.touched.extend(key for key in keys if key in self.readable)


class _FakePrefetchController:
    def __init__(self):
        self.spec = None
        self.lookup_bitmap = None
        self.final_bitmap = None

    def submit_prefetch_request(self, spec):
        self.spec = spec
        return 17

    def peek_lookup_result_bitmap(self, request_id):
        assert request_id == 17
        return self.lookup_bitmap

    def query_prefetch_result(self, request_id):
        assert request_id == 17
        result = self.final_bitmap
        self.final_bitmap = None
        return result


def _key(chunk: int, rank: int = 0) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk),
        model_name="l0-union-test",
        kv_rank=rank,
    )


def _spec(keys: list[ObjectKey], world_size: int = 1) -> PrefetchRequestSpec:
    return PrefetchRequestSpec(
        keys=keys,
        group_layout_descs={
            0: MemoryLayoutDesc(
                shapes=[torch.Size([64])],
                dtypes=[torch.uint8],
            )
        },
        policy=TrimPolicy.PREFIX,
        attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1], world_size=world_size),
    )


def _storage_manager(l0_hits, l1_hits, *, with_l2: bool):
    sm = StorageManager.__new__(StorageManager)
    sm._l0_manager = _FakeLocalManager(l0_hits)
    sm._l1_manager = _FakeLocalManager(l1_hits)
    sm._event_bus = _EventBus()
    sm._prefetch_controller = _FakePrefetchController()
    sm._adapters_lock = threading.Lock()
    sm._l2_adapters = {0: object()} if with_l2 else {}
    return sm


def _prefix_chunks(found: Bitmap, num_chunks: int, world_size: int = 1) -> int:
    hit, _ = fold_unfold_ranked(
        found,
        num_chunks,
        world_size,
        [-1],
    )
    return hit


def test_l0_l1_hole_truncates_global_prefix_and_releases_later_locks() -> None:
    keys = [_key(i) for i in range(4)]
    sm = _storage_manager(
        l0_hits={keys[0], keys[3]},
        l1_hits={keys[2]},
        with_l2=False,
    )

    handle = sm._submit_prefix_l0_union(_spec(keys), "req", skip_l2=False)
    assert handle.l1_hit_chunks == 1

    found = sm.query_prefetch_status(handle)
    assert found is not None
    assert _prefix_chunks(found, 4) == 1
    assert keys[3] in sm._l0_manager.finished
    assert keys[2] in sm._l1_manager.finished
    assert keys[0] in sm._l0_manager.locked


def test_l0_l2_l1_l0_union_reports_full_prefix_before_l2_load_finishes() -> None:
    keys = [_key(i) for i in range(4)]
    sm = _storage_manager(
        l0_hits={keys[0], keys[3]},
        l1_hits={keys[2]},
        with_l2=True,
    )

    handle = sm._submit_prefix_l0_union(_spec(keys), "req", skip_l2=False)
    assert sm._prefetch_controller.spec is not None
    assert sm._prefetch_controller.spec.policy is TrimPolicy.SPARSE
    assert sm._prefetch_controller.spec.keys == [keys[1]]

    lookup = Bitmap(1)
    lookup.set(0)
    sm._prefetch_controller.lookup_bitmap = lookup
    assert sm.query_prefetch_lookup_hits(handle) == 4

    # Model the completed L2 -> temporary-L1 staging before final status.
    sm._l1_manager.readable.add(keys[1])
    sm._l1_manager.locked.add(keys[1])
    final = Bitmap(1)
    final.set(0)
    sm._prefetch_controller.final_bitmap = final

    found = sm.query_prefetch_status(handle)
    assert found is not None
    assert _prefix_chunks(found, 4) == 4

    result = sm._unsafe_read_prefetched(keys)
    assert all(
        err == L1Error.SUCCESS and obj is not None
        for err, obj in result.values()
    )


def test_tp_asymmetry_is_a_logical_l0_miss() -> None:
    # Flat order is chunk-major, then rank. Chunk 0 has only rank 0; chunk 1
    # has both ranks. The missing rank-1 shard of chunk 0 must make the whole
    # logical prefix miss rather than exposing a partial L0 hit.
    keys = [_key(0, 0), _key(0, 1), _key(1, 0), _key(1, 1)]
    sm = _storage_manager(
        l0_hits={keys[0], keys[2], keys[3]},
        l1_hits=set(),
        with_l2=False,
    )

    handle = sm._submit_prefix_l0_union(
        _spec(keys, world_size=2), "req", skip_l2=False
    )
    assert handle.l1_hit_chunks == 0
    found = sm.query_prefetch_status(handle)
    assert found is not None
    assert _prefix_chunks(found, 2, world_size=2) == 0
