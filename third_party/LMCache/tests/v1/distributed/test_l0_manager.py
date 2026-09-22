# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

pytest.importorskip("lmcache.lmcache_native")

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L0ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l0_manager import L0Manager


def _layout(num_bytes: int = 64) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([num_bytes])], dtypes=[torch.uint8])


def _key(chunk: int, rank: int, group: int = 0) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk),
        model_name="l0-test-model",
        kv_rank=rank,
        object_group_id=group,
    )


def _manager(page_bytes: int = 64, pages_per_rank: int = 1) -> L0Manager:
    return L0Manager(
        L0ManagerConfig(
            enabled=True,
            capacity_bytes=page_bytes * pages_per_rank,
            write_ttl_seconds=60,
            read_ttl_seconds=60,
        )
    )


def _prepare(mgr: L0Manager, rank: int, page_bytes: int = 64) -> None:
    assert mgr.prepare_rank_arena(rank, "cpu", {0: _layout(page_bytes)})


def _store(mgr: L0Manager, key: ObjectKey, page_bytes: int = 64) -> None:
    result = mgr.reserve_write([key], _layout(page_bytes))
    assert result[key][0] == L1Error.SUCCESS
    assert result[key][1] is not None
    assert mgr.finish_write([key])[key] == L1Error.SUCCESS


def test_l0_disabled_and_q_zero_do_not_allocate() -> None:
    disabled = L0Manager(L0ManagerConfig())
    assert not disabled.enabled
    assert not disabled.prepare_rank_arena(0, "cpu", {0: _layout()})
    assert disabled.get_memory_usage() == (0, 0)

    q_zero = L0Manager(L0ManagerConfig(enabled=True, capacity_bytes=0))
    assert not q_zero.enabled
    assert not q_zero.prepare_rank_arena(0, "cpu", {0: _layout()})
    assert q_zero.get_memory_usage() == (0, 0)


def test_l0_write_is_not_visible_until_published() -> None:
    mgr = _manager()
    _prepare(mgr, 0)
    key = _key(1, 0)

    write_result = mgr.reserve_write([key], _layout())
    assert write_result[key][0] == L1Error.SUCCESS

    read_before_publish = mgr.reserve_read([key])
    assert read_before_publish[key][0] == L1Error.KEY_NOT_READABLE

    assert mgr.finish_write([key])[key] == L1Error.SUCCESS
    read_after_publish = mgr.reserve_read([key])
    assert read_after_publish[key][0] == L1Error.SUCCESS
    assert read_after_publish[key][1] is not None
    assert mgr.finish_read([key])[key] == L1Error.SUCCESS
    mgr.close()


def test_l0_lru_evicts_a_logical_chunk_across_rank_shards() -> None:
    mgr = _manager()
    _prepare(mgr, 0)
    _prepare(mgr, 1)

    a0 = _key(10, 0)
    a1 = _key(10, 1)
    b0 = _key(20, 0)
    _store(mgr, a0)
    _store(mgr, a1)

    # Rank 0 is full. Admitting B on rank 0 evicts logical chunk A, including
    # its rank-1 shard, rather than leaving a partial logical L0 residency.
    b_result = mgr.reserve_write([b0], _layout())
    assert b_result[b0][0] == L1Error.SUCCESS
    assert mgr.finish_write([b0])[b0] == L1Error.SUCCESS

    assert mgr.reserve_read([a0])[a0][0] == L1Error.KEY_NOT_EXIST
    assert mgr.reserve_read([a1])[a1][0] == L1Error.KEY_NOT_EXIST
    assert mgr.reserve_read([b0])[b0][0] == L1Error.SUCCESS
    assert mgr.finish_read([b0])[b0] == L1Error.SUCCESS
    mgr.close()


def test_l0_full_with_pinned_victim_fails_admission_cleanly() -> None:
    mgr = _manager()
    _prepare(mgr, 0)
    a = _key(1, 0)
    b = _key(2, 0)
    _store(mgr, a)

    assert mgr.reserve_read([a])[a][0] == L1Error.SUCCESS
    b_result = mgr.reserve_write([b], _layout())
    assert b_result[b][0] == L1Error.OUT_OF_MEMORY
    assert b_result[b][1] is None

    # The pinned object remains usable and can be released normally.
    assert mgr.unsafe_read([a])[a][0] == L1Error.SUCCESS
    assert mgr.finish_read([a])[a] == L1Error.SUCCESS

    # Once unpinned, a new admission may evict A.
    b_result = mgr.reserve_write([b], _layout())
    assert b_result[b][0] == L1Error.SUCCESS
    assert mgr.finish_write([b])[b] == L1Error.SUCCESS
    mgr.close()


def test_l0_abort_drops_unpublished_reservation() -> None:
    mgr = _manager()
    _prepare(mgr, 0)
    key = _key(7, 0)

    assert mgr.reserve_write([key], _layout())[key][0] == L1Error.SUCCESS
    assert mgr.abort_write([key])[key] == L1Error.SUCCESS
    assert mgr.reserve_read([key])[key][0] == L1Error.KEY_NOT_EXIST
    assert mgr.get_memory_usage()[0] == 0
    mgr.close()


def test_temp_l0_is_dormant_by_default() -> None:
    mgr = _manager()
    _prepare(mgr, 0)
    key = _key(99, 0)

    result = mgr.reserve_write([key], _layout(), is_temporary=True)
    assert result[key][0] == L1Error.OUT_OF_MEMORY
    assert mgr.temp_object_count == 0
    mgr.close()
