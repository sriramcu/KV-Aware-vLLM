# SPDX-License-Identifier: Apache-2.0
"""CPU-only behavior tests for opt-in cache census and first-reclaim tags."""
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.vpc_policies.diagnostics import prefix_shape, chunk_summary


def test_prefix_hole_measurement():
    assert prefix_shape([True] * 32)["stranded_after_gap"] == 0
    p = prefix_shape([True, True, False, False, True, True])
    assert (p["prefix_blocks"], p["retained"], p["stranded_after_gap"]) == (2, 4, 2)


def test_opt_in_occupancy_and_selective_rejection(monkeypatch):
    monkeypatch.setenv("VLLM_VPC_POLICY", "selective")
    monkeypatch.setenv("VLLM_VPC_DIAGNOSTICS", "1")
    monkeypatch.setenv("VLLM_VPC_DIAG_INTERVAL_S", "0.001")
    monkeypatch.delenv("VLLM_VPC_STATS_JSONL", raising=False)
    pool = BlockPool(7, True, 16)
    assert pool._vpc_diagnostics is not None
    # State and sampling live in a standalone module, not the allocator.
    assert not hasattr(pool, "_vpc_diag_chunks")
    blocks = pool.get_new_blocks(6)
    pool._insert_block_hash(b"gpu-key-1234567890", blocks[0], num_tokens=16)
    pool._insert_block_hash(b"cpu-key-1234567890", blocks[1], num_tokens=16)
    pool.set_block_importance(blocks[0].block_id, "gpu")
    pool.set_block_importance(blocks[1].block_id, "cpu")
    pool.free_blocks(blocks)
    pool.maybe_sample_vpc_diagnostics(force=True)
    sample = pool.get_vpc_stats_snapshot()["vpc_diagnostics"]
    physical = sample["physical_blocks"]
    assert physical["active"] == 0
    assert physical["idle_cached"]["gpu"] == 1
    assert physical["idle_cached"]["cpu"] == 0
    assert physical["uncached_available"] == 5
    assert physical["reserved_special"] == 1
    assert physical["accounted_total"] == 7
    assert pool.vpc_stats.admission_rejects["cpu"] == 1


def test_first_gpu_reclaim_records_request_count(monkeypatch):
    monkeypatch.setenv("VLLM_VPC_POLICY", "selective")
    monkeypatch.setenv("VLLM_VPC_DIAGNOSTICS", "1")
    monkeypatch.delenv("VLLM_VPC_STATS_JSONL", raising=False)
    pool = BlockPool(3, True, 16)
    b0, b1 = pool.get_new_blocks(2)
    pool._insert_block_hash(b"first-gpu-block", b0, num_tokens=16)
    pool.set_block_importance(b0.block_id, "gpu")
    pool.free_blocks([b1, b0])
    assert pool.get_new_blocks(1)[0] is b1  # uncached-first LRU
    assert pool.vpc_stats.first_gpu_reclaim is None
    assert pool.get_new_blocks(1)[0] is b0
    assert pool.vpc_stats.first_gpu_reclaim is not None
    assert pool.vpc_stats.first_gpu_reclaim["requests_first_seen"] == 0
