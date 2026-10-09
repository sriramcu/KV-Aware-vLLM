# SPDX-License-Identifier: Apache-2.0
# [SC] Unit tests for the four VPC policies on real BlockPool metadata.
import pytest

from vllm.v1.core.block_pool import BlockPool


def _make_pool(monkeypatch, mode: str, *, window: int = 3):
    monkeypatch.setenv("VLLM_VPC_POLICY", mode)
    monkeypatch.setenv("VLLM_GNN_AWARE_VPC_WINDOW", str(window))
    monkeypatch.delenv("VLLM_VPC_STATS_JSONL", raising=False)
    pool = BlockPool(num_gpu_blocks=12, enable_caching=True, hash_block_size=16)
    blocks = pool.get_new_blocks(11)  # Block 0 is the null block.
    for i, block in enumerate(blocks):
        # A unique, prefix-like hash for each physical block.
        pool._insert_block_hash(b"sc-vpc-" + i.to_bytes(2, "big") + b"\0\0\0\0",
                                block, num_tokens=(i + 1) * 16)
    return pool, blocks


def test_vpc_legacy_policy_default(monkeypatch):
    monkeypatch.delenv("VLLM_VPC_POLICY", raising=False)
    monkeypatch.setenv("VLLM_GNN_AWARE_VPC", "1")
    pool = BlockPool(8, True, 16)
    assert pool.vpc_policy_name == "window"
    monkeypatch.setenv("VLLM_VPC_POLICY", "vanilla")
    pool = BlockPool(8, True, 16)
    assert pool.vpc_policy_name == "vanilla"


def test_window_matches_bounded_minimum(monkeypatch):
    pool, blocks = _make_pool(monkeypatch, "window", window=3)
    # Reversed free order: newest ID is the oldest eviction candidate.
    for block in blocks:
        pool.set_block_importance(
            block.block_id, "disk" if block is blocks[-2] else "gpu"
        )
    pool.free_blocks(reversed(blocks))
    victim = pool.get_new_blocks(1)[0]
    assert victim is blocks[-2]  # disk within the oldest three
    assert pool.vpc_stats.bypassed_lru == 1
    assert pool.vpc_stats.scanned == 3


def test_tierwise_priorities_touch_and_explicit_invalidation(monkeypatch):
    pool, blocks = _make_pool(monkeypatch, "tierwise")
    # Every block is cached. Label one free block each disk, cpu, gpu;
    # the rest are deliberately left unlabeled.
    for block in blocks:
        pool.set_block_importance(block.block_id, "gpu")
    pool.set_block_importance(blocks[2].block_id, "disk")  # actually max gpu - no downgrade
    pool.free_blocks(reversed(blocks))
    assert pool.get_new_blocks(1)[0] is blocks[-1]
    # A free cached block can be touched; it must disappear from the index.
    touched = blocks[0]
    pool.touch([touched])
    assert touched.ref_cnt == 1
    pool.free_blocks([touched])
    # Explicit invalidation leaves an uncacheable block in the free list.
    victim = blocks[4]
    pool.evict_blocks({victim.block_id})
    assert victim.block_hash is None
    assert pool.get_new_blocks(1)[0] is victim


def test_tierwise_strict_order_and_relabel(monkeypatch):
    pool, blocks = _make_pool(monkeypatch, "tierwise")
    # Assign explicit different priorities before the free operation.
    pool.set_block_importance(blocks[-1].block_id, "gpu")
    pool.set_block_importance(blocks[-2].block_id, "cpu")
    pool.set_block_importance(blocks[-3].block_id, "disk")
    pool.free_blocks(reversed(blocks))
    # Remaining unlabelled blocks are evicted first, even when older labelled
    # blocks precede them in the global LRU queue.
    assert pool.get_new_blocks(1)[0] is blocks[-4]
    # Relabel an already-free disk block to GPU: it must move tier queues.
    pool.set_block_importance(blocks[-3].block_id, "gpu")
    for _ in range(7):
        assert pool.get_new_blocks(1)[0].block_id in {
            block.block_id for block in blocks[:7]
        }
    assert pool.get_new_blocks(1)[0] is blocks[-2]  # CPU before GPU


def test_selective_only_keeps_gpu_hashes(monkeypatch):
    pool, blocks = _make_pool(monkeypatch, "selective")
    pool.set_block_importance(blocks[2].block_id, "gpu")
    pool.set_block_importance(blocks[1].block_id, "cpu")
    pool.set_block_importance(blocks[0].block_id, "disk")
    pool.free_blocks(reversed(blocks))
    assert blocks[2].block_hash is not None
    assert all(block.block_hash is None for block in blocks if block is not blocks[2])
    assert pool.get_num_free_blocks() == 11
    assert sum(pool.vpc_stats.admission_rejects.values()) == 10


def test_all_policies_recycle_same_number_of_blocks(monkeypatch):
    for mode in ("vanilla", "window", "tierwise", "selective"):
        pool, blocks = _make_pool(monkeypatch, mode)
        pool.free_blocks(reversed(blocks))
        assert pool.get_num_free_blocks() == 11
        allocated = pool.get_new_blocks(11)
        assert {block.block_id for block in allocated} == {
            block.block_id for block in blocks
        }
        assert pool.get_num_free_blocks() == 0
        assert all(block.ref_cnt == 1 for block in allocated)
        pool.free_blocks(reversed(allocated))
        assert pool.get_num_free_blocks() == 11


def test_selective_waits_for_last_reference_and_removes_all_hashes(monkeypatch):
    pool, blocks = _make_pool(monkeypatch, "selective")
    block = blocks[0]
    additional_hash = b"sc-vpc-extra\0\0\0\0"
    pool._insert_block_hash(additional_hash, block, num_tokens=16)
    block.ref_cnt += 1  # A second live owner still needs the same KV.
    pool.free_blocks([block])
    assert block.ref_cnt == 1
    assert block.block_hash is not None
    pool.free_blocks([block])
    assert block.ref_cnt == 0
    assert block.block_hash is None
    assert not pool.cached_block_hash_to_block.contain(additional_hash, block.block_id)
    assert block.block_id not in pool.cached_block_hashes_by_block


def test_move_block_hashes_transfers_importance(monkeypatch):
    pool, blocks = _make_pool(monkeypatch, "tierwise")
    src, dst = blocks[0], blocks[1]
    pool.set_block_importance(src.block_id, "gpu")
    assert pool._maybe_evict_cached_block(dst)
    pool.move_block_hashes(src, dst)
    assert src.block_hash is None
    assert dst.block_hash is not None
    assert pool.kv_importance_by_block_id.get(src.block_id) is None
    assert pool.kv_importance_by_block_id[dst.block_id] == "gpu"
    pool.free_blocks(reversed(blocks))
    assert pool.get_num_free_blocks() == 11
    # All unlabeled cached blocks are preferable victims to the GPU block.
    for _ in range(9):
        assert pool.get_new_blocks(1)[0] is not dst


def test_vpc_stats_snapshot_counts_and_percentages(monkeypatch):
    pool, blocks = _make_pool(monkeypatch, "window")
    for block in blocks:
        pool.set_block_importance(block.block_id, "disk")
    pool.free_blocks(reversed(blocks))
    before = pool.get_vpc_stats_snapshot()
    pool.get_new_blocks(2)
    report = pool.get_vpc_stats_snapshot()
    assert report["cached_reclaims"] == 2
    assert report["victim_counts"]["disk"] == 2
    assert report["victim_pct"]["disk"] == 100.0
    assert report["select_calls"] == before["select_calls"] + 2
    assert report["mean_candidates"] == pytest.approx(6 / 13, abs=0.001)


def test_tierwise_free_index_randomized_lifecycle(monkeypatch):
    import random

    rng = random.Random(17)
    monkeypatch.setenv("VLLM_VPC_POLICY", "tierwise")
    monkeypatch.delenv("VLLM_VPC_STATS_JSONL", raising=False)
    pool = BlockPool(65, True, 16)
    active = {}
    next_hash = 0
    for _ in range(1400):
        free = pool.free_block_queue.get_all_free_blocks()
        operation = rng.random()
        if operation < 0.44 and free:
            block = pool.get_new_blocks(1)[0]
            next_hash += 1
            if rng.random() < .75:
                pool._insert_block_hash(
                    f"unique-{next_hash}".encode() + b"\0\0\0\0",
                    block, num_tokens=16,
                )
                if rng.random() < .80:
                    pool.set_block_importance(
                        block.block_id, rng.choice(("disk", "cpu", "gpu"))
                    )
            active[block.block_id] = block
        elif operation < .81 and active:
            block_id = rng.choice(list(active))
            block = active.pop(block_id)
            pool.free_blocks([block])
        elif operation < .93:
            cached_free = [b for b in free if b.block_hash is not None]
            if cached_free:
                block = rng.choice(cached_free)
                pool.evict_blocks({block.block_id})
        elif free:
            cached_free = [b for b in free if b.block_hash is not None]
            if cached_free:
                block = rng.choice(cached_free)
                pool.touch([block])
                active[block.block_id] = block
        assert pool.get_num_free_blocks() == len(pool.free_block_queue.get_all_free_blocks())
        index = pool._tierwise_eviction
        indexed_cached = set().union(*(set(q) for q in index.cached.values()))
        indexed_uncached = set(index.uncached)
        assert not (indexed_cached & indexed_uncached)
        assert indexed_cached | indexed_uncached == {
            block.block_id for block in pool.free_block_queue.get_all_free_blocks()
        }
        actual_by_tier = {tier: [] for tier in index.cached}
        for block in pool.free_block_queue.get_all_free_blocks():
            if block.block_hash is not None:
                actual_by_tier[index.tier(block)].append(block.block_id)
        for tier, expected in actual_by_tier.items():
            assert list(index.cached[tier]) == expected
