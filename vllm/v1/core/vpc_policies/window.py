# SPDX-License-Identifier: Apache-2.0
# [SC] Bounded GNN-aware eviction; exact victim order of the original W policy.
from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


class WindowEviction:
    def __init__(self, pool: BlockPool, window: int):
        if window < 1:
            raise ValueError("VLLM_GNN_AWARE_VPC_WINDOW must be >= 1")
        self.pool = pool
        self.window = window
        self.logged_bias = False

    def pop(self) -> tuple[KVCacheBlock, int, bool]:
        """Return (victim, candidates_examined, bypassed_lru_head).

        The original policy always uses an uncached head first and scans only
        consecutive cached blocks from the head. Ties preserve FIFO/LRU order.
        """
        pool = self.pool
        queue = pool.free_block_queue
        head = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        if head is None or head is tail:
            raise ValueError("No free blocks available")
        if head.block_hash is None or not pool.enable_caching:
            return queue.popleft(), 0, False

        candidates = []
        current = head
        while current is not None and current is not tail:
            if current.block_hash is None:
                break
            candidates.append(current)
            if len(candidates) >= self.window:
                break
            current = current.next_free_block
        victim = min(candidates, key=pool.get_block_importance_rank)
        if victim is head:
            return queue.popleft(), len(candidates), False

        queue.remove(victim)
        if not self.logged_bias:
            self.logged_bias = True
            logger.info(
                "[GNN_AWARE_VPC_BIAS] window=%d oldest_block=%d "
                "oldest_placement=%s selected_block=%d selected_placement=%s",
                len(candidates), head.block_id,
                pool.kv_importance_by_block_id.get(head.block_id, "unlabeled"),
                victim.block_id,
                pool.kv_importance_by_block_id.get(victim.block_id, "unlabeled"),
            )
        return victim, len(candidates), True
