# SPDX-License-Identifier: Apache-2.0
# [SC] O(1) strict-tier victim selection with secondary free-block indexes.
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

TIERS = ("unlabeled", "disk", "cpu", "gpu")


class TierwiseEviction:
    """The upstream doubly linked free list is authoritative.

    Secondary indexes track uncached free blocks and reclaimable cached blocks
    in release (LRU) order. A fixed four-tier scan is O(1); removing a known
    free block from the upstream queue is O(1). Rare free-block relabelling
    may require O(N) index maintenance to preserve LRU-within-tier order.
    """

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.cached: dict[str, OrderedDict[int, KVCacheBlock]] = {
            tier: OrderedDict() for tier in TIERS
        }
        self.uncached: OrderedDict[int, KVCacheBlock] = OrderedDict()
        # Initial queue includes all unallocated blocks (except null).
        for block in pool.free_block_queue.get_all_free_blocks():
            if block.block_hash is None:
                self.uncached[block.block_id] = block
            else:
                self.cached[self.tier(block)][block.block_id] = block

    def tier(self, block: KVCacheBlock) -> str:
        return self.pool.kv_importance_by_block_id.get(block.block_id, "unlabeled")

    def discard(self, block: KVCacheBlock) -> None:
        """O(1) removal for pop/touch/hash-invalidation lifecycle events."""
        block_id = block.block_id
        self.uncached.pop(block_id, None)
        for queue in self.cached.values():
            queue.pop(block_id, None)

    def on_free(self, uncached: list[KVCacheBlock], cached: list[KVCacheBlock]) -> None:
        # Upstream prepends uncached in supplied order (LIFO reuse).
        for block in reversed(uncached):
            self.discard(block)
            self.uncached[block.block_id] = block
            self.uncached.move_to_end(block.block_id, last=False)
        # Upstream appends cached at the tail, preserving release order.
        for block in cached:
            self.discard(block)
            self.cached[self.tier(block)][block.block_id] = block

    def on_hash_invalidated(self, block: KVCacheBlock) -> None:
        """Handle explicit eviction while block remains on the free list."""
        self.discard(block)
        if (not block.is_null and block.ref_cnt == 0
                and block.prev_free_block is not None):
            self.uncached[block.block_id] = block

    def on_label_changed(self, block: KVCacheBlock) -> None:
        """Preserve LRU within tier even for a rare already-free relabel."""
        if (block.is_null or block.ref_cnt != 0 or block.block_hash is None
                or block.prev_free_block is None):
            return
        self.discard(block)
        destination = self.cached[self.tier(block)]
        # A free-block relabel is rare; follow authoritative LRU ordering.
        # This costs O(N) only on metadata updates, never on victim selection.
        following = block.next_free_block
        while following is not None and following is not self.pool.free_block_queue.fake_free_list_tail:
            if following.block_id in destination:
                break
            following = following.next_free_block
        if following is None or following is self.pool.free_block_queue.fake_free_list_tail:
            destination[block.block_id] = block
        else:
            rebuilt = OrderedDict()
            for block_id, value in destination.items():
                if block_id == following.block_id:
                    rebuilt[block.block_id] = block
                rebuilt[block_id] = value
            self.cached[self.tier(block)] = rebuilt

    def pop(self) -> tuple[KVCacheBlock, int, bool]:
        queue = self.pool.free_block_queue
        head = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        if head is None or head is tail:
            raise ValueError("No free blocks available")
        # Reproduce stock LIFO behavior for normal noncached head blocks.
        if head.block_hash is None:
            victim = queue.popleft()
            self.discard(victim)
            return victim, 0, False
        # Prefer uncached free blocks even if explicit hash invalidation left
        # one in the interior of the authoritative free queue.
        if self.uncached:
            victim = next(iter(self.uncached.values()))
            queue.remove(victim)
            self.discard(victim)
            return victim, 0, victim is not head
        for tier in TIERS:
            if self.cached[tier]:
                victim = next(iter(self.cached[tier].values()))
                if victim is head:
                    queue.popleft()
                else:
                    queue.remove(victim)
                self.discard(victim)
                return victim, 1, victim is not head
        raise RuntimeError("Tierwise free-block index diverged from upstream queue")
