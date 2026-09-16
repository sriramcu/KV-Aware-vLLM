# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable, Sequence
import os
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)

class _TieredFreeKVCacheBlockQueue:
    """[SC] Experimental tier-partitioned replacement for the free-block queue.

    Truly unused/unhashed blocks are consumed before any cached victim. Cached
    eviction candidates are split into one LRU queue per configured tier. The
    tier list is configured fastest -> slowest, while allocation/eviction scans
    it in reverse so the slowest non-empty tier loses first.

    This class deliberately mirrors only the FreeKVCacheBlockQueue interface
    used by BlockPool in the supported experiment. Gate-off continues to use
    the vanilla FreeKVCacheBlockQueue directly.
    """

    _UNUSED = "__unused__"

    def __init__(
        self,
        blocks: list[KVCacheBlock],
        tiers_fast_to_slow: tuple[str, ...],
        tier_for_block: Callable[[KVCacheBlock], str],
    ) -> None:
        if not tiers_fast_to_slow:
            raise ValueError("Tierwise KV LRU requires at least one active tier")
        self.tiers_fast_to_slow = tiers_fast_to_slow
        self._tier_for_block = tier_for_block
        self._unused_queue = FreeKVCacheBlockQueue(blocks)
        self._tier_queues = {
            tier: FreeKVCacheBlockQueue([]) for tier in tiers_fast_to_slow
        }
        self._location_by_block_id = {
            block.block_id: self._UNUSED for block in blocks
        }

    @property
    def num_free_blocks(self) -> int:
        return self._unused_queue.num_free_blocks + sum(
            queue.num_free_blocks for queue in self._tier_queues.values()
        )

    def _queue_for_location(self, location: str) -> FreeKVCacheBlockQueue:
        if location == self._UNUSED:
            return self._unused_queue
        return self._tier_queues[location]

    def popleft(self) -> KVCacheBlock:
        if self._unused_queue.num_free_blocks:
            location = self._UNUSED
            queue = self._unused_queue
        else:
            location = ""
            queue = None
            for tier in reversed(self.tiers_fast_to_slow):
                candidate = self._tier_queues[tier]
                if candidate.num_free_blocks:
                    location = tier
                    queue = candidate
                    break
            if queue is None:
                if self.num_free_blocks != 0:
                    raise RuntimeError(
                        "Tierwise free-block count is inconsistent with its queues"
                    )
                raise ValueError("No free blocks available")

        block = queue.popleft()
        recorded = self._location_by_block_id.pop(block.block_id, None)
        if recorded != location:
            raise RuntimeError(
                "Tierwise free-block membership is inconsistent: "
                f"block={block.block_id} expected={location} recorded={recorded}"
            )
        return block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        if n == 0:
            return []
        if n > self.num_free_blocks:
            raise ValueError(f"Cannot pop {n} blocks from {self.num_free_blocks}")
        return [self.popleft() for _ in range(n)]

    def remove(self, block: KVCacheBlock) -> None:
        location = self._location_by_block_id.pop(block.block_id, None)
        if location is None:
            raise RuntimeError(
                f"remove() called on block not in tierwise free queues: {block}"
            )
        self._queue_for_location(location).remove(block)

    def append(self, block: KVCacheBlock) -> None:
        if block.block_id in self._location_by_block_id:
            raise RuntimeError(
                f"Block {block.block_id} is already in a tierwise free queue"
            )
        if block.block_hash is None:
            location = self._UNUSED
        else:
            location = self._tier_for_block(block)
            if location not in self._tier_queues:
                raise RuntimeError(
                    f"Resolved inactive tier {location!r} for block {block.block_id}"
                )
        self._queue_for_location(location).append(block)
        self._location_by_block_id[block.block_id] = location

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        # Preserve the caller-provided (tail-first/LRU) order within each tier.
        for block in blocks:
            self.append(block)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        # Return the order allocation would use: truly unused capacity first,
        # then cached victims from slowest tier to fastest tier.
        ret = self._unused_queue.get_all_free_blocks()
        for tier in reversed(self.tiers_fast_to_slow):
            ret.extend(self._tier_queues[tier].get_all_free_blocks())
        return ret

    def queue_lengths(self) -> dict[str, int]:
        lengths = {
            tier: self._tier_queues[tier].num_free_blocks
            for tier in self.tiers_fast_to_slow
        }
        lengths[self._UNUSED] = self._unused_queue.num_free_blocks
        return lengths

    def reclassify_as_unused_if_queued(self, block: KVCacheBlock) -> None:
        """Move an explicitly evicted free block back to unused capacity."""
        location = self._location_by_block_id.get(block.block_id)
        if location is None or location == self._UNUSED:
            return
        self._tier_queues[location].remove(block)
        self._unused_queue.append(block)
        self._location_by_block_id[block.block_id] = self._UNUSED

    def reset_as_unused(self, blocks: list[KVCacheBlock]) -> None:
        # reset_prefix_cache() is only allowed when every non-null block is free.
        # Rebuild the queue topology after hashes/tier metadata are cleared.
        for block in blocks:
            block.prev_free_block = None
            block.next_free_block = None
        self._unused_queue = FreeKVCacheBlockQueue(blocks)
        self._tier_queues = {
            tier: FreeKVCacheBlockQueue([]) for tier in self.tiers_fast_to_slow
        }
        self._location_by_block_id = {
            block.block_id: self._UNUSED for block in blocks
        }


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        
        self.enable_kv_importance = (
            os.environ.get("VLLM_KV_IMPORTANCE_ENABLE") == "1"
        )
        self.kv_importance_by_block_id = {} # [SY]

        tierwise_requested = (
            os.environ.get("VLLM_KV_TIERWISE_LRU_ENABLE", "0") == "1"
        )
        self.enable_tierwise_lru = tierwise_requested and self.enable_kv_importance
        self.tierwise_lru_trace_enabled = (
            os.environ.get("VLLM_KV_TIERWISE_LRU_TRACE_ENABLE", "0") == "1"
        )
        self.tierwise_lru_tiers: tuple[str, ...] = ()
        self._tierwise_lru_warned_unknown_tiers: set[str] = set()
        self._tierwise_lru_stats: dict[str, dict[str, int]] = {}

        if tierwise_requested and not self.enable_kv_importance:
            logger.warning(
                "VLLM_KV_TIERWISE_LRU_ENABLE=1 requires "
                "VLLM_KV_IMPORTANCE_ENABLE=1; using vanilla free-block LRU"
            )

        if self.enable_tierwise_lru:
            raw_tiers = os.environ.get(
                "VLLM_KV_TIERWISE_LRU_TIERS", "gpu,cpu,disk"
            )
            tiers = tuple(
                tier.strip().lower() for tier in raw_tiers.split(",") if tier.strip()
            )
            if not tiers:
                raise ValueError(
                    "VLLM_KV_TIERWISE_LRU_TIERS must contain at least one tier"
                )
            if len(set(tiers)) != len(tiers):
                raise ValueError(
                    "VLLM_KV_TIERWISE_LRU_TIERS contains duplicate tiers: "
                    f"{tiers}"
                )
            self.tierwise_lru_tiers = tiers
            self._tierwise_lru_stats = {
                name: {tier: 0 for tier in tiers}
                for name in ("assignments", "releases", "hits", "victims")
            }
            logger.info(
                "Tierwise GPU KV LRU enabled; tiers_fast_to_slow=%s", tiers
            )

        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Gate-off is intentionally the existing vanilla queue. Gate-on keeps
        # truly unused blocks separate and partitions cached victims by tier.
        if self.enable_tierwise_lru:
            self.free_block_queue = _TieredFreeKVCacheBlockQueue(
                self.blocks,
                self.tierwise_lru_tiers,
                self._tier_for_free_block,
            )
        else:
            self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

    def _tierwise_lru_trace(self, event: str, **fields: Any) -> None:
        if not self.tierwise_lru_trace_enabled:
            return
        payload = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.warning("[VLLM_KV_TIERWISE_LRU] event=%s %s", event, payload)

    def _tierwise_lru_bump(self, kind: str, tier: str) -> None:
        if not self.enable_tierwise_lru:
            return
        self._tierwise_lru_stats[kind][tier] += 1

    def _effective_tier(self, tier: str) -> str:
        normalized = str(tier).strip().lower()
        if normalized in self.tierwise_lru_tiers:
            return normalized

        # Full-sacrifice behavior: an unavailable/unknown predicted tier maps to
        # the slowest enabled tier instead of failing (e.g. disk-disabled runs).
        fallback = self.tierwise_lru_tiers[-1]
        if normalized not in self._tierwise_lru_warned_unknown_tiers:
            logger.warning(
                "KV importance tier %r is not active in %s; mapping it to %r",
                normalized,
                self.tierwise_lru_tiers,
                fallback,
            )
            self._tierwise_lru_warned_unknown_tiers.add(normalized)
        return fallback

    def _tier_for_free_block(self, block: KVCacheBlock) -> str:
        tier = self.kv_importance_by_block_id.get(block.block_id)
        if tier is None:
            # Missing importance is deliberately fail-open for experimentation:
            # make it maximally evictable rather than crashing the scheduler.
            return self.tierwise_lru_tiers[-1]
        return self._effective_tier(tier)

    def set_block_importance(self, block_id, tier: str) -> None:
        if not self.enable_kv_importance:
            return

        if hasattr(block_id, "tolist"):
            block_id = block_id.tolist()

        if isinstance(block_id, (list, tuple)):
            for bid in block_id:
                self.set_block_importance(bid, tier)
            return

        if hasattr(block_id, "block_id"):
            block_id = block_id.block_id

        block_id = int(block_id)
        if not self.enable_tierwise_lru:
            # The new gate is a true vanilla control: no vLLM importance
            # metadata or release-order modification when tierwise LRU is off.
            return

        effective_tier = self._effective_tier(tier)
        existing = self.kv_importance_by_block_id.get(block_id)
        if existing is None:
            self.kv_importance_by_block_id[block_id] = effective_tier
            self._tierwise_lru_bump("assignments", effective_tier)
            self._tierwise_lru_trace(
                "assign", block=block_id, tier=effective_tier
            )
        elif existing != effective_tier:
            # Full-sacrifice policy: cached-content tier is immutable. A later
            # request reusing the same physical cached content cannot migrate it.
            self._tierwise_lru_trace(
                "ignore_reassign",
                block=block_id,
                existing=existing,
                requested=effective_tier,
            )

    def get_block_importance_rank(self, block) -> int:
        if not self.enable_kv_importance:
            return 1

        tier = self.kv_importance_by_block_id.get(block.block_id, "cpu")
        return {"disk": 0, "cpu": 1, "gpu": 2}.get(tier, 1)



    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null blocks when enabling sparse attention like
            # sliding window attention, or Mamba models with prefix-caching in
            # align mode. We skip null blocks here.
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[
                        num_cached_blocks * block_size : num_full_blocks * block_size
                    ],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                if self.enable_tierwise_lru and block.block_hash is not None:
                    tier = self._tier_for_free_block(block)
                    self._tierwise_lru_bump("victims", tier)
                    queue_lengths = (
                        self.free_block_queue.queue_lengths()
                        if isinstance(
                            self.free_block_queue, _TieredFreeKVCacheBlockQueue
                        )
                        else {}
                    )
                    self._tierwise_lru_trace(
                        "victim",
                        block=block.block_id,
                        tier=tier,
                        queues=queue_lengths,
                    )
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # Uncached content does not own persistent tier metadata.
            if self.enable_tierwise_lru:
                self.kv_importance_by_block_id.pop(block.block_id, None)
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        block.reset_hash()
        if self.enable_tierwise_lru:
            old_tier = self.kv_importance_by_block_id.pop(block.block_id, None)
            if isinstance(self.free_block_queue, _TieredFreeKVCacheBlockQueue):
                self.free_block_queue.reclassify_as_unused_if_queued(block)
            self._tierwise_lru_trace(
                "evict", block=block.block_id, old_tier=old_tier
            )

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            if self.enable_tierwise_lru and block.block_hash is not None:
                tier = self._tier_for_free_block(block)
                self._tierwise_lru_bump("hits", tier)
                self._tierwise_lru_trace(
                    "hit", block=block.block_id, tier=tier, ref_cnt=block.ref_cnt
                )
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)

        for block in blocks_list:
            block.ref_cnt -= 1
            if self.enable_tierwise_lru:
                if block.ref_cnt == 0:
                    if block.block_hash is None:
                        # Partial/unhashed content is not a reusable cache entry.
                        self.kv_importance_by_block_id.pop(block.block_id, None)
                    elif not block.is_null:
                        tier = self._tier_for_free_block(block)
                        self._tierwise_lru_bump("releases", tier)
                        self._tierwise_lru_trace(
                            "release",
                            block=block.block_id,
                            tier=tier,
                            ref_cnt=block.ref_cnt,
                        )
            else:
                # Gate-off is the vanilla single-LRU control.
                self.kv_importance_by_block_id.pop(block.block_id, None)

        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )


    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.enable_tierwise_lru:
            self.kv_importance_by_block_id.clear()
            if not isinstance(
                self.free_block_queue, _TieredFreeKVCacheBlockQueue
            ):
                raise RuntimeError("Tierwise LRU enabled with vanilla free queue")
            self.free_block_queue.reset_as_unused(
                [block for block in self.blocks if not block.is_null]
            )
            self._tierwise_lru_trace(
                "reset", stats=self._tierwise_lru_stats
            )

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
