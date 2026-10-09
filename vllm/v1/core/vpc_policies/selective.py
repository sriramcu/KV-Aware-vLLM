# SPDX-License-Identifier: Apache-2.0
# [SC] Selective VPC admission at the *existing* 16-token block lifecycle.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


def admit_for_reuse(block: KVCacheBlock, labels: dict[int, str]) -> bool:
    """Only GPU-labelled blocks retain searchable prefix hashes on release.

    Chunk->block label propagation remains with the existing placement producer;
    no independent blockwise prediction or oracle is performed here. A missing
    label is conservatively rejected. Referenced blocks are never invalidated.
    """
    return labels.get(block.block_id) == "gpu"
