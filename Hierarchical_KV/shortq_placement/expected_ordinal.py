"""Class-head ordinal expectation for complete LMCache chunks.

Class order is the frozen Short-Q checkpoint order: DROP, DISK, CPU, GPU.
The equally spaced ordinal values (0,1,2,3) are a documented convention,
NOT calibrated hit probabilities and NOT hardware placement decisions.
"""
from __future__ import annotations

import math

SCORE_FORMAT = "expected_ordinal_0_100"
RANK_SCORE_FORMAT = "mean_32_block_rank_scores"


def expected_ordinal_from_logits(logits: list[list[float]], blocks_per_chunk: int = 32,
                                 full_chunks: int | None = None) -> list[float]:
    """Pure-Python reference used to derive offline score-distribution priors.

    Only complete chunks are scored; the trailing incomplete chunk is omitted.
    Softmax is stabilized to avoid overflow with large logits.
    """
    if blocks_per_chunk <= 0:
        raise ValueError("blocks_per_chunk must be positive")
    if full_chunks is None:
        full_chunks = len(logits) // blocks_per_chunk
    if full_chunks < 0 or full_chunks * blocks_per_chunk > len(logits):
        raise ValueError("inconsistent complete chunk count and logit length")
    values: list[float] = []
    for block in logits[:full_chunks * blocks_per_chunk]:
        if len(block) != 4 or any(not math.isfinite(float(v)) for v in block):
            raise ValueError("expected exactly four finite class logits")
        maximum = max(block)
        weights = [math.exp(float(v) - maximum) for v in block]
        total = sum(weights)
        values.append(100.0 / 3.0 * (weights[1] + 2 * weights[2] + 3 * weights[3]) / total)
    return [sum(values[i:i + blocks_per_chunk]) / blocks_per_chunk
            for i in range(0, len(values), blocks_per_chunk)]


def expected_ordinal_from_torch_logits(class_logits, full_chunks: int,
                                       blocks_per_chunk: int = 32) -> list[float]:
    """Vectorized runtime preprocessing, in one GPU -> host result per request."""
    import torch
    if blocks_per_chunk <= 0 or full_chunks < 0:
        raise ValueError("invalid chunk geometry")
    block_count = full_chunks * blocks_per_chunk
    if class_logits.ndim != 2 or class_logits.shape[-1] != 4 or class_logits.shape[0] < block_count:
        raise ValueError("class logits shape mismatches LMCache chunks")
    if not full_chunks:
        return []
    logits = class_logits[:block_count].float()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite class logits")
    probs = logits.softmax(dim=-1)
    # Classes: drop=0, disk=1, cpu=2, gpu=3.
    expected = probs[:, 1] + 2.0 * probs[:, 2] + 3.0 * probs[:, 3]
    scores = expected.reshape(full_chunks, blocks_per_chunk).mean(dim=-1) * (100.0 / 3.0)
    return [float(v) for v in scores.tolist()]
