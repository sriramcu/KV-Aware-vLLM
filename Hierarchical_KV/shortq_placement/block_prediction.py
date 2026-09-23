"""Convert raw Short-Q model outputs into one class prediction per 16-token block.

To try another model-output policy, change only ``_selected_block_policy`` below.
The rest of the project calls ``get_block_predictions``.
"""

from __future__ import annotations

import torch


def argmax_class_logits(class_logits: torch.Tensor, rank_scores: torch.Tensor | None = None) -> torch.Tensor:
    """Plain four-class argmax. ``rank_scores`` is intentionally ignored."""
    del rank_scores
    if class_logits.ndim < 2 or class_logits.shape[-1] != 4:
        raise ValueError(f"expected [..., 4] class logits, got {tuple(class_logits.shape)}")
    return class_logits.argmax(dim=-1)


# Edit this one binding for future output-to-block-prediction experiments.
_selected_block_policy = argmax_class_logits


def get_block_predictions(
    class_logits: torch.Tensor,
    rank_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    return _selected_block_policy(class_logits, rank_scores)
