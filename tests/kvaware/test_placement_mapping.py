from __future__ import annotations

from Hierarchical_KV.shortq_placement.placement_mapping import (
    map_block_predictions_to_placements,
)


def test_checkpoint_classes_map_to_semantic_placements():
    assert map_block_predictions_to_placements([0, 1, 2, 3]) == [
        "disk",
        "disk",
        "cpu",
        "gpu",
    ]
