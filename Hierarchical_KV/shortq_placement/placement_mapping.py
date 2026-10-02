"""Map Short-Q checkpoint classes to semantic KV placements.

Checkpoint-native classes are 0=drop, 1=disk, 2=cpu, 3=gpu.  These values
express placement intent, not LMCache physical tier numbering.
"""

from __future__ import annotations

from collections.abc import Iterable

CLASS_NAMES = ("drop", "disk", "cpu", "gpu")
VALID_PLACEMENTS = ("gpu", "cpu", "disk")


def drop_and_disk_to_disk(class_id: int) -> str:
    if class_id in (0, 1):
        return "disk"
    if class_id == 2:
        return "cpu"
    if class_id == 3:
        return "gpu"
    raise ValueError(f"unknown Short-Q class id {class_id}")


# Edit this binding for future class->placement experiments.
_selected_mapping = drop_and_disk_to_disk


def map_block_prediction_to_placement(class_id: int) -> str:
    return _selected_mapping(int(class_id))


def map_block_predictions_to_placements(class_ids: Iterable[int]) -> list[str]:
    return [map_block_prediction_to_placement(int(x)) for x in class_ids]
