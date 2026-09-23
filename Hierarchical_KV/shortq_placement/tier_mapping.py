"""Map Short-Q block classes to persistent LMCache tiers.

Class IDs are checkpoint-native: 0=drop, 1=disk, 2=cpu, 3=gpu.
To change semantic mapping later, change only ``_selected_mapping``.
"""

from __future__ import annotations

from collections.abc import Iterable

CLASS_NAMES = ("drop", "disk", "cpu", "gpu")
VALID_TIERS = ("L0", "L1", "L2")


def drop_and_disk_to_l2(class_id: int) -> str:
    if class_id in (0, 1):
        return "L2"
    if class_id == 2:
        return "L1"
    if class_id == 3:
        return "L0"
    raise ValueError(f"unknown Short-Q class id {class_id}")


# Edit this one binding for future class->tier mappings.
_selected_mapping = drop_and_disk_to_l2


def map_block_prediction_to_tier(class_id: int) -> str:
    return _selected_mapping(int(class_id))


def map_block_predictions_to_tiers(class_ids: Iterable[int]) -> list[str]:
    return [map_block_prediction_to_tier(int(x)) for x in class_ids]
