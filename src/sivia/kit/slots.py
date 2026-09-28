"""Kit slot matching and ROI placement verification logic."""

from __future__ import annotations

from collections.abc import Sequence


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def is_box_inside_slot(
    box: Sequence[float], slot_box: Sequence[float], min_overlap_ratio: float = 0.5
) -> bool:
    """Check if a detected box is sufficiently contained inside a slot ROI.

    Calculates the proportion of the detected box that lies within the slot box.
    """
    x1 = max(box[0], slot_box[0])
    y1 = max(box[1], slot_box[1])
    x2 = min(box[2], slot_box[2])
    y2 = min(box[3], slot_box[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    if box_area <= 0.0:
        return False
    return (intersection / box_area) >= min_overlap_ratio
