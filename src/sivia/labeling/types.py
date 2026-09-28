"""Data structures and types for the SIVIA teacher labeling pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDetection:
    """Individual object detection predicted by a teacher model."""

    class_id: int
    class_name: str
    box_xyxy: list[float]  # [x1, y1, x2, y2] in pixel coordinates
    score: float
    prompt: str = ""
    model_name: str = ""


@dataclass
class MaskResult:
    """Segmentation mask and derived tight bounding box."""

    mask_rle: dict[str, Any] | str
    mask_box: list[float]  # [x1, y1, x2, y2] tight bounding box derived from mask
    mask_box_iou: float  # IoU between original detector box and mask_box


@dataclass
class EnsembleDetection:
    """Fused detection from multi-teacher ensemble with consensus metrics."""

    class_id: int
    class_name: str
    box_xyxy: list[float]  # [x1, y1, x2, y2]
    score: float  # Calibrated ensemble confidence
    agreement_iou: float  # Overlap between teacher detections (0.0 if single teacher)
    mask_rle: dict[str, Any] | str | None = None
    mask_box: list[float] | None = None
    mask_box_iou: float | None = None
    source_models: list[str] = field(default_factory=list)
