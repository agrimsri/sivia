"""SAM 2 / SAM interface module for SIVIA labeling pipeline."""

from sivia.labeling.sam import (
    SamPredictor,
    compute_box_iou,
    mask_to_rle,
    rle_to_mask,
)

__all__ = ["SamPredictor", "compute_box_iou", "mask_to_rle", "rle_to_mask"]
