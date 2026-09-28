"""SIVIA Kit configuration and verification rules."""

from sivia.kit.rules import Detection, KitVerdict, verify_kit
from sivia.kit.slots import compute_iou, is_box_inside_slot
from sivia.kit.spec import KitClass, KitRules, KitSpec

__all__ = [
    "KitSpec",
    "KitClass",
    "KitRules",
    "Detection",
    "KitVerdict",
    "verify_kit",
    "compute_iou",
    "is_box_inside_slot",
]
