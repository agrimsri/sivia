"""Kit completeness and layout verification rules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sivia.kit.slots import is_box_inside_slot
from sivia.kit.spec import KitSpec


@dataclass
class Detection:
    class_name: str
    class_id: int
    score: float
    bbox_normalized: Sequence[float]  # [x1, y1, x2, y2] in 0..1


@dataclass
class KitVerdict:
    status: str  # 'complete', 'incomplete', 'invalid_count', 'misplaced'
    is_valid: bool
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    misplaced: list[str] = field(default_factory=list)
    details: dict[str, int] = field(default_factory=dict)


def verify_kit(detections: list[Detection], spec: KitSpec) -> KitVerdict:
    """Verify if a set of detected objects satisfies the Kit specification.

    Checks:
    1. Score filtering (min_score threshold)
    2. Required objects presence
    3. Maximum count limits per class
    4. Slot ROI placement
    """
    valid_dets = [d for d in detections if d.score >= spec.rules.min_score]

    # Count occurrences
    counts: dict[str, int] = {}
    dets_by_class: dict[str, list[Detection]] = {}
    for d in valid_dets:
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
        dets_by_class.setdefault(d.class_name, []).append(d)

    # Check required classes
    missing = [req for req in spec.required if counts.get(req, 0) == 0]

    # Check max counts
    extra: list[str] = []
    for class_name, max_allowed in spec.rules.max_count.items():
        observed = counts.get(class_name, 0)
        if observed > max_allowed:
            extra.append(f"{class_name} (found {observed}, max {max_allowed})")

    # Check slots
    misplaced: list[str] = []
    for slot_name, slot_box in spec.slots.items():
        if slot_name in dets_by_class:
            in_slot = any(
                is_box_inside_slot(det.bbox_normalized, slot_box)
                for det in dets_by_class[slot_name]
            )
            if not in_slot:
                misplaced.append(slot_name)

    is_valid = (len(missing) == 0) and (len(extra) == 0) and (len(misplaced) == 0)

    if not is_valid:
        if missing:
            status = "incomplete"
        elif misplaced:
            status = "misplaced"
        else:
            status = "invalid_count"
    else:
        status = "complete"

    return KitVerdict(
        status=status,
        is_valid=is_valid,
        missing=missing,
        extra=extra,
        misplaced=misplaced,
        details=counts,
    )
