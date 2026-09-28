"""Unit tests for multi-teacher ensemble fusion and Weighted Boxes Fusion math."""

from __future__ import annotations

import pytest

from sivia.labeling.ensemble import TeacherEnsemble
from sivia.labeling.sam import compute_box_iou
from sivia.labeling.types import RawDetection


def test_compute_box_iou_exact_match() -> None:
    box1 = [10.0, 20.0, 50.0, 60.0]
    box2 = [10.0, 20.0, 50.0, 60.0]
    iou = compute_box_iou(box1, box2)
    assert pytest.approx(iou, 1e-4) == 1.0


def test_compute_box_iou_disjoint() -> None:
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 20.0, 30.0, 30.0]
    iou = compute_box_iou(box1, box2)
    assert iou == 0.0


def test_compute_box_iou_partial_overlap() -> None:
    # 10x10 square at (0,0) and 10x10 square at (5,0)
    # Intersection: 5x10 = 50
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [5.0, 0.0, 15.0, 10.0]
    iou = compute_box_iou(box1, box2)
    assert pytest.approx(iou, 1e-4) == 1.0 / 3.0


def test_ensemble_fuse_consensus() -> None:
    ensemble = TeacherEnsemble(
        weights={"owlv2": 1.0, "grounding_dino": 1.0},
        iou_threshold=0.5,
        conf_threshold=0.2,
        sam_refinement=False,
    )

    owl_dets = [
        RawDetection(
            class_id=0, class_name="screwdriver", box_xyxy=[10.0, 10.0, 50.0, 50.0], score=0.8
        )
    ]
    gdino_dets = [
        RawDetection(
            class_id=0, class_name="screwdriver", box_xyxy=[12.0, 10.0, 52.0, 50.0], score=0.85
        )
    ]

    fused = ensemble.fuse(
        teacher_detections={"owlv2": owl_dets, "grounding_dino": gdino_dets},
        image_width=100,
        image_height=100,
    )

    assert len(fused) == 1
    det = fused[0]
    assert det.class_id == 0
    assert det.agreement_iou > 0.80
    assert set(det.source_models) == {"owlv2", "grounding_dino"}
    assert det.score > 0.80  # consensus boost


def test_ensemble_fuse_single_teacher() -> None:
    ensemble = TeacherEnsemble(
        weights={"owlv2": 1.0, "grounding_dino": 1.0},
        iou_threshold=0.5,
        conf_threshold=0.2,
        sam_refinement=False,
    )

    owl_dets = [
        RawDetection(
            class_id=1, class_name="tape_roll", box_xyxy=[20.0, 20.0, 60.0, 60.0], score=0.7
        )
    ]
    gdino_dets: list[RawDetection] = []

    fused = ensemble.fuse(
        teacher_detections={"owlv2": owl_dets, "grounding_dino": gdino_dets},
        image_width=100,
        image_height=100,
    )

    assert len(fused) == 1
    det = fused[0]
    assert det.class_id == 1
    assert det.agreement_iou == 0.0  # single teacher
    assert det.source_models == ["owlv2"]
    assert det.score < 0.70  # penalized for unconfirmed proposal


def test_ensemble_fuse_filters_low_confidence() -> None:
    ensemble = TeacherEnsemble(
        weights={"owlv2": 1.0},
        conf_threshold=0.5,
        sam_refinement=False,
    )

    owl_dets = [
        RawDetection(class_id=2, class_name="sensor", box_xyxy=[10.0, 10.0, 30.0, 30.0], score=0.3)
    ]

    fused = ensemble.fuse(
        teacher_detections={"owlv2": owl_dets},
        image_width=100,
        image_height=100,
    )

    assert len(fused) == 0
