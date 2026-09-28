"""Unit tests for SAM mask generation, tight bounding boxes, and RLE encoding."""

from __future__ import annotations

import numpy as np

from sivia.labeling.sam import (
    SamPredictor,
    compute_box_iou,
    mask_to_rle,
    rle_to_mask,
)


def test_rle_round_trip() -> None:
    # Create arbitrary binary mask
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:50, 30:70] = True
    mask[60:80, 10:40] = True

    rle = mask_to_rle(mask)
    assert "size" in rle
    assert "counts" in rle
    assert isinstance(rle["counts"], str)

    decoded = rle_to_mask(rle)
    assert decoded.shape == (100, 100)
    assert np.array_equal(mask, decoded)


def test_tight_box_from_mask() -> None:
    mask = np.zeros((200, 200), dtype=bool)
    # Target rectangle [40, 50, 90, 120] -> rows 50:120, cols 40:90
    mask[50:120, 40:90] = True

    tight_box = SamPredictor._tight_box_from_mask(
        mask, default_box=[0, 0, 10, 10], width=200, height=200
    )
    assert tight_box[0] == 40.0
    assert tight_box[1] == 50.0
    assert tight_box[2] == 90.0
    assert tight_box[3] == 120.0


def test_sam_heuristic_prediction() -> None:
    predictor = SamPredictor(mock=True)
    img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [[10.0, 10.0, 50.0, 60.0]]

    results = predictor.predict_masks(img_arr, boxes)
    assert len(results) == 1
    res = results[0]
    assert res.mask_box_iou > 0.60
    assert len(res.mask_box) == 4
    # Check that mask_box is close to input box
    assert compute_box_iou(boxes[0], res.mask_box) > 0.60
