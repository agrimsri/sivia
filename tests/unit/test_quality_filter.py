"""Unit tests for quality filtering (blur and brightness metrics)."""

import cv2
import numpy as np

from sivia.capture.quality_filter import QualityFilter, compute_blur_score, compute_brightness


def test_blur_score_sharp_vs_blurred():
    """Verify that a blurred image yields a much lower Laplacian variance than a sharp image."""
    # Create sharp checkerboard pattern
    sharp = np.zeros((200, 200), dtype=np.uint8)
    sharp[::20, :] = 255
    sharp[:, ::20] = 255

    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

    sharp_score = compute_blur_score(sharp)
    blurred_score = compute_blur_score(blurred)

    assert sharp_score > blurred_score * 5
    assert blurred_score < 30.0


def test_brightness_levels():
    """Verify that mean brightness calculates accurately across black, gray, and white."""
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = np.full((100, 100, 3), 128, dtype=np.uint8)
    white = np.full((100, 100, 3), 255, dtype=np.uint8)

    assert compute_brightness(black) == 0.0
    assert compute_brightness(gray) == 128.0
    assert compute_brightness(white) == 255.0


def test_quality_filter_rejections():
    """Verify QualityFilter accepts good frames and rejects degraded frames with explicit reason."""
    filter_engine = QualityFilter(min_blur_score=35.0, min_brightness=30.0, max_brightness=230.0)

    # Sharp, well-lit image with textured pattern
    good = np.full((200, 200, 3), 100, dtype=np.uint8)
    good[::10, :] = 220
    good[:, ::10] = 220
    good_res = filter_engine.evaluate(good)
    assert good_res.passed
    assert good_res.rejection_reason is None

    # Severely dark image
    dark = np.full((100, 100, 3), 10, dtype=np.uint8)
    dark_res = filter_engine.evaluate(dark)
    assert not dark_res.passed
    assert "underexposed" in dark_res.rejection_reason

    # Overexposed image
    blown_out = np.full((100, 100, 3), 245, dtype=np.uint8)
    blown_res = filter_engine.evaluate(blown_out)
    assert not blown_res.passed
    assert "overexposed" in blown_res.rejection_reason
