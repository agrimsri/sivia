"""Quality filtering module for visual inspection frames (Task M1.4).

Computes variance of Laplacian (blur metric) and mean brightness to filter out
unusable or degraded frames before entering the labeling pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityResult:
    passed: bool
    blur_score: float
    brightness: float
    rejection_reason: str | None = None


def compute_blur_score(frame_bgr: np.ndarray) -> float:
    """Calculate sharpness score via variance of Laplacian."""
    if len(frame_bgr.shape) == 3:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return round(score, 2)


def compute_brightness(frame_bgr: np.ndarray) -> float:
    """Calculate average luminance/brightness in [0.0, 255.0]."""
    if len(frame_bgr.shape) == 3:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr
    return round(float(np.mean(gray)), 2)


class QualityFilter:
    """Evaluates frames against sharpness and lighting constraints."""

    def __init__(
        self,
        min_blur_score: float = 35.0,
        min_brightness: float = 35.0,
        max_brightness: float = 230.0,
    ) -> None:
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def evaluate(self, frame_bgr: np.ndarray) -> QualityResult:
        blur = compute_blur_score(frame_bgr)
        brightness = compute_brightness(frame_bgr)

        reasons: list[str] = []
        if brightness < self.min_brightness:
            reasons.append(f"underexposed ({brightness:.1f} < min {self.min_brightness:.1f})")
        elif brightness > self.max_brightness:
            reasons.append(f"overexposed ({brightness:.1f} > max {self.max_brightness:.1f})")

        if blur < self.min_blur_score:
            reasons.append(f"blur_score ({blur:.1f} < min {self.min_blur_score:.1f})")

        rejection_str = "; ".join(reasons) if reasons else None
        passed = len(reasons) == 0
        return QualityResult(
            passed=passed,
            blur_score=blur,
            brightness=brightness,
            rejection_reason=rejection_str,
        )
