"""Frame sampling module combining fixed FPS and scene-change detection (Task M1.3)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from sivia.capture.video_ingest import read_frames


@dataclass
class SampledFrame:
    frame_idx: int
    ts: float
    frame_bgr: np.ndarray
    trigger: str  # "fixed_fps" or "scene_change"


class FrameSampler:
    """Samples video frames at a target FPS with scene-change triggering."""

    def __init__(
        self,
        target_fps: float = 2.0,
        scene_change_threshold: float = 30.0,
        min_interval_sec: float = 0.25,
    ) -> None:
        self.target_fps = target_fps
        self.interval_sec = 1.0 / target_fps if target_fps > 0 else 0.5
        self.scene_change_threshold = scene_change_threshold
        self.min_interval_sec = min_interval_sec

    def sample_from_video(self, video_path: str) -> Iterator[SampledFrame]:
        """Process video and yield sampled frames."""
        last_fixed_ts = -999.0
        last_sampled_ts = -999.0
        prev_small_gray: np.ndarray | None = None

        for frame_idx, ts, frame in read_frames(video_path):
            # Compute small grayscale thumbnail for fast scene change detection
            small_gray = cv2.cvtColor(
                cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )

            is_sample = False
            trigger = ""

            # Check fixed FPS
            if (ts - last_fixed_ts) >= (self.interval_sec - 1e-4):
                is_sample = True
                trigger = "fixed_fps"
                last_fixed_ts = ts

            # Check scene change if not already sampled by fixed fps
            elif prev_small_gray is not None and (ts - last_sampled_ts) >= self.min_interval_sec:
                diff = float(
                    np.mean(np.abs(small_gray.astype(float) - prev_small_gray.astype(float)))
                )
                if diff >= self.scene_change_threshold:
                    is_sample = True
                    trigger = "scene_change"

            prev_small_gray = small_gray

            if is_sample:
                last_sampled_ts = ts
                yield SampledFrame(
                    frame_idx=frame_idx,
                    ts=ts,
                    frame_bgr=frame,
                    trigger=trigger,
                )
