"""Video ingestion and metadata extraction module for SIVIA (Task M1.2).

Reads video streams via OpenCV / PyAV, extracts video container metadata,
and yields frames with sequential frame index and timestamps.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    path: str
    session_id: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
    codec: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def get_video_metadata(video_path: str | Path, session_id: str | None = None) -> VideoMetadata:
    """Extract metadata from video file using OpenCV."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip() or "mp4v"
        duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0

        derived_session = session_id or path.parent.name
        if derived_session in ("videos", "raw", ".", ""):
            derived_session = path.stem

        return VideoMetadata(
            path=str(path.resolve()),
            session_id=derived_session,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
            width=width,
            height=height,
            codec=codec,
        )
    finally:
        cap.release()


def read_frames(
    video_path: str | Path,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, timestamp_sec, frame_bgr) for all frames in video."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            # Calculate timestamp in seconds
            ts = round(frame_idx / fps, 4)
            yield frame_idx, ts, frame
            frame_idx += 1
    finally:
        cap.release()


def discover_videos(directory: str | Path) -> list[Path]:
    """Find all video files (*.mp4, *.avi, *.mov, *.mkv) in directory tree."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    videos: list[Path] = []
    for ext in extensions:
        videos.extend(dir_path.rglob(ext))
    return sorted(videos)
