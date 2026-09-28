"""SIVIA video capture, sampling, quality filtering, and deduplication package."""

from sivia.capture.dedup import DedupReport, FrameCandidate, compute_phash, deduplicate_frames
from sivia.capture.frame_sampler import FrameSampler, SampledFrame
from sivia.capture.quality_filter import (
    QualityFilter,
    QualityResult,
    compute_blur_score,
    compute_brightness,
)
from sivia.capture.session_split import SplitSummary, assign_and_freeze_splits, partition_sessions
from sivia.capture.video_ingest import (
    VideoMetadata,
    discover_videos,
    get_video_metadata,
    read_frames,
)

__all__ = [
    "VideoMetadata",
    "get_video_metadata",
    "read_frames",
    "discover_videos",
    "FrameSampler",
    "SampledFrame",
    "QualityFilter",
    "QualityResult",
    "compute_blur_score",
    "compute_brightness",
    "FrameCandidate",
    "DedupReport",
    "compute_phash",
    "deduplicate_frames",
    "partition_sessions",
    "assign_and_freeze_splits",
    "SplitSummary",
]
