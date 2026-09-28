"""Unit tests for two-stage perceptual and semantic deduplication."""

from pathlib import Path

import numpy as np
from PIL import Image

from sivia.capture.dedup import FrameCandidate, deduplicate_frames


def test_dedup_removes_injected_duplicates(tmp_path: Path):
    """Verify that identical and jittered duplicate images are pruned, keeping sharpest exemplar."""
    # Create 3 distinct base images
    candidates: list[FrameCandidate] = []

    # Cluster A: cross pattern + slightly noisy copy
    img_a1 = Image.new("RGB", (100, 100), color=(240, 240, 240))
    for x in range(100):
        img_a1.putpixel((x, 50), (20, 20, 20))
        img_a1.putpixel((50, x), (20, 20, 20))
    p_a1 = tmp_path / "a1.jpg"
    img_a1.save(p_a1)

    img_a2 = img_a1.copy()
    img_a2.putpixel((51, 51), (30, 30, 30))  # minor jitter
    p_a2 = tmp_path / "a2.jpg"
    img_a2.save(p_a2)

    vec_a = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)

    candidates.append(
        FrameCandidate(1, str(p_a1), blur_score=85.0, brightness=100.0, embedding=vec_a)
    )
    candidates.append(
        FrameCandidate(2, str(p_a2), blur_score=40.0, brightness=101.0, embedding=vec_a)
    )

    # Cluster B: diagonal line pattern (completely different pHash)
    img_b = Image.new("RGB", (100, 100), color=(240, 240, 240))
    for x in range(100):
        img_b.putpixel((x, x), (20, 20, 20))
    p_b = tmp_path / "b.jpg"
    img_b.save(p_b)
    vec_b = np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)

    candidates.append(
        FrameCandidate(3, str(p_b), blur_score=75.0, brightness=120.0, embedding=vec_b)
    )

    report = deduplicate_frames(candidates, phash_max_dist=4, cosine_similarity_threshold=0.90)

    # 3 inputs, 1 duplicate removed -> 2 retained
    assert report.total_input == 3
    assert report.retained_count == 2
    assert report.reduction_percentage > 30.0

    # Verify the retained exemplar for Cluster A is candidate 1 (sharpest: blur_score 85 vs 40)
    retained_ids = [c.candidate_id for c in report.retained_candidates]
    assert 1 in retained_ids
    assert 3 in retained_ids
    assert 2 not in retained_ids
