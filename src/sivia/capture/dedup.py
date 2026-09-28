"""Two-stage image deduplication pipeline (Task M1.6).

Stage 1: Perceptual Hash (pHash) for exact and near-exact frame duplicates.
Stage 2: FAISS Cosine radius clustering on DINOv2 embeddings for semantic duplicates.
Prefers the sharpest frame (highest blur score) within each redundant cluster.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image


@dataclass
class FrameCandidate:
    candidate_id: int
    path: str
    blur_score: float
    brightness: float
    phash: str | None = None
    embedding: np.ndarray | None = None
    session_id: str | None = None


@dataclass
class DedupReport:
    total_input: int
    retained_count: int
    exact_duplicates_removed: int
    semantic_duplicates_removed: int
    reduction_percentage: float
    retained_candidates: list[FrameCandidate]


def compute_phash(img_path_or_array: str | Path | np.ndarray) -> str:
    """Compute perceptual hash (pHash) string."""
    if isinstance(img_path_or_array, (str, Path)):
        img = Image.open(img_path_or_array).convert("RGB")
    else:
        rgb = cv2.cvtColor(img_path_or_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
    return str(imagehash.phash(img))


def deduplicate_frames(
    candidates: Sequence[FrameCandidate],
    phash_max_dist: int = 2,
    cosine_similarity_threshold: float = 0.95,
    by_session: bool = True,
) -> DedupReport:
    """Perform two-stage deduplication on a collection of candidate frames.

    Args:
        candidates: Sequence of FrameCandidate objects.
        phash_max_dist: Maximum Hamming distance for pHash exact matching.
        cosine_similarity_threshold: Minimum cosine similarity to merge semantic near-duplicates.
        by_session: If True, deduplicates within each recording session to preserve cross-session diversity.

    Returns:
        DedupReport detailing counts and returning the optimal retained frames.
    """
    total = len(candidates)
    if total == 0:
        return DedupReport(0, 0, 0, 0, 0.0, [])

    if by_session and any(c.session_id for c in candidates):
        # Group by session
        sess_map: dict[str, list[FrameCandidate]] = {}
        for c in candidates:
            key = c.session_id or "default"
            sess_map.setdefault(key, []).append(c)

        all_retained: list[FrameCandidate] = []
        tot_exact = 0
        tot_semantic = 0

        for sub_candidates in sess_map.values():
            sub_rep = deduplicate_frames(
                sub_candidates,
                phash_max_dist=phash_max_dist,
                cosine_similarity_threshold=cosine_similarity_threshold,
                by_session=False,
            )
            all_retained.extend(sub_rep.retained_candidates)
            tot_exact += sub_rep.exact_duplicates_removed
            tot_semantic += sub_rep.semantic_duplicates_removed

        retained = len(all_retained)
        reduction = round(((total - retained) / total) * 100.0, 2) if total > 0 else 0.0
        return DedupReport(
            total_input=total,
            retained_count=retained,
            exact_duplicates_removed=tot_exact,
            semantic_duplicates_removed=tot_semantic,
            reduction_percentage=reduction,
            retained_candidates=all_retained,
        )

    # Sort candidates by blur_score descending so sharper frames are considered first
    sorted_candidates = sorted(candidates, key=lambda c: c.blur_score, reverse=True)

    # -------------------------------------------------------------
    # Stage 1: Perceptual Hash (pHash) Deduplication
    # -------------------------------------------------------------
    stage1_kept: list[FrameCandidate] = []
    exact_removed = 0

    for cand in sorted_candidates:
        if cand.phash is None:
            cand.phash = compute_phash(cand.path)

        cand_hash = imagehash.hex_to_hash(cand.phash)
        is_duplicate = False

        for kept in stage1_kept:
            kept_hash = imagehash.hex_to_hash(kept.phash)  # type: ignore
            if (cand_hash - kept_hash) <= phash_max_dist:
                is_duplicate = True
                exact_removed += 1
                break

        if not is_duplicate:
            stage1_kept.append(cand)

    # -------------------------------------------------------------
    # Stage 2: Semantic Embedding Cosine Radius Deduplication
    # -------------------------------------------------------------
    stage2_kept: list[FrameCandidate] = []
    semantic_removed = 0

    for cand in stage1_kept:
        if cand.embedding is None:
            # If no embedding provided, keep by default
            stage2_kept.append(cand)
            continue

        cand_vec = cand.embedding / (np.linalg.norm(cand.embedding) + 1e-12)
        is_semantic_dup = False

        for kept in stage2_kept:
            if kept.embedding is None:
                continue
            kept_vec = kept.embedding / (np.linalg.norm(kept.embedding) + 1e-12)
            cosine_sim = float(np.dot(cand_vec, kept_vec))

            if cosine_sim >= cosine_similarity_threshold:
                is_semantic_dup = True
                semantic_removed += 1
                break

        if not is_semantic_dup:
            stage2_kept.append(cand)

    retained = len(stage2_kept)
    reduction = round(((total - retained) / total) * 100.0, 2) if total > 0 else 0.0

    return DedupReport(
        total_input=total,
        retained_count=retained,
        exact_duplicates_removed=exact_removed,
        semantic_duplicates_removed=semantic_removed,
        reduction_percentage=reduction,
        retained_candidates=stage2_kept,
    )
