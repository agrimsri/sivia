"""Session-level dataset splitting module (Task M1.8).

Enforces strict session-level partitioning (Train: 70%, Val: 15%, Test: 15%)
to guarantee zero frame leakage between recording sessions, and writes the frozen test split.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sivia.store.repository import Sample, SiviaStore


@dataclass
class SplitSummary:
    train_sessions: list[str]
    val_sessions: list[str]
    test_sessions: list[str]
    train_count: int
    val_count: int
    test_count: int


def partition_sessions(
    session_ids: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """Deterministically partition unique session IDs into train/val/test sets."""
    unique_sessions = sorted(set(session_ids))
    if not unique_sessions:
        return set(), set(), set()

    rng = random.Random(seed)
    shuffled = list(unique_sessions)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 1:
        return set(shuffled), set(), set()
    elif n == 2:
        return {shuffled[0]}, {shuffled[1]}, set()

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))

    # Ensure all splits get represented if >= 3 sessions
    if (n_train + n_val) >= n:
        n_train = n - 2
        n_val = 1

    train_set = set(shuffled[:n_train])
    val_set = set(shuffled[n_train : n_train + n_val])
    test_set = set(shuffled[n_train + n_val :])

    # Assert zero overlap
    assert len(train_set & val_set) == 0, "Train and Val sessions overlap!"
    assert len(train_set & test_set) == 0, "Train and Test sessions overlap!"
    assert len(val_set & test_set) == 0, "Val and Test sessions overlap!"

    return train_set, val_set, test_set


def assign_and_freeze_splits(
    store: SiviaStore,
    samples: list[Sample],
    frozen_test_path: str | Path = "data/splits/frozen_test.txt",
    seed: int = 42,
) -> SplitSummary:
    """Assign splits to database samples and export frozen test split file."""
    sessions = [s.source_video or "unknown" for s in samples]
    train_sess, val_sess, test_sess = partition_sessions(sessions, seed=seed)

    counts = {"train": 0, "val": 0, "test": 0}
    test_sample_paths: list[str] = []

    for s in samples:
        sess = s.source_video or "unknown"
        if sess in train_sess:
            split_name = "train"
        elif sess in val_sess:
            split_name = "val"
        else:
            split_name = "test"
            test_sample_paths.append(s.path)

        s.split = split_name
        counts[split_name] += 1
        if s.id is not None:
            store.update_sample_split(s.id, split_name)

    # Write frozen test split
    out_file = Path(frozen_test_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for path_str in sorted(test_sample_paths):
            f.write(f"{path_str}\n")

    return SplitSummary(
        train_sessions=sorted(train_sess),
        val_sessions=sorted(val_sess),
        test_sessions=sorted(test_sess),
        train_count=counts["train"],
        val_count=counts["val"],
        test_count=counts["test"],
    )
