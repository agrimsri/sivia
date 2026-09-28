"""Unit tests for session-level dataset partitioning."""

from pathlib import Path

from sivia.capture.session_split import assign_and_freeze_splits, partition_sessions
from sivia.store.repository import Sample, SiviaStore


def test_partition_sessions_zero_leakage():
    """Verify session partitioning guarantees mutually exclusive sets."""
    sessions = [f"session_{i:02d}" for i in range(20)]
    train_s, val_s, test_s = partition_sessions(sessions, train_ratio=0.70, val_ratio=0.15, seed=42)

    assert len(train_s) > 0
    assert len(val_s) > 0
    assert len(test_s) > 0

    # Strict zero overlap
    assert len(train_s & val_s) == 0
    assert len(train_s & test_s) == 0
    assert len(val_s & test_s) == 0
    assert (train_s | val_s | test_s) == set(sessions)


def test_assign_and_freeze_splits(tmp_path: Path):
    """Verify split assignment updates store and exports frozen test split file."""
    store = SiviaStore(":memory:")
    samples: list[Sample] = []

    for i in range(15):
        sess = f"session_{i % 5:02d}"
        s = Sample(
            id=None,
            path=f"data/frames/frame_{i:03d}.jpg",
            source_video=sess,
            split="unassigned",
        )
        s_id = store.add_sample(s)
        s.id = s_id
        samples.append(s)

    frozen_path = tmp_path / "frozen_test.txt"
    summary = assign_and_freeze_splits(store, samples, frozen_test_path=frozen_path, seed=42)

    assert summary.train_count > 0
    assert summary.val_count > 0
    assert summary.test_count > 0
    assert summary.train_count + summary.val_count + summary.test_count == 15

    # Check frozen test file exists and contains valid paths
    assert frozen_path.exists()
    lines = [line.strip() for line in frozen_path.read_text().splitlines() if line.strip()]
    assert len(lines) == summary.test_count
