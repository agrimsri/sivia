"""Unit tests for SIVIA SQLite store and repository."""

import sqlite3

import pytest

from sivia.store.repository import (
    Event,
    Label,
    ModelRecord,
    Run,
    Sample,
    SiviaStore,
)


@pytest.fixture
def store() -> SiviaStore:
    """Fixture providing an in-memory SQLite store."""
    return SiviaStore(":memory:")


def test_init_db(store: SiviaStore):
    """Verify that tables and indices are created."""
    with store.get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "samples" in table_names
        assert "labels" in table_names
        assert "runs" in table_names
        assert "models" in table_names
        assert "events" in table_names


def test_sample_crud(store: SiviaStore):
    """Test inserting, retrieving, listing, and updating samples."""
    sample = Sample(
        id=None,
        path="data/raw/videos/session_01/frame_0001.jpg",
        source_video="session_01.mp4",
        frame_idx=1,
        ts=0.5,
        sha256="abc123sha",
        phash="ff00ff00",
        split="unassigned",
        brightness=120.5,
        blur_score=45.2,
        origin="capture",
    )
    sample_id = store.add_sample(sample)
    assert sample_id > 0
    assert sample.id == sample_id

    # Retrieve
    retrieved = store.get_sample(sample_id)
    assert retrieved is not None
    assert retrieved.path == "data/raw/videos/session_01/frame_0001.jpg"
    assert retrieved.brightness == pytest.approx(120.5)
    assert retrieved.split == "unassigned"

    # Update split
    store.update_sample_split(sample_id, "train")
    updated = store.get_sample(sample_id)
    assert updated is not None
    assert updated.split == "train"

    # Count
    assert store.count_samples() == 1
    assert store.count_samples(split="train") == 1
    assert store.count_samples(split="val") == 0


def test_labels_crud_and_foreign_keys(store: SiviaStore):
    """Test label creation and foreign key constraints."""
    sample = Sample(
        id=None,
        path="frame1.jpg",
        split="train",
        origin="capture",
    )
    sample_id = store.add_sample(sample)

    label1 = Label(
        id=None,
        sample_id=sample_id,
        class_id=0,
        bbox_xyxy=[10.0, 20.0, 100.0, 200.0],
        source="owlv2",
        score=0.92,
        agreement_iou=0.88,
        status="auto",
        dataset_version="v1",
    )
    label2 = Label(
        id=None,
        sample_id=sample_id,
        class_id=1,
        bbox_xyxy=[50.0, 60.0, 150.0, 160.0],
        source="ensemble",
        score=0.85,
        status="flagged",
        dataset_version="v1",
    )

    ids = store.add_labels([label1, label2])
    assert len(ids) == 2

    labels = store.get_labels_for_sample(sample_id)
    assert len(labels) == 2
    assert labels[0].bbox_xyxy == [10.0, 20.0, 100.0, 200.0]

    # Update status
    store.update_label_status(ids[1], "reviewed")
    reviewed_list = store.list_labels(status="reviewed")
    assert len(reviewed_list) == 1
    assert reviewed_list[0].id == ids[1]

    # Foreign key rejection
    orphan_label = Label(
        id=None,
        sample_id=99999,  # non-existent
        class_id=0,
        bbox_xyxy=[0, 0, 10, 10],
        source="human",
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.add_label(orphan_label)


def test_runs_and_models_lifecycle(store: SiviaStore):
    """Test tracking training runs and model stage promotion."""
    run = Run(
        id=None,
        kind="train",
        mlflow_run_id="mlflow_run_abc",
        git_sha="commit_sha_123",
        config={"batch_size": 16, "lr": 0.001},
        metrics={"mAP50": 0.72},
    )
    run_id = store.add_run(run)
    assert run_id > 0

    model = ModelRecord(
        id=None,
        name="sivia_student_detector",
        version="v1.0.0",
        format="pt",
        path="models/student_v1.pt",
        parent_run_id=run_id,
        stage="candidate",
        metrics={"mAP50": 0.72},
    )
    model_id = store.add_model(model)
    assert model_id > 0

    # Initially no champion
    champion = store.get_champion_model("sivia_student_detector")
    assert champion is None

    # Promote to champion
    store.set_model_stage(model_id, "champion")
    champion = store.get_champion_model("sivia_student_detector")
    assert champion is not None
    assert champion.id == model_id
    assert champion.version == "v1.0.0"


def test_events_logging(store: SiviaStore):
    """Test telemetry events logging and querying."""
    event1 = Event(id=None, ts=100.0, kind="inference", payload={"latency_ms": 23.4})
    event2 = Event(id=None, ts=105.0, kind="drift", payload={"p_value": 0.01})
    store.add_event(event1)
    store.add_event(event2)

    events = store.list_events()
    assert len(events) == 2
    # Ordered by ts descending
    assert events[0].kind == "drift"

    drift_events = store.list_events(kind="drift")
    assert len(drift_events) == 1
    assert drift_events[0].payload == {"p_value": 0.01}
