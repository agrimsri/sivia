"""SQLite data store and repository for SIVIA.

Implements the core data contracts specified in Section 5.1 of ROADMAP.md.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

SplitType = Literal["train", "val", "test", "unassigned"]
OriginType = Literal["capture", "mined", "synthetic"]
LabelSourceType = Literal["owlv2", "gdino", "sam2", "ensemble", "human", "student"]
LabelStatusType = Literal["auto", "flagged", "reviewed", "rejected"]
RunKindType = Literal["label", "train", "distill", "eval", "export"]
ModelFormatType = Literal["pt", "onnx", "onnx_int8", "openvino"]
ModelStageType = Literal["candidate", "champion", "archived"]
EventKindType = Literal["inference", "drift", "retrain", "swap", "rollback"]


@dataclass
class Sample:
    id: int | None
    path: str
    source_video: str | None = None
    frame_idx: int | None = None
    ts: float | None = None
    sha256: str | None = None
    phash: str | None = None
    embedding_id: int | None = None
    split: SplitType = "unassigned"
    brightness: float | None = None
    blur_score: float | None = None
    origin: OriginType = "capture"
    created_at: str | None = None


@dataclass
class Label:
    id: int | None
    sample_id: int
    class_id: int
    bbox_xyxy: list[float]  # [x1, y1, x2, y2]
    mask_rle: str | None = None
    source: LabelSourceType = "ensemble"
    score: float | None = None
    agreement_iou: float | None = None
    status: LabelStatusType = "auto"
    dataset_version: str | None = None
    created_at: str | None = None


@dataclass
class Run:
    id: int | None
    kind: RunKindType
    mlflow_run_id: str | None = None
    git_sha: str | None = None
    config: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    created_at: str | None = None


@dataclass
class ModelRecord:
    id: int | None
    name: str
    version: str
    format: ModelFormatType
    path: str
    parent_run_id: int | None = None
    stage: ModelStageType = "candidate"
    metrics: dict[str, Any] | None = None
    created_at: str | None = None


@dataclass
class Event:
    id: int | None
    ts: float
    kind: EventKindType
    payload: dict[str, Any] | None = None
    created_at: str | None = None


class SiviaStore:
    """Thread-safe SQLite store for metadata, datasets, models, and telemetry."""

    def __init__(self, db_path: str | Path = "sivia.db") -> None:
        self.db_path = str(db_path)
        self._is_memory = self.db_path == ":memory:" or "mode=memory" in self.db_path
        self._persistent_conn: sqlite3.Connection | None = None
        if self._is_memory:
            self._persistent_conn = sqlite3.connect(self.db_path)
            self._persistent_conn.row_factory = sqlite3.Row
            self._persistent_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def init_db(self) -> None:
        """Initialize database tables using schema.sql."""
        with open(SCHEMA_PATH) as f:
            schema_sql = f.read()
        conn = self.get_connection()
        conn.executescript(schema_sql)
        conn.commit()

    # -------------------------------------------------------------------------
    # Samples CRUD
    # -------------------------------------------------------------------------
    def add_sample(self, sample: Sample) -> int:
        query = """
            INSERT INTO samples (
                path, source_video, frame_idx, ts, sha256, phash,
                embedding_id, split, brightness, blur_score, origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cur = conn.execute(
                query,
                (
                    sample.path,
                    sample.source_video,
                    sample.frame_idx,
                    sample.ts,
                    sample.sha256,
                    sample.phash,
                    sample.embedding_id,
                    sample.split,
                    sample.brightness,
                    sample.blur_score,
                    sample.origin,
                ),
            )
            sample_id = cur.lastrowid
            assert sample_id is not None
            sample.id = sample_id
            return sample_id

    def add_samples(self, samples: list[Sample]) -> list[int]:
        ids: list[int] = []
        with self.get_connection() as conn:
            query = """
                INSERT INTO samples (
                    path, source_video, frame_idx, ts, sha256, phash,
                    embedding_id, split, brightness, blur_score, origin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            for s in samples:
                cur = conn.execute(
                    query,
                    (
                        s.path,
                        s.source_video,
                        s.frame_idx,
                        s.ts,
                        s.sha256,
                        s.phash,
                        s.embedding_id,
                        s.split,
                        s.brightness,
                        s.blur_score,
                        s.origin,
                    ),
                )
                new_id = cur.lastrowid
                assert new_id is not None
                s.id = new_id
                ids.append(new_id)
        return ids

    def get_sample(self, sample_id: int) -> Sample | None:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
            if not row:
                return None
            return self._row_to_sample(row)

    def get_sample_by_path(self, path: str) -> Sample | None:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM samples WHERE path = ?", (path,)).fetchone()
            if not row:
                return None
            return self._row_to_sample(row)

    def list_samples(
        self, split: SplitType | None = None, origin: OriginType | None = None
    ) -> list[Sample]:
        query = "SELECT * FROM samples WHERE 1=1"
        params: list[Any] = []
        if split:
            query += " AND split = ?"
            params.append(split)
        if origin:
            query += " AND origin = ?"
            params.append(origin)
        query += " ORDER BY id ASC"

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_sample(r) for r in rows]

    def update_sample_split(self, sample_id: int, split: SplitType) -> None:
        with self.get_connection() as conn:
            conn.execute("UPDATE samples SET split = ? WHERE id = ?", (split, sample_id))

    def count_samples(self, split: SplitType | None = None) -> int:
        query = "SELECT COUNT(*) FROM samples WHERE 1=1"
        params: list[Any] = []
        if split:
            query += " AND split = ?"
            params.append(split)
        with self.get_connection() as conn:
            return int(conn.execute(query, params).fetchone()[0])

    # -------------------------------------------------------------------------
    # Labels CRUD
    # -------------------------------------------------------------------------
    def add_label(self, label: Label) -> int:
        bbox_json = json.dumps(label.bbox_xyxy)
        query = """
            INSERT INTO labels (
                sample_id, class_id, bbox_xyxy, mask_rle, source,
                score, agreement_iou, status, dataset_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cur = conn.execute(
                query,
                (
                    label.sample_id,
                    label.class_id,
                    bbox_json,
                    label.mask_rle,
                    label.source,
                    label.score,
                    label.agreement_iou,
                    label.status,
                    label.dataset_version,
                ),
            )
            label_id = cur.lastrowid
            assert label_id is not None
            label.id = label_id
            return label_id

    def add_labels(self, labels: list[Label]) -> list[int]:
        ids: list[int] = []
        with self.get_connection() as conn:
            query = """
                INSERT INTO labels (
                    sample_id, class_id, bbox_xyxy, mask_rle, source,
                    score, agreement_iou, status, dataset_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            for lab in labels:
                bbox_json = json.dumps(lab.bbox_xyxy)
                cur = conn.execute(
                    query,
                    (
                        lab.sample_id,
                        lab.class_id,
                        bbox_json,
                        lab.mask_rle,
                        lab.source,
                        lab.score,
                        lab.agreement_iou,
                        lab.status,
                        lab.dataset_version,
                    ),
                )
                lid = cur.lastrowid
                assert lid is not None
                lab.id = lid
                ids.append(lid)
        return ids

    def get_labels_for_sample(self, sample_id: int) -> list[Label]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM labels WHERE sample_id = ? ORDER BY id ASC", (sample_id,)
            ).fetchall()
            return [self._row_to_label(r) for r in rows]

    def update_label_status(self, label_id: int, status: LabelStatusType) -> None:
        with self.get_connection() as conn:
            conn.execute("UPDATE labels SET status = ? WHERE id = ?", (status, label_id))

    def list_labels(
        self,
        sample_id: int | None = None,
        status: LabelStatusType | None = None,
        source: LabelSourceType | None = None,
        dataset_version: str | None = None,
    ) -> list[Label]:
        query = "SELECT * FROM labels WHERE 1=1"
        params: list[Any] = []
        if sample_id is not None:
            query += " AND sample_id = ?"
            params.append(sample_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        if source:
            query += " AND source = ?"
            params.append(source)
        if dataset_version:
            query += " AND dataset_version = ?"
            params.append(dataset_version)
        query += " ORDER BY id ASC"

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_label(r) for r in rows]

    # -------------------------------------------------------------------------
    # Runs CRUD
    # -------------------------------------------------------------------------
    def add_run(self, run: Run) -> int:
        cfg_json = json.dumps(run.config) if run.config else None
        metrics_json = json.dumps(run.metrics) if run.metrics else None
        query = """
            INSERT INTO runs (kind, mlflow_run_id, git_sha, config_json, metrics_json)
            VALUES (?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cur = conn.execute(
                query, (run.kind, run.mlflow_run_id, run.git_sha, cfg_json, metrics_json)
            )
            run_id = cur.lastrowid
            assert run_id is not None
            run.id = run_id
            return run_id

    def get_run(self, run_id: int) -> Run | None:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            return self._row_to_run(row)

    def update_run_metrics(self, run_id: int, metrics: dict[str, Any]) -> None:
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE runs SET metrics_json = ? WHERE id = ?",
                (json.dumps(metrics), run_id),
            )

    # -------------------------------------------------------------------------
    # Models CRUD
    # -------------------------------------------------------------------------
    def add_model(self, model: ModelRecord) -> int:
        metrics_json = json.dumps(model.metrics) if model.metrics else None
        query = """
            INSERT INTO models (name, version, format, path, parent_run_id, stage, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.get_connection() as conn:
            cur = conn.execute(
                query,
                (
                    model.name,
                    model.version,
                    model.format,
                    model.path,
                    model.parent_run_id,
                    model.stage,
                    metrics_json,
                ),
            )
            mid = cur.lastrowid
            assert mid is not None
            model.id = mid
            return mid

    def get_model(self, model_id: int) -> ModelRecord | None:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if not row:
                return None
            return self._row_to_model(row)

    def get_champion_model(self, name: str | None = None) -> ModelRecord | None:
        query = "SELECT * FROM models WHERE stage = 'champion'"
        params: list[Any] = []
        if name:
            query += " AND name = ?"
            params.append(name)
        query += " ORDER BY id DESC LIMIT 1"

        with self.get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            if not row:
                return None
            return self._row_to_model(row)

    def set_model_stage(self, model_id: int, stage: ModelStageType) -> None:
        with self.get_connection() as conn:
            conn.execute("UPDATE models SET stage = ? WHERE id = ?", (stage, model_id))

    def list_models(self, stage: ModelStageType | None = None) -> list[ModelRecord]:
        query = "SELECT * FROM models WHERE 1=1"
        params: list[Any] = []
        if stage:
            query += " AND stage = ?"
            params.append(stage)
        query += " ORDER BY id DESC"

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_model(r) for r in rows]

    # -------------------------------------------------------------------------
    # Events CRUD
    # -------------------------------------------------------------------------
    def add_event(self, event: Event) -> int:
        payload_json = json.dumps(event.payload) if event.payload else None
        query = "INSERT INTO events (ts, kind, payload_json) VALUES (?, ?, ?)"
        with self.get_connection() as conn:
            cur = conn.execute(query, (event.ts, event.kind, payload_json))
            eid = cur.lastrowid
            assert eid is not None
            event.id = eid
            return eid

    def list_events(self, kind: EventKindType | None = None, limit: int = 100) -> list[Event]:
        query = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_event(r) for r in rows]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _row_to_sample(row: sqlite3.Row) -> Sample:
        return Sample(
            id=row["id"],
            path=row["path"],
            source_video=row["source_video"],
            frame_idx=row["frame_idx"],
            ts=row["ts"],
            sha256=row["sha256"],
            phash=row["phash"],
            embedding_id=row["embedding_id"],
            split=row["split"],
            brightness=row["brightness"],
            blur_score=row["blur_score"],
            origin=row["origin"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_label(row: sqlite3.Row) -> Label:
        return Label(
            id=row["id"],
            sample_id=row["sample_id"],
            class_id=row["class_id"],
            bbox_xyxy=json.loads(row["bbox_xyxy"]),
            mask_rle=row["mask_rle"],
            source=row["source"],
            score=row["score"],
            agreement_iou=row["agreement_iou"],
            status=row["status"],
            dataset_version=row["dataset_version"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> Run:
        cfg = json.loads(row["config_json"]) if row["config_json"] else None
        metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else None
        return Run(
            id=row["id"],
            kind=row["kind"],
            mlflow_run_id=row["mlflow_run_id"],
            git_sha=row["git_sha"],
            config=cfg,
            metrics=metrics,
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> ModelRecord:
        metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else None
        return ModelRecord(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            format=row["format"],
            path=row["path"],
            parent_run_id=row["parent_run_id"],
            stage=row["stage"],
            metrics=metrics,
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        payload = json.loads(row["payload_json"]) if row["payload_json"] else None
        return Event(
            id=row["id"],
            ts=row["ts"],
            kind=row["kind"],
            payload=payload,
            created_at=row["created_at"],
        )
