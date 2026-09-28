-- SIVIA SQLite Schema
-- Core Data Contracts (Section 5.1 of ROADMAP.md)

PRAGMA foreign_keys = ON;

-- Samples Table
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    source_video TEXT,
    frame_idx INTEGER,
    ts REAL,
    sha256 TEXT,
    phash TEXT,
    embedding_id INTEGER,
    split TEXT CHECK (split IN ('train', 'val', 'test', 'unassigned')) DEFAULT 'unassigned',
    brightness REAL,
    blur_score REAL,
    origin TEXT CHECK (origin IN ('capture', 'mined', 'synthetic')) NOT NULL DEFAULT 'capture',
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_samples_sha256 ON samples(sha256);
CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split);
CREATE INDEX IF NOT EXISTS idx_samples_source_video ON samples(source_video);

-- Labels Table
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    class_id INTEGER NOT NULL,
    bbox_xyxy TEXT NOT NULL, -- JSON array string: "[x1, y1, x2, y2]"
    mask_rle TEXT,          -- COCO RLE JSON string or NULL
    source TEXT CHECK (source IN ('owlv2', 'gdino', 'sam2', 'ensemble', 'human', 'student')) NOT NULL,
    score REAL,
    agreement_iou REAL,
    status TEXT CHECK (status IN ('auto', 'flagged', 'reviewed', 'rejected')) NOT NULL DEFAULT 'auto',
    dataset_version TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_labels_sample_id ON labels(sample_id);
CREATE INDEX IF NOT EXISTS idx_labels_class_id ON labels(class_id);
CREATE INDEX IF NOT EXISTS idx_labels_status ON labels(status);
CREATE INDEX IF NOT EXISTS idx_labels_dataset_version ON labels(dataset_version);

-- Runs Table
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT CHECK (kind IN ('label', 'train', 'distill', 'eval', 'export')) NOT NULL,
    mlflow_run_id TEXT,
    git_sha TEXT,
    config_json TEXT,
    metrics_json TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_kind ON runs(kind);
CREATE INDEX IF NOT EXISTS idx_runs_mlflow ON runs(mlflow_run_id);

-- Models Table
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    format TEXT CHECK (format IN ('pt', 'onnx', 'onnx_int8', 'openvino')) NOT NULL,
    path TEXT NOT NULL,
    parent_run_id INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    stage TEXT CHECK (stage IN ('candidate', 'champion', 'archived')) NOT NULL DEFAULT 'candidate',
    metrics_json TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_models_stage ON models(stage);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);

-- Events Table
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT CHECK (kind IN ('inference', 'drift', 'retrain', 'swap', 'rollback')) NOT NULL,
    payload_json TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
