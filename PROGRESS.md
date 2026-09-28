# SIVIA Progress Log

## Current Status: Milestone 1 (Data Capture, Frame Sampling, and Dedup) — COMPLETE

### Milestone Tracker
- [x] **M0: Bootstrap, Environment, and Guardrails**
- [x] **M1: Data Capture, Frame Sampling, and Embedding-Based Dedup** (Current)
- [ ] **M2: Zero-Shot Auto-Labeling (Teacher Pipeline) + Gold Set**
- [ ] **M3: Label QA and Active Learning (Human-in-the-Loop)**
- [ ] **M4: Student Baseline Training + Experiment Tracking**
- [ ] **M5: Knowledge Distillation and Ablations**
- [ ] **M6: Rigorous Evaluation and Robustness Testing**
- [ ] **M7: Optimization: ONNX Export, INT8 Quantization, Benchmarking**
- [ ] **M8: Serving: Edge Runtime, REST/gRPC API, Docker, Hot-Swap**
- [ ] **M9: Reasoning Layer: VLM Explanations and Video Search**
- [ ] **M10: Monitoring, Drift Detection, and Hard-Sample Mining**
- [ ] **M11: Closed Loop: Automated Retrain, Champion/Challenger Gate, Rollback**
- [ ] **M12: "New Object in Minutes" Onboarding Workflow**
- [ ] **M13: Demo UI, Documentation, Backup Video, Interview Package**

---

### Completed Tasks (Milestone 1)
- **M1.1:** Authored interactive recording tool `scripts/record_helper.py` supporting webcam capture, on-screen checklist overlay for 6 kit classes, and session metadata generation.
- **M1.2:** Implemented `sivia.capture.video_ingest` for video container parsing and frame stream decoding with timestamps.
- **M1.3:** Built `sivia.capture.frame_sampler` combining fixed 2 FPS sampling with optical/scene-change difference triggers.
- **M1.4:** Built `sivia.capture.quality_filter` evaluating Laplacian variance (sharpness) and mean luminance to reject degraded/blurred frames.
- **M1.5:** Implemented `sivia.embeddings.dino` generating 384-dimensional L2-normalized DINOv2 ViT-S/14 CLS embeddings.
- **M1.6:** Built two-stage deduplication pipeline `sivia.capture.dedup` combining perceptual hash (pHash) exact pruning and FAISS cosine radius clustering, retaining sharpest exemplars.
- **M1.7:** Implemented `sivia.embeddings.faiss_index` maintaining persistent cosine index mapped to SQLite database IDs.
- **M1.8:** Implemented `sivia.capture.session_split` enforcing 70/15/15 session-level data partitioning, zero leakage, and exported `data/splits/frozen_test.txt`.
- **M1.9:** Authored unit tests for quality filtering, two-stage dedup, session partitioning, and FAISS indexing (`tests/unit/test_*.py`).
- **M1.10:** Implemented `scripts/generate_dataset_eda.py` and `notebooks/m1_dataset_eda.ipynb` generating quality histograms and 2D PCA cluster visualizations.

---

### Completed Tasks (Milestone 0)
- **M0.1:** Created complete repository structure matching ROADMAP §4.
- **M0.2:** Configured `pyproject.toml` with PEP 621 metadata, dependency groups, and `Makefile` targets.
- **M0.3:** Added `.pre-commit-config.yaml` with Ruff linter, Ruff formatter, and large-file check.
- **M0.4:** Authored and verified `scripts/probe_hardware.py` generating `configs/compute/auto.yaml`.
- **M0.5:** Implemented SQLite schema (`schema.sql`) and typed repository (`repository.py`) supporting full CRUD on samples, labels, runs, models, and telemetry.
- **M0.6:** Implemented Hydra/OmegaConf config loader and global deterministic seeding (`seed_everything`).
- **M0.7:** Initialized Git repository, DVC pipeline (`dvc.yaml`), and local SQLite MLflow tracking.
- **M0.8:** Created synthetic toy fixture dataset (`tests/fixtures/toy/`) for smoke testing.
- **M0.9:** Created GitHub Actions workflow (`ci.yml`) passing green on Python 3.11 & 3.12.
- **M0.10:** Authored `desk_kit_v1.yaml` and Pydantic validator `KitSpec`.

---

### Metrics Ledger
| Milestone | Key Metric | Target | Measured / Value | Status |
|---|---|---|---|---|
| **M0** | Hardware Probe | Valid auto.yaml | Generated (`configs/compute/auto.yaml`) | PASSED |
| **M0** | Store CRUD Unit Tests | 100% Pass | All CRUD + FK constraint tests passing | PASSED |
| **M0** | Kit Spec Validation | Reject malformed | Comprehensive unit tests passing | PASSED |
| **M0** | Smoke Test Fixture | 10 toy samples | Generated (10 imgs + COCO json) | PASSED |
| **M1** | Total Sessions | $\ge 20$ sessions | 20 sessions processed | PASSED |
| **M1** | Dedup Reduction | $\ge 30\%$ reduction | **79.7% reduction** (380 $\to$ 77 frames) | PASSED |
| **M1** | Session Overlap | Strict 0% leakage | **0% overlap** (Train: 14 sess, Val: 3 sess, Test: 3 sess) | PASSED |
| **M1** | Nearest Neighbor Sanity | Index search | Verified in persistent FAISS index | PASSED |
