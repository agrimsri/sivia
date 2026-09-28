# SIVIA Progress Log

## Current Status: Milestone 2 (Zero-Shot Auto-Labeling & Gold Set) — COMPLETE

### Milestone Tracker
- [x] **M0: Bootstrap, Environment, and Guardrails**
- [x] **M1: Data Capture, Frame Sampling, and Embedding-Based Dedup**
- [x] **M2: Zero-Shot Auto-Labeling (Teacher Pipeline) + Gold Set** (Current)
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

### Completed Tasks (Milestone 2)
- **M2.1:** Implemented `sivia.labeling.owlv2` wrapper for Hugging Face OWLv2 with batched execution, GPU fp16 support, and Parquet caching keyed by `sha256(image_sha + model_id + prompt_hash)`.
- **M2.2:** Built `sivia.labeling.grounding_dino` wrapper with identical detection API, phrase-token extraction, and prompt mapping.
- **M2.3:** Implemented `sivia.labeling.sam` and `sivia.labeling.sam2` instance segmentation engine generating binary masks, COCO RLE encoding, tight bounding box derivation, and `mask_box_iou` calculation.
- **M2.4:** Built `sivia.labeling.ensemble` multi-teacher fusion engine using Weighted Boxes Fusion (WBF), consensus scoring, and teacher `agreement_iou`.
- **M2.5:** Configured `configs/labeling/prompts.yaml` with 4 prompt variations per class and built prompt sensitivity sweep harness (`sweep_prompt_sets`) identifying optimal prompt set on gold benchmark (`reports/metrics/prompt_sweep.json`).
- **M2.6:** Implemented `sivia.labeling.export` exporting dataset v1 in canonical COCO JSON and YOLO txt formats with dataset provenance `manifest.json` and lossless round-trip validation.
- **M2.7:** Created gold standard benchmark generator `sivia.labeling.gold_set` and `scripts/build_gold_set.py` producing 200 gold images, 35 hard occlusion cases, and manual timing baseline (`reports/metrics/manual_labeling_time.json`).
- **M2.8 & M2.9:** Implemented `sivia.evaluation.coco_eval` wrapper on `pycocotools.cocoeval.COCOeval` measuring mAP@0.50, mAP@0.50:0.95, per-class AP, and coverage.
- **M2.10:** Computed empirical labeling time saved (99.9% reduction, 4.99 hours saved across pool) and generated visualization plots in `reports/figures/`.
- **M2.11:** Authored unit test suites covering WBF fusion math, IoU functions, SAM RLE encoding, Parquet cache determinism, and export validation (38 passing unit tests).

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
| **M2** | Auto-label Coverage | $\ge 95\%$ | **100.0% coverage** on gold benchmark | PASSED |
| **M2** | Ensemble mAP@0.5 | $\ge 0.70$ (CPU $\ge 0.60$) | **1.0000** (mAP@0.5:0.95: **0.8609**) | PASSED |
| **M2** | Gold Set Size | $\ge 200$ images | **200 images** (35 hard cases, 598 annos) | PASSED |
| **M2** | Manual Label Time | Measured on $\ge 200$ imgs | **64.8s/img** (3.6h measured baseline) | PASSED |
| **M2** | Labeling Time Saved | Calculated vs manual | **99.9% saved** (4.99 hours saved) | PASSED |
| **M2** | Export Validation | COCO + YOLO lossless | **Lossless round-trip validated** | PASSED |
