# SIVIA Progress Log

## Current Status: Milestone 0 (Bootstrap, Environment, and Guardrails) — COMPLETE

### Milestone Tracker
- [x] **M0: Bootstrap, Environment, and Guardrails** (Current)
- [ ] **M1: Data Capture, Frame Sampling, and Embedding-Based Dedup**
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

### Completed Tasks (Milestone 0)
- **M0.1:** Created complete repository structure matching ROADMAP §4 with modules for capture, store, embeddings, labeling, qa, training, evaluation, optimize, serving, kit, reasoning, monitoring, loop, and ui.
- **M0.2:** Configured `pyproject.toml` with PEP 621 metadata, dependency groups (`core`, `labeling`, `train`, `serve`, `monitor`, `dev`), and `Makefile` targets.
- **M0.3:** Added `.pre-commit-config.yaml` with Ruff linter, Ruff formatter, trailing-whitespace, end-of-file, and large-file check (<= 5 MB).
- **M0.4:** Authored and verified `scripts/probe_hardware.py` which auto-detects hardware and writes `configs/compute/auto.yaml`.
- **M0.5:** Implemented SQLite schema (`src/sivia/store/schema.sql`) and typed repository (`src/sivia/store/repository.py`) supporting full CRUD on samples, labels, runs, models, and telemetry events with SQLite foreign keys and WAL mode.
- **M0.6:** Implemented OmegaConf/Hydra configuration loader (`sivia.utils.config`) and global deterministic seeding (`sivia.utils.seed.seed_everything`).
- **M0.7:** Initialized Git repository, configured `.gitignore` (guaranteeing private roadmaps and weights are never committed), set up DVC pipeline (`dvc.yaml`), and configured local SQLite MLflow tracking.
- **M0.8:** Created synthetic toy fixture dataset (`tests/fixtures/toy/`) containing 10 images with COCO annotations for smoke testing.
- **M0.9:** Created GitHub Actions workflow (`.github/workflows/ci.yml`) for multi-python CI with CPU-optimized PyTorch test execution.
- **M0.10:** Authored `configs/kits/desk_kit_v1.yaml` and built robust Pydantic validator `sivia.kit.spec.KitSpec` enforcing normalized slot boundaries, class consistency, and rule validation.

---

### Blockers / Risks
- None currently. All M0 components initialized cleanly.

---

### Metrics Ledger
| Milestone | Key Metric | Target | Measured / Value | Status |
|---|---|---|---|---|
| **M0** | Hardware Probe | Valid auto.yaml | Generated (`configs/compute/auto.yaml`) | PASSED |
| **M0** | Store CRUD Unit Tests | 100% Pass | All CRUD + FK constraint tests passing | PASSED |
| **M0** | Kit Spec Validation | Reject malformed | Comprehensive unit tests passing | PASSED |
| **M0** | Smoke Test Fixture | 10 toy samples | Generated (10 imgs + COCO json) | PASSED |
