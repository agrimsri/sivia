# SIVIA Architecture Decision Records (ADRs)

This document records architectural, algorithmic, and design decisions made throughout SIVIA's development.
Format: **Context → Options → Decision → Consequence**.

---

## ADR-001: Package & Virtual Environment Management via `uv`

- **Context:** The project requires managing multiple dependency groups (core computer vision, teacher foundation models, student training, serving, and monitoring) across fast local development and CI pipelines.
- **Options Considered:**
  1. `conda / mamba`: Good binary management, but slow environment solving, heavy footprint, and awkward in GitHub Actions.
  2. `pip + virtualenv`: Standard, but lacks unified dependency locking and is slow at wheel resolution.
  3. `uv` (Astral): Extremely fast Rust-based resolver/installer, supports PEP 621/PEP 735, seamless fallback to standard pip.
- **Decision:** Use `uv` as the primary environment and package manager, with standard `pip` compatibility in fallback environments.
- **Consequence:** Dependency installations and CI runs are an order of magnitude faster. Developers without `uv` can still install via `pip install -e .`.

---

## ADR-002: Relational Metadata Store via SQLite with WAL Mode

- **Context:** The system needs persistent storage for video frames, bounding box annotations, experiment runs, model versions, and drift telemetry events without requiring an external database daemon.
- **Options Considered:**
  1. PostgreSQL / MySQL: Enterprise-grade, but requires running background services, extra Docker containers, and complex credentials.
  2. File-based JSON / Parquet only: Simple, but lacks atomic updates, foreign keys, and indexed relational queries.
  3. Embedded SQLite: Zero-configuration single file, supports ACID transactions, foreign keys, and fast index lookups.
- **Decision:** Use SQLite (`sivia.db`) with Foreign Keys enabled and Write-Ahead Logging (`WAL` mode) for concurrency.
- **Consequence:** Zero operational overhead; fully reproducible on any machine; thread-safe for both web serving and offline pipelines.

---

## ADR-003: Compute Profile Strategy for Local 4GB VRAM Hardware

- **Context:** The local development machine possesses an NVIDIA GeForce RTX 3050 Laptop GPU (3.68 GB usable VRAM) and an Intel 12th Gen i5 CPU (12 vCPUs). ROADMAP §3 specifies Profile A for >= 8 GB VRAM and Profile C for CPU / low VRAM.
- **Options Considered:**
  1. Force Profile A: High risk of Out-Of-Memory (OOM) during teacher labeling (OWLv2 + SAM 2).
  2. Pure CPU Profile C: Fails to utilize available CUDA tensor cores for edge inference and small student training.
  3. Profile C with CUDA Acceleration: Configure conservative batch sizes (teacher: 1, student: 4, img_size: 416), enable mixed-precision FP16, and run large teacher ensembling or heavy batches on Colab (Profile B) if local VRAM saturates.
- **Decision:** Select Profile C with CUDA enabled. Let `scripts/probe_hardware.py` dynamically determine device and conservative batch sizes.
- **Consequence:** Stable execution without CUDA OOM errors while preserving GPU acceleration for lightweight tasks.

---

## ADR-004: Kit Specification Validation with Pydantic v2

- **Context:** Workbench inspection relies on strict kit definitions: object classes, natural language prompts, normalized bounding box slot regions on inspection mats, and completeness rules.
- **Options Considered:**
  1. Plain YAML dictionary: Fragile, errors only discovered downstream during inference.
  2. JSONSchema: Standard, but verbose and lacks Pythonic methods and helper properties.
  3. Pydantic v2: Fast compiled Rust core, expressive validation decorators, supports custom geometric constraints (e.g. `0 <= x1 < x2 <= 1.0`).
- **Decision:** Build `KitSpec` and child models in Pydantic v2.
- **Consequence:** Immediate validation failure with clear, human-readable error messages on malformed coordinates, missing classes, or duplicate identifiers.

---

## ADR-005: GitHub Actions CPU-Optimized PyTorch Wheel Strategy

- **Context:** In GitHub Actions runners, downloading full CUDA-enabled PyTorch wheels adds ~4 GB of network bandwidth per build, causing long CI runtimes and high failure rates.
- **Options Considered:**
  1. Download full CUDA PyTorch in CI: Very slow (~5-8 minutes wasted downloading CUDA libraries not usable on GitHub CPU runners).
  2. Install CPU-only PyTorch wheel (`--index-url https://download.pytorch.org/whl/cpu`): Downloads only ~150 MB.
- **Decision:** Explicitly install the CPU wheel index in CI prior to installing project dependencies.
- **Consequence:** GitHub Actions CI job completes in under 60 seconds.

---

## ADR-006: Headless Cloud GPU Execution via `colab` CLI (Profile B)

- **Context:** Heavy training and zero-shot foundation teacher labeling (OWLv2, Grounding DINO, SAM 2) require >= 8–15 GB VRAM. The local laptop GPU has 3.68 GB usable VRAM, which limits batch size and risks CUDA out-of-memory errors on heavy vision-foundation models.
- **Options Considered:**
  1. Manual web browser Colab notebooks: Requires manual browser UI interaction, copying code cells, and manual file uploads/downloads.
  2. Local-only execution (Profile C): Constrains models to smaller variants (OWL-ViT base, MobileSAM) and nano detectors, increasing training wall-clock time.
  3. Paid cloud VMs (AWS EC2 / GCP Compute Engine): Adds recurring cloud cost, IAM credential overhead, and network configuration.
  4. Headless `colab` CLI (`colab new`, `colab exec`, `colab run`, `colab download`): Direct command-line provisioning of free-tier Tesla T4 GPUs (15,360 MiB / 15 GB VRAM) directly from terminal scripts.
- **Decision:** Standardize Profile B compute on the `colab` CLI (`/home/agrim/.local/bin/colab`). Verified live on-device with Tesla T4 allocation (`colab new -s sivia-gpu --gpu T4`).
  - **Heavy Workloads (Remote Colab via CLI):**
    - M2 Teacher zero-shot auto-labeling (`colab exec -s sivia-gpu -f scripts/run_teacher.py`)
    - M4 / M5 Student detector distillation (`colab run --gpu T4 scripts/train_student.py`)
    - Ephemeral artifact download (`colab download -s sivia-gpu /content/sivia/models/student.pt models/`)
    - Session termination to preserve compute (`colab stop -s sivia-gpu`)
  - **Edge Workloads (Local CPU/GPU):**
    - M1 Video capture, frame sampling, Laplacian blur filter
    - M3 Active learning uncertainty ranking & review queue
    - M7 ONNX export, INT8 static QDQ quantization, and latency benchmarking
    - M8 FastAPI server, zero-downtime hot-swap, and live camera inspection demo
- **Consequence:** 100% headless automation from terminal/agents, zero cloud infrastructure cost, access to 15 GB VRAM on demand, and realistic edge deployment benchmarking on local hardware.

---

## ADR-007: Two-Stage Session-Aware Deduplication and Partitioning

- **Context:** Raw continuous video sampling captures high-frequency near-identical frames. Random frame-level train/test splitting leads to catastrophic data leakage (near-identical frames in both train and test splits).
- **Options Considered:**
  1. Global clustering deduplication: Collapses diverse sessions if background surfaces match, reducing test set diversity.
  2. Pure pHash deduplication: Fails to detect semantic duplicates where camera angles or minor lighting changes alter hash bits.
  3. Two-stage session-aware deduplication + session-level splitting:
     - Stage 1: pHash exact duplicate pruning within session.
     - Stage 2: DINOv2 cosine similarity radius pruning (threshold 0.95), retaining the sharpest exemplar per cluster.
     - Split policy: Partitioning strictly by whole video session ID (70/15/15), freezing the test split permanently.
- **Decision:** Adopt two-stage session-aware deduplication and session-level splitting.
- **Consequence:** Zero test leakage, >75% reduction in redundant frames, and strict preservation of diverse recording sessions.
