.PHONY: help setup lint format test label train eval export serve monitor loop demo clean check-readme

SHELL := /bin/bash
PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

help:
	@echo "SIVIA (Self-Improving Visual Inspection Agent) Makefile"
	@echo "Targets:"
	@echo "  setup         Set up virtualenv and install dependencies via uv"
	@echo "  lint          Run ruff linting and mypy type checks"
	@echo "  format        Format code using ruff"
	@echo "  test          Run unit and integration test suite"
	@echo "  label         Run teacher zero-shot auto-labeling pipeline"
	@echo "  train         Run student detector training"
	@echo "  eval          Run model evaluation and robustness benchmarks"
	@echo "  export        Export student model to ONNX / INT8 quantized runtime"
	@echo "  serve         Start FastAPI inference server"
	@echo "  monitor       Run drift detection and hard sample mining"
	@echo "  loop          Run closed-loop retrain and promotion cycle"
	@echo "  demo          Launch Gradio demo UI"
	@echo "  clean         Remove build artifacts and temporary caches"

setup:
	@which uv >/dev/null 2>&1 || (echo "uv not found, installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh)
	uv venv $(VENV) --python 3.12
	uv pip install --upgrade pip
	uv pip install -e ".[dev,core]"
	uv pip install dvc || true
	@if [ -d .git ]; then \
		$(BIN)/pre-commit install || true; \
	fi
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

lint:
	@if [ -f $(BIN)/ruff ]; then \
		$(BIN)/ruff check src tests scripts; \
		$(BIN)/ruff format --check src tests scripts; \
	else \
		ruff check src tests scripts; \
		ruff format --check src tests scripts; \
	fi

format:
	@if [ -f $(BIN)/ruff ]; then \
		$(BIN)/ruff format src tests scripts; \
		$(BIN)/ruff check --fix src tests scripts; \
	else \
		ruff format src tests scripts; \
		ruff check --fix src tests scripts; \
	fi

test:
	@if [ -f $(BIN)/pytest ]; then \
		$(BIN)/pytest tests; \
	else \
		pytest tests; \
	fi

label:
	@echo "Running teacher labeling pipeline..."
	$(PYTHON) -m sivia.labeling.run

train:
	@echo "Running student training pipeline..."
	$(PYTHON) -m sivia.training.run

eval:
	@echo "Running evaluation suite..."
	$(PYTHON) -m sivia.evaluation.run

export:
	@echo "Running export and quantization pipeline..."
	$(PYTHON) -m sivia.optimize.run

serve:
	@echo "Starting SIVIA serving API..."
	uvicorn sivia.serving.api:app --host 0.0.0.0 --port 8000 --reload

monitor:
	@echo "Running monitoring pipeline..."
	$(PYTHON) -m sivia.monitoring.run

loop:
	@echo "Running closed-loop orchestration cycle..."
	$(PYTHON) -m sivia.loop.orchestrator

demo:
	@echo "Launching Gradio demo application..."
	$(PYTHON) -m sivia.ui.gradio_app

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete

check-readme:
	@echo "Checking README metrics consistency against reports/metrics..."
	$(PYTHON) scripts/check_readme_metrics.py
