"""Hardware probe module for SIVIA.

Detects available CPU, GPU, VRAM, and RAM, selects the optimal compute profile
(A: Local GPU >=8GB, B: Cloud Colab/Kaggle, C: CPU / Low VRAM Laptop),
and outputs configs/compute/auto.yaml.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml


def get_cpu_info() -> dict[str, Any]:
    """Extract CPU model and core counts."""
    cpu_name = platform.processor() or "Unknown CPU"
    cores_logical = os.cpu_count() or 1
    cores_physical = cores_logical

    if platform.system() == "Linux" and Path("/proc/cpuinfo").exists():
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()
            for line in content.splitlines():
                if "model name" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
            core_ids = set()
            for line in content.splitlines():
                if "core id" in line:
                    core_ids.add(line.split(":", 1)[1].strip())
            if core_ids:
                cores_physical = len(core_ids)
        except Exception:
            pass

    return {
        "model": cpu_name,
        "logical_cores": cores_logical,
        "physical_cores": max(1, cores_physical),
    }


def get_ram_info() -> float:
    """Extract total system RAM in GB."""
    if platform.system() == "Linux" and Path("/proc/meminfo").exists():
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 * 1024), 2)
        except Exception:
            pass
    return 8.0


def get_gpu_info() -> dict[str, Any]:
    """Detect NVIDIA GPU availability, device name, and VRAM in GB."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(total_memory_bytes / (1024**3), 2)
            return {
                "available": True,
                "name": device_name,
                "vram_gb": vram_gb,
                "cuda_version": torch.version.cuda or "unknown",
            }
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        line = result.stdout.strip().splitlines()[0]
        name, memory_mb = [x.strip() for x in line.split(",")]
        vram_gb = round(float(memory_mb) / 1024.0, 2)
        return {
            "available": True,
            "name": name,
            "vram_gb": vram_gb,
            "cuda_version": "detected via nvidia-smi",
        }
    except Exception:
        pass

    return {
        "available": False,
        "name": None,
        "vram_gb": 0.0,
        "cuda_version": None,
    }


def select_profile(
    gpu_info: dict[str, Any], cpu_info: dict[str, Any], ram_gb: float
) -> dict[str, Any]:
    """Select compute profile according to Section 3 of ROADMAP.md."""
    is_colab = "COLAB_GPU" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

    if is_colab:
        profile = "B"
        profile_name = "B_colab_cloud_gpu"
        device = "cuda" if gpu_info["available"] else "cpu"
        mixed_precision = gpu_info["available"]
        img_size = 640
        batch_sizes = {"teacher": 4, "student_train": 16, "student_val": 32, "eval": 32}
        num_workers = 2
        reason = "Detected Google Colab or Kaggle cloud environment."
    elif gpu_info["available"] and gpu_info["vram_gb"] >= 8.0:
        profile = "A"
        profile_name = "A_gpu_local"
        device = "cuda"
        mixed_precision = True
        img_size = 640
        batch_sizes = {"teacher": 8, "student_train": 16, "student_val": 32, "eval": 32}
        num_workers = min(4, cpu_info["logical_cores"])
        reason = f"High VRAM GPU detected ({gpu_info['name']} with {gpu_info['vram_gb']} GB VRAM >= 8 GB)."
    elif gpu_info["available"] and gpu_info["vram_gb"] >= 4.0:
        profile = "C"
        profile_name = "C_gpu_entry_level"
        device = "cuda"
        mixed_precision = True
        img_size = 416
        batch_sizes = {"teacher": 1, "student_train": 4, "student_val": 8, "eval": 8}
        num_workers = min(2, cpu_info["logical_cores"])
        reason = (
            f"Entry-level GPU detected ({gpu_info['name']} with {gpu_info['vram_gb']} GB VRAM < 8 GB). "
            "Using Profile C parameters with CUDA acceleration."
        )
    else:
        profile = "C"
        profile_name = "C_cpu_only"
        device = "cpu"
        mixed_precision = False
        img_size = 416
        batch_sizes = {"teacher": 1, "student_train": 4, "student_val": 8, "eval": 8}
        num_workers = min(2, cpu_info["logical_cores"])
        reason = "CPU-only execution selected (no dedicated GPU detected with >= 4 GB VRAM)."

    return {
        "profile": profile,
        "profile_name": profile_name,
        "device": device,
        "reason": reason,
        "mixed_precision": mixed_precision,
        "img_size": img_size,
        "batch_sizes": batch_sizes,
        "num_workers": num_workers,
    }


def probe_hardware() -> dict[str, Any]:
    """Execute complete hardware probe and return configuration dictionary."""
    cpu_info = get_cpu_info()
    ram_gb = get_ram_info()
    gpu_info = get_gpu_info()
    decision = select_profile(gpu_info, cpu_info, ram_gb)

    config = {
        "hardware": {
            "platform": platform.platform(),
            "cpu_model": cpu_info["model"],
            "cpu_cores_logical": cpu_info["logical_cores"],
            "cpu_cores_physical": cpu_info["physical_cores"],
            "ram_gb": ram_gb,
            "gpu_available": gpu_info["available"],
            "gpu_name": gpu_info["name"],
            "vram_gb": gpu_info["vram_gb"],
            "cuda_version": gpu_info["cuda_version"],
        },
        "compute": {
            "chosen_profile": decision["profile"],
            "profile_name": decision["profile_name"],
            "device": decision["device"],
            "mixed_precision": decision["mixed_precision"],
            "img_size": decision["img_size"],
            "num_workers": decision["num_workers"],
            "batch_sizes": decision["batch_sizes"],
            "decision_reason": decision["reason"],
        },
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe hardware and generate SIVIA compute config")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("configs/compute/auto.yaml"),
        help="Target YAML path for probe results",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = probe_hardware()

    with open(args.output, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("=" * 60)
    print(" SIVIA Hardware Probe Result")
    print("=" * 60)
    print(
        f" CPU: {config['hardware']['cpu_model']} ({config['hardware']['cpu_cores_logical']} vCPUs)"
    )
    print(f" RAM: {config['hardware']['ram_gb']} GB")
    if config["hardware"]["gpu_available"]:
        print(f" GPU: {config['hardware']['gpu_name']} ({config['hardware']['vram_gb']} GB VRAM)")
    else:
        print(" GPU: None detected")
    print(f" Selected Profile: {config['compute']['profile_name']}")
    print(
        f" Device: {config['compute']['device']} (Mixed Precision: {config['compute']['mixed_precision']})"
    )
    print(f" Input Image Size: {config['compute']['img_size']}px")
    print(f" Batch Sizes: {config['compute']['batch_sizes']}")
    print(f" Reason: {config['compute']['decision_reason']}")
    print(f" Saved config to: {args.output}")
    print("=" * 60)
