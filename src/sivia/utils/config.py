"""Hydra and OmegaConf configuration utilities for SIVIA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load and optionally merge YAML config file into an OmegaConf DictConfig.

    Args:
        config_path: Path to the main YAML config file.
        overrides: List of "key=value" override strings.

    Returns:
        OmegaConf DictConfig object.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def save_config(cfg: DictConfig | dict[str, Any], output_path: str | Path) -> None:
    """Save DictConfig or dictionary to YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    with open(path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
