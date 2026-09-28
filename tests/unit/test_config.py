"""Unit tests for configuration loading and overrides."""

from pathlib import Path

import pytest
from omegaconf import DictConfig

from sivia.utils.config import load_config


def test_load_and_override_config(tmp_path: Path):
    """Test loading YAML file and applying dotlist overrides."""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text("dataset:\n  name: desk_v1\n  batch_size: 16\n")

    cfg = load_config(yaml_file)
    assert isinstance(cfg, DictConfig)
    assert cfg.dataset.name == "desk_v1"
    assert cfg.dataset.batch_size == 16

    # Override
    cfg_overridden = load_config(yaml_file, overrides=["dataset.batch_size=32", "dataset.lr=0.001"])
    assert cfg_overridden.dataset.batch_size == 32
    assert cfg_overridden.dataset.lr == 0.001


def test_missing_config_raises():
    """Verify loading non-existent config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_path.yaml")
