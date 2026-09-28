"""Unit tests for KitSpec validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sivia.kit.spec import KitSpec


def test_load_desk_kit_v1():
    """Verify that desk_kit_v1.yaml conforms strictly to KitSpec schema."""
    config_path = Path("configs/kits/desk_kit_v1.yaml")
    spec = KitSpec.from_yaml(config_path)

    assert spec.kit_name == "desk_kit_v1"
    assert len(spec.classes) == 6
    assert spec.required == ["screwdriver", "tape_roll", "sensor_module", "usb_cable"]
    assert spec.optional == ["multimeter", "pliers"]
    assert "screwdriver" in spec.slots
    assert "sensor_module" in spec.slots
    assert spec.rules.min_score == 0.35
    assert spec.rules.max_count["screwdriver"] == 1

    assert spec.get_class_id_by_name("screwdriver") == 0
    assert spec.get_class_name_by_id(0) == "screwdriver"


def test_reject_malformed_slot_coords():
    """Verify that inverted or out-of-range slot coordinates raise ValidationError."""
    malformed_data = {
        "kit_name": "bad_kit",
        "classes": [{"id": 0, "name": "tool", "prompts": ["a tool"]}],
        "required": ["tool"],
        "slots": {
            "tool": [0.8, 0.2, 0.4, 0.6]  # x1 (0.8) > x2 (0.4)
        },
    }
    with pytest.raises(ValidationError):
        KitSpec.model_validate(malformed_data)

    out_of_bounds = {
        "kit_name": "bad_kit",
        "classes": [{"id": 0, "name": "tool", "prompts": ["a tool"]}],
        "required": ["tool"],
        "slots": {
            "tool": [-0.1, 0.2, 0.4, 0.6]  # x1 < 0
        },
    }
    with pytest.raises(ValidationError):
        KitSpec.model_validate(out_of_bounds)


def test_reject_duplicate_class_id():
    """Verify duplicate class IDs are rejected."""
    bad_data = {
        "kit_name": "dup_ids",
        "classes": [
            {"id": 0, "name": "c1", "prompts": ["p1"]},
            {"id": 0, "name": "c2", "prompts": ["p2"]},
        ],
    }
    with pytest.raises(ValidationError):
        KitSpec.model_validate(bad_data)


def test_reject_undefined_required_class():
    """Verify required class not in classes list is rejected."""
    bad_data = {
        "kit_name": "missing_req",
        "classes": [{"id": 0, "name": "c1", "prompts": ["p1"]}],
        "required": ["c1", "ghost_item"],
    }
    with pytest.raises(ValidationError):
        KitSpec.model_validate(bad_data)


def test_reject_overlapping_required_and_optional():
    """Verify that a class cannot be both required and optional."""
    bad_data = {
        "kit_name": "overlap",
        "classes": [{"id": 0, "name": "item", "prompts": ["p1"]}],
        "required": ["item"],
        "optional": ["item"],
    }
    with pytest.raises(ValidationError):
        KitSpec.model_validate(bad_data)


def test_reject_unknown_slot_class():
    """Verify slot for non-existent class is rejected."""
    bad_data = {
        "kit_name": "unknown_slot",
        "classes": [{"id": 0, "name": "item", "prompts": ["p1"]}],
        "slots": {"non_existent_item": [0.1, 0.1, 0.5, 0.5]},
    }
    with pytest.raises(ValidationError):
        KitSpec.model_validate(bad_data)
