"""Unit tests for kit verification rules and placement checking."""

from pathlib import Path

from sivia.kit.rules import Detection, verify_kit
from sivia.kit.spec import KitSpec


def test_verify_kit_complete_and_valid():
    """Verify that a scene with all required items inside slots is verified as complete."""
    spec = KitSpec.from_yaml(Path("configs/kits/desk_kit_v1.yaml"))

    # Required: screwdriver, tape_roll, sensor_module, usb_cable
    # screwdriver slot: [0.05, 0.10, 0.45, 0.35]
    # sensor_module slot: [0.55, 0.10, 0.95, 0.45]
    detections = [
        Detection("screwdriver", 0, 0.95, [0.10, 0.15, 0.40, 0.30]),
        Detection("tape_roll", 1, 0.90, [0.10, 0.60, 0.30, 0.80]),
        Detection("sensor_module", 2, 0.88, [0.60, 0.15, 0.90, 0.40]),
        Detection("usb_cable", 3, 0.85, [0.40, 0.60, 0.70, 0.90]),
    ]

    verdict = verify_kit(detections, spec)
    assert verdict.is_valid
    assert verdict.status == "complete"
    assert len(verdict.missing) == 0
    assert len(verdict.misplaced) == 0


def test_verify_kit_missing_item():
    """Verify that missing required items marks kit as incomplete."""
    spec = KitSpec.from_yaml(Path("configs/kits/desk_kit_v1.yaml"))

    # Missing usb_cable
    detections = [
        Detection("screwdriver", 0, 0.95, [0.10, 0.15, 0.40, 0.30]),
        Detection("tape_roll", 1, 0.90, [0.10, 0.60, 0.30, 0.80]),
        Detection("sensor_module", 2, 0.88, [0.60, 0.15, 0.90, 0.40]),
    ]

    verdict = verify_kit(detections, spec)
    assert not verdict.is_valid
    assert verdict.status == "incomplete"
    assert "usb_cable" in verdict.missing


def test_verify_kit_misplaced_item():
    """Verify item outside its designated slot marks kit as misplaced."""
    spec = KitSpec.from_yaml(Path("configs/kits/desk_kit_v1.yaml"))

    # Screwdriver placed at bottom right instead of top left slot [0.05, 0.10, 0.45, 0.35]
    detections = [
        Detection("screwdriver", 0, 0.95, [0.70, 0.70, 0.90, 0.90]),
        Detection("tape_roll", 1, 0.90, [0.10, 0.60, 0.30, 0.80]),
        Detection("sensor_module", 2, 0.88, [0.60, 0.15, 0.90, 0.40]),
        Detection("usb_cable", 3, 0.85, [0.40, 0.60, 0.70, 0.90]),
    ]

    verdict = verify_kit(detections, spec)
    assert not verdict.is_valid
    assert verdict.status == "misplaced"
    assert "screwdriver" in verdict.misplaced
