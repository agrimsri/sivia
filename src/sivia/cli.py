"""SIVIA command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sivia.kit.spec import KitSpec
from sivia.probe import probe_hardware


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sivia",
        description="Self-Improving Visual Inspection Agent (SIVIA) CLI",
    )
    parser.add_argument("--version", action="version", version="sivia 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Probe hardware command
    probe_parser = subparsers.add_parser("probe", help="Probe hardware and print compute profile")
    probe_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("configs/compute/auto.yaml"),
        help="Path to save auto.yaml config",
    )

    # Validate kit command
    kit_parser = subparsers.add_parser("validate-kit", help="Validate a kit specification YAML")
    kit_parser.add_argument(
        "kit_path",
        type=Path,
        nargs="?",
        default=Path("configs/kits/desk_kit_v1.yaml"),
        help="Path to kit spec YAML file",
    )

    args = parser.parse_args()

    if args.command == "probe":
        config = probe_hardware()
        print(f"Detected hardware: {config['hardware']['cpu_model']}")
        print(f"Chosen profile: {config['compute']['profile_name']}")
        print(f"Device: {config['compute']['device']}")
    elif args.command == "validate-kit":
        try:
            spec = KitSpec.from_yaml(args.kit_path)
            print(
                f"✓ Kit '{spec.kit_name}' is valid! ({len(spec.classes)} classes, {len(spec.slots)} slots)"
            )
        except Exception as e:
            print(f"✗ Kit validation failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
