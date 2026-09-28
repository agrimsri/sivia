#!/usr/bin/env python3
"""Hardware probe script for SIVIA.

Delegates to sivia.probe to detect CPU, GPU, VRAM, and RAM,
select the optimal compute profile, and write configs/compute/auto.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on python path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sivia.probe import (
    main,
)

if __name__ == "__main__":
    main()
