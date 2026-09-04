"""Pytest path bootstrap for the rank-audit tests."""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_REPRODUCTION = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[4] / "src"

for path in (PAPER_REPRODUCTION, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
