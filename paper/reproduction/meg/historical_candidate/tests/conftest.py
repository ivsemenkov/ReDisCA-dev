"""Pytest path bootstrap for MEG historical-candidate tests."""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_REPRODUCTION = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[5] / "src"

for path in (PAPER_REPRODUCTION, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
