"""Pytest path bootstrap for N170 historical-apply tests."""

from __future__ import annotations

import sys
from pathlib import Path

APPLY = Path(__file__).resolve().parents[1]
N170_DIR = APPLY.parent
PAPER_REPRODUCTION = N170_DIR.parent
SRC = PAPER_REPRODUCTION.parents[1] / "src"

for path in (N170_DIR, PAPER_REPRODUCTION, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
