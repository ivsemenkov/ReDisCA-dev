"""Configurable cache paths for paper-branch reproduction.

Large datasets and upstream clones live under ``.reproduction_data/`` at the
repository root and are gitignored. Scientific workflows should not hard-code
absolute machine paths.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper"
DATA_ROOT = Path(os.environ.get("REDISCA_REPRODUCTION_DATA", REPO_ROOT / ".reproduction_data"))

PAPER_TEXT_DIR = DATA_ROOT / "paper_text"
PAPER_PDF_DIR = DATA_ROOT / "paper_pdf"
UPSTREAM_DIR = DATA_ROOT / "upstream"
OSF_DIR = DATA_ROOT / "osf"
ERPCORE_DIR = DATA_ROOT / "erpcore"
MEG_DIR = DATA_ROOT / "meg"
SOURCE_MODEL_DIR = DATA_ROOT / "source_models"
HASH_DIR = DATA_ROOT / "hashes"

AIRI_CLONE = UPSTREAM_DIR / "AIRI-ReDisCA"
SPOC_CLONE = UPSTREAM_DIR / "matlab_SPoC"


def ensure_data_layout() -> None:
    """Create the cache directories used by download and analysis scripts."""
    for path in (
        PAPER_TEXT_DIR,
        PAPER_PDF_DIR,
        UPSTREAM_DIR,
        OSF_DIR,
        ERPCORE_DIR,
        MEG_DIR,
        SOURCE_MODEL_DIR,
        HASH_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
