"""Configurable cache paths for Stage A reproduction."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper"
DATA_ROOT = Path(os.environ.get("REDISCA_REPRODUCTION_DATA", REPO_ROOT / ".reproduction_data"))
RESULTS_ROOT = Path(os.environ.get("REDISCA_REPRODUCTION_RESULTS", PAPER_ROOT / "results"))

PAPER_TEXT_DIR = DATA_ROOT / "paper_text"
PAPER_PDF_DIR = DATA_ROOT / "paper_pdf"
UPSTREAM_DIR = DATA_ROOT / "upstream"
OSF_DIR = DATA_ROOT / "osf"
ERPCORE_DIR = DATA_ROOT / "erpcore"
MEG_DIR = DATA_ROOT / "meg"
SOURCE_MODEL_DIR = DATA_ROOT / "source_models"
HASH_DIR = DATA_ROOT / "hashes"
ARRAY_CACHE_DIR = DATA_ROOT / "arrays"


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
        ARRAY_CACHE_DIR,
        RESULTS_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
