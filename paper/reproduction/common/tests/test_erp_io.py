"""Tests for ERPLAB ERP loader. Skipped when the cache is absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.erp_io import load_erplab_erp
from common.paths import ERPCORE_DIR

ERP_PATH = ERPCORE_DIR / "all_data_and_scripts" / "1" / "1_N170_erp_ar.erp"

pytestmark = pytest.mark.skipif(not ERP_PATH.exists(), reason="ERP CORE cache not present")


def test_subject1_erp_has_four_bins() -> None:
    payload = load_erplab_erp(ERP_PATH)
    assert payload["data"].shape[0] == 4
    assert payload["data"].shape[2] == 256
    assert payload["srate"] == 256.0
    names = " ".join(payload["bin_descriptions"]).lower()
    assert "face" in names and "car" in names
    assert "scrambled" in names
