"""Tests for Track C N170 preprocessing forensics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
N170_DIR = HERE.parent
REPRO_DIR = N170_DIR.parent
REPO_ROOT = REPRO_DIR.parent.parent
for path in (HERE, N170_DIR, REPRO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.paths import ERPCORE_DIR  # noqa: E402
from prepare import PREFERRED_ERP_NAME, SCALP_LABELS, subject1_dir  # noqa: E402
from eeglab_io import load_eeglab_set, removed_ica_components_1based  # noqa: E402
from ica_xlsx import (  # noqa: E402
    ICA_XLSX_SHA256,
    default_xlsx_path,
    parse_ica_components_xlsx,
)
from forensics import (  # noqa: E402
    PAPER_ICA_QUOTE,
    PREFERRED_ERP_SHA256,
    SCRIPT7_LPFILT_CALL,
    answers,
    build_bundle,
    compare_erp_stages,
    lpfilt_assessment,
)

SUBJECT = ERPCORE_DIR / "all_data_and_scripts" / "1"
ERP_PATH = SUBJECT / PREFERRED_ERP_NAME
XLSX_PATH = default_xlsx_path()

pytestmark = pytest.mark.skipif(not ERP_PATH.exists(), reason="ERP CORE cache absent")


def test_preferred_erp_hash() -> None:
    from common.hashing import sha256_file

    assert sha256_file(ERP_PATH) == PREFERRED_ERP_SHA256


def test_xlsx_subject1_is_2_and_7() -> None:
    if not XLSX_PATH.exists():
        pytest.skip("ICA_Components_N170.xlsx not cached")
    from common.hashing import sha256_file

    assert sha256_file(XLSX_PATH) == ICA_XLSX_SHA256
    mapping = parse_ica_components_xlsx(XLSX_PATH)
    assert mapping["1"] == [2, 7]
    counts = [len(v) for v in mapping.values()]
    assert len(mapping) == 40
    assert counts.count(1) == 15
    assert counts.count(3) == 10


def test_pop_subcomp_dropped_exactly_2_and_7() -> None:
    weighted = SUBJECT / "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_weighted.set"
    corrected = SUBJECT / "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr.set"
    assert removed_ica_components_1based(weighted, corrected) == [2, 7]


def test_p9_p10_present_only_in_raw() -> None:
    raw = load_eeglab_set(SUBJECT / "1_N170.set")
    reref = load_eeglab_set(SUBJECT / "1_N170_shifted_ds_reref_ucbip.set")
    epoch = load_eeglab_set(
        SUBJECT
        / "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch.set"
    )
    assert raw["has_p9"] and raw["has_p10"]
    assert raw["channel_labels"][8] == "P9"
    assert raw["channel_labels"][26] == "P10"
    assert not reref["has_p9"] and not reref["has_p10"]
    assert not epoch["has_p9"] and not epoch["has_p10"]
    assert "P9" not in epoch["channel_labels"]
    stages = compare_erp_stages(SUBJECT)
    for name, payload in stages["stages"].items():
        assert payload["has_p9"] is False, name
        assert payload["has_p10"] is False, name
        assert payload["channel_labels"][:28] == list(SCALP_LABELS)


def test_lpfilt_is_script7_extra_not_paper_default() -> None:
    stages = compare_erp_stages(SUBJECT)
    ar = stages["stages"][PREFERRED_ERP_NAME]
    lp = stages["stages"]["1_N170_erp_ar_lpfilt.erp"]
    dw = stages["stages"]["1_N170_erp_ar_diff_waves_lpfilt.erp"]
    assert ar["isfilt"] == 0
    assert lp["isfilt"] == 1
    assert dw["nbin"] == 9
    assert stages["comparison"]["lpfilt_vs_diff_waves_bins1to4_identical"] is True
    assert stages["comparison"]["ar_vs_lpfilt_identical"] is False
    assessment = lpfilt_assessment()
    assert assessment["paper_mentions_lowpass_on_erps"] is False
    assert assessment["silent_default"] is False
    assert "pop_filterp" in SCRIPT7_LPFILT_CALL
    assert "three ICA components corresponding to ocular" in PAPER_ICA_QUOTE


def test_diff_waves_not_4condition_input() -> None:
    dw = compare_erp_stages(SUBJECT)["stages"]["1_N170_erp_ar_diff_waves_lpfilt.erp"]
    extra = [name.lower() for name in dw["bin_descriptions"][4:]]
    assert dw["nbin"] == 9
    assert any("minus" in name for name in extra)


def test_build_bundle_stops_without_third_component() -> None:
    bundle = build_bundle()
    result = answers(bundle)
    assert result["q4_third_ica_component"]["third_component_source_supported"] is False
    assert result["q4_third_ica_component"]["invented_component"] is None
    assert result["q5_alternative_state_ran"]["ran_track_a_12_variants"] is False
    assert result["d11_status"]["erpcore_subject_1"] == [2, 7]
    assert bundle["track_a_12_variants"]["ran"] is False
    np.testing.assert_array_equal(
        bundle["ica"]["removed_components_1based_from_weights"], [2, 7]
    )


def test_correct_only_and_ar_counts() -> None:
    bundle = build_bundle()
    eventlist = bundle["averaging"]["eventlist"]
    ar = bundle["averaging"]["artifact_rejection"]
    assert eventlist["n_correct_binned_header"] == 240
    assert eventlist["n_incorrect_response_202"] == 76
    assert ar["n_good_flag0"] == 191
    assert ar["n_flagged"] == 49
    erp = bundle["erp_stages"]["stages"][PREFERRED_ERP_NAME]
    assert erp["n_accepted"] == [52, 38, 49, 52]
