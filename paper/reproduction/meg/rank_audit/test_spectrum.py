"""Unit tests for the MEG rank-audit helpers. No MEG array load."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PAPER_REPRODUCTION = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[4] / "src"
for path in (PAPER_REPRODUCTION, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.source_faithful import pair_stack_from_condition_averages, unique_unordered_pairs
from meg.rank_audit.report import render_rank_audit_md
from meg.rank_audit.spectrum import (
    RANK_TOL,
    descending_eigenvalues,
    eig_row,
    matlab_eig_flip_assessment,
    matlab_pca_n_components,
    mean_pair_matrix,
    numerical_rank,
    pca_interval_for_exactly_n,
    solver_comparison,
    spectrum_payload,
    window_table,
)


def _spd_from_spectrum(spectrum: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = spectrum.size
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    matrix = q @ np.diag(spectrum) @ q.T
    return 0.5 * (matrix + matrix.T)


def test_rank_tol_matches_spoc_strict_greater_than() -> None:
    spectrum = np.array([1.0, 2e-6, 1e-6, 9e-7, 1e-12])
    # 1e-6 * max = 1e-6; 2e-6 > cutoff, 1e-6 is NOT > cutoff, 9e-7 is not.
    assert numerical_rank(spectrum) == 2
    # Values sitting exactly on the cutoff are not used for recovered-SPD checks;
    # roundoff can push 1e-6 across the strict `>` test.
    clear = np.array([1.0, 2e-6, 5e-7, 1e-12, 0.0])
    cxx = _spd_from_spectrum(clear)
    recovered = descending_eigenvalues(cxx)
    np.testing.assert_allclose(recovered[:3], clear[:3], rtol=1e-8, atol=1e-14)
    assert numerical_rank(recovered) == 2


def test_window_table_60_75_and_above_cutoff_flag() -> None:
    spectrum = np.concatenate(
        [np.linspace(1.0, 1e-4, 59), np.array([5e-6, 2e-6, 1.1e-6, 1.0000001e-6, 9e-7])]
    )
    spectrum = np.concatenate([spectrum, np.full(20, 1e-12)])
    rows = window_table(spectrum, start_1based=60, stop_1based=64)
    assert [r["index_1based"] for r in rows] == [60, 61, 62, 63, 64]
    assert rows[0]["above_cutoff"] is True
    assert rows[3]["above_cutoff"] is True
    assert rows[4]["above_cutoff"] is False


def test_eig_row_margin() -> None:
    spectrum = np.array([1.0, 1.5e-6, 0.5e-6])
    row = eig_row(spectrum, 2)
    assert row["ratio_to_max"] == pytest.approx(1.5e-6)
    assert row["margin_ratio"] == pytest.approx(0.5e-6)
    assert row["above_cutoff"] is True


def test_pca_default_one_returns_numerical_rank() -> None:
    spectrum = np.array([1.0, 0.5, 0.2, 2e-6, 1e-9, 0.0, 0.0])
    rank = numerical_rank(spectrum)
    assert rank == 4
    assert matlab_pca_n_components(spectrum, pca_var_explained=1.0) == rank
    interval = pca_interval_for_exactly_n(spectrum, 3)
    assert interval["feasible"] is True
    n_at_mid = matlab_pca_n_components(
        spectrum,
        pca_var_explained=0.5 * (interval["open_lower"] + interval["closed_upper"]),
        numerical_rank_r=rank,
    )
    assert n_at_mid == 3


def test_pca_cannot_exceed_numerical_rank() -> None:
    spectrum = np.array([1.0, 0.4, 1e-9, 0.0])
    interval = pca_interval_for_exactly_n(spectrum, 3)
    assert interval["feasible"] is False
    assert matlab_pca_n_components(spectrum, pca_var_explained=1.0) == 2


def test_matlab_flip_not_borderline_when_gap_is_large() -> None:
    eigs = np.ones(80)
    eigs[66] = 3e-6
    eigs[67] = 2e-6
    eigs[68] = 1e-7
    eigs[69:] = 1e-12
    assessment = matlab_eig_flip_assessment(eigs)
    assert assessment["rank_at_1e-6"] == 68
    assert assessment["verdict"] == "not_borderline"
    assert assessment["plausible_that_matlab_eig_alone_flips_68_to_67"] is False


def test_matlab_flip_borderline_when_68_sits_on_cutoff() -> None:
    eigs = np.ones(80)
    eigs[66] = 2e-6
    eigs[67] = 1e-6 + 1e-13
    eigs[68] = 1e-6 - 1e-8
    eigs[69:] = 1e-12
    assessment = matlab_eig_flip_assessment(eigs)
    assert assessment["rank_at_1e-6"] == 68
    assert assessment["verdict"] == "plausible_borderline"
    assert assessment["plausible_that_matlab_eig_alone_flips_68_to_67"] is True


def test_directed_and_unique_share_cxx_for_symmetric_grams() -> None:
    rng = np.random.default_rng(4)
    averages = rng.standard_normal((4, 6, 20))
    cxx_u, n_u = mean_pair_matrix(
        averages, pair_mode="unique_unordered", matrix_mode="unscaled_gram"
    )
    cxx_d, n_d = mean_pair_matrix(
        averages, pair_mode="airi_directed", matrix_mode="unscaled_gram"
    )
    assert n_u == 6
    assert n_d == 12
    np.testing.assert_allclose(cxx_u, cxx_d, atol=1e-12)


def test_mean_pair_matrix_matches_stack_mean() -> None:
    rng = np.random.default_rng(5)
    averages = rng.standard_normal((3, 5, 12))
    pairs = unique_unordered_pairs(3)
    stack = pair_stack_from_condition_averages(
        averages, pairs, matrix_mode="matlab_cov"
    )
    cxx, n_pairs = mean_pair_matrix(
        averages, pair_mode="unique_unordered", matrix_mode="matlab_cov"
    )
    assert n_pairs == stack.shape[0]
    np.testing.assert_allclose(cxx, stack.mean(axis=0))


def test_solvers_agree_on_obvious_rank() -> None:
    spectrum = np.concatenate([np.linspace(1.0, 0.1, 10), np.full(6, 1e-12)])
    cxx = _spd_from_spectrum(spectrum, seed=7)
    comparison = solver_comparison(cxx)
    assert comparison["all_solvers_agree_on_rank"]
    assert comparison["ranks"]["numpy_eigh"] == 10


def test_spectrum_payload_on_large_enough_matrix() -> None:
    spectrum = np.ones(80)
    spectrum[0] = 1.0
    spectrum[1:67] = np.linspace(0.9, 1e-3, 66)
    spectrum[67] = 2.0e-6
    spectrum[68] = 1.0e-7
    spectrum[69:] = 1e-12
    cxx = _spd_from_spectrum(spectrum, seed=9)
    payload = spectrum_payload(
        cxx,
        label="toy80",
        pair_mode="airi_directed",
        matrix_mode="matlab_cov",
        n_times=901,
        n_pairs=30,
        bandpass={"order": 3, "low_hz": 0.25, "high_hz": 20.0},
        window_ms=[99.0, 999.0],
    )
    assert payload["numerical_rank_1e-6"] == 68
    assert payload["whitening_n_components_pca1"] == 68
    assert payload["rank_tol"] == RANK_TOL
    assert len(payload["window_60_75"]) == 16
    assert payload["window_60_75"][0]["index_1based"] == 60
    assert payload["window_60_75"][-1]["index_1based"] == 75
    assert payload["matlab_eig_flip"]["verdict"] == "not_borderline"


def test_render_markdown_contains_ranks_and_disagreement() -> None:
    spectrum = np.ones(80)
    spectrum[67] = 2e-6
    spectrum[68] = 1e-7
    spectrum[69:] = 1e-12
    cxx = _spd_from_spectrum(spectrum, seed=10)
    spec = spectrum_payload(
        cxx,
        label="paper_faithful",
        pair_mode="unique_unordered",
        matrix_mode="unscaled_gram",
        n_times=1501,
        n_pairs=15,
        bandpass=None,
        window_ms=[-500.0, 1000.0],
    )
    airi = dict(spec)
    airi["label"] = "airi_executable"
    airi["pair_mode"] = "airi_directed"
    airi["matrix_mode"] = "matlab_cov"
    airi["n_pairs"] = 30
    payload = {
        "environment": {"packages": {"numpy": "test"}, "matlab": None, "captured_at_utc": "t"},
        "paths": {"paper_faithful": spec, "airi_executable": airi},
        "diagnostic_paths": {},
        "author_saved_a1": {
            "path": "fake.mat",
            "sha256": "abc",
            "shape": [204, 67],
            "n_columns": 67,
            "comps_order": [1, 2, 3, 4],
            "column_norm_min": 1e-12,
            "column_norm_median": 1e-12,
            "column_norm_max": 2e-12,
            "svd_numerical_rank_1e-6_of_singular_values": 67,
            "note": "do not force rank",
        },
        "airi_spoc_inspection": {
            "airi_commit": "15bc19c",
            "spoc_commit": "18e4754",
            "spoc_call": "Xspoc, z, n_bootstrapping_iterations,1000",
            "high_cutoff_hz": 20,
            "pca_rank_mentions": [],
            "explicit_pca_or_rank_setting": "none",
            "source_loc_topo_file": "topo_face_vs_tool_correct_filt15",
            "plot_ylim_15_20": True,
        },
        "conclusion": {
            "summary_markdown": "Ranks disagree with A1. Do not force 67.",
            "summary": "Ranks disagree with A1.",
        },
    }
    md = render_rank_audit_md(payload)
    assert "67 vs 68" in md
    assert "Do not force 67" in md or "disagree" in md.lower()
    assert "paper_faithful" in md
    assert "airi_executable" in md
    assert "λᵢ > λ_max·1e-6" in md
