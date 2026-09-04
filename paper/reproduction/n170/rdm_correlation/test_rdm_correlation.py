"""Unit tests for Track D RDM-correlation definitions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
N170_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "paper" / "reproduction"))
sys.path.insert(0, str(N170_DIR))
sys.path.insert(0, str(HERE))

from common.source_faithful import (  # noqa: E402
    pair_stack_from_condition_averages,
    unique_unordered_pairs,
)
from prepare import default_erp_path, load_n170_subject1, window_slice  # noqa: E402
from rdms import car_rdm, face_rdm  # noqa: E402
from redisca import ReDisCA  # noqa: E402

from definitions import (  # noqa: E402
    airi_matlab_grown_matrix,
    empirical_rdm_squared_euclidean,
    empirical_rdm_wTRw,
    eq2_pearson_of_standardized,
    eq2_printed_inner_product,
    pearson_unique_triangle,
    score_empirical_against_target,
    unique_triangle,
)


def test_eq2_standardized_pearson_matches_raw() -> None:
    rng = np.random.default_rng(0)
    empirical = rng.random((4, 4))
    empirical = 0.5 * (empirical + empirical.T)
    np.fill_diagonal(empirical, 0.0)
    target = face_rdm()
    raw = pearson_unique_triangle(empirical, target)
    zscored = eq2_pearson_of_standardized(empirical, target, convention="sample")
    assert abs(raw - zscored) < 1e-12


def test_eq2_sample_inner_is_pearson_times_n_minus_1_over_n() -> None:
    empirical = np.array(
        [
            [0.0, 2.0, 1.8, 1.7],
            [2.0, 0.0, 0.2, 0.1],
            [1.8, 0.2, 0.0, 0.3],
            [1.7, 0.1, 0.3, 0.0],
        ]
    )
    target = face_rdm()
    r = pearson_unique_triangle(empirical, target)
    inner = eq2_printed_inner_product(empirical, target, convention="sample")
    n = 6
    assert abs(inner - r * (n - 1) / n) < 1e-12
    pop = eq2_printed_inner_product(empirical, target, convention="population")
    assert abs(pop - r) < 1e-12


def test_within_0_vs_0_1_unique_pearson_identical() -> None:
    empirical = np.array(
        [
            [0.0, 1.9, 2.0, 1.8],
            [1.9, 0.0, 0.04, 0.03],
            [2.0, 0.04, 0.0, 0.05],
            [1.8, 0.03, 0.05, 0.0],
        ]
    )
    r0 = pearson_unique_triangle(empirical, face_rdm(within=0.0))
    r1 = pearson_unique_triangle(empirical, face_rdm(within=0.1))
    assert abs(r0 - r1) < 1e-12


def test_wTRw_gram_equals_undemeaned_traces() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 8, 20))
    w = rng.standard_normal(8)
    traces = np.einsum("c,nct->nt", w, X)
    d_tr = empirical_rdm_squared_euclidean(traces, demean_time=False)
    pairs = unique_unordered_pairs(4)
    gram = pair_stack_from_condition_averages(X, pairs, matrix_mode="unscaled_gram")
    d_w = empirical_rdm_wTRw(w, gram, pairs, 4)
    assert np.allclose(d_tr, d_w, atol=1e-12)


def test_wTRw_matlab_cov_matches_demeaned_traces_up_to_T_minus_1() -> None:
    rng = np.random.default_rng(2)
    n_times = 20
    X = rng.standard_normal((4, 8, n_times))
    w = rng.standard_normal(8)
    traces = np.einsum("c,nct->nt", w, X)
    d_tr = empirical_rdm_squared_euclidean(traces, demean_time=True)
    pairs = unique_unordered_pairs(4)
    cov = pair_stack_from_condition_averages(X, pairs, matrix_mode="matlab_cov")
    d_w = empirical_rdm_wTRw(w, cov, pairs, 4)
    assert np.allclose(d_w * (n_times - 1), d_tr, atol=1e-12)
    target = face_rdm()
    assert abs(
        pearson_unique_triangle(d_w, target) - pearson_unique_triangle(d_tr, target)
    ) < 1e-12


def test_airi_grown_shape_is_c_minus_1_by_c() -> None:
    grown = airi_matlab_grown_matrix(face_rdm())
    assert grown.shape == (3, 4)
    # Unique pairs live in the assigned upper entries; last MATLAB row is absent.
    assert grown[0, 1] == 1.0 and grown[0, 2] == 1.0 and grown[0, 3] == 1.0
    assert grown[1, 2] == 0.0 and grown[1, 3] == 0.0 and grown[2, 3] == 0.0
    assert np.allclose(grown[:, 0], 0.0)


def test_score_block_records_affine_invariance() -> None:
    target = face_rdm()
    empirical = np.array(
        [
            [0.0, 3.0, 3.0, 3.0],
            [3.0, 0.0, 0.1, 0.1],
            [2.9, 0.1, 0.0, 0.2],
            [3.1, 0.1, 0.2, 0.0],
        ]
    )
    empirical = 0.5 * (empirical + empirical.T)
    np.fill_diagonal(empirical, 0.0)
    scored = score_empirical_against_target(empirical, target)
    assert scored["eq2_affine_invariance_abs_diff_vs_unique"] < 1e-12
    assert scored["eq2_sample_inner_equals_pearson_times_n_minus_1_over_n"]
    assert scored["n_unique_pairs"] == 6


@pytest.mark.skipif(not default_erp_path().exists(), reason="ERP CORE cache absent")
def test_official_face_window_unique_pearson_near_one() -> None:
    packed = load_n170_subject1()
    win = window_slice(
        packed["data"], packed["times_ms"], center_ms=200.0, duration_ms=100.0
    )
    rdm = face_rdm()
    model = ReDisCA(demean_time=False).fit(win["data"], rdm)
    traces = model.transform(win["data"])[:, 0, :]
    dhat = empirical_rdm_squared_euclidean(traces, demean_time=False)
    corr = pearson_unique_triangle(dhat, rdm)
    assert corr > 0.999
    # Regression vs the frozen Fig. 10 paper_gram number.
    assert abs(corr - 0.9998814542043535) < 1e-8
    assert abs(float(model.eigenvalues_[0]) - 0.8800598487297501) < 1e-8


@pytest.mark.skipif(not default_erp_path().exists(), reason="ERP CORE cache absent")
def test_car_window_unique_pearson_near_one() -> None:
    packed = load_n170_subject1()
    win = window_slice(
        packed["data"], packed["times_ms"], center_ms=170.0, duration_ms=100.0
    )
    rdm = car_rdm()
    model = ReDisCA(demean_time=False).fit(win["data"], rdm)
    traces = model.transform(win["data"])[:, 0, :]
    corr = pearson_unique_triangle(
        empirical_rdm_squared_euclidean(traces, demean_time=False), rdm
    )
    assert corr > 0.999
    assert unique_triangle(rdm).tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
