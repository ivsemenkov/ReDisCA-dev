#!/usr/bin/env python3
"""Track D: how could published Fig. 10 corr=0.82 have been computed?

Canonical library path: ``from redisca import ReDisCA`` with unique pairs and
unscaled Gram (``demean_time=False``). MATLAB-cov is a labeled source-faithful
path. Do not tune windows, channels, or the target fill to hit 0.82.

Example
-------
    python paper/reproduction/n170/rdm_correlation/run.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
N170_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "paper" / "reproduction"))
sys.path.insert(0, str(N170_DIR))
sys.path.insert(0, str(HERE))

from common.hashing import write_json  # noqa: E402
from common.provenance import capture_environment  # noqa: E402
from common.source_faithful import (  # noqa: E402
    fit_condition_averages,
    pair_indices,
    pair_stack_from_condition_averages,
)
from redisca import ReDisCA  # noqa: E402

from prepare import (  # noqa: E402
    CAR_CENTER_MS,
    CAR_DURATION_MS,
    FACE_CENTER_MS,
    FACE_DURATION_MS,
    load_n170_subject1,
    window_slice,
)
from rdms import car_rdm, face_rdm  # noqa: E402

from definitions import (  # noqa: E402
    empirical_rdm_squared_euclidean,
    empirical_rdm_wTRw,
    instantaneous_squared_rdm,
    paper_targets,
    pearson_unique_triangle,
    score_empirical_against_target,
    unique_pairs,
)

RESULTS_DIR = REPO_ROOT / "paper" / "results" / "n170" / "rdm_correlation"
N_REPORT = 2  # face: leading; car: two paper-claimed components


def _filter_row(filters: np.ndarray, index: int, *, library: bool) -> np.ndarray:
    if library:
        return np.asarray(filters[index], dtype=np.float64)
    return np.asarray(filters[:, index], dtype=np.float64)


def _traces_from_filter(
    weights: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """``(n_conditions, n_times)`` for one spatial filter."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    # X: (n_conditions, n_channels, n_times) → (n_conditions, n_times)
    return np.einsum("c,nct->nt", w, np.asarray(X, dtype=np.float64))


def _nearest_time_index(times_ms: np.ndarray, target_ms: float) -> int:
    return int(np.argmin(np.abs(np.asarray(times_ms, dtype=np.float64) - float(target_ms))))


def _score_block(
    *,
    empirical: np.ndarray,
    target: np.ndarray,
    identity_notes: dict[str, Any],
) -> dict[str, Any]:
    payload = score_empirical_against_target(empirical, target)
    payload["identity_checks"] = identity_notes
    payload["empirical_rdm"] = np.asarray(empirical, dtype=np.float64).tolist()
    return payload


def _component_correlations(
    *,
    weights: np.ndarray,
    X_window: np.ndarray,
    X_full: np.ndarray,
    times_full_ms: np.ndarray,
    target: np.ndarray,
    target_within01: np.ndarray,
    gram_window: np.ndarray,
    cov_window: np.ndarray,
    pairs: list[tuple[int, int]],
    named_latencies_ms: dict[str, float],
) -> dict[str, Any]:
    n_conditions = int(X_window.shape[0])
    traces_window = _traces_from_filter(weights, X_window)
    traces_full = _traces_from_filter(weights, X_full)

    d_win = empirical_rdm_squared_euclidean(traces_window, demean_time=False)
    d_win_demeaned = empirical_rdm_squared_euclidean(traces_window, demean_time=True)
    d_full = empirical_rdm_squared_euclidean(traces_full, demean_time=False)
    d_full_demeaned = empirical_rdm_squared_euclidean(traces_full, demean_time=True)
    d_wgram = empirical_rdm_wTRw(weights, gram_window, pairs, n_conditions)
    d_wcov = empirical_rdm_wTRw(weights, cov_window, pairs, n_conditions)

    gram_vs_traces = float(np.max(np.abs(d_wgram - d_win)))
    # MATLAB cov = demeaned Gram / (T-1); Pearson is scale-invariant.
    t_win = int(X_window.shape[-1])
    cov_vs_demeaned_scale = d_wcov * (t_win - 1)
    cov_vs_demeaned = float(np.max(np.abs(cov_vs_demeaned_scale - d_win_demeaned)))

    window_scores = _score_block(
        empirical=d_win,
        target=target,
        identity_notes={
            "matches_item": "1_window_traces_sqeuclidean_unique_pearson",
            "wTRw_unscaled_gram_max_abs_diff_vs_traces": gram_vs_traces,
            "wTRw_unscaled_gram_equals_window_traces": bool(
                np.allclose(d_wgram, d_win, atol=1e-12, rtol=1e-12)
            ),
        },
    )
    full_scores = _score_block(
        empirical=d_full,
        target=target,
        identity_notes={
            "matches_item": "2_full_epoch_traces_sqeuclidean_unique_pearson",
            "filter": "window-fitted w applied to the full ERP epoch",
        },
    )
    wgram_scores = _score_block(
        empirical=d_wgram,
        target=target,
        identity_notes={
            "matches_item": "3_wTRw_unscaled_gram_window",
            "max_abs_diff_vs_window_traces_undemeaned": gram_vs_traces,
            "scale_invariant_note": (
                "Unscaled Gram R_ij; dhat_ij = w^T R_ij w = ||u_i-u_j||^2 "
                "on undemeaned window traces."
            ),
        },
    )
    wcov_scores = _score_block(
        empirical=d_wcov,
        target=target,
        identity_notes={
            "matches_item": "4_wTRw_matlab_cov_window",
            "T_window": t_win,
            "max_abs_diff_vs_demeaned_traces_times_Tminus1": cov_vs_demeaned,
            "pearson_equals_demeaned_window_traces": bool(
                np.isclose(
                    pearson_unique_triangle(d_wcov, target),
                    pearson_unique_triangle(d_win_demeaned, target),
                    atol=1e-12,
                    rtol=1e-12,
                )
            ),
            "scale_invariant_note": (
                "MATLAB cov = demeaned Gram/(T-1). Pearson is scale-invariant, "
                "so this matches demeaned-trace Eq. 1 if traces are demeaned "
                "consistently."
            ),
        },
    )

    within01_window = pearson_unique_triangle(d_win, target_within01)
    within01_full = pearson_unique_triangle(d_full, target_within01)

    instant: dict[str, Any] = {}
    for name, latency in named_latencies_ms.items():
        idx = _nearest_time_index(times_full_ms, latency)
        d_t = instantaneous_squared_rdm(traces_full[:, idx])
        instant[name] = {
            "requested_ms": float(latency),
            "nearest_time_ms": float(times_full_ms[idx]),
            "index": idx,
            "note": (
                "AIRI MEG plotting analog: instantaneous squared differences "
                "of the window-fitted filter's full-epoch traces at a "
                "paper-named latency. Not a Fig. 10 windowed definition. "
                "N170 has no AIRI script."
            ),
            "scores": score_empirical_against_target(d_t, target),
        }

    return {
        "window_traces_sqeuclidean_undemeaned": window_scores,
        "full_epoch_traces_sqeuclidean_undemeaned": full_scores,
        "window_traces_sqeuclidean_demeaned": _score_block(
            empirical=d_win_demeaned,
            target=target,
            identity_notes={
                "matches_item": (
                    "MATLAB-cov centering of Eq. 1 traces; Pearson matches "
                    "item 4 (w^T R_matlab_cov w) by scale invariance"
                ),
            },
        ),
        "full_epoch_traces_sqeuclidean_demeaned": _score_block(
            empirical=d_full_demeaned,
            target=target,
            identity_notes={"filter": "same window w; full epoch; demeaned differences"},
        ),
        "wTRw_unscaled_gram_window": wgram_scores,
        "wTRw_matlab_cov_window": wcov_scores,
        "within0.1_target_unique_pearson_window": within01_window,
        "within0.1_target_unique_pearson_full_epoch": within01_full,
        "within0.1_equals_binary_unique_pearson_window": bool(
            np.isclose(
                within01_window,
                window_scores["pearson_unique_i_lt_j"],
                atol=1e-12,
                rtol=1e-12,
            )
        ),
        "airi_instantaneous_at_paper_named_latencies": instant,
        "headline": {
            "item1_window_unique_pearson": window_scores["pearson_unique_i_lt_j"],
            "item2_full_epoch_unique_pearson": full_scores["pearson_unique_i_lt_j"],
            "item3_wTRw_gram_unique_pearson": wgram_scores["pearson_unique_i_lt_j"],
            "item4_wTRw_matlab_cov_unique_pearson": wcov_scores["pearson_unique_i_lt_j"],
            "item5_eq2_standardized_pearson": window_scores[
                "eq2_pearson_after_sample_sd_standardization"
            ],
            "eq2_printed_inner_product_sample_sd_window": window_scores[
                "eq2_printed_inner_product_sample_sd"
            ],
            "airi_grown_corrcoef_window": window_scores[
                "airi_matlab_grown_Cxminus1_by_C_corrcoef"
            ],
            "possible_misread_full_flatten_window": window_scores[
                "possible_misread_full_symmetric_flatten_corrcoef"
            ],
            "possible_misread_triu_incl_diag_window": window_scores[
                "possible_misread_triu_including_diagonal"
            ],
        },
    }


def _fit_library_gram(X_window: np.ndarray, rdm: np.ndarray) -> dict[str, Any]:
    model = ReDisCA(demean_time=False).fit(X_window, rdm)
    return {
        "path": "library_unique_unscaled_gram",
        "authority": "canonical library; paper printed Gram; unique i<j",
        "demean_time": False,
        "pair_mode": "unique_unordered",
        "matrix_mode": "unscaled_gram",
        "eigenvalues_head": [float(v) for v in model.eigenvalues_[:N_REPORT]],
        "rank": int(model.rank_),
        "filters": np.asarray(model.filters_, dtype=np.float64),
        "library": True,
    }


def _fit_faithful(
    X_window: np.ndarray,
    rdm: np.ndarray,
    *,
    matrix_mode: str,
) -> dict[str, Any]:
    result = fit_condition_averages(
        X_window,
        rdm,
        pair_mode="unique_unordered",
        matrix_mode=matrix_mode,  # type: ignore[arg-type]
        n_bootstrapping_iterations=0,
        inference="none",
    )
    return {
        "path": f"source_faithful_unique_{matrix_mode}",
        "authority": (
            "paper/reproduction/common/source_faithful.py; unique unordered "
            f"pairs; matrix_mode={matrix_mode}"
        ),
        "demean_time": matrix_mode == "matlab_cov",
        "pair_mode": "unique_unordered",
        "matrix_mode": matrix_mode,
        "eigenvalues_head": [float(v) for v in result.eigenvalues[:N_REPORT]],
        "rank": int(result.eigenvalues.size),
        "filters": np.asarray(result.filters, dtype=np.float64),
        "library": False,
    }


def _contrast_payload(
    packed: dict[str, Any],
    *,
    kind: str,
    rdm: np.ndarray,
    rdm_within01: np.ndarray,
    center_ms: float,
    duration_ms: float,
    paper_corr: float,
    paper_label: str,
    named_latencies_ms: dict[str, float],
) -> dict[str, Any]:
    times = packed["times_ms"]
    X_full = packed["data"]
    win = window_slice(X_full, times, center_ms=center_ms, duration_ms=duration_ms)
    Xw = win["data"]
    pairs = unique_pairs(int(Xw.shape[0]))
    gram_window = pair_stack_from_condition_averages(
        Xw, pairs, matrix_mode="unscaled_gram"
    )
    cov_window = pair_stack_from_condition_averages(
        Xw, pairs, matrix_mode="matlab_cov"
    )

    fits = [
        _fit_library_gram(Xw, rdm),
        _fit_faithful(Xw, rdm, matrix_mode="unscaled_gram"),
        _fit_faithful(Xw, rdm, matrix_mode="matlab_cov"),
    ]

    path_payloads = {}
    for fit in fits:
        comps = []
        n_keep = min(N_REPORT, int(fit["filters"].shape[0 if fit["library"] else 1]))
        for k in range(n_keep):
            w = _filter_row(fit["filters"], k, library=bool(fit["library"]))
            corr = _component_correlations(
                weights=w,
                X_window=Xw,
                X_full=X_full,
                times_full_ms=times,
                target=rdm,
                target_within01=rdm_within01,
                gram_window=gram_window,
                cov_window=cov_window,
                pairs=pairs,
                named_latencies_ms=named_latencies_ms,
            )
            h = corr["headline"]
            comps.append(
                {
                    "index": k,
                    "eigenvalue": float(fit["eigenvalues_head"][k]),
                    "correlations": corr,
                    "vs_paper": {
                        "paper": paper_label,
                        "paper_numeric_for_delta": paper_corr,
                        "window_unique_pearson_minus_paper": (
                            h["item1_window_unique_pearson"] - paper_corr
                        ),
                        "full_epoch_unique_pearson_minus_paper": (
                            h["item2_full_epoch_unique_pearson"] - paper_corr
                        ),
                        "eq2_sample_inner_minus_paper": (
                            h["eq2_printed_inner_product_sample_sd_window"] - paper_corr
                        ),
                    },
                }
            )
        path_payloads[fit["path"]] = {
            "authority": fit["authority"],
            "matrix_mode": fit["matrix_mode"],
            "pair_mode": fit["pair_mode"],
            "eigenvalues_head": fit["eigenvalues_head"],
            "rank": fit["rank"],
            "components": comps,
        }

    lib_l0 = path_payloads["library_unique_unscaled_gram"]["eigenvalues_head"][0]
    gram_l0 = path_payloads["source_faithful_unique_unscaled_gram"]["eigenvalues_head"][0]
    return {
        "kind": kind,
        "paper_rdm_corr": paper_label,
        "window": {
            "center_ms": float(win["center_ms"]),
            "duration_ms": float(win["duration_ms"]),
            "t_start_ms": float(win["t_start_ms"]),
            "t_end_ms": float(win["t_end_ms"]),
            "n_samples": int(win["n_samples"]),
            "index_start": int(win["index_start"]),
            "index_end_inclusive": int(win["index_end_inclusive"]),
        },
        "target_rdm_binary_0_1": rdm.tolist(),
        "target_rdm_within0.1_labeled_extra": rdm_within01.tolist(),
        "library_gram_vs_source_faithful_gram_lambda0_abs_diff": abs(lib_l0 - gram_l0),
        "paths": path_payloads,
    }


def _closest_rows(contrast: dict[str, Any], paper_value: float) -> list[dict[str, Any]]:
    rows = []
    for path_name, path in contrast["paths"].items():
        for comp in path["components"]:
            h = comp["correlations"]["headline"]
            for key, value in h.items():
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "path": path_name,
                        "component": int(comp["index"]),
                        "definition": key,
                        "value": float(value),
                        "paper": paper_value,
                        "abs_diff": abs(float(value) - paper_value),
                    }
                )
    rows.sort(key=lambda r: r["abs_diff"])
    return rows


def _verdict(face: dict[str, Any], car: dict[str, Any]) -> dict[str, Any]:
    face0 = face["paths"]["library_unique_unscaled_gram"]["components"][0]
    car0 = car["paths"]["library_unique_unscaled_gram"]["components"][0]
    car1 = car["paths"]["library_unique_unscaled_gram"]["components"][1]
    fh = face0["correlations"]["headline"]
    ch = car0["correlations"]["headline"]
    window_face = fh["item1_window_unique_pearson"]
    full_face = fh["item2_full_epoch_unique_pearson"]
    inner_face = fh["eq2_printed_inner_product_sample_sd_window"]
    inner_car = ch["eq2_printed_inner_product_sample_sd_window"]
    window_car = ch["item1_window_unique_pearson"]
    full_car = ch["item2_full_epoch_unique_pearson"]

    # A definition that maps both near-1 window RDMs through (n-1)/n cannot
    # produce face=0.82 and car>0.99 simultaneously.
    inner_ruled_out_by_car = inner_car < 0.99

    face_window_explains_082 = abs(window_face - 0.82) < 0.01
    face_full_explains_082 = abs(full_face - 0.82) < 0.01
    any_flatten_explains = any(
        abs(fh[k] - 0.82) < 0.01
        for k in (
            "airi_grown_corrcoef_window",
            "possible_misread_full_flatten_window",
            "possible_misread_triu_incl_diag_window",
        )
    )

    explains = (
        face_window_explains_082 or face_full_explains_082 or any_flatten_explains
    )
    classification = (
        "explained_by_documented_definition"
        if explains
        else "unresolved_historical_preprocessing_or_implementation"
    )
    return {
        "classification": classification,
        "canonical_library_bug": False,
        "near_perfect_two_level_match_is_expected": True,
        "expectedness": (
            "ReDisCA/SPoC maximizes a covariance proxy of corr(w^T R_ij w, d_ij) "
            "over 28 channels and ~26 window samples. A 6-entry two-level target "
            "(3 between, 3 within) is a low-dimensional constraint; a leading "
            "component that isolates one condition from the other three routinely "
            "drives unique-triangle Pearson to ~1. A library GEP that matches "
            "that target near-perfectly is the expected optimum, not a bug."
        ),
        "face_library_gram_comp0": {
            "lambda1": face0["eigenvalue"],
            "paper_lambda1": 0.87209,
            "window_unique_pearson": window_face,
            "full_epoch_unique_pearson": full_face,
            "paper": 0.82,
        },
        "car_library_gram": {
            "lambda1": car0["eigenvalue"],
            "lambda2": car1["eigenvalue"],
            "window_unique_pearson_comp0": window_car,
            "window_unique_pearson_comp1": car1["correlations"]["headline"][
                "item1_window_unique_pearson"
            ],
            "full_epoch_unique_pearson_comp0": full_car,
            "paper": ">0.99",
        },
        "eq2_sample_inner_product_ruled_out_by_car_control": inner_ruled_out_by_car,
        "eq2_sample_inner_note": (
            f"Eq. 2 RHS with MATLAB sample SD is (n-1)/n * r = 5/6 * r. "
            f"Face window → {inner_face:.5f} (closest paper-formula number to "
            f"0.82). The same formula on the car window is {inner_car:.5f}, "
            "which cannot be the published >0.99. Therefore this reading is "
            "not how both figures were computed."
        ),
        "none_of_the_endorsed_definitions_yield_0.82": not explains,
        "opposite_full_epoch_pattern": (
            f"Official averages: face full-epoch unique Pearson {full_face:.5f} "
            f"> car full-epoch {full_car:.5f}. Paper: face 0.82 < car >0.99. "
            "Windowed unique Pearson is ~1 for both contrasts."
        ),
        "stock_spoc": (
            "Stock SPoC does not compute an RDM correlation (stock_spoc.md: "
            "SPoC does not know about RDMs)."
        ),
        "airi_n170": (
            "N170 has no AIRI script (D12). The MEG plotting Q(c,i) analog "
            "applied to windowed N170 RDMs still yields ~1, not 0.82."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    packed = load_n170_subject1(lpfilt=False)
    targets = paper_targets()
    face_rdm_bin = face_rdm(within=0.0, between=1.0)
    face_rdm_01 = face_rdm(within=0.1, between=1.0)
    car_rdm_bin = car_rdm(within=0.0, between=1.0)
    car_rdm_01 = car_rdm(within=0.1, between=1.0)

    print("[rdm-corr] Fig 10 face T=100 ms @ 200 ms", flush=True)
    face = _contrast_payload(
        packed,
        kind="face",
        rdm=face_rdm_bin,
        rdm_within01=face_rdm_01,
        center_ms=FACE_CENTER_MS,
        duration_ms=FACE_DURATION_MS,
        paper_corr=0.82,
        paper_label="0.82",
        named_latencies_ms={"face_burst_ms": 170.0, "window_center_ms": 200.0},
    )
    print("[rdm-corr] Fig 11 car T=100 ms @ 170 ms", flush=True)
    car = _contrast_payload(
        packed,
        kind="car",
        rdm=car_rdm_bin,
        rdm_within01=car_rdm_01,
        center_ms=CAR_CENTER_MS,
        duration_ms=CAR_DURATION_MS,
        paper_corr=0.99,
        paper_label=">0.99",
        named_latencies_ms={"car_applied_at_ms": 170.0, "paper_deflection_ms": 150.0},
    )
    verdict = _verdict(face, car)
    table = {
        "face_vs_0.82_closest": _closest_rows(face, 0.82)[:12],
        "car_vs_0.99_closest": _closest_rows(car, 0.99)[:8],
    }

    env = capture_environment()
    env["redisca_import"] = {
        "module": ReDisCA.__module__,
        "class": ReDisCA.__name__,
        "demean_time_used": False,
    }

    payload = {
        "id": "n170-rdm-correlation-track-D",
        "task": (
            "How could published Fig. 10 corr=0.82 have been computed under "
            "definitions named by the paper, Fig. 10/11, or AIRI/SPoC?"
        ),
        "forbidden": [
            "Spearman / cosine / RV / channel-subset RSA (not named)",
            "tuning a time range until correlation becomes 0.82",
            "changing the target RDM fill without source evidence",
        ],
        "paper": targets,
        "data": {
            "erp": packed["path"],
            "erp_sha256": packed["sha256"],
            "lpfilt": packed["lpfilt"],
            "n_channels": len(packed["channel_labels"]),
            "channel_labels": packed["channel_labels"],
            "n_accepted": packed["n_accepted"],
            "srate_hz": packed["srate_hz"],
            "pair_mode": "unique_unordered",
            "subject": "1",
        },
        "definitions_tested": {
            "1": "Eq. 1 ||u_i-u_j||^2 on analysis-window traces vs D; Pearson unique i<j",
            "2": "same on full-epoch traces with the fitted window filter",
            "3": "Dhat_ij = w^T R_ij w, unscaled Gram R_ij (window)",
            "4": "Dhat_ij = w^T R_ij w, MATLAB-cov R_ij (window); Pearson scale-invariant",
            "5": "Pearson after Eq. 2 standardization (affine-invariant ⇒ matches 1)",
            "5_printed_rhs": (
                "literal Eq. 2 inner product of standardized triangles; "
                "sample SD ⇒ (n-1)/n * r; ruled out by car >0.99"
            ),
            "6": (
                "AIRI MEG corrcoef of dynamically grown (C-1)×C upper fill; "
                "stock SPoC computes no RDM correlation; N170 has no AIRI script"
            ),
            "possible_misread": (
                "flatten whole 4×4 including diagonal; triu including diagonal"
            ),
        },
        "face": face,
        "car": car,
        "table": table,
        "verdict": verdict,
        "environment": env,
    }
    write_json(results_dir / "correlations.json", payload)

    summary = {
        "id": payload["id"],
        "erp_sha256": packed["sha256"],
        "classification": verdict["classification"],
        "canonical_library_bug": False,
        "table": {
            "face_library_gram_comp0": face["paths"]["library_unique_unscaled_gram"][
                "components"
            ][0]["correlations"]["headline"],
            "face_source_faithful_matlab_cov_comp0": face["paths"][
                "source_faithful_unique_matlab_cov"
            ]["components"][0]["correlations"]["headline"],
            "car_library_gram_comp0": car["paths"]["library_unique_unscaled_gram"][
                "components"
            ][0]["correlations"]["headline"],
            "car_library_gram_comp1": car["paths"]["library_unique_unscaled_gram"][
                "components"
            ][1]["correlations"]["headline"],
            "paper": {"face": 0.82, "car": ">0.99"},
        },
        "lambda1": {
            "face_library_gram": face["paths"]["library_unique_unscaled_gram"][
                "eigenvalues_head"
            ][0],
            "face_source_faithful_gram": face["paths"][
                "source_faithful_unique_unscaled_gram"
            ]["eigenvalues_head"][0],
            "face_source_faithful_matlab_cov": face["paths"][
                "source_faithful_unique_matlab_cov"
            ]["eigenvalues_head"][0],
            "paper_face": 0.87209,
            "car_library_gram": car["paths"]["library_unique_unscaled_gram"][
                "eigenvalues_head"
            ][:2],
            "paper_car": [0.91639, 0.77036],
        },
        "verdict": verdict,
    }
    write_json(results_dir / "summary.json", summary)

    print(
        json.dumps(
            {
                "classification": verdict["classification"],
                "face_window": fh_safe(face),
                "face_full": face["paths"]["library_unique_unscaled_gram"]["components"][0][
                    "correlations"
                ]["headline"]["item2_full_epoch_unique_pearson"],
                "car_window": car["paths"]["library_unique_unscaled_gram"]["components"][0][
                    "correlations"
                ]["headline"]["item1_window_unique_pearson"],
                "car_full": car["paths"]["library_unique_unscaled_gram"]["components"][0][
                    "correlations"
                ]["headline"]["item2_full_epoch_unique_pearson"],
            },
            indent=2,
        ),
        flush=True,
    )
    print("[rdm-corr] wrote", results_dir, flush=True)
    return 0


def fh_safe(face: dict[str, Any]) -> float:
    return float(
        face["paths"]["library_unique_unscaled_gram"]["components"][0]["correlations"][
            "headline"
        ]["item1_window_unique_pearson"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
