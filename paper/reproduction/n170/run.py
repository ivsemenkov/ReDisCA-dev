#!/usr/bin/env python3
"""Run the Ossadtchi et al. 2024 N170 reproduction track (Figs 7–11).

Canonical deterministic fits: ``from redisca import ReDisCA``.
Inference is a separate condition-label permutation layer (see inference.py).

Example
-------
    python paper/reproduction/n170/run.py
    python paper/reproduction/n170/run.py --B 1000 --step-ms 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "paper" / "reproduction"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from redisca import ReDisCA  # noqa: E402

from common.hashing import write_json  # noqa: E402
from common.metrics import peak_latency_and_amplitude  # noqa: E402
from common.provenance import capture_environment  # noqa: E402
from common.rng import numpy_generator  # noqa: E402
from common.serialize import array_fingerprint  # noqa: E402

from inference import (  # noqa: E402
    assert_core_matches_estimator,
    exact_condition_label_null,
    fit_window,
    monte_carlo_null,
)
from plotting import (  # noqa: E402
    save_component_panel,
    save_fig07,
    save_fig08,
    save_fig09,
)
from prepare import (  # noqa: E402
    CAR_CENTER_MS,
    CAR_DURATION_MS,
    DEFAULT_SLIDING_STEP_MS,
    FACE_CENTER_MS,
    FACE_DURATION_MS,
    FIG8_CENTERS_MS,
    MEANING_DURATION_MS,
    OCCIPITAL_LABELS,
    load_n170_subject1,
    sliding_centers_ms,
    window_slice,
)
from rdms import (  # noqa: E402
    CONDITION_LABELS,
    rdm_catalog,
    theoretical_rdm,
    zscored_unique_pairs,
)

RESULTS_DIR = REPO_ROOT / "paper" / "results" / "n170"
N_REPORT_COMPONENTS = 8
N_PMAP_ROWS = 12
PAPER_FACE_CORR = 0.82
PAPER_CAR_CORR = 0.99  # paper: greater than 0.99


def _finite(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _peak_report(times_ms: np.ndarray, series: np.ndarray, lo_ms: float, hi_ms: float) -> dict[str, Any]:
    times_s = np.asarray(times_ms, dtype=np.float64) / 1000.0
    series = np.asarray(series, dtype=np.float64)
    mask = (times_ms >= lo_ms) & (times_ms <= hi_ms)
    if not np.any(mask):
        mask = np.ones(series.shape[-1], dtype=bool)
    info = peak_latency_and_amplitude(times_s[mask], series[mask], signed=True)
    info["peak_time_ms"] = float(info["peak_time_s"] * 1000.0)
    info["search_lo_ms"] = float(lo_ms)
    info["search_hi_ms"] = float(hi_ms)
    return info


def _pattern_summary(pattern: np.ndarray, labels: list[str]) -> dict[str, Any]:
    pattern = np.asarray(pattern, dtype=np.float64)
    abs_idx = int(np.argmax(np.abs(pattern)))
    occ_idx = [i for i, lab in enumerate(labels) if lab in OCCIPITAL_LABELS]
    occ_energy = float(np.sum(pattern[occ_idx] ** 2) / np.sum(pattern ** 2))
    def _val(lab: str) -> float | None:
        if lab not in labels:
            return None
        return float(pattern[labels.index(lab)])
    return {
        "max_abs_channel": labels[abs_idx],
        "max_abs_value": float(pattern[abs_idx]),
        "occipital_energy_fraction": occ_energy,
        "O1": _val("O1"),
        "Oz": _val("Oz"),
        "O2": _val("O2"),
        "PO7": _val("PO7"),
        "PO8": _val("PO8"),
        "P7": _val("P7"),
        "P8": _val("P8"),
        "right_minus_left_O2_O1": (
            None if _val("O2") is None or _val("O1") is None else float(_val("O2") - _val("O1"))
        ),
    }


def _fit_payload(
    fit,
    *,
    rdm,
    labels: list[str],
    times_full_ms: np.ndarray,
    window_meta: dict[str, Any],
    exact_null: dict[str, Any] | None,
    mc_nulls: dict[str, Any] | None,
    n_sig_alpha: float = 0.05,
) -> dict[str, Any]:
    n_keep = int(fit.patterns.shape[0])
    p_ge = exact_null["p_greater_equal"][:n_keep] if exact_null else None
    n_sig = None
    if p_ge is not None:
        n_sig = int(sum(p < n_sig_alpha for p in p_ge))
    components = []
    for k in range(n_keep):
        peaks = {
            labels[c]: _peak_report(
                times_full_ms, fit.traces_full[c, k, :], 80.0, 250.0
            )
            for c in range(len(labels))
        }
        components.append(
            {
                "index": k,
                "eigenvalue": float(fit.eigenvalues[k]),
                "rdm_corr_window": _finite(fit.rdm_corr_window[k]),
                "rdm_corr_full_epoch": _finite(fit.rdm_corr_full[k]),
                "p_exact24_greater_equal": None if p_ge is None else float(p_ge[k]),
                "p_exact24_strict_greater": (
                    None
                    if exact_null is None
                    else float(exact_null["p_strict_greater"][k])
                ),
                "pattern": _pattern_summary(fit.patterns[k], labels),
                "peaks_80_250ms": peaks,
                "empirical_rdm_window": fit.empirical_rdm_window[k],
            }
        )
    return {
        "demean_time": bool(fit.demean_time),
        "rank": int(fit.model.rank_),
        "n_components_reported": n_keep,
        "window": {
            key: window_meta[key]
            for key in (
                "center_ms",
                "duration_ms",
                "t_start_ms",
                "t_end_ms",
                "n_samples",
                "index_start",
                "index_end_inclusive",
            )
        },
        "eigenvalues_head": [float(v) for v in fit.eigenvalues[:n_keep]],
        "eigenvalues_fingerprint": array_fingerprint(fit.eigenvalues),
        "patterns_fingerprint": array_fingerprint(fit.patterns),
        "components": components,
        "n_components_p_greater_equal_lt_0.05": n_sig,
        "inference_exact24": exact_null,
        "inference_monte_carlo": mc_nulls,
        "note_rdm_corr": (
            "rdm_corr_window is Pearson of unique i<j entries of the theoretical "
            "RDM vs ||u_i-u_j||^2 on the analysis-window component traces "
            "(demeaned iff demean_time). rdm_corr_full_epoch uses the same "
            "spatial filter on the whole epoch. Paper face 0.82 / car >0.99 "
            "are not tuning targets."
        ),
    }


def _run_window_analysis(
    packed: dict[str, Any],
    rdm: np.ndarray,
    *,
    center_ms: float,
    duration_ms: float,
    demean_time: bool,
    rng: np.random.Generator,
    n_mc: int,
    include_exploratory: bool,
) -> tuple[Any, dict[str, Any]]:
    X_full = packed["data"]
    labels = packed["channel_labels"]
    times = packed["times_ms"]
    win = window_slice(X_full, times, center_ms=center_ms, duration_ms=duration_ms)
    Xw = win["data"]
    assert_core_matches_estimator(Xw, rdm, demean_time=demean_time)
    fit = fit_window(
        Xw,
        X_full,
        rdm,
        demean_time=demean_time,
        channel_labels=labels,
        n_report=N_REPORT_COMPONENTS,
    )
    exact = exact_condition_label_null(
        Xw, rdm, fit.eigenvalues[:N_REPORT_COMPONENTS], demean_time=demean_time
    )
    mc: dict[str, Any] = {
        "condition_label": monte_carlo_null(
            Xw,
            rdm,
            fit.eigenvalues[:N_REPORT_COMPONENTS],
            demean_time=demean_time,
            kind="condition_label",
            n_permutations=n_mc,
            rng=rng,
        )
    }
    if include_exploratory:
        mc["pair_vector"] = monte_carlo_null(
            Xw,
            rdm,
            fit.eigenvalues[:N_REPORT_COMPONENTS],
            demean_time=demean_time,
            kind="pair_vector",
            n_permutations=n_mc,
            rng=rng,
        )
        mc["random_phase_exploratory"] = monte_carlo_null(
            Xw,
            rdm,
            fit.eigenvalues[:N_REPORT_COMPONENTS],
            demean_time=demean_time,
            kind="random_phase",
            n_permutations=n_mc,
            rng=rng,
        )
    payload = _fit_payload(
        fit,
        rdm=rdm,
        labels=list(CONDITION_LABELS),
        times_full_ms=times,
        window_meta=win,
        exact_null=exact,
        mc_nulls=mc,
    )
    return fit, payload


def _meaning_scan(
    packed: dict[str, Any],
    rdm: np.ndarray,
    *,
    duration_ms: float,
    step_ms: float,
    demean_time: bool,
    n_pmap_rows: int = N_PMAP_ROWS,
) -> dict[str, Any]:
    times = packed["times_ms"]
    X_full = packed["data"]
    labels = packed["channel_labels"]
    centers = sliding_centers_ms(times, duration_ms=duration_ms, step_ms=step_ms)
    rows = []
    pmap_ge = []
    pmap_gt = []
    lambda0 = []
    corr0 = []
    pattern_400 = None
    n_samples = []
    for center in centers:
        win = window_slice(X_full, times, center_ms=float(center), duration_ms=duration_ms)
        Xw = win["data"]
        n_samples.append(int(win["n_samples"]))
        fit = fit_window(
            Xw,
            X_full,
            rdm,
            demean_time=demean_time,
            channel_labels=labels,
            n_report=max(n_pmap_rows, 1),
        )
        n_row = min(n_pmap_rows, int(fit.eigenvalues.size))
        exact = exact_condition_label_null(
            Xw, rdm, fit.eigenvalues[:n_row], demean_time=demean_time
        )
        p_ge = np.asarray(exact["p_greater_equal"], dtype=np.float64)
        p_gt = np.asarray(exact["p_strict_greater"], dtype=np.float64)
        if p_ge.size < n_pmap_rows:
            p_ge = np.pad(p_ge, (0, n_pmap_rows - p_ge.size), constant_values=np.nan)
            p_gt = np.pad(p_gt, (0, n_pmap_rows - p_gt.size), constant_values=np.nan)
        pmap_ge.append(p_ge[:n_pmap_rows])
        pmap_gt.append(p_gt[:n_pmap_rows])
        lambda0.append(float(fit.eigenvalues[0]))
        corr0.append(_finite(fit.rdm_corr_window[0]))
        if abs(float(center) - 400.0) < 1e-6:
            pattern_400 = {
                "center_ms": float(center),
                "pattern": fit.patterns[0].tolist(),
                "summary": _pattern_summary(fit.patterns[0], labels),
                "rdm_corr_window": _finite(fit.rdm_corr_window[0]),
                "eigenvalue": float(fit.eigenvalues[0]),
            }
        unique_lams = sorted(
            (item["lambda0"] for item in exact["unique"]), reverse=True
        )
        rows.append(
            {
                "center_ms": float(center),
                "n_samples": int(win["n_samples"]),
                "lambda0": float(fit.eigenvalues[0]),
                "rdm_corr_window_comp0": _finite(fit.rdm_corr_window[0]),
                "p_comp0_greater_equal": float(exact["p_greater_equal"][0]),
                "p_comp0_strict_greater": float(exact["p_strict_greater"][0]),
                "unique_lambda0": unique_lams,
                "pattern_summary_comp0": _pattern_summary(fit.patterns[0], labels),
            }
        )
    pmap_ge_arr = np.stack(pmap_ge, axis=1)
    pmap_gt_arr = np.stack(pmap_gt, axis=1)
    p0 = np.array([row["p_comp0_greater_equal"] for row in rows], dtype=np.float64)
    # Continuous p<0.05 segment on component 1 (will often be empty given C=4 floor).
    below = p0 < 0.05
    segments = []
    start = None
    for i, flag in enumerate(below):
        if flag and start is None:
            start = i
        if (not flag or i == len(below) - 1) and start is not None:
            end = i if flag and i == len(below) - 1 else i - 1
            segments.append(
                {
                    "i_start": int(start),
                    "i_end": int(end),
                    "center_ms_start": float(centers[start]),
                    "center_ms_end": float(centers[end]),
                }
            )
            start = None
    nearest_400 = int(np.argmin(np.abs(centers - 400.0)))
    return {
        "demean_time": bool(demean_time),
        "duration_ms": float(duration_ms),
        "step_ms": float(step_ms),
        "step_ms_paper": None,
        "step_note": (
            "Sliding step is not specified in Ossadtchi et al. 2024. "
            f"{step_ms:g} ms is a documented choice, not a paper value."
        ),
        "centers_ms": [float(c) for c in centers],
        "n_windows": int(centers.size),
        "n_samples_per_window": n_samples,
        "n_pmap_rows": int(n_pmap_rows),
        "p_map_greater_equal": pmap_ge_arr.tolist(),
        "p_map_strict_greater": pmap_gt_arr.tolist(),
        "p_comp0_greater_equal": [float(v) for v in p0],
        "lambda0": lambda0,
        "rdm_corr_comp0": corr0,
        "windows": rows,
        "p_lt_0.05_segments_comp0_greater_equal": segments,
        "nearest_400ms_center_ms": float(centers[nearest_400]),
        "nearest_400ms_p_greater_equal": float(p0[nearest_400]),
        "nearest_400ms_p_strict_greater": float(rows[nearest_400]["p_comp0_strict_greater"]),
        "pattern_at_400ms": pattern_400,
        "discrete_floor_note": (
            "With C=4 the meaning RDM is invariant under 8 of 24 condition "
            "relabelings, so uncorrected P(lambda* >= lambda_obs) has floor "
            "8/24 ≈ 0.333 when this partition is uniquely best. Paper Fig 7 "
            "uncorrected p<0.05 around t=400 ms is not reachable under this "
            "null without a different permutation scheme or a strict-'exceeds' "
            "rule that ignores equivalent relabelings."
        ),
    }


def _fig8_windows(
    packed: dict[str, Any],
    rdm: np.ndarray,
    *,
    demean_time: bool,
) -> list[dict[str, Any]]:
    out = []
    X_full = packed["data"]
    labels = packed["channel_labels"]
    times = packed["times_ms"]
    for center in FIG8_CENTERS_MS:
        win = window_slice(
            X_full, times, center_ms=float(center), duration_ms=MEANING_DURATION_MS
        )
        fit = fit_window(
            win["data"],
            X_full,
            rdm,
            demean_time=demean_time,
            channel_labels=labels,
            n_report=1,
        )
        exact = exact_condition_label_null(
            win["data"], rdm, fit.eigenvalues[:1], demean_time=demean_time
        )
        out.append(
            {
                "center_ms": float(center),
                "duration_ms": MEANING_DURATION_MS,
                "n_samples": int(win["n_samples"]),
                "pattern": fit.patterns[0],
                "traces_full": fit.traces_full[:, 0, :],
                "empirical_rdm": np.asarray(fit.empirical_rdm_window[0]),
                "rdm_corr": fit.rdm_corr_window[0],
                "eigenvalue": float(fit.eigenvalues[0]),
                "p_greater_equal": float(exact["p_greater_equal"][0]),
                "p_strict_greater": float(exact["p_strict_greater"][0]),
                "pattern_summary": _pattern_summary(fit.patterns[0], labels),
                "json": {
                    "center_ms": float(center),
                    "duration_ms": MEANING_DURATION_MS,
                    "n_samples": int(win["n_samples"]),
                    "eigenvalue": float(fit.eigenvalues[0]),
                    "rdm_corr_window": _finite(fit.rdm_corr_window[0]),
                    "rdm_corr_full_epoch": _finite(fit.rdm_corr_full[0]),
                    "p_exact24_greater_equal": float(exact["p_greater_equal"][0]),
                    "p_exact24_strict_greater": float(exact["p_strict_greater"][0]),
                    "empirical_rdm_window": fit.empirical_rdm_window[0],
                    "pattern_summary": _pattern_summary(fit.patterns[0], labels),
                    "peaks_80_250ms": {
                        CONDITION_LABELS[c]: _peak_report(
                            times, fit.traces_full[c, 0, :], 80.0, 250.0
                        )
                        for c in range(4)
                    },
                    "peaks_300_500ms": {
                        CONDITION_LABELS[c]: _peak_report(
                            times, fit.traces_full[c, 0, :], 300.0, 500.0
                        )
                        for c in range(4)
                    },
                },
            }
        )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--B",
        type=int,
        default=1000,
        help="Monte Carlo permutations for the requested B=1000 contract (default 1000).",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=DEFAULT_SLIDING_STEP_MS,
        help="Sliding-window step in ms (NOT in the paper; default 25).",
    )
    parser.add_argument("--seed", type=int, default=20240904)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--skip-exploratory",
        action="store_true",
        help="Skip pair-shuffle and SPoC random-phase diagnostics.",
    )
    parser.add_argument(
        "--lpfilt",
        action="store_true",
        help="Use 1_N170_erp_ar_lpfilt.erp instead of the preferred unfiltered ERP.",
    )
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
    rng, rng_record = numpy_generator(int(args.seed))

    packed = load_n170_subject1(lpfilt=bool(args.lpfilt))
    catalog = rdm_catalog()
    z_match = {}
    for kind in ("meaning", "face", "car"):
        z0 = zscored_unique_pairs(catalog[kind])
        z1 = zscored_unique_pairs(catalog[f"{kind}_within0.1"])
        z_match[kind] = {
            "max_abs_z_diff_0_vs_0.1": float(np.max(np.abs(z0 - z1))),
            "equivalent_after_zscore": bool(np.allclose(z0, z1, atol=1e-12, rtol=1e-12)),
        }

    env = capture_environment(extra_packages=("matplotlib",))
    env["redisca_import"] = {
        "module": ReDisCA.__module__,
        "class": ReDisCA.__name__,
        "demean_time_default": ReDisCA().demean_time,
    }

    channel_doc = {
        "kept": packed["channel_selection"],
        "channel_labels": packed["channel_labels"],
        "channel_xyz": packed["channel_xyz"].tolist(),
        "channel_indices_in_erp": packed["channel_indices_in_erp"],
        "all_channel_labels": packed["all_channel_labels"],
        "decision": (
            "Use the 28 scalp EEG channels in the ERPLAB file (FP1…O2). "
            "Drop HEOG_left/right, VEOG_lower, and (corr)/(uncorr) bipolar "
            "EOG. Paper does not name the channel set; this is the documented "
            "scalp subset of the official ERP CORE average."
        ),
    }
    write_json(results_dir / "channel_selection.json", channel_doc)

    fig09 = {
        "id": "fig09-n170-face-car-rdms",
        "condition_order": packed["condition_order"],
        "condition_labels": packed["condition_labels"],
        "binary_0_1": {
            "meaning": catalog["meaning"].tolist(),
            "face": catalog["face"].tolist(),
            "car": catalog["car"].tolist(),
        },
        "within_0.1_labeled_extra": {
            "meaning": catalog["meaning_within0.1"].tolist(),
            "face": catalog["face_within0.1"].tolist(),
            "car": catalog["car_within0.1"].tolist(),
            "zscore_equivalence_to_binary": z_match,
            "note": (
                "0.1-within is an AIRI-style fill, not printed in the N170 "
                "figures. After sample-SD z-scoring of unique pairs it is "
                "identical to 0/1 for these two-level RDMs, so ReDisCA "
                "eigenvalues/filters match."
            ),
        },
        "paper_vs_this": (
            "Paper Figs 7a/9a/9b are images; entries are not printed. "
            "Structure follows §4.2.1 prose."
        ),
    }
    write_json(results_dir / "fig09_rdms.json", fig09)
    if not args.no_plots:
        save_fig09(
            results_dir / "fig09_rdms.png",
            catalog["face"],
            catalog["car"],
            catalog["meaning"],
            packed["condition_labels"],
        )

    include_expl = not bool(args.skip_exploratory)
    n_mc = int(args.B)
    print(
        f"[n170] subject=1  channels={len(packed['channel_labels'])}  "
        f"erp={Path(packed['path']).name}  B_mc={n_mc}  step_ms={args.step_ms:g}",
        flush=True,
    )

    # Fig 10 face, both demean flags.
    fig10_paths = {}
    for demean in (False, True):
        tag = "paper_gram" if not demean else "demeaned_gram_extra"
        print(f"[n170] Fig 10 face window demean_time={demean}", flush=True)
        fit, payload = _run_window_analysis(
            packed,
            catalog["face"],
            center_ms=FACE_CENTER_MS,
            duration_ms=FACE_DURATION_MS,
            demean_time=demean,
            rng=rng,
            n_mc=n_mc,
            include_exploratory=include_expl,
        )
        payload["paper_expected"] = {
            "rdm_corr": PAPER_FACE_CORR,
            "n_significant_components": 1,
            "window": "T=100 ms centered at 200 ms",
            "face_burst_ms": 170,
        }
        payload["paper_minus_observed_corr_window_comp0"] = (
            PAPER_FACE_CORR - float(fit.rdm_corr_window[0])
            if np.isfinite(fit.rdm_corr_window[0])
            else None
        )
        fig10_paths[tag] = payload
        if not args.no_plots and not demean:
            save_component_panel(
                results_dir / "fig10_face.png",
                title=(
                    "Fig 10 analog: face RDM, T=100 ms @ 200 ms, "
                    f"demean_time=False, window corr={fit.rdm_corr_window[0]:.4f} "
                    f"(paper 0.82)"
                ),
                patterns=[fit.patterns[0]],
                traces_list=[fit.traces_full[:, 0, :]],
                empirical_rdms=[np.asarray(fit.empirical_rdm_window[0])],
                corrs=[fit.rdm_corr_window[0]],
                p_values=[payload["components"][0]["p_exact24_greater_equal"]],
                times_ms=packed["times_ms"],
                labels=packed["condition_labels"],
                xyz=packed["channel_xyz"],
                channel_labels=packed["channel_labels"],
                window_lo=FACE_CENTER_MS - FACE_DURATION_MS / 2.0,
                window_hi=FACE_CENTER_MS + FACE_DURATION_MS / 2.0,
            )
    write_json(
        results_dir / "fig10_face.json",
        {
            "id": "fig10-n170-face",
            "erp_sha256": packed["sha256"],
            "paths": fig10_paths,
        },
    )

    fig11_paths = {}
    for demean in (False, True):
        tag = "paper_gram" if not demean else "demeaned_gram_extra"
        print(f"[n170] Fig 11 car window demean_time={demean}", flush=True)
        fit, payload = _run_window_analysis(
            packed,
            catalog["car"],
            center_ms=CAR_CENTER_MS,
            duration_ms=CAR_DURATION_MS,
            demean_time=demean,
            rng=rng,
            n_mc=n_mc,
            include_exploratory=include_expl,
        )
        payload["paper_expected"] = {
            "rdm_corr": f">{PAPER_CAR_CORR}",
            "n_significant_components": 2,
            "p_threshold": 0.01,
            "window": (
                "applied at t=170 ms; duration not restated in the paper; "
                "using T=100 ms centered at 170 ms (only other real-data T "
                "besides 150 ms)."
            ),
            "car_deflection_ms": 150,
        }
        payload["paper_minus_observed_corr_window_comp0"] = (
            PAPER_CAR_CORR - float(fit.rdm_corr_window[0])
            if np.isfinite(fit.rdm_corr_window[0])
            else None
        )
        n_p01 = sum(
            c["p_exact24_greater_equal"] is not None
            and c["p_exact24_greater_equal"] < 0.01
            for c in payload["components"]
        )
        payload["n_components_p_greater_equal_lt_0.01"] = int(n_p01)
        fig11_paths[tag] = payload
        if not args.no_plots and not demean:
            n_show = min(2, fit.patterns.shape[0])
            save_component_panel(
                results_dir / "fig11_car.png",
                title=(
                    "Fig 11 analog: car RDM, T=100 ms @ 170 ms, "
                    f"demean_time=False, window corr={fit.rdm_corr_window[0]:.4f} "
                    "(paper >0.99)"
                ),
                patterns=[fit.patterns[k] for k in range(n_show)],
                traces_list=[fit.traces_full[:, k, :] for k in range(n_show)],
                empirical_rdms=[
                    np.asarray(fit.empirical_rdm_window[k]) for k in range(n_show)
                ],
                corrs=[fit.rdm_corr_window[k] for k in range(n_show)],
                p_values=[
                    payload["components"][k]["p_exact24_greater_equal"]
                    for k in range(n_show)
                ],
                times_ms=packed["times_ms"],
                labels=packed["condition_labels"],
                xyz=packed["channel_xyz"],
                channel_labels=packed["channel_labels"],
                window_lo=CAR_CENTER_MS - CAR_DURATION_MS / 2.0,
                window_hi=CAR_CENTER_MS + CAR_DURATION_MS / 2.0,
            )
    write_json(
        results_dir / "fig11_car.json",
        {
            "id": "fig11-n170-car",
            "erp_sha256": packed["sha256"],
            "paths": fig11_paths,
        },
    )

    print("[n170] Fig 7 meaning sliding scan demean_time=False", flush=True)
    scan_false = _meaning_scan(
        packed,
        catalog["meaning"],
        duration_ms=MEANING_DURATION_MS,
        step_ms=float(args.step_ms),
        demean_time=False,
    )
    print("[n170] Fig 7 meaning sliding scan demean_time=True", flush=True)
    scan_true = _meaning_scan(
        packed,
        catalog["meaning"],
        duration_ms=MEANING_DURATION_MS,
        step_ms=float(args.step_ms),
        demean_time=True,
    )
    # Extra MC at 400 ms for the B=1000 contract on the highlighted window.
    print("[n170] Fig 7 highlighted window t=400 ms with MC", flush=True)
    _, meaning400_false = _run_window_analysis(
        packed,
        catalog["meaning"],
        center_ms=400.0,
        duration_ms=MEANING_DURATION_MS,
        demean_time=False,
        rng=rng,
        n_mc=n_mc,
        include_exploratory=include_expl,
    )
    _, meaning400_true = _run_window_analysis(
        packed,
        catalog["meaning"],
        center_ms=400.0,
        duration_ms=MEANING_DURATION_MS,
        demean_time=True,
        rng=rng,
        n_mc=n_mc,
        include_exploratory=include_expl,
    )
    fig07 = {
        "id": "fig07-n170-meaning-pmap",
        "erp_sha256": packed["sha256"],
        "paper_expected": {
            "first_component_uncorrected_p": "<0.05 around t=400 ms",
            "topography": "highly occipital",
            "T_ms": 150,
            "step_ms": "not specified",
        },
        "paper_gram_demean_time_false": scan_false,
        "demeaned_gram_extra": scan_true,
        "highlighted_window_400ms": {
            "paper_gram": meaning400_false,
            "demeaned_gram_extra": meaning400_true,
        },
    }
    write_json(results_dir / "fig07_meaning_pmap.json", fig07)
    if not args.no_plots:
        centers = np.asarray(scan_false["centers_ms"], dtype=np.float64)
        pmap = np.asarray(scan_false["p_map_greater_equal"], dtype=np.float64)
        pattern = (
            scan_false["pattern_at_400ms"]["pattern"]
            if scan_false["pattern_at_400ms"]
            else np.zeros(len(packed["channel_labels"]))
        )
        save_fig07(
            results_dir / "fig07_meaning_pmap.png",
            meaning_rdm=catalog["meaning"],
            labels=packed["condition_labels"],
            pmap=pmap,
            centers_ms=centers,
            p_comp0=np.asarray(scan_false["p_comp0_greater_equal"]),
            pattern=pattern,
            xyz=packed["channel_xyz"],
            channel_labels=packed["channel_labels"],
            p_rule="exact-24 P(lambda* >= lambda_obs)",
        )

    print("[n170] Fig 8 three adjacent windows", flush=True)
    fig8_false = _fig8_windows(packed, catalog["meaning"], demean_time=False)
    fig8_true = _fig8_windows(packed, catalog["meaning"], demean_time=True)
    fig08 = {
        "id": "fig08-n170-meaning-patterns",
        "centers_ms": list(FIG8_CENTERS_MS),
        "duration_ms": MEANING_DURATION_MS,
        "adjacent_step_ms": 25.0,
        "paper_gram": [w["json"] for w in fig8_false],
        "demeaned_gram_extra": [w["json"] for w in fig8_true],
        "note": (
            "Three consecutive T=150 ms windows centered at 375, 400, 425 ms "
            "on the documented 25 ms step grid around t=400 ms."
        ),
    }
    write_json(results_dir / "fig08_meaning_windows.json", fig08)
    if not args.no_plots:
        save_fig08(
            results_dir / "fig08_meaning_windows.png",
            windows=fig8_false,
            times_full_ms=packed["times_ms"],
            labels=packed["condition_labels"],
            xyz=packed["channel_xyz"],
            channel_labels=packed["channel_labels"],
        )

    # Compact summary + fingerprints for regression.
    face_false = fig10_paths["paper_gram"]["components"][0]
    car_false = fig11_paths["paper_gram"]["components"]
    summary = {
        "track": "n170",
        "subject": "1",
        "erp_file": packed["path"],
        "erp_sha256": packed["sha256"],
        "srate_hz": packed["srate_hz"],
        "n_times": int(packed["times_ms"].size),
        "times_ms_span": [float(packed["times_ms"][0]), float(packed["times_ms"][-1])],
        "n_channels": len(packed["channel_labels"]),
        "channel_labels": packed["channel_labels"],
        "n_accepted": packed["n_accepted"],
        "ica": packed["ica"],
        "sliding_step_ms": float(args.step_ms),
        "sliding_step_in_paper": False,
        "B_monte_carlo": n_mc,
        "B_exact_condition_label": 24,
        "seed": int(args.seed),
        "rng": rng_record.to_dict(),
        "lpfilt": bool(args.lpfilt),
        "figure_status": {
            "fig07-n170-meaning-pmap": "ran; p<0.05 around 400 ms not obtained under condition-label permutation (C=4 discrete floor)",
            "fig08-n170-meaning-patterns": "ran; three adjacent windows 375/400/425 ms",
            "fig09-n170-face-car-rdms": "encoded 0/1 plus labeled 0.1-within extra",
            "fig10-n170-face": "ran; report actual RDM corr (not tuned to 0.82)",
            "fig11-n170-car": "ran; report actual RDM corr (not tuned to >0.99)",
        },
        "numbers_paper_gram_demean_time_false": {
            "face_T100_c200": {
                "rdm_corr_window_comp0": face_false["rdm_corr_window"],
                "rdm_corr_full_epoch_comp0": face_false["rdm_corr_full_epoch"],
                "paper_rdm_corr": PAPER_FACE_CORR,
                "eigenvalue_comp0": face_false["eigenvalue"],
                "p_exact24_greater_equal_comp0": face_false["p_exact24_greater_equal"],
                "p_exact24_strict_greater_comp0": face_false["p_exact24_strict_greater"],
                "peak_faces_ms": face_false["peaks_80_250ms"]["Faces"]["peak_time_ms"],
                "max_abs_channel": face_false["pattern"]["max_abs_channel"],
            },
            "car_T100_c170": {
                "rdm_corr_window_comp0": car_false[0]["rdm_corr_window"],
                "rdm_corr_window_comp1": car_false[1]["rdm_corr_window"],
                "paper_rdm_corr": f">{PAPER_CAR_CORR}",
                "eigenvalue_comp0": car_false[0]["eigenvalue"],
                "eigenvalue_comp1": car_false[1]["eigenvalue"],
                "p_exact24_greater_equal_comp0": car_false[0]["p_exact24_greater_equal"],
                "p_exact24_greater_equal_comp1": car_false[1]["p_exact24_greater_equal"],
                "n_comp_p_ge_lt_0.01": fig11_paths["paper_gram"][
                    "n_components_p_greater_equal_lt_0.01"
                ],
                "peak_cars_comp0_ms": car_false[0]["peaks_80_250ms"]["Cars"]["peak_time_ms"],
                "max_abs_channel_comp0": car_false[0]["pattern"]["max_abs_channel"],
                "max_abs_channel_comp1": car_false[1]["pattern"]["max_abs_channel"],
            },
            "meaning_T150_c400": {
                "rdm_corr_window_comp0": meaning400_false["components"][0]["rdm_corr_window"],
                "eigenvalue_comp0": meaning400_false["components"][0]["eigenvalue"],
                "p_exact24_greater_equal_comp0": meaning400_false["components"][0][
                    "p_exact24_greater_equal"
                ],
                "p_exact24_strict_greater_comp0": meaning400_false["components"][0][
                    "p_exact24_strict_greater"
                ],
                "max_abs_channel": meaning400_false["components"][0]["pattern"][
                    "max_abs_channel"
                ],
                "occipital_energy_fraction": meaning400_false["components"][0]["pattern"][
                    "occipital_energy_fraction"
                ],
            },
        },
        "library": env["redisca_import"],
        "environment": env,
    }
    write_json(results_dir / "summary.json", summary)

    fingerprints = {
        "erp_sha256": packed["sha256"],
        "n_channels": len(packed["channel_labels"]),
        "channel_labels": packed["channel_labels"],
        "face_binary_rdm": catalog["face"].tolist(),
        "car_binary_rdm": catalog["car"].tolist(),
        "meaning_binary_rdm": catalog["meaning"].tolist(),
        "zscore_0_vs_0.1": z_match,
        "face_paper_gram": {
            "evals_head": fig10_paths["paper_gram"]["eigenvalues_head"],
            "corr_window_comp0": face_false["rdm_corr_window"],
            "p_ge_comp0": face_false["p_exact24_greater_equal"],
        },
        "car_paper_gram": {
            "evals_head": fig11_paths["paper_gram"]["eigenvalues_head"],
            "corr_window_comp0": car_false[0]["rdm_corr_window"],
            "corr_window_comp1": car_false[1]["rdm_corr_window"],
            "p_ge_comp0": car_false[0]["p_exact24_greater_equal"],
            "p_ge_comp1": car_false[1]["p_exact24_greater_equal"],
        },
        "meaning400_paper_gram": {
            "evals_head": meaning400_false["eigenvalues_head"],
            "corr_window_comp0": meaning400_false["components"][0]["rdm_corr_window"],
            "p_ge_comp0": meaning400_false["components"][0]["p_exact24_greater_equal"],
        },
        "meaning_scan_paper_gram": {
            "centers_ms": scan_false["centers_ms"],
            "lambda0": scan_false["lambda0"],
            "p_comp0_greater_equal": scan_false["p_comp0_greater_equal"],
        },
        "eigenvalues_fingerprints": {
            "face_paper_gram": fig10_paths["paper_gram"]["eigenvalues_fingerprint"],
            "car_paper_gram": fig11_paths["paper_gram"]["eigenvalues_fingerprint"],
            "meaning400_paper_gram": meaning400_false["eigenvalues_fingerprint"],
        },
    }
    write_json(results_dir / "fingerprints.json", fingerprints)
    write_json(
        results_dir / "environment.json",
        {
            "environment": env,
            "rng": rng_record.to_dict(),
            "commands": {
                "run": "python paper/reproduction/n170/run.py",
                "run_from_repo_root": True,
                "working_directory_cloud": "/tmp/redisca-worktrees/n170",
            },
            "data": {
                "erp": packed["path"],
                "sha256": packed["sha256"],
                "lpfilt": packed["lpfilt"],
                "epoch_note": packed["epoch_note"],
            },
        },
    )

    print("[n170] wrote", results_dir, flush=True)
    print(
        json.dumps(
            {
                "face_corr": face_false["rdm_corr_window"],
                "car_corr": car_false[0]["rdm_corr_window"],
                "meaning400_p_ge": meaning400_false["components"][0][
                    "p_exact24_greater_equal"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
