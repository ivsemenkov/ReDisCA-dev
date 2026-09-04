"""Frozen-estimator fits for N170 Fig. 7 sliding and Fig. 8 windows.

PRIMARY inference is stock SPoC random-phase via
``fit_condition_averages``. Condition-label permutation is secondary.
Does not import ``redisca``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from common.source_faithful import (
    fit_condition_averages,
    pair_indices,
    pair_stack_from_condition_averages,
)

from historical.analysis import (
    _pattern_fingerprint,
    empirical_rdm_trace_sq,
    empirical_rdm_wTRw,
    exact_condition_label_permutation,
    unique_triangle_pearson,
)
from historical_apply.freeze import (
    FIG7_WINDOW_SEED_BASE,
    FROZEN_INFERENCE,
    FROZEN_MATRIX_MODE,
    FROZEN_PAIR_MODE,
    frozen_estimator_record,
)
from prepare import (
    DEFAULT_SLIDING_STEP_MS,
    FIG8_CENTERS_MS,
    MEANING_DURATION_MS,
    sliding_centers_ms,
    window_slice,
)
from rdms import theoretical_rdm

N_REPORT = 8


def _pcg64(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed)))


def _json_float(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def p_lt_segments(
    centers_ms: NDArray[np.floating],
    p_values: NDArray[np.floating],
    *,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Continuous runs of component-1 uncorrected p < alpha along the step grid."""
    centers = np.asarray(centers_ms, dtype=np.float64)
    p = np.asarray(p_values, dtype=np.float64)
    below = np.isfinite(p) & (p < float(alpha))
    segments: list[dict[str, Any]] = []
    start: int | None = None
    n = int(below.size)
    for i in range(n):
        flag = bool(below[i])
        if flag and start is None:
            start = i
        closing = start is not None and ((not flag) or i == n - 1)
        if closing:
            if flag and i == n - 1:
                end = i
            else:
                end = i - 1
            if end >= start:  # type: ignore[operator]
                segments.append(
                    {
                        "i_start": int(start),
                        "i_end": int(end),
                        "center_ms_start": float(centers[start]),
                        "center_ms_end": float(centers[end]),
                        "n_windows": int(end - start + 1),
                        "p_min": float(np.min(p[start : end + 1])),
                    }
                )
            start = None
    return segments


def fit_frozen_window(
    *,
    X_window: NDArray[np.floating],
    rdm: NDArray[np.floating],
    channel_labels: list[str],
    n_bootstrapping_iterations: int,
    rng: np.random.Generator,
    n_report: int = N_REPORT,
    pair_mode: str = FROZEN_PAIR_MODE,
    matrix_mode: str = FROZEN_MATRIX_MODE,
) -> dict[str, Any]:
    """One analysis window under the frozen directed + MATLAB-cov estimator."""
    X_window = np.asarray(X_window, dtype=np.float64)
    rdm = np.asarray(rdm, dtype=np.float64)
    n_conditions, n_channels, n_times = X_window.shape
    pairs = pair_indices(n_conditions, pair_mode)  # type: ignore[arg-type]
    pair_stack = pair_stack_from_condition_averages(
        X_window, pairs, matrix_mode=matrix_mode  # type: ignore[arg-type]
    )
    if n_bootstrapping_iterations > 0:
        inference = FROZEN_INFERENCE
        fit_rng: np.random.Generator | None = rng
    else:
        inference = "none"
        fit_rng = None
    result = fit_condition_averages(
        X_window,
        rdm,
        pair_mode=pair_mode,  # type: ignore[arg-type]
        matrix_mode=matrix_mode,  # type: ignore[arg-type]
        n_bootstrapping_iterations=int(n_bootstrapping_iterations),
        rng=fit_rng,
        inference=inference,
    )
    n_keep = min(int(n_report), int(result.eigenvalues.size))
    traces_window = np.einsum("ck,nct->nkt", result.filters[:, :n_keep], X_window)
    if result.p_values is None:
        p_head: list[float | None] = [None] * n_keep
    else:
        p_head = [float(v) for v in result.p_values[:n_keep]]
    secondary = exact_condition_label_permutation(
        pair_stack,
        rdm,
        pairs,
        result.eigenvalues,
        pair_mode=pair_mode,  # type: ignore[arg-type]
        matrix_mode=matrix_mode,  # type: ignore[arg-type]
        n_report=n_keep,
    )
    components: list[dict[str, Any]] = []
    for k in range(n_keep):
        d_wtrw = empirical_rdm_wTRw(
            result.filters[:, k], pair_stack, pairs, n_conditions
        )
        d_trace = empirical_rdm_trace_sq(
            traces_window[:, k, :], demean_time=False
        )
        pattern = _pattern_fingerprint(result.patterns[:, k], channel_labels)
        components.append(
            {
                "index": k,
                "eigenvalue": float(result.eigenvalues[k]),
                "primary_random_phase_p": p_head[k],
                "rdm_corr_wTRw_unique_triangle": unique_triangle_pearson(d_wtrw, rdm),
                "rdm_corr_trace_sq_unique_triangle": unique_triangle_pearson(
                    d_trace, rdm
                ),
                "pattern_max_abs_channel": pattern["max_abs_channel"],
                "pattern_max_abs_value": pattern["max_abs_value"],
                "pattern": pattern,
                "secondary_exact24_p_signed_ge": float(
                    secondary["p_signed_greater_equal"][k]
                ),
                "secondary_exact24_p_maxabs": float(
                    secondary["p_maxabs_familywise"][k]
                ),
            }
        )
    return {
        "pair_mode": pair_mode,
        "matrix_mode": matrix_mode,
        "n_channels": int(n_channels),
        "n_times": int(n_times),
        "n_pairs": len(pairs),
        "pair_sequence": [list(pair) for pair in pairs],
        "numerical_rank_whitening_rows": int(result.whitening.shape[0]),
        "n_components": int(result.eigenvalues.size),
        "n_reported": n_keep,
        "eigenvalues_head": [float(v) for v in result.eigenvalues[:n_keep]],
        "primary_random_phase_p_head": p_head,
        "components": components,
        "extras": {
            "condition_label_permutation": {
                "label": secondary["label"],
                "n_permutations": secondary["n_permutations"],
                "n_unique_rdms": secondary["n_unique_rdms"],
                "p_signed_greater_equal": secondary["p_signed_greater_equal"],
                "p_maxabs_familywise": secondary["p_maxabs_familywise"],
                "permutation_floor_note": secondary["permutation_floor_note"],
            }
        },
    }


def compact_window_row(payload: dict[str, Any], *, center_ms: float, n_samples: int) -> dict[str, Any]:
    comp0 = payload["components"][0]
    p1 = payload["primary_random_phase_p_head"][0]
    return {
        "center_ms": float(center_ms),
        "n_samples": int(n_samples),
        "lambda1": float(comp0["eigenvalue"]),
        "primary_p1": None if p1 is None else float(p1),
        "corr_wTRw": float(comp0["rdm_corr_wTRw_unique_triangle"]),
        "corr_trace_sq": float(comp0["rdm_corr_trace_sq_unique_triangle"]),
        "pattern_max_abs_channel": comp0["pattern_max_abs_channel"],
        "pattern_max_abs_value": float(comp0["pattern_max_abs_value"]),
        "secondary_exact24_p1_signed_ge": float(comp0["secondary_exact24_p_signed_ge"]),
        "secondary_exact24_p1_maxabs": float(comp0["secondary_exact24_p_maxabs"]),
        "rank": int(payload["numerical_rank_whitening_rows"]),
        "eigenvalues_head": payload["eigenvalues_head"],
        "primary_p_head": payload["primary_random_phase_p_head"],
    }


def run_meaning_scan(
    packed: dict[str, Any],
    freeze: dict[str, Any],
    *,
    n_bootstrapping_iterations: int,
    duration_ms: float = MEANING_DURATION_MS,
    step_ms: float = DEFAULT_SLIDING_STEP_MS,
    n_report: int = N_REPORT,
) -> dict[str, Any]:
    """Fig. 7: sliding T=150 ms meaning RDM under the frozen estimator.

    ``step_ms`` defaults to the existing documented 25 ms choice. The paper
    does not print a step; this function does not search over steps.
    """
    rdm = theoretical_rdm("meaning", within=0.0, between=1.0)
    times = packed["times_ms"]
    X_full = packed["data"]
    labels = list(packed["channel_labels"])
    centers = sliding_centers_ms(times, duration_ms=duration_ms, step_ms=step_ms)
    rows: list[dict[str, Any]] = []
    pattern_400: dict[str, Any] | None = None
    for index, center in enumerate(centers):
        win = window_slice(
            X_full, times, center_ms=float(center), duration_ms=duration_ms
        )
        rng = _pcg64(FIG7_WINDOW_SEED_BASE + int(index))
        payload = fit_frozen_window(
            X_window=win["data"],
            rdm=rdm,
            channel_labels=labels,
            n_bootstrapping_iterations=int(n_bootstrapping_iterations),
            rng=rng,
            n_report=n_report,
            pair_mode=freeze["pair_mode"],
            matrix_mode=freeze["matrix_mode"],
        )
        row = compact_window_row(
            payload, center_ms=float(center), n_samples=int(win["n_samples"])
        )
        row["rng_seed"] = int(FIG7_WINDOW_SEED_BASE + int(index))
        rows.append(row)
        if abs(float(center) - 400.0) < 1e-6:
            pattern_400 = {
                "center_ms": float(center),
                "pattern": payload["components"][0]["pattern"],
                "lambda1": row["lambda1"],
                "primary_p1": row["primary_p1"],
            }
    p1 = np.array(
        [np.nan if row["primary_p1"] is None else row["primary_p1"] for row in rows],
        dtype=np.float64,
    )
    segments = p_lt_segments(centers, p1, alpha=0.05)
    nearest_400 = int(np.argmin(np.abs(centers - 400.0)))
    near_band = (centers >= 350.0) & (centers <= 450.0)
    p_near = p1[near_band]
    finite_near = p_near[np.isfinite(p_near)]
    any_p_lt_near_400 = bool(np.any(finite_near < 0.05)) if finite_near.size else False
    segment_covers_400 = any(
        seg["center_ms_start"] <= 400.0 <= seg["center_ms_end"] for seg in segments
    )
    return {
        "item_id": "fig07-n170-meaning-pmap",
        "path_label": "historical_apply",
        "contrast": "meaning",
        "rdm_fill": "binary_0_1",
        "duration_ms": float(duration_ms),
        "step_ms": float(step_ms),
        "step_ms_paper": None,
        "step_note": (
            "Sliding step is not specified in Ossadtchi et al. 2024. "
            f"{step_ms:g} ms is the existing documented DEFAULT_SLIDING_STEP_MS, "
            "not a paper value, and was not tuned for significance."
        ),
        "centers_ms": [float(c) for c in centers],
        "n_windows": int(centers.size),
        "n_samples_per_window": [int(row["n_samples"]) for row in rows],
        "windows": rows,
        "p1_primary_random_phase": [
            None if row["primary_p1"] is None else float(row["primary_p1"]) for row in rows
        ],
        "lambda1": [float(row["lambda1"]) for row in rows],
        "corr_wTRw": [float(row["corr_wTRw"]) for row in rows],
        "pattern_max_abs_channel": [row["pattern_max_abs_channel"] for row in rows],
        "secondary_exact24_p1_signed_ge": [
            float(row["secondary_exact24_p1_signed_ge"]) for row in rows
        ],
        "p_lt_0.05_segments_comp1_primary": segments,
        "nearest_400ms_center_ms": float(centers[nearest_400]),
        "nearest_400ms_primary_p1": _json_float(p1[nearest_400]),
        "nearest_400ms_lambda1": float(rows[nearest_400]["lambda1"]),
        "nearest_400ms_corr_wTRw": float(rows[nearest_400]["corr_wTRw"]),
        "nearest_400ms_pattern_max_abs_channel": rows[nearest_400][
            "pattern_max_abs_channel"
        ],
        "nearest_400ms_secondary_exact24_p1": float(
            rows[nearest_400]["secondary_exact24_p1_signed_ge"]
        ),
        "any_primary_p1_lt_0.05_in_350_450ms": any_p_lt_near_400,
        "continuous_p1_lt_0.05_segment_covers_400ms": bool(segment_covers_400),
        "random_phase_recovers_p1_lt_0.05_near_400ms": bool(
            any_p_lt_near_400 or segment_covers_400
        ),
        "pattern_at_400ms": pattern_400,
        "estimator": frozen_estimator_record(
            freeze, used_B=int(n_bootstrapping_iterations)
        ),
        "seed_policy": {
            "bit_generator": "PCG64",
            "master_seed": int(FIG7_WINDOW_SEED_BASE - 2000),
            "fig7_window_i": "PCG64(MASTER_SEED + 2000 + i) for sliding-center index i",
            "disjoint_from_track_a_b": (
                "Track A uses MASTER+offset with offset<200; "
                "Track B uses MASTER+10000+i."
            ),
        },
        "secondary_floor_note": (
            "Meaning RDM is invariant under 8 of 24 condition relabelings, so "
            "signed P(λ* >= λ_obs) has floor 8/24 ≈ 0.333 when this partition "
            "is uniquely best. That floor is expected for the secondary test "
            "and is not the historical primary."
        ),
        "paper_claim": (
            "Fig. 7: first-component uncorrected p<0.05 around t=400 ms; T=150 ms."
        ),
        "matlab": None,
        "imports_redisca": False,
    }


def run_fig8_windows(
    packed: dict[str, Any],
    freeze: dict[str, Any],
    *,
    n_bootstrapping_iterations: int,
    n_report: int = N_REPORT,
) -> dict[str, Any]:
    """Fig. 8: three adjacent 375/400/425 ms windows, same frozen estimator."""
    rdm = theoretical_rdm("meaning", within=0.0, between=1.0)
    times = packed["times_ms"]
    X_full = packed["data"]
    labels = list(packed["channel_labels"])
    sliding = sliding_centers_ms(
        times, duration_ms=MEANING_DURATION_MS, step_ms=DEFAULT_SLIDING_STEP_MS
    )
    windows: list[dict[str, Any]] = []
    for center in FIG8_CENTERS_MS:
        matches = np.flatnonzero(np.isclose(sliding, float(center)))
        if matches.size != 1:
            raise RuntimeError(
                f"Fig. 8 center {center} ms is not a unique sliding-grid point "
                f"(matches={matches.tolist()}). Step was not retuned."
            )
        index = int(matches[0])
        win = window_slice(
            X_full,
            times,
            center_ms=float(center),
            duration_ms=MEANING_DURATION_MS,
        )
        rng = _pcg64(FIG7_WINDOW_SEED_BASE + index)
        payload = fit_frozen_window(
            X_window=win["data"],
            rdm=rdm,
            channel_labels=labels,
            n_bootstrapping_iterations=int(n_bootstrapping_iterations),
            rng=rng,
            n_report=n_report,
            pair_mode=freeze["pair_mode"],
            matrix_mode=freeze["matrix_mode"],
        )
        row = compact_window_row(
            payload, center_ms=float(center), n_samples=int(win["n_samples"])
        )
        row["rng_seed"] = int(FIG7_WINDOW_SEED_BASE + index)
        row["sliding_grid_index"] = index
        row["duration_ms"] = float(MEANING_DURATION_MS)
        row["t_start_ms"] = float(win["t_start_ms"])
        row["t_end_ms"] = float(win["t_end_ms"])
        row["pattern"] = payload["components"][0]["pattern"]
        windows.append(row)
    return {
        "item_id": "fig08-n170-meaning-patterns",
        "path_label": "historical_apply",
        "centers_ms": [float(c) for c in FIG8_CENTERS_MS],
        "adjacent_step_ms": 25.0,
        "duration_ms": float(MEANING_DURATION_MS),
        "windows": windows,
        "estimator": frozen_estimator_record(
            freeze, used_B=int(n_bootstrapping_iterations)
        ),
        "seed_policy": {
            "bit_generator": "PCG64",
            "fig8_window": (
                "Same PCG64(MASTER_SEED + 2000 + i) as the Fig. 7 sliding "
                "center with the same timestamp, so 375/400/425 ms p-values "
                "match the sliding map."
            ),
        },
        "matlab": None,
        "imports_redisca": False,
    }
