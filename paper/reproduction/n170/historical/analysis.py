"""Source-faithful fits and diagnostics for one N170 historical variant.

Does not import ``redisca``. Primary inference is stock SPoC random-phase
``max|lambda|`` via ``fit_condition_averages``.
"""

from __future__ import annotations

import hashlib
from itertools import permutations
from typing import Any

import numpy as np
from numpy.typing import NDArray

from common.metrics import peak_latency_and_amplitude, pearson
from common.source_faithful import (
    create_cxxz,
    fit_condition_averages,
    matlab_zscore,
    pair_indices,
    pair_stack_from_condition_averages,
    random_phase_surrogate,
    spoc_from_pair_stack,
    theoretical_rdm_vector,
)
from common.source_faithful import PairMatrixMode, PairMode, SPoCResult

from prepare import OCCIPITAL_LABELS

from .variants import (
    LIBRARY_UNIQUE_GRAM,
    N_REPORT_COMPONENTS,
    PRINTED_CAR,
    PRINTED_FACE,
    VariantSpec,
)


def _json_float(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _sha256_array(array: NDArray[np.floating]) -> str:
    payload = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(payload.tobytes()).hexdigest()


def canonicalize_columns(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    """Flip each column so the max-abs entry is non-negative (hash stability)."""
    out = np.array(matrix, dtype=np.float64, copy=True)
    if out.ndim != 2:
        raise ValueError("expected 2-D column-oriented matrix")
    for k in range(out.shape[1]):
        index = int(np.argmax(np.abs(out[:, k])))
        if out[index, k] < 0.0:
            out[:, k] *= -1.0
    return out


def align_occipital_columns(
    patterns: NDArray[np.floating],
    filters: NDArray[np.floating],
    traces: NDArray[np.floating],
    *,
    channel_labels: list[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Flip each component so the occipital ROI mean of the pattern is >= 0."""
    patterns = np.array(patterns, dtype=np.float64, copy=True)
    filters = np.array(filters, dtype=np.float64, copy=True)
    traces = np.array(traces, dtype=np.float64, copy=True)
    occ = [i for i, lab in enumerate(channel_labels) if lab in OCCIPITAL_LABELS]
    if not occ:
        occ = [int(np.argmax(np.abs(patterns[:, 0])))]
    for k in range(patterns.shape[1]):
        roi = float(np.mean(patterns[occ, k]))
        if roi < 0.0:
            patterns[:, k] *= -1.0
            filters[:, k] *= -1.0
            traces[:, k, :] *= -1.0
    return patterns, filters, traces


def empirical_rdm_wTRw(
    weight: NDArray[np.floating],
    pair_stack: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    n_conditions: int,
) -> NDArray[np.float64]:
    """Paper-style ``dhat_ij = w^T R_ij w`` using the same ``R_ij`` as the fit."""
    weight = np.asarray(weight, dtype=np.float64).ravel()
    dhat = np.zeros((n_conditions, n_conditions), dtype=np.float64)
    for matrix, (i, j) in zip(pair_stack, pairs, strict=True):
        value = float(weight @ matrix @ weight)
        dhat[i, j] = value
        dhat[j, i] = value
    return dhat


def empirical_rdm_trace_sq(
    traces: NDArray[np.floating],
    *,
    demean_time: bool,
) -> NDArray[np.float64]:
    """``||u_i - u_j||^2`` on window traces ``u = w^T X``.

    ``demean_time=True`` temporally centers the pairwise difference (MATLAB
    ``cov`` centering) before the unscaled squared Euclidean.
    """
    u = np.asarray(traces, dtype=np.float64)
    n_conditions = u.shape[0]
    dhat = np.zeros((n_conditions, n_conditions), dtype=np.float64)
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            delta = u[i] - u[j]
            if demean_time:
                delta = delta - delta.mean()
            value = float(delta @ delta)
            dhat[i, j] = value
            dhat[j, i] = value
    return dhat


def unique_triangle_pearson(
    empirical: NDArray[np.floating],
    theoretical: NDArray[np.floating],
) -> float:
    n = int(empirical.shape[0])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    a = np.array([empirical[i, j] for i, j in pairs], dtype=np.float64)
    b = np.array([theoretical[i, j] for i, j in pairs], dtype=np.float64)
    return pearson(a, b)


def _peak_in_window(
    times_ms: NDArray[np.floating],
    series: NDArray[np.floating],
    *,
    lo_ms: float,
    hi_ms: float,
) -> dict[str, Any]:
    times_ms = np.asarray(times_ms, dtype=np.float64)
    series = np.asarray(series, dtype=np.float64)
    mask = (times_ms >= lo_ms) & (times_ms <= hi_ms)
    if not np.any(mask):
        mask = np.ones(series.shape[-1], dtype=bool)
    info = peak_latency_and_amplitude(times_ms[mask] / 1000.0, series[mask], signed=True)
    return {
        "peak_index_in_search": int(info["peak_index"]),
        "peak_time_ms": float(info["peak_time_s"] * 1000.0),
        "peak_amplitude": float(info["peak_amplitude"]),
        "search_lo_ms": float(lo_ms),
        "search_hi_ms": float(hi_ms),
    }


def _pattern_fingerprint(
    pattern: NDArray[np.floating],
    labels: list[str],
) -> dict[str, Any]:
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
        "O2": _val("O2"),
        "Oz": _val("Oz"),
        "PO7": _val("PO7"),
        "PO8": _val("PO8"),
        "P7": _val("P7"),
        "P8": _val("P8"),
    }


def matched_component_random_phase(
    result: SPoCResult,
    *,
    n_iterations: int,
    rng: np.random.Generator,
    n_report: int,
) -> dict[str, Any]:
    """Labeled extra: compare the k-th sorted surrogate eigenvalue to λ_obs_k.

    This is not the stock SPoC formula (which uses ``max|lambda|``). Stored
    only as a diagnostic beside the primary p-values.
    """
    if n_iterations <= 0:
        return {
            "label": "matched_component_random_phase_diagnostic_not_primary",
            "n_iterations": 0,
            "p_abs_matched": None,
            "p_signed_ge_matched": None,
        }
    observed = np.asarray(result.eigenvalues, dtype=np.float64)
    n_keep = min(int(n_report), int(observed.size))
    observed = observed[:n_keep]
    matching = np.empty((n_iterations, n_keep), dtype=np.float64)
    z_amps = None
    z = np.asarray(result.z, dtype=np.float64)
    for k in range(n_iterations):
        z_shuffled, z_amps = random_phase_surrogate(z, rng, z_amps=z_amps)
        cxxz_s = create_cxxz(result.cxxe, z_shuffled)
        white = result.whitening @ cxxz_s @ result.whitening.T
        white = 0.5 * (white + white.T)
        evals = np.sort(np.linalg.eigvalsh(white))[::-1]
        matching[k] = evals[:n_keep]
    p_abs = [
        float(np.mean(np.abs(matching[:, n]) >= abs(float(observed[n]))))
        for n in range(n_keep)
    ]
    p_ge = [
        float(np.mean(matching[:, n] >= float(observed[n])))
        for n in range(n_keep)
    ]
    return {
        "label": "matched_component_random_phase_diagnostic_not_primary",
        "n_iterations": int(n_iterations),
        "p_abs_matched": p_abs,
        "p_signed_ge_matched": p_ge,
        "note": (
            "Each surrogate GEP is sorted signed-descending like the observed "
            "spectrum. p_abs_matched uses |λ_surr_k| >= |λ_obs_k|. This is "
            "not stock SPoC (stock uses max|λ| over components)."
        ),
    }


def exact_condition_label_permutation(
    pair_stack: NDArray[np.floating],
    rdm: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    observed: NDArray[np.floating],
    *,
    pair_mode: PairMode,
    matrix_mode: PairMatrixMode,
    n_report: int,
) -> dict[str, Any]:
    """SECONDARY diagnostic: enumerate all C! condition-label permutations.

    Not the historical reproduction oracle. Stock SPoC random-phase is primary.
    """
    rdm = np.asarray(rdm, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    n_keep = min(int(n_report), int(observed.size))
    observed = observed[:n_keep]
    n_conditions = int(rdm.shape[0])
    orders = list(permutations(range(n_conditions)))
    n_perm = len(orders)
    samples = np.empty((n_perm, n_keep), dtype=np.float64)
    max_abs = np.empty(n_perm, dtype=np.float64)
    unique: dict[tuple[float, ...], dict[str, Any]] = {}
    for row, order in enumerate(orders):
        permuted = rdm[np.ix_(order, order)]
        z_raw = theoretical_rdm_vector(permuted, pairs)
        fitted = spoc_from_pair_stack(
            pair_stack,
            z_raw,
            n_bootstrapping_iterations=0,
            pair_mode=pair_mode,
            matrix_mode=matrix_mode,
            inference="none",
        )
        evals = np.asarray(fitted.eigenvalues[:n_keep], dtype=np.float64)
        samples[row] = evals
        max_abs[row] = float(np.max(np.abs(fitted.eigenvalues)))
        key = tuple(np.round(permuted[np.triu_indices(n_conditions, 1)], 12).tolist())
        if key not in unique:
            unique[key] = {
                "multiplicity": 1,
                "example_order": list(order),
                "lambda0": float(fitted.eigenvalues[0]),
            }
        else:
            unique[key]["multiplicity"] += 1
    p_maxabs = [
        float(np.mean(max_abs >= abs(float(observed[n])))) for n in range(n_keep)
    ]
    p_matched_abs = [
        float(np.mean(np.abs(samples[:, n]) >= abs(float(observed[n]))))
        for n in range(n_keep)
    ]
    p_signed_ge = [
        float(np.mean(samples[:, n] >= float(observed[n]))) for n in range(n_keep)
    ]
    return {
        "label": "secondary_condition_label_permutation_not_historical_oracle",
        "kind": "exact_enumeration",
        "n_permutations": int(n_perm),
        "n_unique_rdms": len(unique),
        "unique": list(unique.values()),
        "p_maxabs_familywise": p_maxabs,
        "p_matched_abs": p_matched_abs,
        "p_signed_greater_equal": p_signed_ge,
        "permutation_floor_note": (
            "Face/car one-vs-three RDMs have 3! = 6 equivalent relabelings of "
            "the three non-target conditions, so signed P(λ* >= λ_obs) cannot "
            "fall below 6/24 = 0.25 when the observed structure is uniquely "
            "best. That floor is a property of this secondary test, not of "
            "stock SPoC random-phase."
        ),
        "samples_lambda0": [float(v) for v in samples[:, 0]],
    }


def printed_deltas(spec: VariantSpec, payload: dict[str, Any]) -> dict[str, Any]:
    evals = payload["eigenvalues_head"]
    p_primary = payload["primary_random_phase_p_head"]
    corr = payload["components"][0]["rdm_corr_wTRw_unique_triangle"]
    corr_trace = payload["components"][0]["rdm_corr_trace_sq_unique_triangle"]
    if spec.contrast == "face":
        burst = payload["components"][0]["peaks_80_250ms_full_epoch"].get("Faces", {})
        return {
            "printed": PRINTED_FACE,
            "delta_lambda1": _json_float(evals[0] - PRINTED_FACE["lambda1"]),
            "delta_p1": None
            if p_primary[0] is None
            else _json_float(p_primary[0] - PRINTED_FACE["p1"]),
            "delta_corr_wTRw_vs_0.82": _json_float(corr - PRINTED_FACE["corr"]),
            "delta_corr_trace_sq_vs_0.82": _json_float(
                corr_trace - PRINTED_FACE["corr"]
            ),
            "delta_faces_peak_ms_vs_170": _json_float(
                burst.get("peak_time_ms", float("nan")) - PRINTED_FACE["burst_ms"]
            ),
        }
    p2 = p_primary[1] if len(p_primary) > 1 else None
    return {
        "printed": PRINTED_CAR,
        "delta_lambda1": _json_float(evals[0] - PRINTED_CAR["lambda1"]),
        "delta_lambda2": _json_float(evals[1] - PRINTED_CAR["lambda2"]),
        "delta_p1": None if p_primary[0] is None else _json_float(p_primary[0] - PRINTED_CAR["p1"]),
        "delta_p2": None if p2 is None else _json_float(p2 - PRINTED_CAR["p2"]),
        "delta_corr_wTRw_vs_0.99_threshold": _json_float(corr - PRINTED_CAR["corr_gt"]),
        "corr_wTRw_exceeds_0.99": bool(corr > PRINTED_CAR["corr_gt"]),
        "corr_trace_sq_exceeds_0.99": bool(corr_trace > PRINTED_CAR["corr_gt"]),
    }


def library_sanity_block(spec: VariantSpec, payload: dict[str, Any]) -> dict[str, Any] | None:
    if spec.pair_mode != "unique_unordered" or spec.matrix_mode != "unscaled_gram":
        return None
    evals = payload["eigenvalues_head"]
    corr = payload["components"][0]["rdm_corr_trace_sq_unique_triangle"]
    if spec.contrast == "face" and spec.window_center_ms == 200.0:
        ref = LIBRARY_UNIQUE_GRAM["face"]
        return {
            "label": "sanity_unique_unscaled_gram_vs_existing_library_numbers",
            "imports_redisca": False,
            "reference": ref,
            "delta_lambda1": _json_float(evals[0] - ref["lambda1"]),
            "delta_corr_trace_sq": _json_float(corr - ref["corr_window"]),
            "note": (
                "Library path is unique pairs + unscaled Gram "
                "(ReDisCA demean_time=False). Historical fit uses "
                "source_faithful, not redisca. Deltas should be ~0 if the "
                "GEPs agree."
            ),
        }
    if spec.contrast == "car" and spec.window_center_ms == 170.0:
        ref = LIBRARY_UNIQUE_GRAM["car"]
        return {
            "label": "sanity_unique_unscaled_gram_vs_existing_library_numbers",
            "imports_redisca": False,
            "reference": ref,
            "delta_lambda1": _json_float(evals[0] - ref["lambda1"]),
            "delta_lambda2": _json_float(evals[1] - ref["lambda2"]),
            "delta_corr_trace_sq": _json_float(corr - ref["corr_window"]),
            "note": (
                "Library path is unique pairs + unscaled Gram "
                "(ReDisCA demean_time=False). Historical fit uses "
                "source_faithful, not redisca."
            ),
        }
    return None


def fit_variant(
    *,
    X_window: NDArray[np.floating],
    X_full: NDArray[np.floating],
    times_full_ms: NDArray[np.floating],
    channel_labels: list[str],
    condition_labels: list[str],
    rdm: NDArray[np.floating],
    spec: VariantSpec,
    rng: np.random.Generator,
    n_bootstrapping_iterations: int,
    window_meta: dict[str, Any],
    n_report: int = N_REPORT_COMPONENTS,
    matched_rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """PRIMARY fit: stock SPoC random-phase through source_faithful."""
    X_window = np.asarray(X_window, dtype=np.float64)
    X_full = np.asarray(X_full, dtype=np.float64)
    rdm = np.asarray(rdm, dtype=np.float64)
    n_conditions, n_channels, _ = X_window.shape
    pairs = pair_indices(n_conditions, spec.pair_mode)
    pair_stack = pair_stack_from_condition_averages(
        X_window, pairs, matrix_mode=spec.matrix_mode
    )
    z_raw = theoretical_rdm_vector(rdm, pairs)
    z_scored = matlab_zscore(z_raw)

    inference = "spoc_random_phase" if n_bootstrapping_iterations > 0 else "none"
    result = fit_condition_averages(
        X_window,
        rdm,
        pair_mode=spec.pair_mode,
        matrix_mode=spec.matrix_mode,
        n_bootstrapping_iterations=int(n_bootstrapping_iterations),
        rng=rng if n_bootstrapping_iterations > 0 else None,
        inference=inference,
    )
    n_comp = int(result.eigenvalues.size)
    n_keep = min(int(n_report), n_comp)
    traces_window = np.einsum("ck,nct->nkt", result.filters[:, :n_keep], X_window)
    traces_full = np.einsum("ck,nct->nkt", result.filters[:, :n_keep], X_full)
    patterns, filters, traces_window = align_occipital_columns(
        result.patterns[:, :n_keep],
        result.filters[:, :n_keep],
        traces_window,
        channel_labels=channel_labels,
    )
    signs = np.sign(np.sum(patterns * result.patterns[:, :n_keep], axis=0))
    signs[signs == 0.0] = 1.0
    traces_full = traces_full * signs[np.newaxis, :, np.newaxis]

    hash_filters = canonicalize_columns(result.filters[:, :n_keep])
    hash_patterns = canonicalize_columns(result.patterns[:, :n_keep])
    demean_for_cov = spec.matrix_mode == "matlab_cov"

    components: list[dict[str, Any]] = []
    for k in range(n_keep):
        d_wtrw = empirical_rdm_wTRw(
            filters[:, k], pair_stack, pairs, n_conditions
        )
        d_trace = empirical_rdm_trace_sq(
            traces_window[:, k, :], demean_time=False
        )
        d_trace_demeaned = empirical_rdm_trace_sq(
            traces_window[:, k, :], demean_time=True
        )
        peaks = {
            condition_labels[c]: _peak_in_window(
                times_full_ms, traces_full[c, k, :], lo_ms=80.0, hi_ms=250.0
            )
            for c in range(len(condition_labels))
        }
        p_primary = None
        if result.p_values is not None:
            p_primary = float(result.p_values[k])
        components.append(
            {
                "index": k,
                "eigenvalue": float(result.eigenvalues[k]),
                "primary_random_phase_p": p_primary,
                "empirical_rdm_wTRw": d_wtrw.tolist(),
                "empirical_rdm_trace_sq": d_trace.tolist(),
                "empirical_rdm_trace_sq_temporally_demeaned": d_trace_demeaned.tolist(),
                "rdm_corr_wTRw_unique_triangle": unique_triangle_pearson(d_wtrw, rdm),
                "rdm_corr_trace_sq_unique_triangle": unique_triangle_pearson(
                    d_trace, rdm
                ),
                "rdm_corr_trace_sq_demeaned_unique_triangle": unique_triangle_pearson(
                    d_trace_demeaned, rdm
                ),
                "pattern": _pattern_fingerprint(patterns[:, k], channel_labels),
                "filter_max_abs_channel": channel_labels[
                    int(np.argmax(np.abs(filters[:, k])))
                ],
                "peaks_80_250ms_full_epoch": peaks,
            }
        )

    p_head: list[float | None]
    if result.p_values is None:
        p_head = [None] * n_keep
    else:
        p_head = [float(v) for v in result.p_values[:n_keep]]

    matched = matched_component_random_phase(
        result,
        n_iterations=int(n_bootstrapping_iterations),
        rng=matched_rng
        if matched_rng is not None
        else np.random.Generator(np.random.PCG64(spec.rng_seed + 500)),
        n_report=n_keep,
    )
    secondary = exact_condition_label_permutation(
        pair_stack,
        rdm,
        pairs,
        result.eigenvalues,
        pair_mode=spec.pair_mode,
        matrix_mode=spec.matrix_mode,
        n_report=n_keep,
    )

    payload: dict[str, Any] = {
        "variant_id": spec.variant_id,
        "contrast": spec.contrast,
        "pair_mode": spec.pair_mode,
        "matrix_mode": spec.matrix_mode,
        "window_center_ms": spec.window_center_ms,
        "window_duration_ms": spec.window_duration_ms,
        "n_samples": int(window_meta["n_samples"]),
        "n_channels": int(n_channels),
        "channel_labels": list(channel_labels),
        "condition_labels": list(condition_labels),
        "condition_order": ["faces", "cars", "scrambled_faces", "scrambled_cars"],
        "rdm_fill": {"within": 0.0, "between": 1.0, "diagonal": 0.0, "kind": "binary_0_1"},
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
        "inference_primary": {
            "name": "spoc_random_phase",
            "formula": "p = count(max|lambda_surr| >= |lambda_obs|) / B",
            "B": int(n_bootstrapping_iterations),
            "p_equals_zero_allowed": True,
            "executable": "common.source_faithful.fit_condition_averages",
            "note": (
                "Stock SPoC random-phase of z. This is PRIMARY historical "
                "inference. It is not paper §2.3 condition-label permutation."
            ),
        },
        "seed_policy": {
            "master_seed": int(spec.rng_seed - spec.seed_offset),
            "variant_offset": spec.seed_offset,
            "rng_seed": spec.rng_seed,
            "bit_generator": "PCG64",
            "matched_component_extra_offset": 500,
        },
        "numerical_rank_whitening_rows": int(result.whitening.shape[0]),
        "n_components": n_comp,
        "n_reported": n_keep,
        "eigenvalues_head": [float(v) for v in result.eigenvalues[:n_keep]],
        "primary_random_phase_p_head": p_head,
        "z_after_matlab_zscore": [float(v) for v in z_scored],
        "z_raw": [float(v) for v in z_raw],
        "pair_sequence": [list(pair) for pair in pairs],
        "filters_fingerprint": {
            "sha256_sign_canonical_columns": _sha256_array(hash_filters),
            "shape": list(hash_filters.shape),
        },
        "patterns_fingerprint": {
            "sha256_sign_canonical_columns": _sha256_array(hash_patterns),
            "shape": list(hash_patterns.shape),
        },
        "components": components,
        "extras": {
            "matched_component_random_phase": matched,
            "condition_label_permutation": secondary,
        },
        "matlab": None,
        "reconstruction_note": (
            "Source-faithful Python reconstruction of AIRI/SPoC semantics. "
            "MATLAB is unavailable; this is not MATLAB parity."
        ),
    }
    payload["comparison_to_printed"] = printed_deltas(spec, payload)
    sanity = library_sanity_block(spec, payload)
    if sanity is not None:
        payload["extras"]["library_unique_gram_sanity"] = sanity
    return payload


def compact_row(payload: dict[str, Any]) -> dict[str, Any]:
    comps = payload["components"]
    corr0 = comps[0]["rdm_corr_wTRw_unique_triangle"] if comps else None
    corr0_trace = comps[0]["rdm_corr_trace_sq_unique_triangle"] if comps else None
    row = {
        "variant_id": payload["variant_id"],
        "contrast": payload["contrast"],
        "pair_mode": payload["pair_mode"],
        "matrix_mode": payload["matrix_mode"],
        "window_center_ms": payload["window_center_ms"],
        "window_duration_ms": payload["window_duration_ms"],
        "n_samples": payload["n_samples"],
        "n_channels": payload["n_channels"],
        "numerical_rank": payload["numerical_rank_whitening_rows"],
        "eigenvalues_head": payload["eigenvalues_head"],
        "primary_p_head": payload["primary_random_phase_p_head"],
        "corr_wTRw_comp0": corr0,
        "corr_trace_sq_comp0": corr0_trace,
        "comparison_to_printed": payload["comparison_to_printed"],
    }
    if payload["contrast"] == "face" and comps:
        row["faces_peak_ms"] = comps[0]["peaks_80_250ms_full_epoch"]["Faces"][
            "peak_time_ms"
        ]
    if payload["contrast"] == "car" and comps:
        row["cars_peak_ms"] = comps[0]["peaks_80_250ms_full_epoch"]["Cars"][
            "peak_time_ms"
        ]
        if len(comps) > 1:
            row["corr_wTRw_comp1"] = comps[1]["rdm_corr_wTRw_unique_triangle"]
    return row
