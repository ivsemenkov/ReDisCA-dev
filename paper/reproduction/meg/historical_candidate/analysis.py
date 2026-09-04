"""Frozen directed+cov+random-phase fit on the paper MEG epoch.

Full −500…+1000 ms (1501 samples), 204 planars, no AIRI bandpass.
Does not import ``redisca``.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np
from numpy.typing import NDArray

from common.metrics import peak_latency_and_amplitude
from common.serialize import array_fingerprint
from common.source_faithful import (
    create_cxxz,
    matlab_zscore,
    pair_indices,
    pair_stack_from_condition_averages,
    spoc_from_pair_stack,
    theoretical_rdm_vector,
)

from meg.historical_candidate.freeze import (
    FROZEN_INFERENCE,
    FROZEN_MATRIX_MODE,
    FROZEN_PAIR_MODE,
    N_REPORT,
    RDM_ORDER,
)
from meg.inference import empirical_rdm_from_traces, unique_pair_pearson
from meg.rdms import class_labels, theoretical_rdm


def empirical_rdm_wTRw(
    weight: NDArray[np.floating],
    pair_stack: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    n_conditions: int,
) -> NDArray[np.float64]:
    weight = np.asarray(weight, dtype=np.float64).ravel()
    dhat = np.zeros((n_conditions, n_conditions), dtype=np.float64)
    for matrix, (i, j) in zip(pair_stack, pairs, strict=True):
        value = float(weight @ matrix @ weight)
        dhat[i, j] = value
        dhat[j, i] = value
    return dhat


def _peak_ms(time_ms: NDArray[np.floating], series: NDArray[np.floating]) -> dict[str, float | None]:
    t = np.asarray(time_ms, dtype=np.float64)
    y = np.asarray(series, dtype=np.float64)
    post = t >= 0.0
    if not np.any(post):
        return {"peak_ms": None, "peak_amp": None}
    helper = peak_latency_and_amplitude(t[post] / 1000.0, y[post])
    return {
        "peak_ms": float(helper["peak_time_s"] * 1000.0),
        "peak_amp": float(helper["peak_amplitude"]),
    }


def exact_condition_label_permutation(
    pair_stack: NDArray[np.floating],
    rdm: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    observed: NDArray[np.floating],
    *,
    pair_mode: str,
    matrix_mode: str,
    n_report: int,
) -> dict[str, Any]:
    """SECONDARY diagnostic: enumerate all C! condition-label permutations.

    Reuses whitening / Cxxe (only z changes). Not the historical primary.
    """
    rdm = np.asarray(rdm, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    n_keep = min(int(n_report), int(observed.size))
    observed = observed[:n_keep]
    n_conditions = int(rdm.shape[0])
    base_z = theoretical_rdm_vector(rdm, pairs)
    base = spoc_from_pair_stack(
        pair_stack,
        base_z,
        n_bootstrapping_iterations=0,
        pair_mode=pair_mode,  # type: ignore[arg-type]
        matrix_mode=matrix_mode,  # type: ignore[arg-type]
        inference="none",
    )
    orders = list(permutations(range(n_conditions)))
    n_perm = len(orders)
    samples = np.empty((n_perm, n_keep), dtype=np.float64)
    max_abs = np.empty(n_perm, dtype=np.float64)
    unique: dict[tuple[float, ...], int] = {}
    for row, order in enumerate(orders):
        permuted = rdm[np.ix_(order, order)]
        z = matlab_zscore(theoretical_rdm_vector(permuted, pairs))
        cxxz = create_cxxz(base.cxxe, z)
        white = base.whitening @ cxxz @ base.whitening.T
        white = 0.5 * (white + white.T)
        evals = np.sort(np.linalg.eigvalsh(white))[::-1]
        samples[row] = evals[:n_keep]
        max_abs[row] = float(np.max(np.abs(evals)))
        key = tuple(np.round(permuted[np.triu_indices(n_conditions, 1)], 12).tolist())
        unique[key] = unique.get(key, 0) + 1
    p_maxabs = [
        float(np.mean(max_abs >= abs(float(observed[n])))) for n in range(n_keep)
    ]
    p_signed_ge = [
        float(np.mean(samples[:, n] >= float(observed[n]))) for n in range(n_keep)
    ]
    return {
        "label": "secondary_condition_label_permutation_not_historical_oracle",
        "kind": "exact_enumeration",
        "n_permutations": int(n_perm),
        "n_unique_rdms": len(unique),
        "p_maxabs_familywise": p_maxabs,
        "p_signed_greater_equal": p_signed_ge,
        "note": (
            "Paper §2.3-style condition-label permutation on the frozen "
            "directed+matlab_cov pair stack. PRIMARY remains SPoC random-phase."
        ),
    }


def fit_one_rdm(
    *,
    pair_stack: NDArray[np.floating],
    averages: NDArray[np.floating],
    time_ms: NDArray[np.floating],
    rdm_name: str,
    n_bootstrapping_iterations: int,
    rng: np.random.Generator,
    n_report: int = N_REPORT,
) -> dict[str, Any]:
    rdm = theoretical_rdm(rdm_name, fill="airi")
    n_conditions = int(averages.shape[0])
    pairs = pair_indices(n_conditions, FROZEN_PAIR_MODE)
    z_raw = theoretical_rdm_vector(rdm, pairs)
    if n_bootstrapping_iterations > 0:
        inference = FROZEN_INFERENCE
        fit_rng: np.random.Generator | None = rng
    else:
        inference = "none"
        fit_rng = None
    result = spoc_from_pair_stack(
        pair_stack,
        z_raw,
        n_bootstrapping_iterations=int(n_bootstrapping_iterations),
        rng=fit_rng,
        pair_mode=FROZEN_PAIR_MODE,
        matrix_mode=FROZEN_MATRIX_MODE,
        inference=inference,
    )
    n_rank = int(result.filters.shape[1])
    n_keep = min(int(n_report), n_rank)
    filters = np.asarray(result.filters, dtype=np.float64)
    patterns = np.asarray(result.patterns, dtype=np.float64)
    evals = np.asarray(result.eigenvalues, dtype=np.float64)
    if result.p_values is None:
        p_primary: list[float | None] = [None] * n_keep
        n_sig_all = None
        n_sig_first3 = None
    else:
        p_all = np.asarray(result.p_values, dtype=np.float64)
        p_primary = [float(v) for v in p_all[:n_keep]]
        n_sig_all = int(np.sum(p_all < 0.05))
        n_sig_first3 = int(np.sum(p_all[: min(3, p_all.size)] < 0.05))

    traces = np.einsum("ck,nct->nkt", filters[:, :n_keep], averages)
    class1_paper, class2_paper = class_labels(rdm_name, convention="paper")
    class1_airi, class2_airi = class_labels(rdm_name, convention="airi")
    contrast_paper = traces[list(class2_paper)].mean(axis=0) - traces[list(class1_paper)].mean(
        axis=0
    )
    peaks: list[dict[str, Any]] = []
    emp: list[dict[str, Any]] = []
    for k in range(n_keep):
        c1 = traces[list(class1_paper), k].mean(axis=0)
        c2 = traces[list(class2_paper), k].mean(axis=0)
        mean_all = traces[:, k, :].mean(axis=0)
        d_wtrw = empirical_rdm_wTRw(filters[:, k], pair_stack, pairs, n_conditions)
        d_trace = empirical_rdm_from_traces(traces[:, k, :], demean_time=True)
        abs_idx = int(np.argmax(np.abs(patterns[:, k])))
        peaks.append(
            {
                "component": k + 1,
                "contrast_peak_ms_paper_convention": _peak_ms(time_ms, contrast_paper[k])[
                    "peak_ms"
                ],
                "class1_peak_ms_paper_convention": _peak_ms(time_ms, c1)["peak_ms"],
                "class2_peak_ms_paper_convention": _peak_ms(time_ms, c2)["peak_ms"],
                "mean_of_six_conditions_peak_ms": _peak_ms(time_ms, mean_all)["peak_ms"],
            }
        )
        emp.append(
            {
                "component": k + 1,
                "pearson_wTRw_unique_triangle": unique_pair_pearson(d_wtrw, rdm),
                "pearson_trace_sq_demeaned_unique_triangle": unique_pair_pearson(
                    d_trace, rdm
                ),
                "pattern_max_abs_planar_index": abs_idx,
                "pattern_max_abs_value": float(patterns[abs_idx, k]),
            }
        )

    secondary = exact_condition_label_permutation(
        pair_stack,
        rdm,
        pairs,
        evals,
        pair_mode=FROZEN_PAIR_MODE,
        matrix_mode=FROZEN_MATRIX_MODE,
        n_report=n_keep,
    )
    item_id = {
        "face": "fig13-meg-face",
        "tool": "fig14-meg-tool",
        "meaning": "fig15-meg-meaning",
        "facevstool": "fig17-meg-nonbinary-components",
    }[rdm_name]
    return {
        "path_label": "historical_candidate_paper_epoch",
        "item_id": item_id,
        "rdm_name": rdm_name,
        "rdm_fill": "airi_numeric",
        "window_ms": [float(time_ms[0]), float(time_ms[-1])],
        "n_samples": int(time_ms.size),
        "n_planars": int(averages.shape[1]),
        "bandpass": None,
        "pairs": "airi_directed_i_neq_j",
        "n_pairs": len(pairs),
        "pair_sequence_head": [list(p) for p in pairs[:6]],
        "pair_matrix": FROZEN_MATRIX_MODE,
        "estimator": "common.source_faithful.spoc_from_pair_stack (does not import redisca)",
        "rank": n_rank,
        "n_reported": n_keep,
        "eigenvalues_head": [float(v) for v in evals[:n_keep]],
        "primary_random_phase_p_head": p_primary,
        "n_components_p_lt_0.05_all_rank": n_sig_all,
        "n_first3_p_lt_0.05": n_sig_first3,
        "eigenvalues_fingerprint": array_fingerprint(evals),
        "patterns_fingerprint": array_fingerprint(patterns[:, :n_keep]),
        "filters_fingerprint": array_fingerprint(filters[:, :n_keep]),
        "empirical_rdm": emp,
        "peaks": peaks,
        "class_labels_paper": {"class1": list(class1_paper), "class2": list(class2_paper)},
        "class_labels_airi": {"class1": list(class1_airi), "class2": list(class2_airi)},
        "extras": {"condition_label_permutation": secondary},
        "paper_qualitative": {
            "face_peak_ms": 160.0 if rdm_name in {"face", "facevstool"} else None,
            "n_significant_components_published": 3,
        },
        "imports_redisca": False,
        "matlab": None,
    }


def run_paper_epoch(
    averages: NDArray[np.floating],
    time_ms: NDArray[np.floating],
    *,
    n_bootstrapping_iterations: int,
    rng: np.random.Generator,
    rdm_names: tuple[str, ...] = RDM_ORDER,
    n_report: int = N_REPORT,
) -> dict[str, Any]:
    """Fit all four RDMs on one shared paper-epoch pair stack."""
    averages = np.asarray(averages, dtype=np.float64)
    time_ms = np.asarray(time_ms, dtype=np.float64)
    if averages.shape[1] != 204:
        raise ValueError(f"expected 204 planars, got {averages.shape[1]}")
    if int(time_ms.size) != 1501:
        raise ValueError(f"expected 1501 samples for the paper epoch, got {time_ms.size}")
    pairs = pair_indices(averages.shape[0], FROZEN_PAIR_MODE)
    print(
        f"[historical_candidate] building directed matlab_cov pair stack "
        f"{averages.shape} n_pairs={len(pairs)}",
        flush=True,
    )
    stack = pair_stack_from_condition_averages(
        averages, pairs, matrix_mode=FROZEN_MATRIX_MODE
    )
    rdms: dict[str, Any] = {}
    for name in rdm_names:
        print(
            f"[historical_candidate] paper-epoch {name} "
            f"B={n_bootstrapping_iterations} (no AIRI bandpass, full epoch)",
            flush=True,
        )
        payload = fit_one_rdm(
            pair_stack=stack,
            averages=averages,
            time_ms=time_ms,
            rdm_name=name,
            n_bootstrapping_iterations=int(n_bootstrapping_iterations),
            rng=rng,
            n_report=n_report,
        )
        rdms[name] = payload
        phead = payload["primary_random_phase_p_head"][:4]
        print(
            f"    rank={payload['rank']} λ={payload['eigenvalues_head'][:4]} p={phead} "
            f"n_p<0.05={payload['n_components_p_lt_0.05_all_rank']}",
            flush=True,
        )
    return {
        "path_label": "historical_candidate_paper_epoch",
        "window_ms": [float(time_ms[0]), float(time_ms[-1])],
        "n_samples": int(time_ms.size),
        "n_planars": 204,
        "bandpass": None,
        "pair_mode": FROZEN_PAIR_MODE,
        "matrix_mode": FROZEN_MATRIX_MODE,
        "inference": FROZEN_INFERENCE if n_bootstrapping_iterations > 0 else "none",
        "B": int(n_bootstrapping_iterations),
        "rdms": rdms,
        "imports_redisca": False,
        "matlab": None,
    }
