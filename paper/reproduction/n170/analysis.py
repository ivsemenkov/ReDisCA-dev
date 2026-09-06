"""N170 analyses using the frozen AIRI-SPoC ReDisCA factory."""

from __future__ import annotations

from typing import Any

import numpy as np
from redisca import random_phase_test

from paper.reproduction.common.constants import RANDOM_PHASE_B
from paper.reproduction.common.hashing import sha256_array
from paper.reproduction.common.inference_secondary import (
    condition_label_permutation,
    empirical_rdm_from_traces,
)
from paper.reproduction.common.method import AIRI_SPOC_KWARGS, fit_redisca
from paper.reproduction.common.metrics import peak_latency_and_amplitude, rdm_pearson
from paper.reproduction.n170.prepare import (
    FIG8_CENTERS_MS,
    MEANING_DURATION_MS,
    OCCIPITAL_LABELS,
    sliding_centers_ms,
    window_slice,
)
from paper.reproduction.n170.rdms import theoretical_rdm

PAPER_ANCHORS = {
    "face": {"t_s": 0.2, "lambda": 0.87209, "p": 0.0, "rdm_corr": 0.81556},
    "car": {
        "comp1": {"lambda": 0.91639, "p": 0.0, "rdm_corr": 0.99074},
        "comp2": {"lambda": 0.77036, "p": 0.009, "rdm_corr": 0.93002},
    },
}


def _pattern_report(pattern: np.ndarray, labels: list[str]) -> dict[str, Any]:
    idx = int(np.argmax(np.abs(pattern)))
    occ = [labels.index(lab) for lab in OCCIPITAL_LABELS if lab in labels]
    energy = float(np.sum(pattern ** 2))
    occ_energy = float(np.sum(pattern[occ] ** 2)) / energy if energy else float("nan")
    return {
        "maxabs_channel": labels[idx],
        "maxabs_value": float(pattern[idx]),
        "occipital_energy_fraction": occ_energy,
    }


def analyze_window(
    data: np.ndarray,
    times_ms: np.ndarray,
    rdm: np.ndarray,
    *,
    center_ms: float,
    duration_ms: float,
    channel_labels: list[str],
    seed: int,
    n_surrogates: int = RANDOM_PHASE_B,
    secondary: bool = True,
) -> dict[str, Any]:
    sliced = window_slice(data, times_ms, center_ms=center_ms, duration_ms=duration_ms)
    X = sliced["data"]
    model = fit_redisca(X, rdm)
    traces = model.transform(X)
    primary = random_phase_test(model, n_surrogates=n_surrogates, random_state=seed)
    components = []
    for k in range(min(4, model.rank_)):
        emp = empirical_rdm_from_traces(traces[:, k, :])
        peak = peak_latency_and_amplitude(sliced["times_ms"] / 1000.0, traces[0, k])
        components.append(
            {
                "index": k,
                "lambda": float(model.eigenvalues_[k]),
                "p_random_phase": float(primary.p_values[k]),
                "rdm_corr": rdm_pearson(emp, rdm),
                "pattern": _pattern_report(model.patterns_[k], channel_labels),
                "peak": peak,
                "empirical_rdm": emp.tolist(),
            }
        )
    payload: dict[str, Any] = {
        "center_ms": center_ms,
        "duration_ms": duration_ms,
        "n_samples": sliced["n_samples"],
        "t_start_ms": sliced["t_start_ms"],
        "t_end_ms": sliced["t_end_ms"],
        "input_hash": sha256_array(X),
        "rdm_hash": sha256_array(rdm),
        "rank": int(model.rank_),
        "eigenvalues": model.eigenvalues_.tolist(),
        "p_random_phase": primary.p_values.tolist(),
        "null_statistic_hash": sha256_array(primary.null_statistic),
        "B": int(primary.n_surrogates),
        "seed": seed,
        "redisca_kwargs": dict(AIRI_SPOC_KWARGS),
        "components": components,
        "filters_hash": sha256_array(model.filters_),
        "patterns_hash": sha256_array(model.patterns_),
    }
    if secondary:
        payload["secondary_condition_labels"] = condition_label_permutation(
            model, rdm, kind="condition_labels"
        )
        payload["secondary_upper_triangle_shuffle"] = condition_label_permutation(
            model, rdm, kind="upper_triangle_shuffle"
        )
    return payload


def analyze_sliding_meaning(
    data: np.ndarray,
    times_ms: np.ndarray,
    *,
    channel_labels: list[str],
    seed: int,
    step_ms: float,
    n_surrogates: int = RANDOM_PHASE_B,
) -> dict[str, Any]:
    rdm = theoretical_rdm("meaning")
    centers = sliding_centers_ms(times_ms, duration_ms=MEANING_DURATION_MS, step_ms=step_ms)
    rows = []
    for center in centers:
        result = analyze_window(
            data,
            times_ms,
            rdm,
            center_ms=float(center),
            duration_ms=MEANING_DURATION_MS,
            channel_labels=channel_labels,
            seed=seed,
            n_surrogates=n_surrogates,
            secondary=False,
        )
        rows.append(
            {
                "center_ms": float(center),
                "p1": result["p_random_phase"][0],
                "lambda1": result["eigenvalues"][0],
                "p_all": result["p_random_phase"][:8],
            }
        )
    p1 = np.array([row["p1"] for row in rows], dtype=np.float64)
    sig = np.flatnonzero(p1 < 0.05)
    fig8 = [
        analyze_window(
            data,
            times_ms,
            rdm,
            center_ms=center,
            duration_ms=MEANING_DURATION_MS,
            channel_labels=channel_labels,
            seed=seed,
            n_surrogates=n_surrogates,
            secondary=True,
        )
        for center in FIG8_CENTERS_MS
    ]
    return {
        "step_ms": step_ms,
        "centers_ms": centers.tolist(),
        "p1_profile": p1.tolist(),
        "n_significant_p1": int(sig.size),
        "significant_centers_ms": [float(centers[i]) for i in sig],
        "windows": rows,
        "fig8": fig8,
        "near_400_ms": {
            str(center): next((row for row in rows if abs(row["center_ms"] - center) < 1e-6), None)
            for center in FIG8_CENTERS_MS
        },
    }


def analyze_candidate(
    bundle: dict[str, Any],
    *,
    candidate_id: str,
    seed: int,
    n_surrogates: int = RANDOM_PHASE_B,
    meaning_steps_ms: tuple[float, ...] = (25.0, 1000.0 / 256.0),
) -> dict[str, Any]:
    data = bundle["data"]
    times = bundle["times_ms"]
    labels = bundle["channel_labels"]
    face = analyze_window(
        data,
        times,
        theoretical_rdm("face"),
        center_ms=200.0,
        duration_ms=100.0,
        channel_labels=labels,
        seed=seed,
        n_surrogates=n_surrogates,
    )
    cars = {
        str(center): analyze_window(
            data,
            times,
            theoretical_rdm("car"),
            center_ms=center,
            duration_ms=100.0,
            channel_labels=labels,
            seed=seed,
            n_surrogates=n_surrogates,
        )
        for center in (170.0, 200.0)
    }
    meaning = {
        str(step): analyze_sliding_meaning(
            data,
            times,
            channel_labels=labels,
            seed=seed,
            step_ms=step,
            n_surrogates=n_surrogates,
        )
        for step in meaning_steps_ms
    }
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "erp_sha256": bundle["sha256"],
        "lpfilt": bundle["lpfilt"],
        "n_accepted": bundle["n_accepted"],
        "ica": bundle["ica"],
        "face_c200": face,
        "car": cars,
        "meaning": meaning,
        "paper_anchors": PAPER_ANCHORS,
        "deltas": {
            "face_lambda": face["eigenvalues"][0] - PAPER_ANCHORS["face"]["lambda"],
            "face_p": face["p_random_phase"][0] - PAPER_ANCHORS["face"]["p"],
            "face_rdm_corr": face["components"][0]["rdm_corr"] - PAPER_ANCHORS["face"]["rdm_corr"],
            "car170_lambda1": cars["170.0"]["eigenvalues"][0] - PAPER_ANCHORS["car"]["comp1"]["lambda"],
            "car200_lambda1": cars["200.0"]["eigenvalues"][0] - PAPER_ANCHORS["car"]["comp1"]["lambda"],
        },
    }
