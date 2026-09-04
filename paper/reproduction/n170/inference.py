"""Inference layer for N170 ReDisCA — separate from the deterministic fit.

Paper (§2.3): surrogate GEPs after permuting **condition labels**, which
“destroys the mutual correspondence between the set of difference correlation
matrices R_ij and the condition pair labels (i,j)”. B is unspecified.

This module:

- Observed decomposition always comes from ``from redisca import ReDisCA``.
- Paper N170 test (primary): permute the condition order of the theoretical
  RDM. With C=4 there are 24 permutations. Those collapse to 3 unique meaning
  RDMs and 4 unique face/car RDMs because of partition symmetry. Authoritative
  p-values are the exact 24-permutation distribution. A Monte Carlo draw of
  size B (default 1000) is also stored because the contract asks for B=1000;
  it is a resample of the same 24, not an independent null.
- Pair-vector shuffle: labeled alternative reading of “pair labels (i,j)”.
- SPoC random-phase of z: **exploratory only**. Do not call this the paper
  N170 test (discrepancy D5).

p-value conventions (both stored):

- ``greater_equal``: ``mean(lambda_surr >= lambda_obs)`` (SPoC-style, can be 0
  only if no surrogate matches; for exact-24, equivalent RDMs force a floor).
- ``strict_greater``: ``mean(lambda_surr > lambda_obs)`` (“exceeds” in the
  paper sentence). Equivalent label permutations have identical lambda, so
  they do not count.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import permutations
from math import factorial
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "paper" / "reproduction") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "paper" / "reproduction"))

from redisca import ReDisCA
from redisca._core import (
    mean_pair_matrix,
    pair_indices,
    pair_matrices,
    solve_generalized_eigenproblem,
    standardize_target,
    vectorize_rdm,
    weighted_centered_mean,
)

from common.source_faithful import random_phase_surrogate

PRule = Literal["greater_equal", "strict_greater"]
NullKind = Literal["condition_label", "pair_vector", "random_phase"]


def _p_from_samples(
    observed: NDArray[np.floating],
    samples: NDArray[np.floating],
    *,
    rule: PRule,
) -> NDArray[np.float64]:
    """``samples`` shape ``(B, n_comp)`` matching ``observed``."""
    observed = np.asarray(observed, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if rule == "greater_equal":
        hits = samples >= observed[np.newaxis, :]
    elif rule == "strict_greater":
        hits = samples > observed[np.newaxis, :]
    else:
        raise ValueError(f"Unknown p-value rule {rule!r}")
    return np.mean(hits, axis=0)


def fit_redisca(
    X: NDArray[np.floating],
    rdm: NDArray[np.floating],
    *,
    demean_time: bool,
) -> ReDisCA:
    """Canonical deterministic fit."""
    model = ReDisCA(demean_time=bool(demean_time))
    return model.fit(X, rdm)


def _eigenvalues_from_pair_stack(
    pair_stack: NDArray[np.floating],
    r_bar: NDArray[np.floating],
    z: NDArray[np.floating],
) -> NDArray[np.float64]:
    r_bar_d = weighted_centered_mean(pair_stack, r_bar, z)
    _, eigenvalues = solve_generalized_eigenproblem(r_bar_d, r_bar)
    return np.asarray(eigenvalues, dtype=np.float64)


def precompute_pair_geometry(
    X: NDArray[np.floating],
    *,
    demean_time: bool,
) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    pairs = pair_indices(int(X.shape[0]))
    stack = pair_matrices(X, pairs, demean_time=bool(demean_time))
    r_bar = mean_pair_matrix(stack)
    return {"pairs": pairs, "pair_stack": stack, "r_bar": r_bar}


def unique_condition_relabelings(
    rdm: NDArray[np.floating],
) -> list[dict[str, Any]]:
    """All 24 permutations, collapsed to unique upper-triangle keys."""
    rdm = np.asarray(rdm, dtype=np.float64)
    n = rdm.shape[0]
    unique: dict[tuple[float, ...], dict[str, Any]] = {}
    for order in permutations(range(n)):
        permuted = rdm[np.ix_(order, order)]
        key = tuple(np.round(permuted[np.triu_indices(n, 1)], decimals=12).tolist())
        if key not in unique:
            unique[key] = {
                "rdm": permuted,
                "multiplicity": 1,
                "example_order": list(order),
            }
        else:
            unique[key]["multiplicity"] += 1
    total = sum(item["multiplicity"] for item in unique.values())
    if total != 24 and n == 4:
        raise RuntimeError(f"Expected 24 permutations for C=4, got {total}")
    return list(unique.values())


def exact_condition_label_null(
    X: NDArray[np.floating],
    rdm: NDArray[np.floating],
    observed_eigenvalues: NDArray[np.floating],
    *,
    demean_time: bool,
) -> dict[str, Any]:
    """Enumerate all C! condition-label permutations (24 for N170)."""
    X = np.asarray(X, dtype=np.float64)
    rdm = np.asarray(rdm, dtype=np.float64)
    observed = np.asarray(observed_eigenvalues, dtype=np.float64)
    geom = precompute_pair_geometry(X, demean_time=demean_time)
    pairs = geom["pairs"]
    n_comp = int(observed.size)
    n_perm = int(factorial(int(rdm.shape[0])))
    samples = np.empty((n_perm, n_comp), dtype=np.float64)
    unique_rows = []
    row = 0
    for item in unique_condition_relabelings(rdm):
        z = standardize_target(vectorize_rdm(item["rdm"], pairs))
        ev = _eigenvalues_from_pair_stack(geom["pair_stack"], geom["r_bar"], z)
        k = min(n_comp, ev.size)
        block = np.full(n_comp, np.nan, dtype=np.float64)
        block[:k] = ev[:k]
        unique_rows.append(
            {
                "multiplicity": int(item["multiplicity"]),
                "example_order": item["example_order"],
                "lambda0": float(ev[0]),
                "eigenvalues_head": [float(v) for v in ev[: min(6, ev.size)]],
            }
        )
        for _ in range(item["multiplicity"]):
            samples[row] = block
            row += 1
    if row != n_perm:
        raise RuntimeError("Permutation table size mismatch")
    return {
        "kind": "condition_label_exact",
        "n_permutations": int(n_perm),
        "n_unique_rdms": len(unique_rows),
        "unique": unique_rows,
        "p_greater_equal": _p_from_samples(
            observed, samples, rule="greater_equal"
        ).tolist(),
        "p_strict_greater": _p_from_samples(
            observed, samples, rule="strict_greater"
        ).tolist(),
        "permutation_floor_greater_equal": {
            "meaning_partition_2_2": 8 / 24,
            "one_vs_three_detector": 6 / 24,
            "note": (
                "Equivalent relabelings leave D unchanged, so "
                "P(lambda* >= lambda_obs) cannot fall below the multiplicity "
                "of the observed RDM if that structure is uniquely best."
            ),
        },
        "samples_lambda0": [float(v) for v in samples[:, 0]],
    }


def monte_carlo_null(
    X: NDArray[np.floating],
    rdm: NDArray[np.floating],
    observed_eigenvalues: NDArray[np.floating],
    *,
    demean_time: bool,
    kind: NullKind,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Monte Carlo surrogates. ``random_phase`` is exploratory (not paper N170)."""
    X = np.asarray(X, dtype=np.float64)
    rdm = np.asarray(rdm, dtype=np.float64)
    observed = np.asarray(observed_eigenvalues, dtype=np.float64)
    geom = precompute_pair_geometry(X, demean_time=demean_time)
    pairs = geom["pairs"]
    z_obs = standardize_target(vectorize_rdm(rdm, pairs))
    n_comp = int(observed.size)
    n_conditions = int(rdm.shape[0])
    samples = np.empty((int(n_permutations), n_comp), dtype=np.float64)
    max_abs = np.empty(int(n_permutations), dtype=np.float64)
    z_amps = None
    for index in range(int(n_permutations)):
        if kind == "condition_label":
            order = rng.permutation(n_conditions)
            permuted = rdm[np.ix_(order, order)]
            z = standardize_target(vectorize_rdm(permuted, pairs))
        elif kind == "pair_vector":
            z = standardize_target(rng.permutation(vectorize_rdm(rdm, pairs)))
        elif kind == "random_phase":
            z_surr, z_amps = random_phase_surrogate(z_obs, rng, z_amps=z_amps)
            z = np.asarray(z_surr, dtype=np.float64)
        else:
            raise ValueError(f"Unknown null kind {kind!r}")
        ev = _eigenvalues_from_pair_stack(geom["pair_stack"], geom["r_bar"], z)
        k = min(n_comp, ev.size)
        row = np.full(n_comp, np.nan, dtype=np.float64)
        row[:k] = ev[:k]
        samples[index] = row
        max_abs[index] = float(np.max(np.abs(ev)))
    p_maxabs = [
        float(np.mean(max_abs >= abs(float(value)))) for value in observed
    ]
    label = {
        "condition_label": "paper_condition_label_monte_carlo",
        "pair_vector": "labeled_alternative_pair_vector_shuffle",
        "random_phase": "exploratory_spoc_random_phase_not_paper_n170",
    }[kind]
    return {
        "kind": label,
        "null_kind": kind,
        "n_permutations": int(n_permutations),
        "p_greater_equal": _p_from_samples(
            observed, samples, rule="greater_equal"
        ).tolist(),
        "p_strict_greater": _p_from_samples(
            observed, samples, rule="strict_greater"
        ).tolist(),
        "p_maxabs_null_greater_equal": p_maxabs,
        "is_paper_n170_test": kind == "condition_label",
        "lambda0_null_mean": float(np.mean(samples[:, 0])),
        "lambda0_null_max": float(np.max(samples[:, 0])),
    }


def empirical_rdm_from_traces(
    traces: NDArray[np.floating],
    *,
    demean_time: bool,
) -> NDArray[np.float64]:
    """Squared-Euclidean condition RDM of a component time series.

    ``traces`` has shape ``(n_conditions, n_times)``. If ``demean_time`` is
    True, each pairwise difference is temporally centered before the Gram,
    matching library pair matrices.
    """
    traces = np.asarray(traces, dtype=np.float64)
    n_conditions = traces.shape[0]
    matrix = np.zeros((n_conditions, n_conditions), dtype=np.float64)
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            delta = traces[i] - traces[j]
            if demean_time:
                delta = delta - delta.mean()
            value = float(delta @ delta)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def pearson_upper_triangle(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
) -> float:
    """Pearson correlation of unique ``i < j`` entries (paper Eq. 2 style)."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    tri = np.triu_indices(left.shape[0], k=1)
    a = left[tri]
    b = right[tri]
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def align_component_signs(
    patterns: NDArray[np.floating],
    filters: NDArray[np.floating],
    traces: NDArray[np.floating],
    *,
    channel_labels: list[str],
    occipital_labels: tuple[str, ...] = (
        "PO7",
        "PO3",
        "O1",
        "Oz",
        "PO8",
        "PO4",
        "O2",
    ),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Flip each component so the occipital ROI mean of the pattern is >= 0."""
    patterns = np.array(patterns, dtype=np.float64, copy=True)
    filters = np.array(filters, dtype=np.float64, copy=True)
    traces = np.array(traces, dtype=np.float64, copy=True)
    occ = [i for i, lab in enumerate(channel_labels) if lab in occipital_labels]
    if not occ:
        occ = [int(np.argmax(np.abs(patterns[0])))]
    for k in range(patterns.shape[0]):
        roi = float(np.mean(patterns[k, occ]))
        if roi < 0.0:
            patterns[k] *= -1.0
            filters[k] *= -1.0
            traces[:, k, :] *= -1.0
    return patterns, filters, traces


@dataclass
class WindowFit:
    demean_time: bool
    model: ReDisCA
    eigenvalues: NDArray[np.float64]
    filters: NDArray[np.float64]
    patterns: NDArray[np.float64]
    traces_window: NDArray[np.float64]
    traces_full: NDArray[np.float64]
    empirical_rdm_window: list[list[list[float]]]
    rdm_corr_window: list[float]
    rdm_corr_full: list[float]


def fit_window(
    X_window: NDArray[np.floating],
    X_full: NDArray[np.floating],
    rdm: NDArray[np.floating],
    *,
    demean_time: bool,
    channel_labels: list[str],
    n_report: int = 8,
) -> WindowFit:
    model = fit_redisca(X_window, rdm, demean_time=demean_time)
    n_keep = min(int(n_report), int(model.rank_))
    traces_window = model.transform(X_window)[:, :n_keep, :]
    traces_full = model.transform(X_full)[:, :n_keep, :]
    patterns, filters, traces_window = align_component_signs(
        model.patterns_[:n_keep],
        model.filters_[:n_keep],
        traces_window,
        channel_labels=channel_labels,
    )
    # Apply the same sign flips to full-epoch traces.
    signs = np.sign(
        np.sum(patterns * model.patterns_[:n_keep], axis=1, keepdims=True)
    )
    signs[signs == 0.0] = 1.0
    traces_full = traces_full * signs[np.newaxis, :, :]
    empirical = []
    corr_window = []
    corr_full = []
    for k in range(n_keep):
        d_win = empirical_rdm_from_traces(
            traces_window[:, k, :], demean_time=demean_time
        )
        d_full = empirical_rdm_from_traces(
            traces_full[:, k, :], demean_time=demean_time
        )
        empirical.append(d_win.tolist())
        corr_window.append(pearson_upper_triangle(rdm, d_win))
        corr_full.append(pearson_upper_triangle(rdm, d_full))
    return WindowFit(
        demean_time=bool(demean_time),
        model=model,
        eigenvalues=np.asarray(model.eigenvalues_, dtype=np.float64),
        filters=filters,
        patterns=patterns,
        traces_window=traces_window,
        traces_full=traces_full,
        empirical_rdm_window=empirical,
        rdm_corr_window=corr_window,
        rdm_corr_full=corr_full,
    )


def assert_core_matches_estimator(
    X: NDArray[np.floating],
    rdm: NDArray[np.floating],
    *,
    demean_time: bool,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> None:
    """Guard: permutation GEPs use the same numbers as ``ReDisCA.fit``."""
    model = fit_redisca(X, rdm, demean_time=demean_time)
    geom = precompute_pair_geometry(X, demean_time=demean_time)
    z = standardize_target(vectorize_rdm(np.asarray(rdm, dtype=np.float64), geom["pairs"]))
    ev = _eigenvalues_from_pair_stack(geom["pair_stack"], geom["r_bar"], z)
    if ev.shape != model.eigenvalues_.shape or not np.allclose(
        ev, model.eigenvalues_, rtol=rtol, atol=atol
    ):
        raise RuntimeError(
            "Inference-layer eigenvalues do not match ReDisCA.fit "
            f"(max abs diff {np.max(np.abs(ev - model.eigenvalues_)):.3e})."
        )
