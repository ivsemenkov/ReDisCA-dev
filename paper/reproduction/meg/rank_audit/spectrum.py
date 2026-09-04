"""Cxx / Rbar numerical-rank tables for the MEG 67 vs 68 audit.

Stock SPoC ``whiten_data.m`` (pinned ``18e4754``)::

    tol = ev_sorted(1) * 10^-6
    r = sum(ev_sorted > tol)

The same relative cutoff is ``source_faithful.whiten_from_covariance``
(``rank_tol=1e-6``) and the library GEP. This module does **not** run
SPoC bootstrap or MEG Monte Carlo.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sla

from common.metrics import pearson, subspace_similarity
from common.source_faithful import (
    pair_indices,
    pair_stack_from_condition_averages,
    whiten_from_covariance,
)

RANK_TOL = 1e-6
WINDOW_START_1BASED = 60
WINDOW_STOP_1BASED = 75  # inclusive
FOCUS_1BASED = (67, 68, 69)
# Dense Hermitian eig relative perturbation of λ/λ_max is ~ n·ε ≈ 4e-14.
# Use 1e-12 as a conservative "solver-scale" pad; still 1e6× below the cutoff.
CONSERVATIVE_SOLVER_DELTA = 1e-12


SolverName = Literal["numpy_eigh", "scipy_eigh", "scipy_eig_real"]


def symmetrize(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def descending_eigenvalues(
    cxx: NDArray[np.floating],
    *,
    solver: SolverName = "numpy_eigh",
) -> NDArray[np.float64]:
    """Return eigenvalues of a covariance/Gram, largest first."""
    cxx = np.asarray(cxx, dtype=np.float64)
    if solver == "numpy_eigh":
        values = np.linalg.eigvalsh(symmetrize(cxx))
        return np.sort(np.real(values))[::-1]
    if solver == "scipy_eigh":
        values = sla.eigvalsh(symmetrize(cxx), check_finite=False)
        return np.sort(np.real(values))[::-1]
    if solver == "scipy_eig_real":
        # Closer in *kind* to MATLAB ``eig`` on a general matrix than ``eigh``.
        # Not MATLAB parity. Tiny imaginary parts are discarded.
        values = sla.eigvals(cxx, check_finite=False)
        return np.sort(np.real(values))[::-1]
    raise ValueError(f"Unknown solver {solver!r}")


def numerical_rank(
    eigenvalues: NDArray[np.floating],
    *,
    rank_tol: float = RANK_TOL,
) -> int:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if eigenvalues.size == 0:
        return 0
    eig_max = float(eigenvalues[0])
    if not np.isfinite(eig_max) or eig_max <= 0.0:
        return 0
    cutoff = eig_max * float(rank_tol)
    return int(np.sum(eigenvalues > cutoff))


def ratios_to_max(eigenvalues: NDArray[np.floating]) -> NDArray[np.float64]:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).ravel()
    eig_max = float(eigenvalues[0])
    if eig_max == 0.0:
        return np.full(eigenvalues.shape, np.nan, dtype=np.float64)
    return eigenvalues / eig_max


def eig_row(
    eigenvalues: NDArray[np.floating],
    index_1based: int,
    *,
    rank_tol: float = RANK_TOL,
) -> dict[str, Any]:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if index_1based < 1 or index_1based > eigenvalues.size:
        raise IndexError(f"1-based index {index_1based} out of range 1..{eigenvalues.size}")
    value = float(eigenvalues[index_1based - 1])
    eig_max = float(eigenvalues[0])
    ratio = float(value / eig_max) if eig_max != 0.0 else float("nan")
    cutoff = eig_max * float(rank_tol)
    return {
        "index_1based": int(index_1based),
        "index_0based": int(index_1based - 1),
        "eigenvalue": value,
        "ratio_to_max": ratio,
        "cutoff": float(cutoff),
        "above_cutoff": bool(value > cutoff),
        "margin_ratio": float(ratio - rank_tol),
    }


def window_table(
    eigenvalues: NDArray[np.floating],
    *,
    start_1based: int = WINDOW_START_1BASED,
    stop_1based: int = WINDOW_STOP_1BASED,
    rank_tol: float = RANK_TOL,
) -> list[dict[str, Any]]:
    return [
        eig_row(eigenvalues, i, rank_tol=rank_tol)
        for i in range(start_1based, stop_1based + 1)
    ]


def matlab_pca_n_components(
    eigenvalues: NDArray[np.floating],
    *,
    pca_var_explained: float,
    numerical_rank_r: int | None = None,
) -> int:
    """Stock SPoC ``find(var_explained >= min_var_explained, 1)`` then ``min(., r)``.

    ``pca_var_explained=1`` (AIRI / SPoC default) therefore returns the
    numerical rank ``r``, not a tighter PCA cutoff.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).ravel()
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 1
    var_explained = np.cumsum(eigenvalues) / total
    hits = np.flatnonzero(var_explained >= float(pca_var_explained))
    if hits.size == 0:
        n_components = int(eigenvalues.size)
    else:
        n_components = int(hits[0] + 1)
    if numerical_rank_r is None:
        numerical_rank_r = numerical_rank(eigenvalues)
    return min(max(n_components, 1), int(numerical_rank_r))


def cumulative_variance(eigenvalues: NDArray[np.floating]) -> NDArray[np.float64]:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).ravel()
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return np.zeros_like(eigenvalues)
    return np.cumsum(eigenvalues) / total


def pca_interval_for_exactly_n(
    eigenvalues: NDArray[np.floating],
    n_components: int,
) -> dict[str, Any]:
    """``pca_X_var_explained`` values that yield exactly ``n`` SPoC components.

    MATLAB ``find(var_explained >= pca, 1)`` returns ``n`` iff
    ``var_explained[n-2] < pca <= var_explained[n-1]`` (0-based; for n>=2),
    after the subsequent ``min(n, r)``.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).ravel()
    var_explained = cumulative_variance(eigenvalues)
    rank_r = numerical_rank(eigenvalues)
    if n_components < 1 or n_components > eigenvalues.size:
        return {
            "n_components": int(n_components),
            "feasible": False,
            "reason": "n outside 1..n_channels",
            "open_lower": None,
            "closed_upper": None,
            "numerical_rank": rank_r,
        }
    if n_components > rank_r:
        return {
            "n_components": int(n_components),
            "feasible": False,
            "reason": (
                f"SPoC then does min(n, r) with r={rank_r}; cannot keep "
                f"{n_components} components even with pca=1"
            ),
            "open_lower": None,
            "closed_upper": None,
            "numerical_rank": rank_r,
        }
    upper = float(var_explained[n_components - 1])
    lower = 0.0 if n_components == 1 else float(var_explained[n_components - 2])
    # If r > n, pca=1 still yields r, not n. Need pca <= var_explained[n-1]
    # and pca > var_explained[n-2], and also pca small enough that find
    # returns n rather than a later index — that is exactly the interval
    # (var[n-2], var[n-1]].
    return {
        "n_components": int(n_components),
        "feasible": True,
        "open_lower": lower,
        "closed_upper": upper,
        "interval_note": (
            f"pca in ({lower:.16g}, {upper:.16g}] yields find()={n_components} "
            f"before min(., r={rank_r})"
        ),
        "airi_default_pca": 1.0,
        "airi_default_would_select": matlab_pca_n_components(
            eigenvalues, pca_var_explained=1.0, numerical_rank_r=rank_r
        ),
        "numerical_rank": rank_r,
        "cumulative_at_n": upper,
        "cumulative_at_n_minus_1": lower,
    }


def solver_comparison(
    cxx: NDArray[np.floating],
    *,
    rank_tol: float = RANK_TOL,
) -> dict[str, Any]:
    """Compare numpy eigh, scipy eigh, and scipy general eig on the same Cxx."""
    cxx = np.asarray(cxx, dtype=np.float64)
    spectra: dict[str, NDArray[np.float64]] = {}
    ranks: dict[str, int] = {}
    for name in ("numpy_eigh", "scipy_eigh", "scipy_eig_real"):
        eigs = descending_eigenvalues(cxx, solver=name)  # type: ignore[arg-type]
        spectra[name] = eigs
        ranks[name] = numerical_rank(eigs, rank_tol=rank_tol)
    np_eigs = spectra["numpy_eigh"]
    sc_eigs = spectra["scipy_eigh"]
    ge_eigs = spectra["scipy_eig_real"]
    n_compare = int(np_eigs.size)
    focus = [i for i in FOCUS_1BASED if i <= n_compare]
    return {
        "ranks": ranks,
        "max_abs_numpy_vs_scipy_eigh": float(np.max(np.abs(np_eigs - sc_eigs))),
        "max_abs_numpy_eigh_vs_scipy_eig_real": float(
            np.max(np.abs(np_eigs - ge_eigs[:n_compare]))
        ),
        "ratio_window_numpy": [
            eig_row(np_eigs, i, rank_tol=rank_tol) for i in focus
        ],
        "ratio_window_scipy_eigh": [
            eig_row(sc_eigs, i, rank_tol=rank_tol) for i in focus
        ],
        "ratio_window_scipy_eig_real": [
            eig_row(ge_eigs, i, rank_tol=rank_tol) for i in focus
        ],
        "all_solvers_agree_on_rank": len(set(ranks.values())) == 1,
        "asymmetry_fro_over_fro": float(
            np.linalg.norm(cxx - cxx.T, ord="fro")
            / max(np.linalg.norm(cxx, ord="fro"), np.finfo(float).tiny)
        ),
    }


def matlab_eig_flip_assessment(
    eigenvalues: NDArray[np.floating],
    *,
    rank_tol: float = RANK_TOL,
    conservative_solver_delta: float = CONSERVATIVE_SOLVER_DELTA,
) -> dict[str, Any]:
    """Argue 67 vs 68 from the gap around the cutoff, not from MATLAB parity.

    MATLAB is unavailable. A flip of *this* spectrum from rank 68 to 67 would
    require the 68th relative eigenvalue to fall through ``rank_tol`` while the
    67th stays above it. Dense Hermitian eig perturbations of ``λ/λ_max`` are
    ~``n·ε ≈ 4e-14``; ``conservative_solver_delta`` pads that to 1e-12.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).ravel()
    rows = {i: eig_row(eigenvalues, i, rank_tol=rank_tol) for i in FOCUS_1BASED}
    r67 = float(rows[67]["ratio_to_max"])
    r68 = float(rows[68]["ratio_to_max"])
    r69 = float(rows[69]["ratio_to_max"])
    cutoff = float(rank_tol)
    rank_here = numerical_rank(eigenvalues, rank_tol=rank_tol)
    margin_68_above = r68 - cutoff
    margin_69_below = cutoff - r69
    gap_67_68 = r67 - r68
    gap_68_69 = r68 - r69
    # Relative drop of eig_68 needed to land on or below the cutoff.
    if r68 > 0.0:
        relative_drop_to_cross = (r68 - cutoff) / r68
    else:
        relative_drop_to_cross = float("nan")
    if rank_here >= 68:
        distance_to_flip_down = margin_68_above
    else:
        distance_to_flip_down = float("nan")
    if distance_to_flip_down > 100.0 * conservative_solver_delta:
        verdict = "not_borderline"
        plausible = False
    elif distance_to_flip_down > 10.0 * conservative_solver_delta:
        verdict = "unlikely"
        plausible = False
    elif np.isfinite(distance_to_flip_down) and distance_to_flip_down > 0.0:
        verdict = "plausible_borderline"
        plausible = True
    else:
        verdict = "already_at_or_below_cutoff"
        plausible = False
    return {
        "rank_at_1e-6": rank_here,
        "ratio_67": r67,
        "ratio_68": r68,
        "ratio_69": r69,
        "cutoff": cutoff,
        "margin_68_above_cutoff": float(margin_68_above),
        "margin_69_below_cutoff": float(margin_69_below),
        "gap_ratio_67_minus_68": float(gap_67_68),
        "gap_ratio_68_minus_69": float(gap_68_69),
        "relative_drop_of_eig68_to_cross_cutoff": float(relative_drop_to_cross),
        "conservative_solver_delta_ratio": float(conservative_solver_delta),
        "typical_lapack_n_eps": float(eigenvalues.size * np.finfo(float).eps),
        "verdict": verdict,
        "plausible_that_matlab_eig_alone_flips_68_to_67": bool(plausible),
        "rows": {str(k): v for k, v in rows.items()},
        "note": (
            "MATLAB eig is unavailable. This is a gap argument, not a parity claim. "
            "A solver-only flip 68→67 needs |δ(λ_68/λ_max)| ≥ margin_68_above_cutoff."
        ),
    }


def mean_pair_matrix(
    averages: NDArray[np.floating],
    *,
    pair_mode: str,
    matrix_mode: str,
) -> tuple[NDArray[np.float64], int]:
    """Cxx / Rbar = mean of pair matrices. No z-weighting, no bootstrap."""
    averages = np.asarray(averages, dtype=np.float64)
    pairs = pair_indices(averages.shape[0], pair_mode)  # type: ignore[arg-type]
    stack = pair_stack_from_condition_averages(
        averages, pairs, matrix_mode=matrix_mode  # type: ignore[arg-type]
    )
    return stack.mean(axis=0), int(stack.shape[0])


def spectrum_payload(
    cxx: NDArray[np.floating],
    *,
    label: str,
    pair_mode: str,
    matrix_mode: str,
    n_times: int,
    n_pairs: int,
    bandpass: dict[str, Any] | None,
    window_ms: list[float],
    extra: dict[str, Any] | None = None,
    rank_tol: float = RANK_TOL,
) -> dict[str, Any]:
    cxx = np.asarray(cxx, dtype=np.float64)
    eigs = descending_eigenvalues(cxx, solver="numpy_eigh")
    rank = numerical_rank(eigs, rank_tol=rank_tol)
    whitening = whiten_from_covariance(cxx, pca_var_explained=1.0, rank_tol=rank_tol)
    var_explained = cumulative_variance(eigs)
    payload: dict[str, Any] = {
        "label": label,
        "pair_mode": pair_mode,
        "matrix_mode": matrix_mode,
        "n_channels": int(cxx.shape[0]),
        "n_times": int(n_times),
        "n_pairs": int(n_pairs),
        "window_ms": [float(x) for x in window_ms],
        "bandpass": bandpass,
        "rank_tol": float(rank_tol),
        "eig_max": float(eigs[0]),
        "cutoff": float(eigs[0] * rank_tol),
        "numerical_rank_1e-6": rank,
        "whitening_n_components_pca1": int(whitening.shape[0]),
        "n_positive_eigs": int(np.sum(eigs > 0.0)),
        "window_60_75": window_table(eigs, rank_tol=rank_tol),
        "focus_67_69": [eig_row(eigs, i, rank_tol=rank_tol) for i in FOCUS_1BASED],
        "eigenvalues_descending": eigs.tolist(),
        "ratios_to_max": ratios_to_max(eigs).tolist(),
        "cumulative_variance": var_explained.tolist(),
        "pca_interval_for_exactly_67": pca_interval_for_exactly_n(eigs, 67),
        "pca_n_at_default_1": matlab_pca_n_components(
            eigs, pca_var_explained=1.0, numerical_rank_r=rank
        ),
        "solver_comparison": solver_comparison(cxx, rank_tol=rank_tol),
        "matlab_eig_flip": matlab_eig_flip_assessment(eigs, rank_tol=rank_tol),
    }
    if extra:
        payload["extra"] = extra
    return payload


def author_a1_payload(
    a1: NDArray[np.floating],
    *,
    comps_order: NDArray[np.integer] | None,
    path: str,
    sha256: str,
    airi_patterns: NDArray[np.floating] | None = None,
    paper_patterns: NDArray[np.floating] | None = None,
) -> dict[str, Any]:
    """Describe OSF A1 and (optionally) its subspace overlap with local Haufe patterns.

    Does **not** force local rank to ``A1.shape[1]``.
    """
    a1 = np.asarray(a1, dtype=np.float64)
    col_norms = np.linalg.norm(a1, axis=0)
    s = np.linalg.svd(a1, compute_uv=False)
    s_rank = numerical_rank(s, rank_tol=RANK_TOL)
    payload: dict[str, Any] = {
        "path": path,
        "sha256": sha256,
        "shape": [int(a1.shape[0]), int(a1.shape[1])],
        "n_columns": int(a1.shape[1]),
        "comps_order": None if comps_order is None else [int(x) for x in np.asarray(comps_order).ravel()],
        "column_norm_min": float(np.min(col_norms)),
        "column_norm_max": float(np.max(col_norms)),
        "column_norm_median": float(np.median(col_norms)),
        "column_norm_mean": float(np.mean(col_norms)),
        "svd_head": [float(x) for x in s[:8]],
        "svd_tail": [float(x) for x in s[-4:]],
        "svd_numerical_rank_1e-6_of_singular_values": s_rank,
        "do_not_force_local_rank_to_n_columns": True,
        "note": (
            "A1 column count is the whitening size of the MATLAB SPoC run that "
            "saved this file (D17: committed script returns before save). It is "
            "not by itself a reason to truncate a local Cxx rank to 67."
        ),
    }
    if airi_patterns is not None:
        payload["vs_airi_executable_haufe"] = _pattern_overlap(a1, airi_patterns)
    if paper_patterns is not None:
        payload["vs_paper_faithful_haufe"] = _pattern_overlap(a1, paper_patterns)
    return payload


def _pattern_overlap(
    a1: NDArray[np.float64],
    patterns: NDArray[np.floating],
) -> dict[str, Any]:
    patterns = np.asarray(patterns, dtype=np.float64)
    n_take = min(a1.shape[1], patterns.shape[1])
    subspace = subspace_similarity(a1[:, :n_take].T, patterns[:, :n_take].T)
    n_lead = min(4, n_take)
    lead = []
    for k in range(n_lead):
        lead.append(
            {
                "component_1based": k + 1,
                "abs_pearson": abs(pearson(a1[:, k], patterns[:, k])),
                "a1_col_norm": float(np.linalg.norm(a1[:, k])),
                "local_col_norm": float(np.linalg.norm(patterns[:, k])),
            }
        )
    return {
        "local_n_patterns": int(patterns.shape[1]),
        "compared_dim": int(n_take),
        "subspace_first_n_take": subspace,
        "leading_column_abs_pearson": lead,
    }
