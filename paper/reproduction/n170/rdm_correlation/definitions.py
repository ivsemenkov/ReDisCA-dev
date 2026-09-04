"""Legitimate empirical–theoretical RDM correlations (paper / AIRI / SPoC).

Only definitions named by the published paper, the Fig. 10/11 captions, or
executable/reference code are implemented. Spearman, cosine, RV, and
channel-subset RSA are not defined here.

Paper (NeuroImage 301, 120868)
------------------------------
* Eq. 1: ``d_ij = ||u_i - u_j||^2`` (squared Euclidean).
* Eq. 2: Pearson of **standardized unique upper-triangular** (``i < j``)
  RDM entries. Standardization: subtract mean, divide by SD (sample vs
  population not stated). The printed RHS is
  ``2/(C(C-1)) * sum  d̃_ij * d̃_m_ij``.
  With **population** SD that RHS equals Pearson. With MATLAB/SPoC
  **sample** SD (``N-1``) it equals ``(n-1)/n * r`` and is **not** Pearson.
* Simulation metric (Methods §3.2 / Fig. 5): corr of unique triangle of
  ``D_p`` vs ``Dhat_p = {w_p^T R_ij w_p}``.
* Fig. 10/11: observed RDM in the bottom panel; traces shown for the full
  response; face corr **0.82**, car **>0.99**.

AIRI MATLAB (MEG script only; N170 has no AIRI script)
-----------------------------------------------------
``Redisca_tools_faces_3_random_norm_correct.m`` lines 257–268:
instantaneous squared differences, then ``corrcoef(rdm_e(:), rdm_t(:))``
after filling **only** MATLAB ``(k,l)`` with ``l > k`` starting from
``rdm_e = []``. Dynamic growth yields a ``(C-1) x C`` array; unassigned
entries are 0. Stock SPoC does not compute an RDM correlation.

Possible misread (not endorsed)
-------------------------------
Flattening the whole displayed 4×4 (including diagonal zeros / the
symmetric lower triangle). Paper Eq. 2 is unique ``i < j``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

SdConvention = Literal["sample", "population"]


def unique_pairs(n_conditions: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_conditions) for j in range(i + 1, n_conditions)]


def unique_triangle(rdm: NDArray[np.floating]) -> NDArray[np.float64]:
    """Paper Eq. 2 vector: unique unordered ``i < j`` in row-major order."""
    rdm = np.asarray(rdm, dtype=np.float64)
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def triu_including_diagonal(rdm: NDArray[np.floating]) -> NDArray[np.float64]:
    """``triu(k=0)``: unique pairs **plus** the diagonal. Possible misread."""
    rdm = np.asarray(rdm, dtype=np.float64)
    return rdm[np.triu_indices(rdm.shape[0], k=0)]


def pearson(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Sample Pearson (``np.corrcoef``). Affine-invariant; scale-invariant."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size != a.size:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pearson_unique_triangle(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
) -> float:
    """Paper Eq. 2 ``corr`` of unique ``i < j`` entries."""
    return pearson(unique_triangle(left), unique_triangle(right))


def _zscore(values: NDArray[np.floating], *, convention: SdConvention) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64).ravel()
    ddof = 1 if convention == "sample" else 0
    scale = float(np.std(values, ddof=ddof))
    if scale == 0.0:
        raise ValueError("RDM triangle has zero standard deviation")
    return (values - float(np.mean(values))) / scale


def eq2_pearson_of_standardized(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    *,
    convention: SdConvention = "sample",
) -> float:
    """Pearson of Eq. 2-standardized unique triangles (affine ⇒ matches raw Pearson)."""
    a = unique_triangle(left)
    b = unique_triangle(right)
    return pearson(_zscore(a, convention=convention), _zscore(b, convention=convention))


def eq2_printed_inner_product(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    *,
    convention: SdConvention,
) -> float:
    """Literal RHS of Eq. 2: ``2/(C(C-1)) * sum d̃_ij d̃_m_ij``.

    ``convention='population'`` (divide by n) equals Pearson.
    ``convention='sample'`` (MATLAB ``std`` / SPoC) equals ``(n-1)/n * r``.
    """
    a = unique_triangle(left)
    b = unique_triangle(right)
    n_conditions = int(left.shape[0])
    za = _zscore(a, convention=convention)
    zb = _zscore(b, convention=convention)
    return float(2.0 / (n_conditions * (n_conditions - 1)) * np.sum(za * zb))


def empirical_rdm_squared_euclidean(
    traces: NDArray[np.floating],
    *,
    demean_time: bool,
) -> NDArray[np.float64]:
    """Eq. 1 on component traces of shape ``(n_conditions, n_times)``.

    ``demean_time=True`` centers each pairwise difference (MATLAB ``cov``
    centering). The Gram is unscaled (no ``1/(T-1)``).
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


def empirical_rdm_wTRw(
    weights: NDArray[np.floating],
    pair_stack: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    n_conditions: int,
) -> NDArray[np.float64]:
    """Paper simulation ``Dhat_n = {w^T R_ij w}`` for one filter ``w``."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    pair_stack = np.asarray(pair_stack, dtype=np.float64)
    if len(pairs) != pair_stack.shape[0]:
        raise ValueError("pair_stack length must match pairs")
    matrix = np.zeros((n_conditions, n_conditions), dtype=np.float64)
    for index, (i, j) in enumerate(pairs):
        value = float(w @ pair_stack[index] @ w)
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def airi_matlab_grown_matrix(rdm: NDArray[np.floating]) -> NDArray[np.float64]:
    """AIRI executable growth: ``rdm_e=[]; rdm_e(k,l)=...`` for ``l>k``.

    MATLAB size is ``(C-1, C)``. Unassigned entries (strict lower triangle
    of that rectangle, plus the unused last row of a square) are 0.
    """
    rdm = np.asarray(rdm, dtype=np.float64)
    n = rdm.shape[0]
    grown = np.zeros((n - 1, n), dtype=np.float64)
    for k in range(n):
        for l in range(k + 1, n):
            grown[k, l] = rdm[k, l]
    return grown


def airi_notes_square_upper_only(rdm: NDArray[np.floating]) -> NDArray[np.float64]:
    """Square C×C with only ``i<j`` filled (zeros on and below the diagonal).

    This is what ``paper/reference/source_notes/airi_matlab.md`` describes.
    The live MEG script grows a ``(C-1)×C`` array instead (see
    ``airi_matlab_grown_matrix``).
    """
    rdm = np.asarray(rdm, dtype=np.float64)
    n = rdm.shape[0]
    square = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            square[i, j] = rdm[i, j]
    return square


def airi_corrcoef_column_major(left: NDArray[np.floating], right: NDArray[np.floating]) -> float:
    """MATLAB ``corrcoef(A(:), B(:))``: column-major flatten, sample Pearson."""
    return pearson(
        np.asarray(left, dtype=np.float64).ravel(order="F"),
        np.asarray(right, dtype=np.float64).ravel(order="F"),
    )


def instantaneous_squared_rdm(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """AIRI ``(ss(k)-ss(l)).^2`` at one time sample. ``values`` shape ``(C,)``."""
    values = np.asarray(values, dtype=np.float64).ravel()
    n = values.size
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(values[i] - values[j]) ** 2
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


def score_empirical_against_target(
    empirical: NDArray[np.floating],
    target: NDArray[np.floating],
) -> dict[str, Any]:
    """All documented correlation readings of one empirical RDM vs a target."""
    empirical = np.asarray(empirical, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    unique_r = pearson_unique_triangle(empirical, target)
    sample_inner = eq2_printed_inner_product(empirical, target, convention="sample")
    population_inner = eq2_printed_inner_product(
        empirical, target, convention="population"
    )
    grown_e = airi_matlab_grown_matrix(empirical)
    grown_t = airi_matlab_grown_matrix(target)
    square_e = airi_notes_square_upper_only(empirical)
    square_t = airi_notes_square_upper_only(target)
    n = int(empirical.shape[0])
    n_unique = n * (n - 1) // 2
    return {
        "n_conditions": n,
        "n_unique_pairs": n_unique,
        "unique_triangle_empirical": unique_triangle(empirical).tolist(),
        "unique_triangle_target": unique_triangle(target).tolist(),
        # (1) / (5 Pearson): paper Eq. 2 corr of unique i<j
        "pearson_unique_i_lt_j": unique_r,
        "eq2_pearson_after_sample_sd_standardization": eq2_pearson_of_standardized(
            empirical, target, convention="sample"
        ),
        "eq2_pearson_after_population_sd_standardization": eq2_pearson_of_standardized(
            empirical, target, convention="population"
        ),
        "eq2_affine_invariance_abs_diff_vs_unique": abs(
            unique_r
            - eq2_pearson_of_standardized(empirical, target, convention="sample")
        ),
        # Literal printed RHS of Eq. 2 (not Pearson if sample SD)
        "eq2_printed_inner_product_sample_sd": sample_inner,
        "eq2_printed_inner_product_population_sd": population_inner,
        "eq2_sample_inner_equals_pearson_times_n_minus_1_over_n": bool(
            np.isclose(sample_inner, unique_r * (n_unique - 1) / n_unique, atol=1e-12)
        ),
        # AIRI MEG plotting analog, applied to this (windowed or instantaneous) RDM
        "airi_matlab_grown_Cxminus1_by_C_corrcoef": airi_corrcoef_column_major(
            grown_e, grown_t
        ),
        "airi_notes_square_upper_zeros_below_corrcoef": airi_corrcoef_column_major(
            square_e, square_t
        ),
        # Documented possible misreads of “upper triangular” / the displayed 4×4
        "possible_misread_full_symmetric_flatten_corrcoef": airi_corrcoef_column_major(
            empirical, target
        ),
        "possible_misread_triu_including_diagonal": pearson(
            triu_including_diagonal(empirical), triu_including_diagonal(target)
        ),
        "labels": {
            "pearson_unique_i_lt_j": (
                "paper Eq. 2 / Fig. 10–11: Pearson of unique i<j "
                "(endorsed RSA score)"
            ),
            "eq2_printed_inner_product_sample_sd": (
                "literal Eq. 2 RHS with MATLAB/SPoC sample SD; "
                "equals (n-1)/n * Pearson; not endorsed as a second metric"
            ),
            "airi_matlab_grown_Cxminus1_by_C_corrcoef": (
                "AIRI MEG script corrcoef(rdm_e(:), rdm_t(:)) after "
                "rdm_e=[]; rdm_e(k,l)=... (N170 has no AIRI script)"
            ),
            "possible_misread_full_symmetric_flatten_corrcoef": (
                "possible misread: flatten the whole displayed 4×4 including "
                "diagonal zeros; not Eq. 2"
            ),
            "possible_misread_triu_including_diagonal": (
                "possible misread: MATLAB triu(D,0) including the zero diagonal"
            ),
        },
    }


def paper_targets() -> dict[str, Any]:
    return {
        "face": 0.82,
        "car": 0.99,
        "car_inequality": ">0.99",
        "face_source": (
            "Fig. 10 caption and §4.2.1: observed RDM 'exhibits a high "
            "correlation coefficient of 0.82 with the theoretical RDM'"
        ),
        "car_source": (
            "§4.2.1: 'The observed and the theoretical RDM appear highly "
            "correlated with a correlation coefficient greater than 0.99.'"
        ),
    }
