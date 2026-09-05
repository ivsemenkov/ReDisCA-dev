"""Reproduction-only inferential branches that are not in ``main``.

PRIMARY Stage A component inference is ``redisca.random_phase_test``.

These helpers implement paper-described condition-label analyses on an
already fitted AIRI-SPoC ``ReDisCA``. They do not refit the observed
estimator and they do not change constructor settings.
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_is_fitted

from redisca._core import (
    metric_subspace,
    pair_indices,
    standardize_target,
    subspace_eigenvalues,
    vectorize_rdm,
    weighted_aggregate,
)

PermutationKind = Literal["condition_labels", "upper_triangle_shuffle"]


def _fitted_subspace(estimator) -> Any:
    check_is_fitted(
        estimator,
        ["eigenvalues_", "r_bar_", "centered_pair_stack_", "rank_", "rank_tol_", "solver_", "aggregation_"],
    )
    subspace = metric_subspace(
        estimator.r_bar_,
        rank=estimator.rank_,
        rank_tol=estimator.rank_tol_,
    )
    if subspace.used_rank != int(estimator.rank_):
        raise RuntimeError("Fitted rank_ does not match reconstructed R_bar subspace.")
    return subspace


def _eigenvalues_for_rdm(estimator, rdm: NDArray[np.floating], subspace) -> NDArray[np.float64]:
    pairs = pair_indices(int(estimator.n_conditions_), directed=True)
    z = standardize_target(vectorize_rdm(np.asarray(rdm, dtype=np.float64), pairs))
    cxxz = weighted_aggregate(
        estimator.centered_pair_stack_,
        z,
        aggregation=estimator.aggregation_,
    )
    return subspace_eigenvalues(cxxz, subspace, solver=estimator.solver_)


def condition_label_permutation(
    estimator,
    rdm: NDArray[np.floating],
    *,
    kind: PermutationKind = "condition_labels",
    n_permutations: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Paper-described condition-structure permutation of the target RDM.

    Pair matrices stay fixed. Only ``z`` is rebuilt. The GEP is solved in
    the fitted ``Cxx`` subspace (same aggregation/solver/rank as the fit).

    Interpretations
    --------------
    ``condition_labels``
        Simultaneous row/column permutation of the theoretical RDM
        (permute condition names). Exact enumeration when ``C!`` is
        feasible (C=4 → 24, C=6 → 720).
    ``upper_triangle_shuffle``
        Independently shuffle unique ``i<j`` entries, then symmetrize.
        Exact enumeration when ``(C choose 2)!`` is feasible (C=4 → 720).
    """
    rdm = np.asarray(rdm, dtype=np.float64)
    n_conditions = int(rdm.shape[0])
    observed = np.asarray(estimator.eigenvalues_, dtype=np.float64)
    subspace = _fitted_subspace(estimator)

    if kind == "condition_labels":
        all_orders = list(permutations(range(n_conditions)))
        if n_permutations is None or n_permutations >= len(all_orders):
            orders = all_orders
            exact = True
        else:
            if rng is None:
                raise ValueError("rng is required when sampling condition-label permutations")
            chosen = rng.choice(len(all_orders), size=int(n_permutations), replace=False)
            orders = [all_orders[int(i)] for i in chosen]
            exact = False

        def _rdms() -> list[NDArray[np.float64]]:
            out = []
            for order in orders:
                idx = np.asarray(order, dtype=np.intp)
                out.append(rdm[np.ix_(idx, idx)])
            return out

        surrogates = _rdms()
        description = (
            "Simultaneous row/column permutation of the theoretical RDM "
            "(condition-label permutation). Pair matrices fixed."
        )
    elif kind == "upper_triangle_shuffle":
        ii, jj = np.triu_indices(n_conditions, k=1)
        base = rdm[ii, jj]
        n_pairs = int(base.size)
        n_exact = int(math.factorial(n_pairs)) if n_pairs <= 8 else None
        if n_permutations is None and n_exact is not None:
            from itertools import permutations as _perm

            unique_perms = [p for p in _perm(base) if not np.allclose(p, base)]
            # Deduplicate identical sequences.
            seen: set[tuple[float, ...]] = set()
            sequences = []
            for perm in unique_perms:
                key = tuple(np.round(perm, decimals=12))
                if key in seen:
                    continue
                seen.add(key)
                sequences.append(np.asarray(perm, dtype=np.float64))
            exact = True
        else:
            if rng is None:
                raise ValueError("rng is required when sampling upper-triangle shuffles")
            n = int(n_permutations or 1000)
            sequences = []
            for _ in range(n):
                sequences.append(rng.permutation(base))
            exact = False

        surrogates = []
        for values in sequences:
            noisy = np.zeros_like(rdm)
            noisy[ii, jj] = values
            noisy[jj, ii] = values
            surrogates.append(noisy)
        description = (
            "Independent shuffle of unique upper-triangle RDM entries, then "
            "symmetrize. Pair matrices fixed. A second reading of the paper "
            "phrase 'randomly reshuffled upper triangle'."
        )
    else:
        raise ValueError(f"Unknown permutation kind {kind!r}")

    n_obs = observed.size
    max_abs_null = np.empty(len(surrogates), dtype=np.float64)
    matched = np.empty((len(surrogates), n_obs), dtype=np.float64)
    for index, surrogate in enumerate(surrogates):
        evals = np.sort(_eigenvalues_for_rdm(estimator, surrogate, subspace))[::-1]
        matched[index] = evals[:n_obs]
        max_abs_null[index] = float(np.max(np.abs(evals)))

    p_maxabs = np.empty(n_obs, dtype=np.float64)
    p_matched = np.empty(n_obs, dtype=np.float64)
    for n, value in enumerate(observed):
        p_maxabs[n] = float(np.sum(max_abs_null >= abs(value)) / max(len(surrogates), 1))
        p_matched[n] = float(np.sum(np.abs(matched[:, n]) >= abs(value)) / max(len(surrogates), 1))
    return {
        "inference": f"paper_{kind}",
        "role": "secondary_paper_described",
        "null_statistic_primary": "max_abs_lambda",
        "p_formula": "count/B (p=0 possible; no +1 correction)",
        "B": int(len(surrogates)),
        "exact_enumeration": exact,
        "p_maxabs": p_maxabs.tolist(),
        "p_matched_exploratory": p_matched.tolist(),
        "description": description,
    }


def paper_timeseries_fwer(
    filtered_trials: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    class1: tuple[int, ...],
    class2: tuple[int, ...],
    nmc: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paper MEG time-series test: permute subcategory labels, FWER max-stat.

    ``filtered_trials`` shape ``(n_components, n_times, n_trials)``.
    """
    if nmc < 1:
        raise ValueError("nmc must be positive")
    u = np.asarray(filtered_trials, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.intp)
    n_comp, n_times, n_trials = u.shape
    if lab.size != n_trials:
        raise ValueError("labels length must match n_trials")
    obs_avgs = _condition_means(u, lab)
    obs_contrast = _class_contrast(obs_avgs, class1, class2)
    max_null = np.empty((nmc, n_comp), dtype=np.float64)
    for k in range(nmc):
        perm = rng.permutation(lab)
        avg = _condition_means(u, perm)
        contrast = _class_contrast(avg, class1, class2)
        max_null[k] = np.max(np.abs(contrast), axis=1)
    thresh = np.quantile(max_null, 1.0 - alpha, axis=0)
    significant = np.abs(obs_contrast) >= thresh[:, np.newaxis]
    p_from_max = np.empty_like(obs_contrast)
    for c in range(n_comp):
        stat = np.abs(obs_contrast[c])
        p_from_max[c] = np.mean(max_null[:, c][:, np.newaxis] >= stat[np.newaxis, :], axis=0)
    return {
        "inference": "paper_subcategory_label_permutation_fwer_max_stat",
        "role": "secondary_paper_described",
        "Nmc": int(nmc),
        "alpha": float(alpha),
        "class1": list(class1),
        "class2": list(class2),
        "contrast_definition": "mean(class2 traces) - mean(class1 traces)",
        "observed_contrast": obs_contrast,
        "fwer_threshold": thresh,
        "significant": significant,
        "p_max_stat": p_from_max,
    }


def airi_halfsplit_timecourse(
    std_planars: NDArray[np.floating],
    class1_idx: NDArray[np.integer],
    class2_idx: NDArray[np.integer],
    filters: NDArray[np.floating],
    *,
    nmc: int,
    rng: np.random.Generator,
    n_components: int = 4,
) -> dict[str, Any]:
    """Literal AIRI Nmc half-split on channel-std data.

    ``filters`` is row-oriented ``(n_components, n_channels)`` as stored by
    ``ReDisCA.filters_``.
    """
    if nmc < 1:
        raise ValueError("nmc must be positive")
    data = np.asarray(std_planars, dtype=np.float64)
    w = np.asarray(filters, dtype=np.float64)[:n_components]
    idx1 = np.asarray(class1_idx, dtype=np.intp)
    idx2 = np.asarray(class2_idx, dtype=np.intp)
    pooled = np.concatenate([idx1, idx2])
    n_pooled = pooled.size
    half = n_pooled // 2
    n_times = data.shape[1]
    projected = np.tensordot(w, data, axes=(1, 0))
    mean1 = projected[:, :, idx1].mean(axis=2)
    mean2 = projected[:, :, idx2].mean(axis=2)
    dd = mean2 - mean1
    aa = np.empty((nmc, w.shape[0], n_times), dtype=np.float64)
    for mc in range(nmc):
        order = rng.permutation(n_pooled)
        a = pooled[order[:half]]
        b = pooled[order[half:]]
        mxs1 = projected[:, :, a].mean(axis=2)
        mxs2 = projected[:, :, b].mean(axis=2)
        aa[mc] = mxs1 - mxs2
    pminus = np.empty((w.shape[0], n_times), dtype=np.float64)
    pplus = np.empty((w.shape[0], n_times), dtype=np.float64)
    for i in range(w.shape[0]):
        max_over_time = aa[:, i, :].max(axis=1)
        min_over_time = aa[:, i, :].min(axis=1)
        pminus[i] = 1.0 - np.sum(dd[i][np.newaxis, :] > max_over_time[:, np.newaxis], axis=0) / nmc
        pplus[i] = 1.0 - np.sum(dd[i][np.newaxis, :] < min_over_time[:, np.newaxis], axis=0) / nmc
    return {
        "inference": "airi_halfsplit_channel_std_nmc100",
        "role": "primary_airi_executable",
        "Nmc": int(nmc),
        "n_components": int(w.shape[0]),
        "observed_contrast_std": dd,
        "pminus": pminus,
        "pplus": pplus,
        "asterisk_positive": pplus < 0.05,
        "asterisk_negative": pminus < 0.05,
    }


def empirical_rdm_from_traces(traces: NDArray[np.floating]) -> NDArray[np.float64]:
    """Paper Eq. 1 squared-Euclidean RDM on component time series."""
    u = np.asarray(traces, dtype=np.float64)
    n = u.shape[0]
    dhat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            delta = u[i] - u[j]
            val = float(delta @ delta)
            dhat[i, j] = dhat[j, i] = val
    return dhat


def _condition_means(u: NDArray[np.float64], labels: NDArray[np.intp]) -> NDArray[np.float64]:
    n_comp, n_times, _ = u.shape
    n_cond = int(labels.max()) + 1
    out = np.empty((n_cond, n_comp, n_times), dtype=np.float64)
    for c in range(n_cond):
        out[c] = u[:, :, labels == c].mean(axis=2)
    return out


def _class_contrast(
    averages: NDArray[np.float64],
    class1: tuple[int, ...],
    class2: tuple[int, ...],
) -> NDArray[np.float64]:
    m1 = averages[list(class1)].mean(axis=0)
    m2 = averages[list(class2)].mean(axis=0)
    return m2 - m1
