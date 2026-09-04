"""Labeled inference procedures for the MEG track.

Paper path (D5, D5b paper side)
    Component p: permute condition labels of the theoretical RDM, refit the
    GEP on *fixed* pair matrices, ``p = count / B``. Primary null is
    ``max|lambda|`` (SPoC-style FWER across components). Matched-component
    p-values are recorded as exploratory.

    Time-series asterisks: permute subcategory labels of the 480 epochs
    (preserving 80/condition counts), recompute surrogate averages, apply
    *fixed* spatial filters, FWER max-stat over time. Linear filtering is
    applied to trials first; averaging after filtering is algebraically
    identical to averaging then filtering.

AIRI path (D5, D5b AIRI side)
    Component p: stock SPoC random-phase of ``z``, ``p = count / B`` with
    ``B=1000`` when feasible. Implemented by ``fit_condition_averages``.

    Time-series asterisks: ``Nmc=100`` half-split of pooled class trials on
    channel-time sample-SD normalized *filtered* data. This is **not** the
    paper test.

    Pair-order diagnostic: shuffle the directed pair sequence and recompute
    random-phase p-values. Random-phase preserves the FFT amplitude of ``z``
    as a *sequence*, so pair order is an accidental degree of freedom.
    Labeled diagnostic, not a replacement test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from common.source_faithful import (
    create_cxxz,
    directed_pairs,
    matlab_zscore,
    pair_indices,
    pair_stack_from_condition_averages,
    spoc_from_pair_stack,
    theoretical_rdm_vector,
    unique_unordered_pairs,
)


def paper_component_permutation(
    pair_stack: NDArray[np.floating],
    rdm: NDArray[np.floating],
    observed_eigenvalues: NDArray[np.floating],
    *,
    n_permutations: int,
    rng: np.random.Generator,
    pair_mode: str = "unique_unordered",
    matrix_mode: str = "unscaled_gram",
) -> dict[str, Any]:
    """Paper §2.3: permute condition labels; pair matrices stay fixed.

    Whitening / ``Cxx`` are data functions and are reused across permutations
    (equivalent to a full GEP refit because only ``z`` changes).
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    rdm = np.asarray(rdm, dtype=np.float64)
    observed = np.asarray(observed_eigenvalues, dtype=np.float64)
    n_conditions = rdm.shape[0]
    pairs = pair_indices(n_conditions, pair_mode)  # type: ignore[arg-type]
    base_z = theoretical_rdm_vector(rdm, pairs)
    base = spoc_from_pair_stack(
        pair_stack,
        base_z,
        n_bootstrapping_iterations=0,
        pair_mode=pair_mode,  # type: ignore[arg-type]
        matrix_mode=matrix_mode,  # type: ignore[arg-type]
        inference="none",
    )
    n_obs = observed.size
    max_abs_null = np.empty(n_permutations, dtype=np.float64)
    matched = np.empty((n_permutations, n_obs), dtype=np.float64)
    for k in range(n_permutations):
        order = rng.permutation(n_conditions)
        permuted = rdm[np.ix_(order, order)]
        z = matlab_zscore(theoretical_rdm_vector(permuted, pairs))
        cxxz = create_cxxz(base.cxxe, z)
        white = base.whitening @ cxxz @ base.whitening.T
        white = 0.5 * (white + white.T)
        evals = np.sort(np.linalg.eigvalsh(white))[::-1]
        matched[k] = evals[:n_obs]
        max_abs_null[k] = float(np.max(np.abs(evals)))
    p_maxabs = np.empty(n_obs, dtype=np.float64)
    p_matched = np.empty(n_obs, dtype=np.float64)
    for n, value in enumerate(observed):
        p_maxabs[n] = float(np.sum(max_abs_null >= abs(value)) / n_permutations)
        p_matched[n] = float(np.sum(np.abs(matched[:, n]) >= abs(value)) / n_permutations)
    return {
        "inference": "paper_condition_label_permutation",
        "null_statistic_primary": "max_abs_lambda",
        "p_formula": "count/B (p=0 possible; no +1 continuity correction)",
        "B": int(n_permutations),
        "pair_mode": pair_mode,
        "matrix_mode": matrix_mode,
        "p_maxabs": p_maxabs,
        "p_matched_exploratory": p_matched,
        "note": (
            "Primary p-values use a family-wise max|lambda| null (SPoC-style). "
            "Matched-component p-values are exploratory; the paper does not "
            "specify which. D4 sum-vs-mean is a global scale and cancels in "
            "this ranking."
        ),
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
    """Paper MEG time-series test (D5b paper side).

    ``filtered_trials`` is already spatially filtered:
    shape ``(n_components, n_times, n_trials)`` with ``n_trials=480``.
    ``labels`` is the 0…5 subcategory label of each trial (80 each).

    Surrogates permute those labels (preserving counts), rebuild condition
    averages, then the class contrast. FWER threshold is the ``1-alpha``
    quantile of ``max_t |contrast_t|``.
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
        "Nmc": int(nmc),
        "alpha": float(alpha),
        "class1": list(class1),
        "class2": list(class2),
        "contrast_definition": "mean(class2 traces) - mean(class1 traces)",
        "observed_contrast": obs_contrast,
        "observed_averages": obs_avgs,
        "fwer_threshold": thresh,
        "significant": significant,
        "p_max_stat": p_from_max,
        "null_max_mean": max_null.mean(axis=0),
        "null_max_std": max_null.std(axis=0, ddof=1) if nmc > 1 else np.zeros(n_comp),
        "note": (
            "Permute subcategory labels of individual epochs; surrogate "
            "averages; fixed filters; FWER via maximum statistics over the "
            "entire time interval (paper §4.2.2). Not the AIRI half-split."
        ),
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
    """AIRI ``Nmc=100`` half-split on channel-std data (D5b AIRI side).

    ``std_planars`` shape ``(n_channels, n_times, n_file_trials)`` already
    divided by MATLAB ``std(d,0,3)``. ``filters`` is column-oriented
    ``(n_channels, n_components)`` as in SPoC ``W``.

    MATLAB indexes ``randperm(length(idxAll))`` values as trial numbers
    (a possible off-by-construction hazard). This reconstruction splits the
    *pooled class trial indices* as described in D5b / source notes.

    pplus/pminus copy the MATLAB broadcasting:
    ``pminus = 1 - count(dd(t) > max_tau aa(mc, tau)) / Nmc``.
    """
    if nmc < 1:
        raise ValueError("nmc must be positive")
    data = np.asarray(std_planars, dtype=np.float64)
    w = np.asarray(filters, dtype=np.float64)[:, :n_components]
    idx1 = np.asarray(class1_idx, dtype=np.intp)
    idx2 = np.asarray(class2_idx, dtype=np.intp)
    pooled = np.concatenate([idx1, idx2])
    n_pooled = pooled.size
    half = n_pooled // 2
    n_times = data.shape[1]
    projected = np.tensordot(w.T, data, axes=(1, 0))  # (n_comp, T, n_trials)
    mean1 = projected[:, :, idx1].mean(axis=2)
    mean2 = projected[:, :, idx2].mean(axis=2)
    dd = mean2 - mean1  # AIRI: W'*(meanClass2-meanClass1)
    aa = np.empty((nmc, n_components, n_times), dtype=np.float64)
    for mc in range(nmc):
        order = rng.permutation(n_pooled)
        a = pooled[order[:half]]
        b = pooled[order[half:]]
        mxs1 = projected[:, :, a].mean(axis=2)
        mxs2 = projected[:, :, b].mean(axis=2)
        aa[mc] = mxs1 - mxs2
    pminus = np.empty((n_components, n_times), dtype=np.float64)
    pplus = np.empty((n_components, n_times), dtype=np.float64)
    for i in range(n_components):
        max_over_time = aa[:, i, :].max(axis=1)
        min_over_time = aa[:, i, :].min(axis=1)
        pminus[i] = 1.0 - np.sum(dd[i][np.newaxis, :] > max_over_time[:, np.newaxis], axis=0) / nmc
        pplus[i] = 1.0 - np.sum(dd[i][np.newaxis, :] < min_over_time[:, np.newaxis], axis=0) / nmc
    return {
        "inference": "airi_halfsplit_channel_std_nmc100",
        "Nmc": int(nmc),
        "n_components": int(n_components),
        "class1_n": int(idx1.size),
        "class2_n": int(idx2.size),
        "pooled_n": int(n_pooled),
        "observed_contrast_std": dd,
        "pminus": pminus,
        "pplus": pplus,
        "asterisk_positive": pplus < 0.05,
        "asterisk_negative": pminus < 0.05,
        "matlab_indexing_hazard": (
            "Committed MATLAB uses idx1=rpm(1:half) as linear indices into d "
            "(values 1:length(idxAll)), not idxAll(rpm). This reconstruction "
            "splits the pooled class trials as D5b describes."
        ),
        "note": (
            "AIRI path only. Pointwise pplus/pminus from half-splits of pooled "
            "class trials on channel-time std-normalized data. Do not call this "
            "the paper FWER test. Do not compare asterisks across paths."
        ),
    }


def pair_order_sensitivity(
    averages: NDArray[np.floating],
    rdm: NDArray[np.floating],
    *,
    n_shuffles: int,
    n_bootstrapping_iterations: int,
    rng: np.random.Generator,
    n_report: int = 4,
) -> dict[str, Any]:
    """Diagnostic: random-phase p-values vs directed pair sequence order.

    Not a replacement for the AIRI default pair order (i_cnd, j_cnd double
    loop). Random-phase surrogates treat ``z`` as a length-30 sequence.
    """
    pairs = directed_pairs(averages.shape[0])
    base_stack = pair_stack_from_condition_averages(averages, pairs, matrix_mode="matlab_cov")
    base_z = theoretical_rdm_vector(rdm, pairs)
    base = spoc_from_pair_stack(
        base_stack,
        base_z,
        n_bootstrapping_iterations=n_bootstrapping_iterations,
        rng=rng,
        pair_mode="airi_directed",
        matrix_mode="matlab_cov",
        inference="spoc_random_phase",
    )
    shuffled_p: list[list[float]] = []
    for _ in range(n_shuffles):
        order = rng.permutation(len(pairs))
        sh_pairs = [pairs[int(i)] for i in order]
        stack = pair_stack_from_condition_averages(averages, sh_pairs, matrix_mode="matlab_cov")
        z = theoretical_rdm_vector(rdm, sh_pairs)
        result = spoc_from_pair_stack(
            stack,
            z,
            n_bootstrapping_iterations=n_bootstrapping_iterations,
            rng=rng,
            pair_mode="airi_directed",
            matrix_mode="matlab_cov",
            inference="spoc_random_phase",
        )
        p = result.p_values
        shuffled_p.append([float(v) for v in p[:n_report]] if p is not None else [])
    base_p = [float(v) for v in base.p_values[:n_report]] if base.p_values is not None else []
    return {
        "inference": "diagnostic_pair_order_sensitivity_of_random_phase_p",
        "replacement_test": False,
        "n_shuffles": int(n_shuffles),
        "B": int(n_bootstrapping_iterations),
        "default_pair_order_p": base_p,
        "shuffled_pair_order_p": shuffled_p,
        "note": (
            "Diagnostic only. Pair-order shuffle changes the accidental FFT "
            "spectrum of the length-30 z vector that stock SPoC random-phase "
            "preserves. Filters stay on the same rays for a symmetric RDM."
        ),
    }


def empirical_rdm_from_traces(
    traces: NDArray[np.floating],
    *,
    demean_time: bool,
) -> NDArray[np.float64]:
    """Windowed empirical RDM ``d_ij = ||u_i - u_j||^2`` (paper squared Euclidean).

    ``traces`` shape ``(n_conditions, n_times)`` for one component.
    Temporal demeaning, if requested, matches ``demean_time`` on the pair
    difference (library / MATLAB cov centering).
    """
    u = np.asarray(traces, dtype=np.float64)
    n = u.shape[0]
    dhat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            delta = u[i] - u[j]
            if demean_time:
                delta = delta - delta.mean()
            val = float(delta @ delta)
            dhat[i, j] = dhat[j, i] = val
    return dhat


def unique_pair_pearson(empirical: NDArray[np.floating], theoretical: NDArray[np.floating]) -> float:
    """Pearson of unique upper-triangle entries (paper RSA score, Eq. 2)."""
    from common.metrics import pearson

    n = empirical.shape[0]
    pairs = unique_unordered_pairs(n)
    a = np.array([empirical[i, j] for i, j in pairs], dtype=np.float64)
    b = np.array([theoretical[i, j] for i, j in pairs], dtype=np.float64)
    return pearson(a, b)


def _condition_means(u: NDArray[np.float64], labels: NDArray[np.intp]) -> NDArray[np.float64]:
    """``u`` is ``(n_comp, n_times, n_trials)`` → averages ``(6, n_comp, n_times)``."""
    n_comp, n_times, _ = u.shape
    out = np.empty((6, n_comp, n_times), dtype=np.float64)
    for c in range(6):
        out[c] = u[:, :, labels == c].mean(axis=2)
    return out


def _class_contrast(
    averages: NDArray[np.float64],
    class1: tuple[int, ...],
    class2: tuple[int, ...],
) -> NDArray[np.float64]:
    """``averages`` is ``(6, n_comp, n_times)`` → ``(n_comp, n_times)``."""
    m1 = averages[list(class1)].mean(axis=0)
    m2 = averages[list(class2)].mean(axis=0)
    return m2 - m1
