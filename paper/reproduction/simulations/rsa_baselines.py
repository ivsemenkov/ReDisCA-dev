"""Four source-space RSA baselines from paper Fig. 1 (MNE/BF × AV/S.T.).

Inverse operators are not specified numerically in the paper. Regularization
and trial-pairing rules are documented assumptions (see config.py).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from .config import SimulationConfig
except ImportError:
    from config import SimulationConfig


def unique_pairs(n_conditions: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_conditions) for j in range(i + 1, n_conditions)]


def vectorize_upper(rdm: NDArray[np.floating]) -> NDArray[np.float64]:
    rdm = np.asarray(rdm, dtype=np.float64)
    ii, jj = np.triu_indices(rdm.shape[0], k=1)
    return rdm[ii, jj]


def pearson_upper(rdm: NDArray[np.floating], target: NDArray[np.floating]) -> float:
    """Eq. 2: Pearson of upper-triangular entries (sample corr via corrcoef)."""
    a = vectorize_upper(rdm)
    b = vectorize_upper(target)
    if a.size < 2:
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_mne_kernel(
    gain: NDArray[np.floating],
    *,
    snr: float,
) -> NDArray[np.float64]:
    """Tikhonov MNE: W = G^T (G G^T + λ^2 I)^{-1}, λ^2 = tr(GG^T)/(N SNR^2).

    ``gain`` is (n_channels, n_vertices). Returns W (n_vertices, n_channels).
    """
    gain = np.asarray(gain, dtype=np.float64)
    n_channels = gain.shape[0]
    gram = gain @ gain.T
    lambda_sq = float(np.trace(gram)) / (n_channels * float(snr) ** 2)
    regularized = gram + lambda_sq * np.eye(n_channels)
    return np.linalg.solve(regularized, gain).T


def trial_covariance(
    trials: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Sample covariance of concatenated demeaned trials.

    ``trials`` shape (I_c, C, N, T) or (C, I_c, N, T) is not accepted —
    use (I_c, C, N, T) as produced by generate.py.
    """
    trials = np.asarray(trials, dtype=np.float64)
    if trials.ndim != 4:
        raise ValueError(f"trials must be (I_c, C, N, T), got {trials.shape}")
    n_channels = trials.shape[2]
    flat = np.ascontiguousarray(trials.transpose(2, 0, 1, 3).reshape(n_channels, -1))
    flat = flat - flat.mean(axis=1, keepdims=True)
    n_samples = flat.shape[1]
    if n_samples < 2:
        raise ValueError("not enough samples for a covariance")
    return (flat @ flat.T) / (n_samples - 1)


def build_lcmv_kernel(
    gain: NDArray[np.floating],
    data_cov: NDArray[np.floating],
    *,
    reg_frac: float,
) -> NDArray[np.float64]:
    """Scalar LCMV: w_m = C^{-1} g_m / (g_m^T C^{-1} g_m). Returns (M, N)."""
    gain = np.asarray(gain, dtype=np.float64)
    data_cov = np.asarray(data_cov, dtype=np.float64)
    n_channels = data_cov.shape[0]
    ridge = float(reg_frac) * float(np.trace(data_cov) / n_channels)
    cov = data_cov + ridge * np.eye(n_channels)
    filtered = np.linalg.solve(cov, gain)
    denom = np.sum(gain * filtered, axis=0)
    denom = np.maximum(denom, 1e-20)
    return (filtered / denom).T


def apply_inverse_averages(
    kernel: NDArray[np.floating],
    averages: NDArray[np.floating],
) -> NDArray[np.float64]:
    """``kernel`` (M, N), ``averages`` (C, N, T) -> (C, M, T)."""
    kernel = np.asarray(kernel, dtype=np.float64)
    averages = np.asarray(averages, dtype=np.float64)
    return np.einsum("mn,cnt->cmt", kernel, averages, optimize=True)


def apply_inverse_trials(
    kernel: NDArray[np.floating],
    trials: NDArray[np.floating],
) -> NDArray[np.float64]:
    """``kernel`` (M, N), ``trials`` (I_c, C, N, T) -> (M, C, I_c, T).

    One GEMM; caller should drop the array after building RDMs.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    n_trials, n_cond, n_channels, n_times = trials.shape
    n_vertices = kernel.shape[0]
    flat = np.ascontiguousarray(trials.transpose(2, 1, 0, 3).reshape(n_channels, -1))
    sources = kernel @ flat
    return sources.reshape(n_vertices, n_cond, n_trials, n_times)


def rdm_from_average_sources(sources: NDArray[np.floating]) -> NDArray[np.float64]:
    """AV RSA: d_m^{ij} = ||s_m^i - s_m^j||^2. Returns (C, C, M)."""
    sources = np.asarray(sources, dtype=np.float64)
    n_cond, n_vertices, n_times = sources.shape
    rdm = np.zeros((n_cond, n_cond, n_vertices), dtype=np.float64)
    for i, j in unique_pairs(n_cond):
        delta = sources[i] - sources[j]
        dist = np.sum(delta * delta, axis=-1)
        rdm[i, j] = dist
        rdm[j, i] = dist
    return rdm


def rdm_from_single_trial_sources(sources: NDArray[np.floating]) -> NDArray[np.float64]:
    """S.T. RSA: mean_l ||s_m^{i,l} - s_m^{j,l}||^2 with index pairing.

    ``sources`` is (M, C, I_c, T). Returns (C, C, M).
    """
    sources = np.asarray(sources, dtype=np.float64)
    n_vertices, n_cond, n_trials, n_times = sources.shape
    rdm = np.zeros((n_cond, n_cond, n_vertices), dtype=np.float64)
    for i, j in unique_pairs(n_cond):
        delta = sources[:, i] - sources[:, j]
        dist = np.mean(np.sum(delta * delta, axis=-1), axis=-1)
        rdm[i, j] = dist
        rdm[j, i] = dist
    return rdm


def spotlight_scores(
    vertex_rdms: NDArray[np.floating],
    target: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Pearson Eq. 2 at each vertex. ``vertex_rdms`` is (C, C, M)."""
    vertex_rdms = np.asarray(vertex_rdms, dtype=np.float64)
    target_vec = vectorize_upper(target)
    n_cond = vertex_rdms.shape[0]
    n_vertices = vertex_rdms.shape[2]
    ii, jj = np.triu_indices(n_cond, k=1)
    stacked = vertex_rdms[ii, jj, :]  # (n_pairs, M)
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    target_c = target_vec - target_vec.mean()
    numer = centered.T @ target_c
    denom = np.sqrt(np.sum(centered * centered, axis=0) * float(np.dot(target_c, target_c)))
    scores = np.zeros(n_vertices, dtype=np.float64)
    np.divide(numer, denom, out=scores, where=denom > 0.0)
    return scores


def rsa_scan_av(
    kernel: NDArray[np.floating],
    averages: NDArray[np.floating],
    target: NDArray[np.floating],
) -> NDArray[np.float64]:
    sources = apply_inverse_averages(kernel, averages)
    rdms = rdm_from_average_sources(sources)
    return spotlight_scores(rdms, target)


def rsa_scan_st(
    kernel: NDArray[np.floating],
    trials: NDArray[np.floating],
    target: NDArray[np.floating],
) -> NDArray[np.float64]:
    sources = apply_inverse_trials(kernel, trials)
    rdms = rdm_from_single_trial_sources(sources)
    del sources
    return spotlight_scores(rdms, target)


def _vertex_rdms_for_methods(
    gain: NDArray[np.floating],
    averages: NDArray[np.floating],
    trials: NDArray[np.floating],
    *,
    config: SimulationConfig,
    mne_kernel: NDArray[np.floating] | None,
    methods: tuple[str, ...],
) -> dict[str, NDArray[np.float64]]:
    """Build (C, C, M) vertex RDMs once per inverse; reuse across source RDMs."""
    if mne_kernel is None and any(name.startswith("mne_") for name in methods):
        mne_kernel = build_mne_kernel(gain, snr=config.mne_snr)
    lcmv_kernel = None
    if any(name.startswith("bf_") for name in methods):
        cov = trial_covariance(trials)
        lcmv_kernel = build_lcmv_kernel(gain, cov, reg_frac=config.lcmv_reg_frac)
    rdms: dict[str, NDArray[np.float64]] = {}
    if "mne_av" in methods:
        assert mne_kernel is not None
        rdms["mne_av"] = rdm_from_average_sources(apply_inverse_averages(mne_kernel, averages))
    if "mne_st" in methods:
        assert mne_kernel is not None
        src = apply_inverse_trials(mne_kernel, trials)
        rdms["mne_st"] = rdm_from_single_trial_sources(src)
        del src
    if "bf_av" in methods:
        assert lcmv_kernel is not None
        rdms["bf_av"] = rdm_from_average_sources(apply_inverse_averages(lcmv_kernel, averages))
    if "bf_st" in methods:
        assert lcmv_kernel is not None
        src = apply_inverse_trials(lcmv_kernel, trials)
        rdms["bf_st"] = rdm_from_single_trial_sources(src)
        del src
    return rdms


def four_rsa_scans(
    gain: NDArray[np.floating],
    averages: NDArray[np.floating],
    trials: NDArray[np.floating],
    target: NDArray[np.floating],
    *,
    config: SimulationConfig,
    mne_kernel: NDArray[np.floating] | None = None,
    methods: tuple[str, ...] = ("mne_av", "mne_st", "bf_av", "bf_st"),
) -> dict[str, NDArray[np.float64]]:
    """Compute requested Fig. 1 RSA scans for one target RDM."""
    rdms = _vertex_rdms_for_methods(
        gain, averages, trials, config=config, mne_kernel=mne_kernel, methods=methods
    )
    return {name: spotlight_scores(vertex_rdm, target) for name, vertex_rdm in rdms.items()}


def rsa_scans_for_targets(
    gain: NDArray[np.floating],
    averages: NDArray[np.floating],
    trials: NDArray[np.floating],
    targets: NDArray[np.floating],
    *,
    config: SimulationConfig,
    mne_kernel: NDArray[np.floating] | None = None,
    methods: tuple[str, ...] = ("mne_av", "mne_st", "bf_av", "bf_st"),
) -> list[dict[str, NDArray[np.float64]]]:
    """One inverse apply, many target RDMs (four-source MC).

    ``targets`` is (P, C, C). Returns a list of length P of method→scan maps.
    """
    rdms = _vertex_rdms_for_methods(
        gain, averages, trials, config=config, mne_kernel=mne_kernel, methods=methods
    )
    out: list[dict[str, NDArray[np.float64]]] = []
    for target in targets:
        out.append({name: spotlight_scores(vertex_rdm, target) for name, vertex_rdm in rdms.items()})
    return out
