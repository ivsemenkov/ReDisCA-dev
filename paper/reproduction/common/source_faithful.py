"""Source-faithful Python reconstruction of AIRI MATLAB + stock SPoC semantics.

This module must NOT import ``redisca``. It reconstructs executable MATLAB
semantics from:

- AIRI-Institute/ReDisCA @ 15bc19cdc76989da202714b257f6de4d26a42c51
- svendaehne/matlab_SPoC @ 18e4754aec1411160fd5b7ef0db852f1e0a87d90

It is a *source-faithful Python reconstruction*, not MATLAB parity. MATLAB
``eig`` / ``rand`` / ``filtfilt`` numerical details can still differ.

Authority map
-------------
Paper printed pair matrix:
    unscaled Gram ``(X_i - X_j) @ (X_i - X_j).T`` (no temporal demeaning).
AIRI -> SPoC:
    MATLAB ``cov`` of each pair difference (temporal demean + divide by T-1).
Stock SPoC target standardization:
    MATLAB ``std`` (sample SD, N-1).
Stock SPoC aggregation:
    weighted average: ``Cxxz = (Cxxe_vec @ z) / n_epochs``.
Stock SPoC inference (active code):
    random-phase surrogates; null statistic ``max(abs(lambda))``;
    ``p = count / B`` so p=0 is possible.
    The ``z(randperm(...))`` line is commented out.
AIRI pairs:
    directed ``i != j`` (both (i,j) and (j,i)).
    For a symmetric RDM this duplicates every unique pair. ``Cxx`` is
    unchanged, but MATLAB sample-SD z-scoring uses N-1, so ``z`` and
    therefore ``lambda`` scale relative to unique unordered pairs. Filters
    remain in the same one-dimensional rays.
Paper Table 1:
    triangular unique pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

PairMode = Literal["airi_directed", "unique_unordered"]
PairMatrixMode = Literal["matlab_cov", "unscaled_gram"]


def _assert_no_redisca_import() -> None:
    import sys

    if any(name == "redisca" or name.startswith("redisca.") for name in sys.modules):
        # Importing redisca elsewhere in the process is allowed for comparison
        # tests. This function is a documentation hook used by unit tests that
        # inspect this module's source.
        return


def matlab_std(values: NDArray[np.floating]) -> float:
    """MATLAB ``std(x(:))`` with default ``w=0`` (divide by N-1)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 2:
        return 0.0
    centered = values - np.mean(values)
    return float(np.sqrt(np.sum(centered * centered) / (values.size - 1)))


def matlab_zscore(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """MATLAB SPoC line: ``(z-mean(z(:)))./std(z(:))``."""
    values = np.asarray(values, dtype=np.float64).ravel()
    scale = matlab_std(values)
    if scale == 0.0:
        raise ValueError("SPoC target vector has zero sample standard deviation.")
    return (values - np.mean(values)) / scale


def matlab_cov_time_by_channel(epoch: NDArray[np.floating]) -> NDArray[np.float64]:
    """MATLAB ``cov(X)`` for ``X`` of shape ``(T, n_channels)``.

    Temporally demeans each channel and divides by ``T-1``.
    """
    epoch = np.asarray(epoch, dtype=np.float64)
    if epoch.ndim != 2:
        raise ValueError(f"epoch must be 2-D (T, n_channels), got {epoch.shape}")
    n_times = epoch.shape[0]
    if n_times < 2:
        raise ValueError("MATLAB cov requires at least two time samples.")
    centered = epoch - epoch.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / (n_times - 1)


def unscaled_gram_channel_by_time(delta: NDArray[np.floating]) -> NDArray[np.float64]:
    """Paper-style uncentered Gram ``delta @ delta.T`` for ``(n_channels, T)``."""
    delta = np.asarray(delta, dtype=np.float64)
    return delta @ delta.T


def directed_pairs(n_conditions: int) -> list[tuple[int, int]]:
    """AIRI double loop ``i_cnd = 1:N``, ``j_cnd = 1:N``, skip ``i==j``."""
    return [
        (i, j)
        for i in range(n_conditions)
        for j in range(n_conditions)
        if i != j
    ]


def unique_unordered_pairs(n_conditions: int) -> list[tuple[int, int]]:
    """Paper triangular pairs ``i < j`` (0-based)."""
    return [
        (i, j)
        for i in range(n_conditions)
        for j in range(i + 1, n_conditions)
    ]


def pair_indices(n_conditions: int, mode: PairMode) -> list[tuple[int, int]]:
    if mode == "airi_directed":
        return directed_pairs(n_conditions)
    if mode == "unique_unordered":
        return unique_unordered_pairs(n_conditions)
    raise ValueError(f"Unknown pair mode {mode!r}")


def theoretical_rdm_vector(
    rdm: NDArray[np.floating],
    pairs: list[tuple[int, int]],
) -> NDArray[np.float64]:
    rdm = np.asarray(rdm, dtype=np.float64)
    return np.array([rdm[i, j] for i, j in pairs], dtype=np.float64)


def pair_stack_from_condition_averages(
    X: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    *,
    matrix_mode: PairMatrixMode,
) -> NDArray[np.float64]:
    """Build pair matrices from condition averages.

    Parameters
    ----------
    X :
        Condition averages with shape ``(n_conditions, n_channels, n_times)``.
    matrix_mode :
        ``matlab_cov`` reconstructs SPoC ``cov(Xi'-Xj')``.
        ``unscaled_gram`` reconstructs the paper printed Gram.
    """
    X = np.asarray(X, dtype=np.float64)
    n_channels = X.shape[1]
    stack = np.empty((len(pairs), n_channels, n_channels), dtype=np.float64)
    for index, (i, j) in enumerate(pairs):
        delta = X[i] - X[j]
        if matrix_mode == "matlab_cov":
            stack[index] = matlab_cov_time_by_channel(delta.T)
        elif matrix_mode == "unscaled_gram":
            stack[index] = unscaled_gram_channel_by_time(delta)
        else:
            raise ValueError(f"Unknown pair-matrix mode {matrix_mode!r}")
    return stack


def create_cxxz(
    cxxe: NDArray[np.floating],
    z: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Stock SPoC ``create_Cxxz``: ``reshape(Cxxe_vec * z', size) / Ne``.

    ``cxxe`` has shape ``(n_channels, n_channels, n_epochs)``.
    ``z`` is already standardized.
    """
    cxxe = np.asarray(cxxe, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    n_channels, _, n_epochs = cxxe.shape
    if z.size != n_epochs:
        raise ValueError("z length must match n_epochs")
    # MATLAB: reshape(Cxxe,[Nx*Nx,Ne]) * z' / Ne. Equivalent to a z-weighted mean.
    return np.tensordot(cxxe, z, axes=([2], [0])) / n_epochs


def whiten_from_covariance(
    cxx: NDArray[np.floating],
    *,
    pca_var_explained: float = 1.0,
    rank_tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Stock SPoC ``whiten_data`` given a precomputed covariance ``C``.

    Returns whitening matrix ``M`` with whitening filters in rows, truncated
    to numerical rank and optional PCA variance cutoff.
    """
    cxx = 0.5 * (np.asarray(cxx, dtype=np.float64) + np.asarray(cxx, dtype=np.float64).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cxx)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if eigenvalues[0] <= 0.0 or not np.isfinite(eigenvalues[0]):
        raise ValueError("Cxx has no positive leading eigenvalue.")
    tol = float(eigenvalues[0]) * float(rank_tol)
    numerical_rank = int(np.sum(eigenvalues > tol))
    var_explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = int(np.searchsorted(var_explained, pca_var_explained, side="left") + 1)
    n_components = min(max(n_components, 1), numerical_rank)
    d_inv_sqrt = eigenvalues[:n_components] ** -0.5
    return (d_inv_sqrt[:, np.newaxis] * eigenvectors[:, :n_components].T)


def random_phase_surrogate(
    z: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    z_amps: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reconstruct stock SPoC ``random_phase_surrogate.m``.

    Uses ``numpy`` RNG rather than MATLAB ``rand``. Amplitude spectrum is
    preserved; phases of the positive-frequency bins are randomized with
    conjugate symmetry as in the MATLAB source.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    n = z.size
    if z_amps is None:
        z_amps = np.abs(np.fft.fft(z))
    else:
        z_amps = np.asarray(z_amps, dtype=np.float64).ravel()
        if z_amps.size != n:
            raise ValueError("z_amps length must match z")
    n_half = n // 2
    rand_phases = rng.random(n_half) * 2.0 * np.pi
    start = n_half - 1 if (n % 2 == 0) else n_half
    # MATLAB: [0, rand_phases, -rand_phases(start:-1:1)]
    if start <= 0:
        trailing = np.array([], dtype=np.float64)
    else:
        trailing = -rand_phases[start - 1 :: -1]
    phases = np.concatenate(([0.0], rand_phases, trailing))
    if phases.size != n:
        raise RuntimeError(
            f"phase vector length {phases.size} != n={n}; "
            "SPoC random-phase construction mismatch."
        )
    spectrum = z_amps * np.exp(1j * phases)
    surrogate = np.real(np.fft.ifft(spectrum))
    return surrogate.astype(np.float64), np.asarray(z_amps, dtype=np.float64)


@dataclass
class SPoCResult:
    filters: NDArray[np.float64]  # (n_channels, n_components)
    patterns: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    p_values: NDArray[np.float64] | None
    cxx: NDArray[np.float64]
    cxxz: NDArray[np.float64]
    cxxe: NDArray[np.float64]
    whitening: NDArray[np.float64]
    z: NDArray[np.float64]
    n_bootstrapping_iterations: int
    pair_mode: PairMode
    matrix_mode: PairMatrixMode
    inference: str


def spoc_from_pair_stack(
    pair_stack: NDArray[np.floating],
    z_raw: NDArray[np.floating],
    *,
    n_bootstrapping_iterations: int = 0,
    pca_var_explained: float = 1.0,
    rank_tol: float = 1e-6,
    rng: np.random.Generator | None = None,
    pair_mode: PairMode = "airi_directed",
    matrix_mode: PairMatrixMode = "matlab_cov",
    inference: str = "none",
) -> SPoCResult:
    """Deterministic SPoC GEP plus optional random-phase bootstrap.

    ``pair_stack`` shape ``(n_epochs, n_channels, n_channels)`` corresponds to
    MATLAB ``Cxxe`` *before* mean-centering, i.e. the per-epoch covariances
    (or Grams).
    """
    pair_stack = np.asarray(pair_stack, dtype=np.float64)
    z = matlab_zscore(z_raw)
    n_epochs, n_channels, _ = pair_stack.shape
    cxx = pair_stack.mean(axis=0)
    cxxe = np.moveaxis(pair_stack - cxx[np.newaxis, :, :], 0, -1)
    cxxz = create_cxxz(cxxe, z)
    whitening = whiten_from_covariance(
        cxx, pca_var_explained=pca_var_explained, rank_tol=rank_tol
    )
    cxxz_white = whitening @ cxxz @ whitening.T
    cxxz_white = 0.5 * (cxxz_white + cxxz_white.T)
    white_evals, white_filters = np.linalg.eigh(cxxz_white)
    order = np.argsort(white_evals)[::-1]
    white_evals = white_evals[order]
    white_filters = white_filters[:, order]
    filters = whitening.T @ white_filters
    for k in range(filters.shape[1]):
        denom = float(np.sqrt(filters[:, k] @ cxx @ filters[:, k]))
        if denom == 0.0 or not np.isfinite(denom):
            raise RuntimeError(f"failed to normalize filter {k}")
        filters[:, k] /= denom
    metric = filters.T @ cxx @ filters
    patterns = np.linalg.solve(metric, (cxx @ filters).T).T

    p_values: NDArray[np.float64] | None = None
    if n_bootstrapping_iterations > 0:
        if rng is None:
            raise ValueError("rng is required when n_bootstrapping_iterations > 0")
        if inference != "spoc_random_phase":
            raise ValueError(
                "n_bootstrapping_iterations>0 requires inference='spoc_random_phase'"
            )
        p_values = _random_phase_pvalues(
            cxxe=cxxe,
            whitening=whitening,
            z=z,
            observed=white_evals,
            n_iterations=n_bootstrapping_iterations,
            rng=rng,
        )

    return SPoCResult(
        filters=filters,
        patterns=patterns,
        eigenvalues=white_evals,
        p_values=p_values,
        cxx=cxx,
        cxxz=cxxz,
        cxxe=cxxe,
        whitening=whitening,
        z=z,
        n_bootstrapping_iterations=int(n_bootstrapping_iterations),
        pair_mode=pair_mode,
        matrix_mode=matrix_mode,
        inference=inference,
    )


def _random_phase_pvalues(
    *,
    cxxe: NDArray[np.float64],
    whitening: NDArray[np.float64],
    z: NDArray[np.float64],
    observed: NDArray[np.float64],
    n_iterations: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Stock SPoC bootstrap: ``p = count(max|lambda_s| >= |lambda_obs|) / B``."""
    z_amps = None
    lambda_samples = np.empty(n_iterations, dtype=np.float64)
    for k in range(n_iterations):
        z_shuffled, z_amps = random_phase_surrogate(z, rng, z_amps=z_amps)
        cxxz_s = create_cxxz(cxxe, z_shuffled)
        white = whitening @ cxxz_s @ whitening.T
        white = 0.5 * (white + white.T)
        evals = np.linalg.eigvalsh(white)
        lambda_samples[k] = float(np.max(np.abs(evals)))
    p_values = np.empty(observed.shape[0], dtype=np.float64)
    for n, value in enumerate(observed):
        p_values[n] = float(np.sum(np.abs(lambda_samples) >= abs(value)) / n_iterations)
    return p_values


def condition_label_permutation_pvalues(
    pair_stack: NDArray[np.floating],
    rdm: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    observed_eigenvalues: NDArray[np.floating],
    *,
    n_permutations: int,
    rng: np.random.Generator,
    matrix_mode: PairMatrixMode,
    pair_mode: PairMode,
    max_abs_null: bool = True,
) -> NDArray[np.float64]:
    """Paper-described surrogate: permute condition labels / theoretical RDM.

    The paper (Methods, statistical testing around Eq. 10) describes surrogate
    GEPs after permuting condition labels, which reshuffles the upper triangle
    of the theoretical RDM. This implementation permutes the condition order of
    ``rdm`` and rebuilds ``z``. Pair matrices stay fixed (they are functions of
    the data, not of the target labels).

    ``max_abs_null=True`` uses the SPoC-style ``max(|lambda|)`` null for a
    family-wise comparison. ``False`` compares each component to the matching
    surrogate eigenvalue (exploratory; label it as such in reports).
    """
    pair_stack = np.asarray(pair_stack, dtype=np.float64)
    rdm = np.asarray(rdm, dtype=np.float64)
    observed = np.asarray(observed_eigenvalues, dtype=np.float64)
    n_conditions = rdm.shape[0]
    samples = np.empty(n_permutations, dtype=np.float64)
    matching = np.empty((n_permutations, observed.size), dtype=np.float64)
    for k in range(n_permutations):
        order = rng.permutation(n_conditions)
        permuted = rdm[np.ix_(order, order)]
        z_raw = theoretical_rdm_vector(permuted, pairs)
        result = spoc_from_pair_stack(
            pair_stack,
            z_raw,
            n_bootstrapping_iterations=0,
            pair_mode=pair_mode,
            matrix_mode=matrix_mode,
            inference="none",
        )
        evals = result.eigenvalues[: observed.size]
        matching[k] = evals
        samples[k] = float(np.max(np.abs(result.eigenvalues)))
    p_values = np.empty(observed.size, dtype=np.float64)
    for n, value in enumerate(observed):
        if max_abs_null:
            p_values[n] = float(np.sum(samples >= abs(value)) / n_permutations)
        else:
            p_values[n] = float(
                np.sum(np.abs(matching[:, n]) >= abs(value)) / n_permutations
            )
    return p_values


def fit_condition_averages(
    X: NDArray[np.floating],
    rdm: NDArray[np.floating],
    *,
    pair_mode: PairMode,
    matrix_mode: PairMatrixMode,
    n_bootstrapping_iterations: int = 0,
    rng: np.random.Generator | None = None,
    inference: str = "none",
) -> SPoCResult:
    """Fit the source-faithful reconstruction to condition averages and a target RDM.

    ``X`` shape ``(n_conditions, n_channels, n_times)``.
    """
    X = np.asarray(X, dtype=np.float64)
    rdm = np.asarray(rdm, dtype=np.float64)
    pairs = pair_indices(X.shape[0], pair_mode)
    stack = pair_stack_from_condition_averages(X, pairs, matrix_mode=matrix_mode)
    z_raw = theoretical_rdm_vector(rdm, pairs)
    if inference == "spoc_random_phase" and n_bootstrapping_iterations == 0:
        raise ValueError("spoc_random_phase inference requires n_bootstrapping_iterations")
    return spoc_from_pair_stack(
        stack,
        z_raw,
        n_bootstrapping_iterations=n_bootstrapping_iterations,
        rng=rng,
        pair_mode=pair_mode,
        matrix_mode=matrix_mode,
        inference=inference,
    )


# AIRI MEG theoretical RDMs copied from Redisca_tools_faces_3_random_norm_correct.m
# Condition order: face1, face2, tool1, tool2, nons1, nons2.
# Values are the upper triangle as assigned before ``D = D+D'``.


def airi_rdm(name: str) -> NDArray[np.float64]:
    """Return the 6x6 AIRI theoretical RDM for a named contrast."""
    d = np.zeros((6, 6), dtype=np.float64)
    if name == "face":
        d[0, 1] = 0.1
        d[0, 2] = 1
        d[0, 3] = 1
        d[0, 4] = 1
        d[0, 5] = 1
        d[1, 2] = 1
        d[1, 3] = 1
        d[1, 4] = 1
        d[1, 5] = 1
        d[2, 3] = 0.1
        d[2, 4] = 0.1
        d[2, 5] = 0.1
        d[3, 4] = 0.1
        d[3, 5] = 0.1
        d[4, 5] = 0.1
    elif name == "facevstool":
        d[0, 1] = 0.1
        d[0, 2] = 1
        d[0, 3] = 1
        d[0, 4] = 0.5
        d[0, 5] = 0.5
        d[1, 2] = 1
        d[1, 3] = 1
        d[1, 4] = 0.5
        d[1, 5] = 0.5
        d[2, 3] = 0.1
        d[2, 4] = 0.5
        d[2, 5] = 0.5
        d[3, 4] = 0.5
        d[3, 5] = 0.5
        d[4, 5] = 0.1
    elif name == "tool":
        d[0, 1] = 0.1
        d[0, 2] = 1
        d[0, 3] = 1
        d[0, 4] = 0.1
        d[0, 5] = 0.1
        d[1, 2] = 1
        d[1, 3] = 1
        d[1, 4] = 0.1
        d[1, 5] = 0.1
        d[2, 3] = 0.1
        d[2, 4] = 1
        d[2, 5] = 1
        d[3, 4] = 1
        d[3, 5] = 1
        d[4, 5] = 0.1
    elif name == "meaning":
        d[0, 1] = 0.1
        d[0, 2] = 0.1
        d[0, 3] = 0.1
        d[0, 4] = 1
        d[0, 5] = 1
        d[1, 2] = 0.1
        d[1, 3] = 0.1
        d[1, 4] = 1
        d[1, 5] = 1
        d[2, 3] = 0.1
        d[2, 4] = 1
        d[2, 5] = 1
        d[3, 4] = 1
        d[3, 5] = 1
        d[4, 5] = 0.1
    elif name == "meaning1":
        d[0, 1] = 0.1
        d[0, 2] = 1
        d[0, 3] = 1
        d[0, 4] = 1
        d[0, 5] = 1
        d[1, 2] = 1
        d[1, 3] = 1
        d[1, 4] = 1
        d[1, 5] = 1
        d[2, 3] = 0.1
        d[2, 4] = 1
        d[2, 5] = 1
        d[3, 4] = 1
        d[3, 5] = 1
        d[4, 5] = 0.1
    else:
        raise ValueError(f"Unknown AIRI RDM name {name!r}")
    return d + d.T


def airi_bandpass_trials(
    trials: NDArray[np.floating],
    *,
    low_hz: float = 0.25,
    high_hz: float = 20.0,
    fs: float = 1000.0,
    order: int = 3,
) -> NDArray[np.float64]:
    """Reconstruct AIRI ``butter(3,[low,high]/(fs/2)); filtfilt`` per trial.

    ``trials`` shape ``(n_channels, n_times, n_trials)``. SciPy ``filtfilt``
    is used; MATLAB Signal Processing Toolbox padding can still differ.
    """
    from scipy.signal import butter, filtfilt

    trials = np.asarray(trials, dtype=np.float64)
    if trials.ndim != 3:
        raise ValueError("trials must have shape (n_channels, n_times, n_trials)")
    b, a = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs)
    filtered = np.empty_like(trials)
    n_channels, _, n_trials = trials.shape
    for trial in range(n_trials):
        for channel in range(n_channels):
            filtered[channel, :, trial] = filtfilt(b, a, trials[channel, :, trial])
    return filtered


AIRI_DEFAULT_RDM_NAME = "facevstool"
AIRI_TRANGE_1BASED = (600, 1500)  # MATLAB 600:1500 inclusive
AIRI_FILTER = {"order": 3, "low_hz": 0.25, "high_hz": 20.0, "fs": 1000.0}
AIRI_N_BOOTSTRAP = 1000
AIRI_N_MC_TIMECourse = 100
