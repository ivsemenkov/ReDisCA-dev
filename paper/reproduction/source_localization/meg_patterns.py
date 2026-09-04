"""Independent Fig. 16/17-style MEG fits used as MUSIC subspaces.

MEG sensor-space figures are owned by ``paper/reproduction/meg/``. This module
does not edit that tree. It refits the non-binary ``facevstool`` RDM here so
Fig. 18 has a locally computed ``A_K``.

Two labeled fits:

- ``paper_faithful``: ``redisca.ReDisCA(demean_time=False)``, unique pairs,
  full epoch [−500, +1000] ms (1501 samples), no AIRI Butterworth. Optional
  ``demean_time=True`` extra. Inference: condition-label permutation via
  ``common.source_faithful`` (paper §2.3; B documented, paper B unspecified).
- ``airi_executable``: ``source_faithful`` directed pairs, MATLAB ``cov``,
  ``trange`` samples 600…1500 (1-based) = 99–999 ms, butter(3) 0.25–20 Hz
  ``filtfilt``. No claim that SciPy ``filtfilt`` equals MATLAB.

Haufe patterns are used on both paths (D9): MEG numerical rank is ~67, so
paper ``A = W^{-1}`` does not apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

from common.meg_io import airi_condition_indices, load_meg_ad_run1, load_spm_trial_labels
from common.source_faithful import (
    AIRI_TRANGE_1BASED,
    airi_rdm,
    condition_label_permutation_pvalues,
    fit_condition_averages,
    pair_indices,
    pair_stack_from_condition_averages,
)

CONDITION_ORDER = ("face1", "face2", "tool1", "tool2", "nons1", "nons2")
N_PLANAR = 204
N_TIMES = 1501
FS_HZ = 1000.0
TIME_ONSET_S = -0.5


@dataclass
class PatternFit:
    label: str
    patterns: NDArray[np.float64]  # (n_components, n_channels)
    filters: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    p_values: NDArray[np.float64] | None
    rdm_name: str
    rdm: NDArray[np.float64]
    n_times: int
    time_window_s: tuple[float, float]
    preprocessing: dict[str, Any]
    estimator: str
    pair_mode: str
    matrix_mode: str
    inference: str
    permutation_B: int | None


def airi_time_slice_0based() -> slice:
    """MATLAB ``600:1500`` inclusive on a 1-based 1501-sample vector."""
    start_1based, end_1based = AIRI_TRANGE_1BASED
    return slice(start_1based - 1, end_1based)


def sample_time_s(n_times: int = N_TIMES) -> NDArray[np.float64]:
    return TIME_ONSET_S + np.arange(n_times, dtype=np.float64) / FS_HZ


def theoretical_rdm(name: str = "facevstool") -> NDArray[np.float64]:
    """AIRI numeric Fig. 16 analog (within 0.1, face–tool 1, vs nons 0.5)."""
    return airi_rdm(name)


def _bandpass_airi(trials: NDArray[np.floating]) -> NDArray[np.float64]:
    """AIRI ``butter(3,[0.25,20]/500); filtfilt`` along time (axis 1).

    ``trials`` is ``(n_channels, n_times, n_trials)``. SciPy vs MATLAB padding
    can differ; this is the source-faithful Python reconstruction, not parity.
    """
    trials = np.asarray(trials, dtype=np.float64)
    b, a = butter(3, [0.25, 20.0], btype="bandpass", fs=FS_HZ)
    return np.asarray(filtfilt(b, a, trials, axis=1, padtype="odd"), dtype=np.float64)


def load_condition_trial_indices(spm_path: Path) -> dict[str, NDArray[np.int64]]:
    labels = load_spm_trial_labels(spm_path)
    indices = airi_condition_indices(labels)
    return {name: np.asarray(indices[name], dtype=np.int64) for name in CONDITION_ORDER}


def condition_averages_from_planars(
    planars: NDArray[np.floating],
    indices: dict[str, NDArray[np.integer]],
) -> NDArray[np.float64]:
    """Return ``(6, 204, T)`` in ``CONDITION_ORDER``."""
    planars = np.asarray(planars, dtype=np.float64)
    cubes = []
    for name in CONDITION_ORDER:
        idx = np.asarray(indices[name], dtype=np.int64)
        if idx.size == 0:
            raise ValueError(f"No trials for condition {name}")
        cubes.append(planars[:, :, idx].mean(axis=2))
    return np.stack(cubes, axis=0)


def load_meg_planars(meg_mat: Path) -> NDArray[np.float64]:
    payload = load_meg_ad_run1(meg_mat)
    data = payload["data"]
    if data.shape[0] < N_PLANAR or data.shape[1] != N_TIMES:
        raise ValueError(f"Unexpected MEG layout {data.shape}")
    return np.asarray(data[:N_PLANAR], dtype=np.float64)


def prepare_condition_averages(
    meg_mat: Path,
    spm_mat: Path,
    *,
    path: Literal["paper_faithful", "airi_executable"],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Build 6 condition averages for one labeled MEG path."""
    both = prepare_both_path_averages(meg_mat, spm_mat)
    if path == "paper_faithful":
        return both["paper_faithful"]
    if path == "airi_executable":
        return both["airi_executable"]
    raise ValueError(f"Unknown path {path!r}")


def prepare_both_path_averages(
    meg_mat: Path,
    spm_mat: Path,
) -> dict[str, tuple[NDArray[np.float64], dict[str, Any]]]:
    """Load the MEG cube once; return paper-faithful and AIRI-executable averages."""
    indices = load_condition_trial_indices(spm_mat)
    counts = {name: int(indices[name].size) for name in CONDITION_ORDER}
    planars = load_meg_planars(meg_mat)
    time_s = sample_time_s(planars.shape[1])
    x_paper = condition_averages_from_planars(planars, indices)
    meta_paper = {
        "path": "paper_faithful",
        "n_times": int(x_paper.shape[2]),
        "time_window_s": (float(time_s[0]), float(time_s[-1])),
        "bandpass": None,
        "trial_counts": counts,
        "n_samples_note": "OSF is 1501 samples; paper prints 1500 (D16).",
    }
    used = np.unique(np.concatenate([indices[name] for name in CONDITION_ORDER]))
    filtered_used = _bandpass_airi(planars[:, :, used])
    filtered = np.zeros_like(planars)
    filtered[:, :, used] = filtered_used
    del filtered_used
    x_full = condition_averages_from_planars(filtered, indices)
    del planars, filtered
    sl = airi_time_slice_0based()
    x_airi = x_full[:, :, sl]
    t_win = time_s[sl]
    meta_airi = {
        "path": "airi_executable",
        "n_times": int(x_airi.shape[2]),
        "time_window_s": (float(t_win[0]), float(t_win[-1])),
        "bandpass": {
            "order": 3,
            "low_hz": 0.25,
            "high_hz": 20.0,
            "fs": FS_HZ,
            "note": "SciPy filtfilt; not MATLAB bit-parity (D8).",
        },
        "trange_1based": list(AIRI_TRANGE_1BASED),
        "trial_counts": counts,
    }
    return {
        "paper_faithful": (x_paper, meta_paper),
        "airi_executable": (x_airi, meta_airi),
    }


def fit_paper_faithful(
    X: NDArray[np.floating],
    rdm: NDArray[np.floating],
    *,
    demean_time: bool = False,
    permutation_B: int = 0,
    rng: np.random.Generator | None = None,
    rdm_name: str = "facevstool",
) -> PatternFit:
    """Canonical library fit (unique pairs, printed Gram if ``demean_time=False``)."""
    from redisca import ReDisCA

    X = np.asarray(X, dtype=np.float64)
    rdm = np.asarray(rdm, dtype=np.float64)
    estimator = ReDisCA(demean_time=demean_time).fit(X, rdm)
    p_values: NDArray[np.float64] | None = None
    inference = "none"
    B: int | None = None
    matrix_mode = "unscaled_gram" if not demean_time else "matlab_cov_centering_no_Tm1"
    if permutation_B > 0:
        if demean_time:
            raise ValueError(
                "Permutation p-values are only wired to the unscaled-Gram GEP; "
                "refuse demean_time=True so library λ and surrogate λ stay comparable."
            )
        if rng is None:
            raise ValueError("rng is required when permutation_B > 0")
        # Pair matrices and the paper permutation live in source_faithful.
        # Library unique+unscaled matches this GEP (common tests).
        pairs = pair_indices(X.shape[0], "unique_unordered")
        stack = pair_stack_from_condition_averages(X, pairs, matrix_mode="unscaled_gram")
        p_values = condition_label_permutation_pvalues(
            stack,
            rdm,
            pairs,
            estimator.eigenvalues_,
            n_permutations=permutation_B,
            rng=rng,
            matrix_mode="unscaled_gram",
            pair_mode="unique_unordered",
            max_abs_null=True,
        )
        inference = "paper_condition_label_permutation_max_abs_lambda"
        B = int(permutation_B)
    time_s = sample_time_s(X.shape[2]) if X.shape[2] == N_TIMES else (0.0, float(X.shape[2] - 1))
    if isinstance(time_s, tuple):
        window = time_s
    else:
        window = (float(time_s[0]), float(time_s[-1]))
    return PatternFit(
        label="paper_faithful" + ("_demeaned_gram" if demean_time else ""),
        patterns=np.asarray(estimator.patterns_, dtype=np.float64),
        filters=np.asarray(estimator.filters_, dtype=np.float64),
        eigenvalues=np.asarray(estimator.eigenvalues_, dtype=np.float64),
        p_values=p_values,
        rdm_name=rdm_name,
        rdm=rdm,
        n_times=int(X.shape[2]),
        time_window_s=window,
        preprocessing={
            "demean_time": bool(demean_time),
            "bandpass": None,
            "window": "full_epoch_1501",
        },
        estimator="redisca.ReDisCA",
        pair_mode="unique_unordered",
        matrix_mode=matrix_mode,
        inference=inference,
        permutation_B=B,
    )


def fit_airi_executable(
    X: NDArray[np.floating],
    rdm: NDArray[np.floating],
    *,
    rdm_name: str = "facevstool",
) -> PatternFit:
    """Directed pairs + MATLAB cov; no random-phase bootstrap (not needed for MUSIC)."""
    X = np.asarray(X, dtype=np.float64)
    rdm = np.asarray(rdm, dtype=np.float64)
    result = fit_condition_averages(
        X,
        rdm,
        pair_mode="airi_directed",
        matrix_mode="matlab_cov",
        n_bootstrapping_iterations=0,
        inference="none",
    )
    sl = airi_time_slice_0based()
    t = sample_time_s()[sl]
    return PatternFit(
        label="airi_executable",
        patterns=np.asarray(result.patterns.T, dtype=np.float64),
        filters=np.asarray(result.filters.T, dtype=np.float64),
        eigenvalues=np.asarray(result.eigenvalues, dtype=np.float64),
        p_values=None,
        rdm_name=rdm_name,
        rdm=rdm,
        n_times=int(X.shape[2]),
        time_window_s=(float(t[0]), float(t[-1])),
        preprocessing={
            "bandpass": "butter3_0.25_20_filtfilt_scipy",
            "window": "matlab_600_1500_99_to_999_ms",
        },
        estimator="common.source_faithful.fit_condition_averages",
        pair_mode="airi_directed",
        matrix_mode="matlab_cov",
        inference="none_bootstrap_skipped_for_localization",
        permutation_B=None,
    )


def three_lowest_p_indices(
    eigenvalues: NDArray[np.floating],
    p_values: NDArray[np.floating] | None,
    *,
    k: int = 3,
) -> NDArray[np.int64]:
    """Fig. 17 analog: three components with lowest p, ties by |λ| descending.

    If ``p_values`` is None, take the first ``k`` eigenvalue-sorted components.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).ravel()
    k = min(int(k), eigenvalues.size)
    if p_values is None:
        return np.arange(k, dtype=np.int64)
    p_values = np.asarray(p_values, dtype=np.float64).ravel()
    n = min(p_values.size, eigenvalues.size)
    order = np.lexsort((-np.abs(eigenvalues[:n]), p_values[:n]))
    return order[:k].astype(np.int64)
