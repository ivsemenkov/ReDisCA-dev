"""Statistical inference helpers that operate on an already fitted ReDisCA.

These routines must not refit the estimator. Changing the number of
surrogates reuses the fitted ``Cxx``, pair stack, standardized target, and
rank/whitening.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_is_fitted

from ._core import (
    metric_subspace,
    subspace_eigenvalues,
    weighted_aggregate,
)
from ._validation import validate_positive_int


@dataclass(frozen=True)
class RandomPhaseResult:
    """Stock-SPoC random-phase test result.

    ``p_values`` has one entry per fitted component. ``null_statistic`` is
    ``max(abs(lambda))`` for each surrogate. ``p = count / B`` with no
    ``+1`` correction, so a value of 0 is allowed.
    """

    p_values: NDArray[np.float64]
    null_statistic: NDArray[np.float64]
    n_surrogates: int


def _as_generator(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    if isinstance(random_state, (bool, np.bool_)):
        raise TypeError("random_state must not be a bool.")
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None or isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(random_state)
    raise TypeError(
        "random_state must be an int, numpy Generator, or None, "
        f"got {type(random_state).__name__}."
    )


def random_phase_surrogate(
    z: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    amplitudes: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Stock SPoC ``random_phase_surrogate.m`` using a NumPy RNG.

    Amplitude spectrum is preserved except at the Nyquist bin when ``n`` is
    even, matching the MATLAB construction (free Nyquist phase, then
    ``real(ifft)``). This is not MATLAB ``rand`` bitwise parity.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    n = int(z.size)
    if n < 2:
        raise ValueError("random-phase surrogate requires at least two samples.")
    if amplitudes is None:
        amplitudes = np.abs(np.fft.fft(z))
    else:
        amplitudes = np.asarray(amplitudes, dtype=np.float64).ravel()
        if amplitudes.size != n:
            raise ValueError("amplitudes length must match z.")

    n_half = n // 2
    rand_phases = rng.random(n_half) * 2.0 * np.pi
    start = n_half - 1 if (n % 2 == 0) else n_half
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
    spectrum = amplitudes * np.exp(1j * phases)
    surrogate = np.real(np.fft.ifft(spectrum))
    return np.asarray(surrogate, dtype=np.float64), np.asarray(
        amplitudes, dtype=np.float64
    )


def random_phase_test(
    estimator,
    *,
    n_surrogates: int,
    random_state: int | np.random.Generator | None = None,
) -> RandomPhaseResult:
    """Stock-SPoC random-phase significance test on a fitted ReDisCA.

    The observed model is not refitted. Each surrogate randomizes the phase
    of the stored standardized target, rebuilds only the weighted matrix,
    and recomputes the eigenspectrum in the fitted ``Cxx`` subspace.

    The null statistic is ``max(abs(lambda_surrogate))``. Component p-values
    are ``count(null >= abs(lambda_observed)) / B`` with no ``+1`` correction.
    """
    check_is_fitted(
        estimator,
        ["eigenvalues_", "r_bar_", "z_", "centered_pair_stack_", "rank_"],
    )
    n_surrogates = validate_positive_int(n_surrogates, name="n_surrogates")
    rng = _as_generator(random_state)

    subspace = metric_subspace(
        estimator.r_bar_,
        rank=estimator.rank,
        rank_tol=estimator.rank_tol,
    )
    if subspace.used_rank != int(estimator.rank_):
        raise RuntimeError(
            "Fitted rank_ does not match the subspace reconstructed from "
            "the stored R_bar. The estimator state is inconsistent."
        )

    z = np.asarray(estimator.z_, dtype=np.float64)
    amplitudes = None
    null_statistic = np.empty(n_surrogates, dtype=np.float64)
    for index in range(n_surrogates):
        z_surrogate, amplitudes = random_phase_surrogate(
            z, rng, amplitudes=amplitudes
        )
        cxxz_s = weighted_aggregate(
            estimator.centered_pair_stack_,
            z_surrogate,
            aggregation=estimator.aggregation,
        )
        evals = subspace_eigenvalues(
            cxxz_s, subspace, solver=estimator.solver
        )
        null_statistic[index] = float(np.max(np.abs(evals)))

    observed = np.asarray(estimator.eigenvalues_, dtype=np.float64)
    p_values = np.empty(observed.shape[0], dtype=np.float64)
    for index, value in enumerate(observed):
        p_values[index] = float(
            np.sum(null_statistic >= abs(value)) / n_surrogates
        )
    return RandomPhaseResult(
        p_values=p_values,
        null_statistic=null_statistic,
        n_surrogates=n_surrogates,
    )
