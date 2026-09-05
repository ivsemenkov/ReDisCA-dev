"""Comparison metrics for reproduction targets."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sign_align_vectors(
    reference: NDArray[np.floating],
    estimated: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Flip each estimated row so its inner product with reference is >= 0."""
    reference = np.asarray(reference, dtype=np.float64)
    estimated = np.array(estimated, dtype=np.float64, copy=True)
    if reference.shape != estimated.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} vs estimated {estimated.shape}"
        )
    if reference.ndim == 1:
        if float(np.dot(reference, estimated)) < 0.0:
            estimated *= -1.0
        return estimated
    if reference.ndim != 2:
        raise ValueError("sign alignment supports 1-D or 2-D arrays only")
    for index in range(reference.shape[0]):
        if float(np.dot(reference[index], estimated[index])) < 0.0:
            estimated[index] *= -1.0
    return estimated


def cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def pearson(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    a_rank = a.argsort().argsort().astype(np.float64)
    b_rank = b.argsort().argsort().astype(np.float64)
    return pearson(a_rank, b_rank)


def relative_error(reference: float, estimated: float) -> float:
    denom = abs(float(reference))
    if denom == 0.0:
        return abs(float(estimated))
    return abs(float(estimated) - float(reference)) / denom


def principal_angles_cosines(
    basis_a: NDArray[np.floating],
    basis_b: NDArray[np.floating],
) -> NDArray[np.floating]:
    qa, _ = np.linalg.qr(np.asarray(basis_a, dtype=np.float64), mode="reduced")
    qb, _ = np.linalg.qr(np.asarray(basis_b, dtype=np.float64), mode="reduced")
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def subspace_similarity(
    rows_a: NDArray[np.floating],
    rows_b: NDArray[np.floating],
) -> dict[str, float]:
    a = np.asarray(rows_a, dtype=np.float64).T
    b = np.asarray(rows_b, dtype=np.float64).T
    if a.shape[0] != b.shape[0]:
        raise ValueError("feature dimensions must match")
    n_dim = min(a.shape[1], b.shape[1])
    cosines = principal_angles_cosines(a[:, :n_dim], b[:, :n_dim])
    angles = np.arccos(np.clip(cosines, 0.0, 1.0))
    return {
        "min_cosine": float(np.min(cosines)),
        "mean_cosine": float(np.mean(cosines)),
        "max_angle_rad": float(np.max(angles)),
        "mean_angle_rad": float(np.mean(angles)),
        "n_dim": int(n_dim),
    }


def peak_latency_and_amplitude(
    time_s: NDArray[np.floating],
    series: NDArray[np.floating],
    *,
    signed: bool = True,
) -> dict[str, float]:
    series = np.asarray(series, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    if signed:
        index = int(np.argmax(np.abs(series)))
    else:
        index = int(np.argmax(series))
    return {
        "peak_index": index,
        "peak_time_s": float(time_s[index]),
        "peak_amplitude": float(series[index]),
    }


def unique_upper(rdm: NDArray[np.floating]) -> NDArray[np.float64]:
    rdm = np.asarray(rdm, dtype=np.float64)
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def rdm_pearson(empirical: NDArray[np.floating], theoretical: NDArray[np.floating]) -> float:
    return pearson(unique_upper(empirical), unique_upper(theoretical))
