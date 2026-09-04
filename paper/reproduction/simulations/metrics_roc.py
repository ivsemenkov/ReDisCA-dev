"""ROC (Eqs 17–18) and localization-error gauges for Figs 4–6."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def cosine_abs_scan(
    pattern: NDArray[np.floating],
    gain: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Eq. 13 with absolute value (pattern sign is free). ``gain`` is (N, M)."""
    pattern = np.asarray(pattern, dtype=np.float64).ravel()
    gain = np.asarray(gain, dtype=np.float64)
    numer = np.abs(gain.T @ pattern)
    denom = np.linalg.norm(gain, axis=0) * np.linalg.norm(pattern)
    return numer / np.maximum(denom, 1e-20)


def sphere_mask(distances_m: NDArray[np.floating], r_max_m: float) -> NDArray[np.bool_]:
    return np.asarray(distances_m, dtype=np.float64) <= float(r_max_m)


def counts_at_threshold(
    scores: NDArray[np.floating],
    inside: NDArray[np.bool_],
    threshold: float,
) -> dict[str, int]:
    above = np.asarray(scores, dtype=np.float64) >= float(threshold)
    inside = np.asarray(inside, dtype=bool)
    tp = int(np.count_nonzero(above & inside))
    fp = int(np.count_nonzero(above & ~inside))
    fn = int(np.count_nonzero(~above & inside))
    tn = int(np.count_nonzero(~above & ~inside))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def roc_from_mc(
    scores: NDArray[np.floating],
    inside: NDArray[np.bool_],
    thresholds: NDArray[np.floating],
) -> dict[str, Any]:
    """Paper Eqs 17–18: sum TP/FP/TN/FN across MC, then TPR/FPR.

    ``scores`` and ``inside`` have shape (n_mc, n_vertices).
    """
    scores = np.asarray(scores, dtype=np.float64)
    inside = np.asarray(inside, dtype=bool)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    tpr = np.empty(thresholds.size, dtype=np.float64)
    fpr = np.empty(thresholds.size, dtype=np.float64)
    for k, theta in enumerate(thresholds):
        above = scores >= theta
        tp = int(np.count_nonzero(above & inside))
        fp = int(np.count_nonzero(above & ~inside))
        fn = int(np.count_nonzero(~above & inside))
        tn = int(np.count_nonzero(~above & ~inside))
        tpr[k] = tp / (tp + fn) if (tp + fn) else 0.0
        fpr[k] = fp / (fp + tn) if (fp + tn) else 0.0
    order = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[order], fpr[order]))
    return {
        "thresholds": thresholds,
        "tpr": tpr,
        "fpr": fpr,
        "auc": auc,
    }


def tpr_at_fpr_cap(fpr: NDArray[np.floating], tpr: NDArray[np.floating], cap: float) -> float:
    """Largest TPR among operating points with FPR <= cap (nan if none)."""
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    ok = fpr <= float(cap)
    if not np.any(ok):
        return float("nan")
    return float(np.max(tpr[ok]))


def summarize_roc(roc: dict[str, Any]) -> dict[str, float]:
    fpr = np.asarray(roc["fpr"], dtype=np.float64)
    tpr = np.asarray(roc["tpr"], dtype=np.float64)
    return {
        "auc": float(roc["auc"]),
        "tpr_at_fpr_0": tpr_at_fpr_cap(fpr, tpr, 0.0),
        "tpr_at_fpr_0.001": tpr_at_fpr_cap(fpr, tpr, 0.001),
        "tpr_at_fpr_0.01": tpr_at_fpr_cap(fpr, tpr, 0.01),
        "tpr_at_fpr_0.05": tpr_at_fpr_cap(fpr, tpr, 0.05),
    }


def localization_error_m(
    scores: NDArray[np.floating],
    true_vertex: int,
    vertices: NDArray[np.floating],
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    hat = int(np.argmax(scores))
    true_xyz = vertices[int(true_vertex)]
    hat_xyz = vertices[hat]
    err = float(np.linalg.norm(hat_xyz - true_xyz))
    return {
        "true_vertex": int(true_vertex),
        "est_vertex": hat,
        "error_m": err,
        "error_cm": 100.0 * err,
        "peak_score": float(scores[hat]),
        "score_at_true": float(scores[int(true_vertex)]),
    }


def sign_align_to_reference(
    vector: NDArray[np.floating],
    reference: NDArray[np.floating],
) -> NDArray[np.float64]:
    vector = np.asarray(vector, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    if float(np.dot(vector, reference)) < 0.0:
        return -vector
    return vector


def pearson(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def default_thresholds(kind: str, n: int = 201) -> NDArray[np.float64]:
    if kind == "cosine_abs":
        return np.linspace(0.0, 1.0, n, dtype=np.float64)
    if kind == "pearson":
        return np.linspace(-1.0, 1.0, n, dtype=np.float64)
    raise ValueError(kind)


def downsample_roc(roc: dict[str, Any], n_keep: int = 51) -> dict[str, list[float]]:
    fpr = np.asarray(roc["fpr"], dtype=np.float64)
    tpr = np.asarray(roc["tpr"], dtype=np.float64)
    th = np.asarray(roc["thresholds"], dtype=np.float64)
    idx = np.linspace(0, fpr.size - 1, n_keep).round().astype(int)
    idx = np.unique(idx)
    return {
        "threshold": [float(x) for x in th[idx]],
        "fpr": [float(x) for x in fpr[idx]],
        "tpr": [float(x) for x in tpr[idx]],
    }
