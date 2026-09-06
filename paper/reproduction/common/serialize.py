"""Compact serialization of reproduction results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .hashing import write_json


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    write_json(path, metrics)


def array_fingerprint(array: np.ndarray, *, n_values: int = 8) -> dict[str, Any]:
    flat = np.asarray(array, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    summary: dict[str, Any] = {
        "shape": list(np.asarray(array).shape),
        "dtype": str(np.asarray(array).dtype),
        "n_finite": int(finite.size),
        "n_nonfinite": int(flat.size - finite.size),
    }
    if finite.size:
        summary.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "head": [float(v) for v in finite[:n_values]],
            }
        )
    return summary
