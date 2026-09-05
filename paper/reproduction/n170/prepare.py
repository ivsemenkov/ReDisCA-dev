"""Load ERP CORE subject-1 averages and documented N170 analysis windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from paper.reproduction.common.erp_io import load_erplab_erp
from paper.reproduction.common.hashing import sha256_file
from paper.reproduction.common.paths import ERPCORE_DIR
from paper.reproduction.n170.rdms import CONDITION_LABELS, CONDITION_ORDER

SUBJECT_ID = "1"
PREFERRED_ERP_NAME = "1_N170_erp_ar.erp"
LPFILT_ERP_NAME = "1_N170_erp_ar_lpfilt.erp"
AR_CSV_NAME = "1_AR_Percentages_N170.csv"

SCALP_LABELS: tuple[str, ...] = (
    "FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "PO7", "PO3", "O1", "Oz",
    "Pz", "CPz", "FP2", "Fz", "F4", "F8", "FC4", "FCz", "Cz", "C4", "C6", "P4",
    "P8", "PO8", "PO4", "O2",
)
OCCIPITAL_LABELS: tuple[str, ...] = ("PO7", "PO3", "O1", "Oz", "PO8", "PO4", "O2", "P7", "P8")
ERPCORE_SUBJECT1_ICA_COMPONENTS: tuple[int, ...] = (2, 7)
MEANING_DURATION_MS = 150.0
FACE_DURATION_MS = 100.0
FACE_CENTER_MS = 200.0
CAR_DURATION_MS = 100.0
CAR_CENTERS_MS: tuple[float, ...] = (170.0, 200.0)
DEFAULT_SLIDING_STEP_MS = 25.0
SAMPLE_STEP_MS = 1000.0 / 256.0
FIG8_CENTERS_MS: tuple[float, float, float] = (375.0, 400.0, 425.0)
EXPECTED_SHA256 = {
    PREFERRED_ERP_NAME: "53e74e931e6f0adaf1e5be4d606d028fcc3e04ee8b066569c2ed2d033d9bbc72",
    LPFILT_ERP_NAME: "228b52ad69b9dc9b88f6b4c0b1d32dc778450e0d9aa32850b9ad9a8a61a8b9fe",
}


def subject1_dir() -> Path:
    return ERPCORE_DIR / "all_data_and_scripts" / SUBJECT_ID


def default_erp_path(*, lpfilt: bool = False) -> Path:
    name = LPFILT_ERP_NAME if lpfilt else PREFERRED_ERP_NAME
    return subject1_dir() / name


def scalp_indices(channel_labels: list[str]) -> list[int]:
    missing = [lab for lab in SCALP_LABELS if lab not in channel_labels]
    if missing:
        raise ValueError(f"Expected scalp labels missing from ERP: {missing}")
    return [channel_labels.index(lab) for lab in SCALP_LABELS]


def window_mask(
    times_ms: NDArray[np.floating],
    *,
    center_ms: float,
    duration_ms: float,
) -> NDArray[np.bool_]:
    times_ms = np.asarray(times_ms, dtype=np.float64)
    half = float(duration_ms) / 2.0
    return (times_ms >= float(center_ms) - half) & (times_ms <= float(center_ms) + half)


def window_slice(
    data: NDArray[np.floating],
    times_ms: NDArray[np.floating],
    *,
    center_ms: float,
    duration_ms: float,
) -> dict[str, Any]:
    mask = window_mask(times_ms, center_ms=center_ms, duration_ms=duration_ms)
    indices = np.flatnonzero(mask)
    if indices.size < 2:
        raise ValueError(
            f"Window center={center_ms} ms, T={duration_ms} ms has {int(indices.size)} samples."
        )
    return {
        "center_ms": float(center_ms),
        "duration_ms": float(duration_ms),
        "t_start_ms": float(center_ms) - float(duration_ms) / 2.0,
        "t_end_ms": float(center_ms) + float(duration_ms) / 2.0,
        "n_samples": int(indices.size),
        "index_start": int(indices[0]),
        "index_end_inclusive": int(indices[-1]),
        "times_ms": np.asarray(times_ms, dtype=np.float64)[mask],
        "data": np.asarray(data, dtype=np.float64)[..., mask],
    }


def sliding_centers_ms(
    times_ms: NDArray[np.floating],
    *,
    duration_ms: float,
    step_ms: float,
) -> NDArray[np.float64]:
    times_ms = np.asarray(times_ms, dtype=np.float64)
    half = float(duration_ms) / 2.0
    step = float(step_ms)
    t0 = float(times_ms[0])
    t1 = float(times_ms[-1])
    raw_min = t0 + half
    raw_max = t1 - half
    c_min = np.ceil(raw_min / step) * step
    c_max = np.floor(raw_max / step) * step
    if c_max < c_min:
        raise ValueError("No fully interior sliding window fits the epoch.")
    n = int(np.round((c_max - c_min) / step)) + 1
    return c_min + step * np.arange(n, dtype=np.float64)


def load_n170_subject1(*, lpfilt: bool = False) -> dict[str, Any]:
    path = default_erp_path(lpfilt=lpfilt)
    if not path.exists():
        raise FileNotFoundError(
            f"ERP CORE subject-1 file not found: {path}. "
            "Run: python -m paper.reproduction download n170"
        )
    payload = load_erplab_erp(path)
    labels = list(payload["channel_labels"])
    keep = scalp_indices(labels)
    data_all = np.asarray(payload["data"], dtype=np.float64)
    data = data_all[:, keep, :][:4]
    digest = sha256_file(path)
    expected = EXPECTED_SHA256[path.name]
    if digest != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: got {digest}")
    n_accepted = np.asarray(payload["n_accepted"], dtype=np.int64)[:4]
    return {
        "path": str(path),
        "lpfilt": bool(lpfilt),
        "sha256": digest,
        "srate_hz": float(payload["srate"]),
        "times_ms": np.asarray(payload["times_ms"], dtype=np.float64),
        "data": data,
        "channel_labels": [labels[i] for i in keep],
        "channel_xyz": np.asarray(payload["channel_xyz"], dtype=np.float64)[keep],
        "bin_descriptions": [str(x) for x in payload["bin_descriptions"]][:4],
        "condition_order": list(CONDITION_ORDER),
        "condition_labels": list(CONDITION_LABELS),
        "n_accepted": n_accepted.tolist(),
        "ica": {
            "paper_claim": "three components ocular+cardiac",
            "erpcore_subject_1_components_1based": list(ERPCORE_SUBJECT1_ICA_COMPONENTS),
            "applied_in_this_script": False,
        },
    }
