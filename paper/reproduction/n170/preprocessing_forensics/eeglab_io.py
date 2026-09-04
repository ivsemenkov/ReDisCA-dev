"""Read EEGLAB ``.set`` metadata without requiring MATLAB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    array = np.asarray(value)
    if array.size == 0:
        return []
    return array.ravel().tolist()


def _channel_labels(eeg: Any) -> list[str]:
    chanlocs = getattr(eeg, "chanlocs", None)
    if chanlocs is None:
        return []
    items = chanlocs if np.ndim(chanlocs) else [chanlocs]
    labels: list[str] = []
    for channel in items:
        labels.append(str(getattr(channel, "labels", "")))
    return labels


def _ica_array(eeg: Any, name: str) -> dict[str, Any]:
    raw = getattr(eeg, name, None)
    array = np.asarray(raw) if raw is not None else np.asarray([])
    if array.size == 0:
        return {"present": False, "shape": list(array.shape), "dtype": str(array.dtype)}
    return {
        "present": True,
        "shape": [int(x) for x in array.shape],
        "dtype": str(array.dtype),
        "n_finite": int(np.isfinite(array).sum()) if np.issubdtype(array.dtype, np.number) else None,
    }


def load_eeglab_set(path: Path) -> dict[str, Any]:
    """Return channel / ICA / epoch metadata from an EEGLAB ``.set`` file."""
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "EEG" not in payload:
        raise ValueError(f"{path} has no EEG struct")
    eeg = payload["EEG"]
    labels = _channel_labels(eeg)
    ica_channels = np.asarray(getattr(eeg, "icachansind", []), dtype=np.int64).ravel()
    reject = getattr(eeg, "reject", None)
    gcompreject = None
    if reject is not None:
        gcompreject = _as_list(getattr(reject, "gcompreject", []))
    icaweights = np.asarray(getattr(eeg, "icaweights", []))
    return {
        "path": str(path),
        "name": path.name,
        "setname": str(getattr(eeg, "setname", "")),
        "nbchan": int(getattr(eeg, "nbchan", len(labels))),
        "pnts": int(getattr(eeg, "pnts", 0)),
        "trials": int(getattr(eeg, "trials", 0)),
        "srate_hz": float(getattr(eeg, "srate", float("nan"))),
        "channel_labels": labels,
        "has_p9": "P9" in labels,
        "has_p10": "P10" in labels,
        "icaweights": _ica_array(eeg, "icaweights"),
        "icasphere": _ica_array(eeg, "icasphere"),
        "icawinv": _ica_array(eeg, "icawinv"),
        "icachansind_1based": [int(x) for x in ica_channels.tolist()] if ica_channels.size else [],
        "n_ica_components": int(icaweights.shape[0]) if icaweights.ndim == 2 and icaweights.size else 0,
        "n_ica_channels": int(icaweights.shape[1]) if icaweights.ndim == 2 and icaweights.size else 0,
        "gcompreject": gcompreject,
        "datfile": str(getattr(eeg, "datfile", "") or getattr(eeg, "data", "")),
    }


def ica_weights(path: Path) -> np.ndarray:
    eeg = loadmat(path, squeeze_me=True, struct_as_record=False)["EEG"]
    return np.asarray(getattr(eeg, "icaweights", []), dtype=np.float64)


def ica_mixing(path: Path) -> np.ndarray:
    """EEGLAB ``icawinv`` (channels × components) when present."""
    eeg = loadmat(path, squeeze_me=True, struct_as_record=False)["EEG"]
    return np.asarray(getattr(eeg, "icawinv", []), dtype=np.float64)


def removed_ica_components_1based(weighted_path: Path, corrected_path: Path) -> list[int]:
    """Match remaining ``icaweights`` rows to the pre-``pop_subcomp`` matrix.

    EEGLAB ``pop_subcomp`` drops rows of ``icaweights``. The official subject-1
    files drop exactly the rows corresponding to components 2 and 7 (1-based).
    """
    before = ica_weights(weighted_path)
    after = ica_weights(corrected_path)
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("ICA weight matrices are not 2-D")
    if after.shape[1] != before.shape[1]:
        raise ValueError(
            f"ICA channel axis changed ({before.shape} -> {after.shape}); "
            "cannot identify removed component indices from weights alone."
        )
    kept: list[int] = []
    for row in after:
        distances = np.linalg.norm(before - row[None, :], axis=1)
        kept.append(int(np.argmin(distances)) + 1)
    original = set(range(1, before.shape[0] + 1))
    return sorted(original.difference(kept))
