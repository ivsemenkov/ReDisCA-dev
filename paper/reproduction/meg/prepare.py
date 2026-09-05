"""Load Kozunov/AIRI MEG run-1 planars and build condition averages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from paper.reproduction.common.hashing import sha256_file
from paper.reproduction.common.meg_io import airi_condition_indices, load_meg_ad_run1, load_spm_trial_labels
from paper.reproduction.common.paths import MEG_DIR
from paper.reproduction.common.preprocessing import AIRI_FILTER, AIRI_SLICE, airi_bandpass_trials
from paper.reproduction.meg.rdms import CONDITION_NAMES

N_PLANARS = 204
N_TIMES_FILE = 1501
N_TRIALS_FILE = 880
N_CONDITIONS = 6
EPOCHS_PER_CONDITION = 80
FS_HZ = 1000.0
TIME_ONSET_MS = -500.0
EXPECTED_SHA256 = {
    "MEG_AD_run1.mat": "0eca2756c9190ce637a3e14abd24e7cf975d758d3ccea03107963e8b5841a4f6",
    "ibfctfprespm8_AD_run1_raw_tsss_mc.mat": "87890337c385e81c718c421d7be35e54423ca9ceb985e047b276b02018334950",
}


@dataclass(frozen=True)
class MegBundle:
    planars: NDArray[np.float64]
    indices: dict[str, NDArray[np.intp]]
    time_ms: NDArray[np.float64]
    fs: float
    source_mat: str
    labels_mat: str
    source_sha256: str
    labels_sha256: str


def meg_paths() -> tuple[Path, Path]:
    data = MEG_DIR / "MEG_AD_run1.mat"
    labels = MEG_DIR / "ibfctfprespm8_AD_run1_raw_tsss_mc.mat"
    if not data.exists() or not labels.exists():
        raise FileNotFoundError(
            "MEG OSF cache missing. Run: python -m paper.reproduction download meg"
        )
    return data, labels


def time_vector_ms(n_times: int = N_TIMES_FILE) -> NDArray[np.float64]:
    return TIME_ONSET_MS + np.arange(n_times, dtype=np.float64) * (1000.0 / FS_HZ)


def load_meg_bundle() -> MegBundle:
    data_path, labels_path = meg_paths()
    data_hash = sha256_file(data_path)
    labels_hash = sha256_file(labels_path)
    if data_hash != EXPECTED_SHA256["MEG_AD_run1.mat"]:
        raise ValueError(f"SHA-256 mismatch for MEG_AD_run1.mat: {data_hash}")
    if labels_hash != EXPECTED_SHA256["ibfctfprespm8_AD_run1_raw_tsss_mc.mat"]:
        raise ValueError(f"SHA-256 mismatch for SPM labels: {labels_hash}")
    payload = load_meg_ad_run1(data_path)
    labels = load_spm_trial_labels(labels_path)
    indices = airi_condition_indices(labels)
    for name, idx in indices.items():
        if idx.size != EPOCHS_PER_CONDITION:
            raise ValueError(f"{name} has {idx.size} trials; expected {EPOCHS_PER_CONDITION}")
    planars = np.asarray(payload["data"][:N_PLANARS], dtype=np.float64)
    return MegBundle(
        planars=planars,
        indices=indices,
        time_ms=time_vector_ms(),
        fs=float(payload["fs"]),
        source_mat=str(data_path),
        labels_mat=str(labels_path),
        source_sha256=data_hash,
        labels_sha256=labels_hash,
    )


def used_trial_indices(indices: dict[str, NDArray[np.intp]]) -> NDArray[np.intp]:
    return np.concatenate([indices[name] for name in CONDITION_NAMES]).astype(np.intp)


def trial_condition_labels(indices: dict[str, NDArray[np.intp]]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    used = used_trial_indices(indices)
    labels = np.concatenate(
        [np.full(indices[name].size, c, dtype=np.intp) for c, name in enumerate(CONDITION_NAMES)]
    )
    return used, labels


def condition_averages(
    planars: NDArray[np.floating],
    indices: dict[str, NDArray[np.intp]],
    *,
    time_slice: slice | None = None,
) -> NDArray[np.float64]:
    data = np.asarray(planars, dtype=np.float64)
    if time_slice is not None:
        data = data[:, time_slice, :]
    n_times = data.shape[1]
    averages = np.empty((N_CONDITIONS, N_PLANARS, n_times), dtype=np.float64)
    for c, name in enumerate(CONDITION_NAMES):
        averages[c] = data[:, :, indices[name]].mean(axis=2)
    return averages


def extract_used_trials(
    planars: NDArray[np.floating],
    indices: dict[str, NDArray[np.intp]],
    *,
    time_slice: slice | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    used, labels = trial_condition_labels(indices)
    data = np.asarray(planars, dtype=np.float64)[:, :, used]
    if time_slice is not None:
        data = data[:, time_slice, :]
    return np.asarray(data, dtype=np.float64), labels


def prepare_candidate(bundle: MegBundle, candidate_id: str) -> dict[str, Any]:
    if candidate_id == "MEG-AIRI":
        filtered = airi_bandpass_trials(bundle.planars, **AIRI_FILTER)
        time_slice = AIRI_SLICE
        time_ms = bundle.time_ms[time_slice]
        planars = filtered
        filter_desc = dict(AIRI_FILTER)
    elif candidate_id == "MEG-PAPER-1501":
        time_slice = slice(None)
        time_ms = bundle.time_ms
        planars = bundle.planars
        filter_desc = None
    elif candidate_id == "MEG-PAPER-1500":
        time_slice = slice(0, 1500)
        time_ms = bundle.time_ms[:1500]
        planars = bundle.planars
        filter_desc = None
    else:
        raise ValueError(f"Unknown MEG candidate {candidate_id!r}")
    averages = condition_averages(planars, bundle.indices, time_slice=time_slice)
    trials, labels = extract_used_trials(planars, bundle.indices, time_slice=time_slice)
    return {
        "candidate_id": candidate_id,
        "averages": averages,
        "used_trials": trials,
        "trial_labels": labels,
        "time_ms": time_ms,
        "planars_for_std": planars,
        "filter": filter_desc,
        "n_samples": int(time_ms.size),
        "window_ms": [float(time_ms[0]), float(time_ms[-1])],
        "source_sha256": bundle.source_sha256,
        "labels_sha256": bundle.labels_sha256,
        "indices": bundle.indices,
    }
