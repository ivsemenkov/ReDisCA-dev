"""Load ERP CORE subject-``1`` averages and apply documented N170 choices.

Data
----
Preferred file (manifest): ``1_N170_erp_ar.erp`` — official ERP CORE averaged
ERPs after ICA correction, epoching [−200, +800] ms, artifact rejection, and
averaging of *good* trials. Bins are correct-response Faces / Cars /
Scrambled Faces / Scrambled Cars. Sampling rate 256 Hz.

ICA (D11)
---------
The paper says “three ICA components corresponding to ocular and cardiac
artifacts”. ERP CORE subject ``"1"`` removes components **2 and 7 only**
(``ICA_Components_N170.xlsx``). Those components are already removed in the
precomputed ``.erp`` file. This loader does not re-run ICA and does not invent
a third component.

Channel selection
-----------------
The ERPLAB file has 35 channels. The first 28 are scalp EEG (10-10 labels
FP1…O2). Channels 29–35 are EOG electrodes and bipolar HEOG/VEOG
(corrected and uncorrected). P9 and P10 from the original 33 EEG-typed
montage were already dropped in ERP CORE Script 1
(``Rereference_Add_Uncorrected_Bipolars_N170.txt``) and are not in the ERP.

Default analysis set: the 28 scalp channels. EOG / corr / uncorr channels are
dropped. Labels are recorded in the results JSON.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "paper" / "reproduction") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "paper" / "reproduction"))

from common.erp_io import load_erplab_erp
from common.paths import ERPCORE_DIR, UPSTREAM_DIR

from rdms import CONDITION_LABELS, CONDITION_ORDER

SUBJECT_ID = "1"
PREFERRED_ERP_NAME = "1_N170_erp_ar.erp"
LPFILT_ERP_NAME = "1_N170_erp_ar_lpfilt.erp"
AR_CSV_NAME = "1_AR_Percentages_N170.csv"

# 28 scalp EEG in the ERPLAB file order (indices 0..27).
SCALP_LABELS: tuple[str, ...] = (
    "FP1",
    "F3",
    "F7",
    "FC3",
    "C3",
    "C5",
    "P3",
    "P7",
    "PO7",
    "PO3",
    "O1",
    "Oz",
    "Pz",
    "CPz",
    "FP2",
    "Fz",
    "F4",
    "F8",
    "FC4",
    "FCz",
    "Cz",
    "C4",
    "C6",
    "P4",
    "P8",
    "PO8",
    "PO4",
    "O2",
)

EOG_AND_BIPOLAR_LABELS: tuple[str, ...] = (
    "HEOG_left",
    "HEOG_right",
    "VEOG_lower",
    "(corr) HEOG",
    "(corr) VEOG",
    "(uncorr) HEOG",
    "(uncorr) VEOG",
)

OCCIPITAL_LABELS: tuple[str, ...] = ("PO7", "PO3", "O1", "Oz", "PO8", "PO4", "O2", "P7", "P8")

# ERP CORE subject-1 ICA list (1-based component indices). Not re-applied here.
ERPCORE_SUBJECT1_ICA_COMPONENTS: tuple[int, ...] = (2, 7)
PAPER_ICA_CLAIM = "three components ocular+cardiac"

# Paper windows. Sliding step is NOT in the paper (choose and document).
MEANING_DURATION_MS = 150.0
FACE_DURATION_MS = 100.0
FACE_CENTER_MS = 200.0
CAR_DURATION_MS = 100.0  # duration not restated; only other real-data T is 100 ms
CAR_CENTER_MS = 170.0
DEFAULT_SLIDING_STEP_MS = 25.0  # not specified in the paper
FIG8_CENTERS_MS: tuple[float, float, float] = (375.0, 400.0, 425.0)

ERPCORE_SCRIPTS_COMMIT = "c18b43d70d791ca914d90410afe4ff06d6f7f429"


def subject1_dir() -> Path:
    return ERPCORE_DIR / "all_data_and_scripts" / SUBJECT_ID


def default_erp_path(*, lpfilt: bool = False) -> Path:
    name = LPFILT_ERP_NAME if lpfilt else PREFERRED_ERP_NAME
    return subject1_dir() / name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def scalp_indices(channel_labels: list[str]) -> list[int]:
    missing = [lab for lab in SCALP_LABELS if lab not in channel_labels]
    if missing:
        raise ValueError(f"Expected scalp labels missing from ERP: {missing}")
    return [channel_labels.index(lab) for lab in SCALP_LABELS]


def dropped_channel_report(channel_labels: list[str]) -> dict[str, Any]:
    dropped = [lab for lab in channel_labels if lab not in SCALP_LABELS]
    unexpected = [lab for lab in dropped if lab not in EOG_AND_BIPOLAR_LABELS]
    return {
        "kept_scalp_labels": list(SCALP_LABELS),
        "n_kept": len(SCALP_LABELS),
        "dropped_labels": dropped,
        "dropped_expected_eog_or_bipolar": list(EOG_AND_BIPOLAR_LABELS),
        "dropped_unexpected": unexpected,
        "p9_p10_note": (
            "P9 and P10 are not in the 35-channel ERP. ERP CORE Script 1 "
            "skips original ch9=P9 and ch27=P10 when building the 28 "
            "average-referenced scalp channels."
        ),
    }


def window_mask(
    times_ms: NDArray[np.floating],
    *,
    center_ms: float,
    duration_ms: float,
) -> NDArray[np.bool_]:
    """Inclusive samples with ``|t - center| <= duration/2``.

    At 256 Hz, 100 ms covers 26 samples and 150 ms covers 38 samples for the
    official ERP CORE time axis (dt = 1000/256 ms).
    """
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
            f"Window center={center_ms} ms, T={duration_ms} ms has "
            f"{int(indices.size)} samples (need >= 2)."
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
        "mask": mask,
    }


def sliding_centers_ms(
    times_ms: NDArray[np.floating],
    *,
    duration_ms: float,
    step_ms: float = DEFAULT_SLIDING_STEP_MS,
) -> NDArray[np.float64]:
    """Centers on a ``step_ms`` grid whose windows overlap the recorded epoch.

    The sliding *step* is not printed in the paper. Default 25 ms is a
    documented choice, not a paper value.
    """
    times_ms = np.asarray(times_ms, dtype=np.float64)
    half = float(duration_ms) / 2.0
    step = float(step_ms)
    t0 = float(times_ms[0])
    t1 = float(times_ms[-1])
    # Earliest center on the step grid such that the window still contains samples.
    raw_min = t0 + half
    raw_max = t1 - half
    c_min = np.ceil(raw_min / step) * step
    c_max = np.floor(raw_max / step) * step
    if c_max < c_min:
        raise ValueError("No fully interior sliding window fits the epoch.")
    n = int(np.round((c_max - c_min) / step)) + 1
    centers = c_min + step * np.arange(n, dtype=np.float64)
    return centers


def load_n170_subject1(*, lpfilt: bool = False) -> dict[str, Any]:
    """Return scalp condition-averages ready for ``ReDisCA.fit``."""
    path = default_erp_path(lpfilt=lpfilt)
    if not path.exists():
        raise FileNotFoundError(
            f"ERP CORE subject-1 file not found: {path}. "
            "Expected gitignored cache under .reproduction_data/erpcore/."
        )
    payload = load_erplab_erp(path)
    labels = list(payload["channel_labels"])
    keep = scalp_indices(labels)
    data_all = np.asarray(payload["data"], dtype=np.float64)
    data = data_all[:, keep, :]
    xyz = np.asarray(payload["channel_xyz"], dtype=np.float64)[keep]
    bin_names = [str(x) for x in payload["bin_descriptions"]]
    expected_substrings = ("face", "car", "scrambled face", "scrambled car")
    joined = " | ".join(n.lower() for n in bin_names[:4])
    for needle in expected_substrings:
        if needle not in joined:
            raise ValueError(f"Unexpected ERPLAB bins {bin_names!r}")
    if data.shape[0] < 4:
        raise ValueError(f"Expected 4 condition bins, got {data.shape[0]}")
    data = data[:4]
    n_accepted = np.asarray(payload["n_accepted"], dtype=np.int64)[:4]
    ar_csv = subject1_dir() / AR_CSV_NAME
    scripts_root = UPSTREAM_DIR / "ERP_CORE"
    return {
        "path": str(path),
        "lpfilt": bool(lpfilt),
        "sha256": sha256_file(path),
        "erpname": payload["erpname"],
        "srate_hz": float(payload["srate"]),
        "times_ms": np.asarray(payload["times_ms"], dtype=np.float64),
        "xmin_s": float(payload["xmin"]),
        "xmax_s": float(payload["xmax"]),
        "isfilt": payload.get("isfilt"),
        "data": data,
        "data_all_channels": data_all[:4],
        "channel_labels": [labels[i] for i in keep],
        "channel_xyz": xyz,
        "channel_indices_in_erp": keep,
        "all_channel_labels": labels,
        "channel_selection": dropped_channel_report(labels),
        "bin_descriptions": bin_names[:4],
        "condition_order": list(CONDITION_ORDER),
        "condition_labels": list(CONDITION_LABELS),
        "n_accepted": n_accepted.tolist(),
        "n_accepted_total": int(np.sum(n_accepted)),
        "ar_csv": str(ar_csv) if ar_csv.exists() else None,
        "ica": {
            "paper_claim": PAPER_ICA_CLAIM,
            "erpcore_subject_1_components_1based": list(ERPCORE_SUBJECT1_ICA_COMPONENTS),
            "applied_in_this_script": False,
            "note": (
                "ICA correction is already present in the official precomputed "
                "ERP. Subject 1 list is components 2 and 7 (ocular), not three "
                "ocular+cardiac components (discrepancy D11)."
            ),
        },
        "erp_core_scripts_commit": ERPCORE_SCRIPTS_COMMIT,
        "erp_core_scripts_root": str(scripts_root) if scripts_root.exists() else None,
        "epoch_note": (
            "Paper: epoch [-200, 800] ms at 256 Hz. Official ERPLAB times are "
            f"{float(payload['times_ms'][0]):.5f} … "
            f"{float(payload['times_ms'][-1]):.5f} ms "
            f"(n={int(np.asarray(payload['times_ms']).size)}, "
            "dt=1000/256 ms) after 26 ms event shift + resample."
        ),
    }
