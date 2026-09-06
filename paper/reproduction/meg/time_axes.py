"""Two source-supported MEG time axes for already-computed sample indices.

Do not move significance masks. Only remap stored times / sample indices.

A. PAPER/NOMINAL AXIS
    Dataset metadata epoch axis: ``-500 + sample_index`` ms at 1000 Hz
    for a 1501-sample file (``-500 … +1000`` ms).

B. AIRI-LITERAL-PLOTTING AXIS
    Committed AIRI MATLAB::

        time_axis = linspace(-536, 964, size(mx{1},2));

    For 1501 samples this is exactly ``linspace(-536, 964, 1501)``.
    Step is 1 ms, so this is a uniform −36 ms shift of the nominal axis.

``peak_ms`` stored on temporal intervals is the first True sample of a
boolean significance run (argmax of a True block), **not** a
component-waveform peak.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from paper.reproduction.common.preprocessing import AIRI_SLICE, AIRI_TRANGE_1BASED
from paper.reproduction.meg.prepare import FS_HZ, N_TIMES_FILE, TIME_ONSET_MS

StoredAxis = Literal["nominal_full_epoch", "nominal_analysis_window"]

AIRI_LITERAL_START_MS = -536.0
AIRI_LITERAL_END_MS = 964.0
AIRI_LITERAL_N_SAMPLES = 1501
NOMINAL_TO_AIRI_SHIFT_MS = 36.0


def nominal_axis_ms(
    n_times: int = N_TIMES_FILE,
    *,
    onset_ms: float = TIME_ONSET_MS,
    fs_hz: float = FS_HZ,
) -> NDArray[np.float64]:
    """Paper/nominal physical epoch axis from dataset metadata."""
    return onset_ms + np.arange(int(n_times), dtype=np.float64) * (1000.0 / float(fs_hz))


def airi_literal_plotting_axis_ms(n_times: int = AIRI_LITERAL_N_SAMPLES) -> NDArray[np.float64]:
    """Literal AIRI ``linspace(-536, 964, n_times)``."""
    return np.linspace(
        AIRI_LITERAL_START_MS,
        AIRI_LITERAL_END_MS,
        int(n_times),
        dtype=np.float64,
    )


def sample_index_from_nominal_ms(
    t_ms: float,
    *,
    onset_ms: float = TIME_ONSET_MS,
    fs_hz: float = FS_HZ,
) -> int:
    """Invert ``onset_ms + sample_index * (1000/fs)``."""
    return int(np.rint((float(t_ms) - float(onset_ms)) * float(fs_hz) / 1000.0))


def dual_times_from_sample(
    sample_index: int,
    *,
    n_times: int = AIRI_LITERAL_N_SAMPLES,
) -> dict[str, float]:
    """Map one full-epoch sample index onto both coordinate systems."""
    idx = int(sample_index)
    nominal = nominal_axis_ms(n_times)
    airi = airi_literal_plotting_axis_ms(n_times)
    if idx < 0 or idx >= n_times:
        raise IndexError(f"sample_index {idx} outside [0, {n_times})")
    return {
        "sample_index": idx,
        "nominal_ms": float(nominal[idx]),
        "airi_literal_ms": float(airi[idx]),
    }


def remap_interval(
    interval: dict[str, float],
    *,
    stored_axis: StoredAxis = "nominal_full_epoch",
    n_times_stored: int = N_TIMES_FILE,
    window_start_sample: int = 0,
) -> dict[str, Any]:
    """Remap a stored ``t_start_ms`` / ``t_end_ms`` / ``peak_ms`` interval.

    Stored values are left unchanged. Dual-axis copies are added.
    ``peak_ms`` is the first True sample of the saved run, not a waveform peak.
    """
    out = dict(interval)
    out["stored_axis"] = stored_axis
    out["peak_ms_is_waveform_peak"] = False
    out["peak_ms_meaning"] = (
        "first True sample of the saved significance-interval mask, "
        "not a component-waveform peak"
    )
    for key in ("t_start_ms", "t_end_ms", "peak_ms"):
        if key not in interval or interval[key] is None:
            continue
        t_stored = float(interval[key])
        # Current results store nominal physical times: −500 + full-epoch
        # sample index. AIRI-windowed paper-FWER values are still those
        # physical times (e.g. 99 ms is full-epoch sample 599), not
        # window-local indices. Invert with the global nominal onset.
        if stored_axis not in {"nominal_full_epoch", "nominal_analysis_window"}:
            raise ValueError(f"Unknown stored axis {stored_axis!r}")
        sample = sample_index_from_nominal_ms(t_stored)
        if stored_axis == "nominal_analysis_window" and window_start_sample:
            # Only used if a caller stored window-local times; default path
            # above already matches the committed JSON files.
            _ = n_times_stored
        dual = dual_times_from_sample(sample)
        out[f"{key}_nominal"] = dual["nominal_ms"]
        out[f"{key}_airi_literal"] = dual["airi_literal_ms"]
        out[f"{key}_sample_index"] = dual["sample_index"]
    return out


def remap_interval_list(
    intervals: list[dict[str, float]] | None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    if not intervals:
        return []
    return [remap_interval(item, **kwargs) for item in intervals]


def airi_window_start_sample() -> int:
    """0-based start of AIRI ``trange = 600:1500`` on the 1501-sample epoch."""
    return int(AIRI_TRANGE_1BASED[0] - 1)


def stored_axis_for_block(block: dict[str, Any] | None, *, candidate_id: str | None) -> StoredAxis:
    """How the saved millisecond values were produced."""
    axis = (block or {}).get("time_axis")
    if axis == "full_epoch":
        return "nominal_full_epoch"
    if axis == "analysis_window":
        # MEG-AIRI paper-FWER uses the AIRI slice but still stores nominal
        # physical times of those samples (−500 + full-epoch index).
        return "nominal_full_epoch"
    if candidate_id == "MEG-AIRI" and axis is None:
        return "nominal_full_epoch"
    return "nominal_full_epoch"


def paper_anchors_on_both_axes() -> dict[str, Any]:
    """Published timing anchors expressed on both coordinate systems."""
    anchors = {
        "face_c1_first_onset": 65.0,
        "face_c1_peak": 160.0,
        "face_c1_second": 311.0,
        "face_c2": 218.0,
        "face_c3": 273.0,
        "tool_c1": 210.0,
        "tool_c3": 240.0,
        "meaning_c1": 160.0,
        "meaning_c3_early": 182.0,
        "meaning_c3_late": 675.0,
        "facevstool": 202.0,
    }
    out = {}
    for name, printed_ms in anchors.items():
        # Paper numbers are treated as the printed / figure-annotation values.
        # If those annotations used the AIRI plotting axis, the same sample
        # on the nominal axis is printed_ms + 36.
        sample_if_airi = sample_index_from_nominal_ms(
            printed_ms, onset_ms=AIRI_LITERAL_START_MS
        )
        sample_if_nominal = sample_index_from_nominal_ms(printed_ms)
        out[name] = {
            "printed_ms": printed_ms,
            "if_printed_is_nominal": dual_times_from_sample(sample_if_nominal),
            "if_printed_is_airi_literal": dual_times_from_sample(sample_if_airi),
        }
    out["axis_definitions"] = {
        "nominal": "TIME_ONSET_MS + sample_index at 1000 Hz; −500…+1000 for 1501 samples",
        "airi_literal": "linspace(-536, 964, 1501)",
        "shift_ms_airi_minus_nominal": -NOMINAL_TO_AIRI_SHIFT_MS,
    }
    out["airi_window"] = {
        "trange_1based": list(AIRI_TRANGE_1BASED),
        "slice_0based_start": AIRI_SLICE.start,
        "slice_0based_stop": AIRI_SLICE.stop,
        "n_window_samples": AIRI_SLICE.stop - AIRI_SLICE.start,
    }
    return out
