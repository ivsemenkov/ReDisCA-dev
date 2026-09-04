"""Reuse existing MEG airi_executable PRIMARY p-values (do not rerun B=1000).

Does not import ``redisca``.
"""

from __future__ import annotations

from typing import Any

from common.hashing import read_json

from meg.historical_candidate.freeze import AIRI_EXECUTABLE_DIR, FROZEN_B, RDM_ORDER

_FILES = {
    "face": "fig13_face.json",
    "tool": "fig14_tool.json",
    "meaning": "fig15_meaning.json",
    "facevstool": "airi_executable_meg_facevstool.json",
}


def _as_float_list(values: Any, n: int = 8) -> list[float]:
    out = [float(v) for v in list(values)]
    return out[:n]


def verify_airi_executable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Check that stored AIRI-executable JSON is directed+cov+random-phase B=1000."""
    inf = payload.get("inference_component") or {}
    bandpass = payload.get("bandpass") or {}
    checks = {
        "path_label_airi_executable": payload.get("path_label") == "airi_executable",
        "pairs_airi_directed": str(payload.get("pairs", "")).startswith("airi_directed"),
        "pair_matrix_matlab_cov": payload.get("pair_matrix") == "matlab_cov",
        "inference_spoc_random_phase": inf.get("name") == "spoc_random_phase",
        "B_1000": int(inf.get("B", -1)) == FROZEN_B and int(inf.get("used_B", -1)) == FROZEN_B,
        "reduced_B_false": inf.get("reduced_B") is False,
        "n_planars_204": int(payload.get("n_planars", -1)) == 204,
        "n_samples_901_airi_crop": int(payload.get("n_samples", -1)) == 901,
        "bandpass_0p25_20": (
            bandpass.get("low_hz") == 0.25
            and bandpass.get("high_hz") == 20.0
            and int(bandpass.get("butter_order", -1)) == 3
        ),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "window_ms": payload.get("window_ms"),
        "n_pairs": payload.get("n_pairs"),
        "rank": payload.get("rank"),
    }


def reuse_airi_executable(*, n_head: int = 8) -> dict[str, Any]:
    """Copy PRIMARY p-values and λ from existing JSON. Never recomputes B=1000."""
    rdms: dict[str, Any] = {}
    all_ok = True
    for name in RDM_ORDER:
        path = AIRI_EXECUTABLE_DIR / _FILES[name]
        payload = read_json(path)
        verification = verify_airi_executable_payload(payload)
        p_head = _as_float_list(payload["p_component_spoc_random_phase"], n_head)
        lam = _as_float_list(payload["eigenvalues"], n_head)
        n_sig = int(sum(p < 0.05 for p in p_head))
        # Count over the stored head (8); also first-three for the paper count.
        n_sig_first3 = int(sum(p < 0.05 for p in p_head[:3]))
        peak = None
        peaks = payload.get("peaks") or []
        if peaks:
            peak = peaks[0].get("contrast_peak_ms")
            class1_peak = peaks[0].get("class1_peak_ms")
        else:
            class1_peak = None
        rdms[name] = {
            "item_id": payload.get("item_id"),
            "source_file": _FILES[name],
            "recomputed_B1000": False,
            "verification": verification,
            "rank": payload.get("rank"),
            "eigenvalues_head": lam,
            "p_spoc_head": p_head,
            "n_components_p_lt_0.05_in_head": n_sig,
            "n_first3_p_lt_0.05": n_sig_first3,
            "empirical_rdm_pearson_comp1": (
                (payload.get("empirical_rdm_pearson") or [{}])[0].get(
                    "pearson_unique_triangle"
                )
            ),
            "contrast_peak_ms_comp1": peak,
            "class1_peak_ms_comp1": class1_peak,
            "window_ms": payload.get("window_ms"),
            "n_samples": payload.get("n_samples"),
            "bandpass": payload.get("bandpass"),
            "pairs": payload.get("pairs"),
            "pair_matrix": payload.get("pair_matrix"),
        }
        all_ok = all_ok and bool(verification["ok"])
    summary_path = AIRI_EXECUTABLE_DIR / "summary.json"
    stored_summary = read_json(summary_path) if summary_path.exists() else None
    return {
        "path_label": "airi_executable_reuse",
        "recomputed": False,
        "verified_directed_cov_random_phase_B1000": all_ok,
        "note": (
            "Existing paper/results/meg/airi_executable/ is the frozen "
            "estimator PLUS AIRI MEG extras (99–999 ms crop, butter 0.25–20 Hz "
            "filtfilt). PRIMARY p-values and λ are copied, not recomputed."
        ),
        "rdms": rdms,
        "stored_summary": stored_summary,
        "imports_redisca": False,
        "matlab": None,
    }
