"""Summarize Stage A JSON results against published anchors.

Never treats ``QUICK_NONREPRO_*`` or leftover ``n_mc < 100`` files as
reproduction results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper.reproduction.common.hashing import read_json, write_json
from paper.reproduction.common.paths import RESULTS_ROOT


def _is_reproduction_payload(payload: dict[str, Any]) -> bool:
    if payload.get("quick_non_reproduction"):
        return False
    n_mc = payload.get("n_mc")
    if n_mc is not None and int(n_mc) < 100:
        return False
    return True


def _n170_rows() -> list[dict[str, Any]]:
    rows = []
    root = RESULTS_ROOT / "n170"
    if not root.exists():
        return rows
    for candidate_dir in sorted(root.iterdir()):
        if not candidate_dir.is_dir():
            continue
        for path in sorted(candidate_dir.glob("seed*.json")):
            payload = read_json(path)
            if not _is_reproduction_payload(payload):
                continue
            face = payload["face_c200"]
            car170 = payload["car"]["170.0"]
            car200 = payload["car"]["200.0"]
            meaning25 = payload["meaning"].get("25.0", {})
            meaning_fine = payload["meaning"].get("3.90625", {})
            near = meaning25.get("near_400_ms") or {}
            rows.append(
                {
                    "candidate": payload["candidate_id"],
                    "seed": payload["seed"],
                    "face_lambda": face["eigenvalues"][0],
                    "face_p": face["p_random_phase"][0],
                    "face_rdm_corr": face["components"][0]["rdm_corr"],
                    "face_maxabs": face["components"][0]["pattern"]["maxabs_channel"],
                    "face_secondary_p_maxabs": (face.get("secondary_condition_labels") or {}).get(
                        "p_maxabs", [None]
                    )[0],
                    "car170_lambda1": car170["eigenvalues"][0],
                    "car170_p1": car170["p_random_phase"][0],
                    "car170_lambda2": car170["eigenvalues"][1] if len(car170["eigenvalues"]) > 1 else None,
                    "car170_p2": car170["p_random_phase"][1] if len(car170["p_random_phase"]) > 1 else None,
                    "car170_rdm1": car170["components"][0]["rdm_corr"],
                    "car170_rdm2": car170["components"][1]["rdm_corr"] if len(car170["components"]) > 1 else None,
                    "car200_lambda1": car200["eigenvalues"][0],
                    "car200_p1": car200["p_random_phase"][0],
                    "car200_lambda2": car200["eigenvalues"][1] if len(car200["eigenvalues"]) > 1 else None,
                    "car200_p2": car200["p_random_phase"][1] if len(car200["p_random_phase"]) > 1 else None,
                    "car200_rdm1": car200["components"][0]["rdm_corr"],
                    "meaning_sig_centers_25ms": meaning25.get("significant_centers_ms"),
                    "meaning_p1_375": (near.get("375.0") or {}).get("p1"),
                    "meaning_p1_400": (near.get("400.0") or {}).get("p1"),
                    "meaning_p1_425": (near.get("425.0") or {}).get("p1"),
                    "meaning_fine_sig_centers": meaning_fine.get("significant_centers_ms"),
                    "deltas": payload.get("deltas"),
                }
            )
    return rows


def _first_intervals(intervals: list[dict[str, float]] | None, n: int = 3) -> list[dict[str, float]]:
    if not intervals:
        return []
    return list(intervals[:n])


def _meg_rows() -> list[dict[str, Any]]:
    rows = []
    root = RESULTS_ROOT / "meg"
    if not root.exists():
        return rows
    for candidate_dir in sorted(root.iterdir()):
        if not candidate_dir.is_dir():
            continue
        for path in sorted(candidate_dir.glob("seed*.json")):
            payload = read_json(path)
            if not _is_reproduction_payload(payload):
                continue
            for rdm_key, rdm in payload.get("rdms", {}).items():
                airi = rdm.get("temporal_airi") or {}
                paper = rdm.get("temporal_paper_fwer") or {}
                airi_pplus = airi.get("intervals_pplus") or []
                airi_pminus = airi.get("intervals_pminus") or []
                paper_iv = paper.get("intervals") or []
                rows.append(
                    {
                        "candidate": payload["candidate_id"],
                        "seed": payload["seed"],
                        "rdm": rdm_key,
                        "n_samples": payload.get("n_samples"),
                        "window_ms": payload.get("window_ms"),
                        "lambdas": rdm.get("eigenvalues", [])[:6],
                        "p_random_phase": rdm.get("p_random_phase", [])[:6],
                        "n_sig": rdm.get("n_significant_p05"),
                        "rdm_corr": [c.get("rdm_corr") for c in rdm.get("components", [])[:4]],
                        "secondary_p_maxabs": (rdm.get("secondary_condition_labels") or {}).get(
                            "p_maxabs", []
                        )[:4],
                        "airi_time_axis": airi.get("time_axis"),
                        "airi_pplus_c1": _first_intervals(airi_pplus[0] if airi_pplus else None),
                        "airi_pminus_c1": _first_intervals(airi_pminus[0] if airi_pminus else None),
                        "airi_pplus_c2": _first_intervals(airi_pplus[1] if len(airi_pplus) > 1 else None),
                        "airi_pminus_c2": _first_intervals(airi_pminus[1] if len(airi_pminus) > 1 else None),
                        "airi_pplus_c3": _first_intervals(airi_pplus[2] if len(airi_pplus) > 2 else None),
                        "airi_pminus_c3": _first_intervals(airi_pminus[2] if len(airi_pminus) > 2 else None),
                        "paper_c1": _first_intervals(paper_iv[0] if paper_iv else None),
                        "paper_c2": _first_intervals(paper_iv[1] if len(paper_iv) > 1 else None),
                        "paper_c3": _first_intervals(paper_iv[2] if len(paper_iv) > 2 else None),
                        "paper_qualitative": rdm.get("paper_qualitative"),
                    }
                )
    return rows


def _sim_rows() -> list[dict[str, Any]]:
    rows = []
    root = RESULTS_ROOT / "simulations"
    if not root.exists():
        return rows
    for candidate_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(candidate_dir.rglob("*.json")):
            if path.name.startswith("QUICK_NONREPRO_") or "_quick_leftovers" in path.parts:
                continue
            payload = read_json(path)
            if not _is_reproduction_payload(payload):
                continue
            row: dict[str, Any] = {
                "candidate": payload.get("candidate_id", candidate_dir.name),
                "experiment": payload.get("experiment"),
                "seed": payload.get("seed"),
                "snr": payload.get("snr"),
                "n_mc": payload.get("n_mc"),
                "path": str(path.relative_to(RESULTS_ROOT)),
            }
            if "roc" in payload:
                row["roc"] = payload.get("roc_summary") or payload.get("roc")
            if "by_C" in payload:
                row["by_C"] = {
                    c: {
                        "mean_median_error_cm": block.get("mean_median_error_cm"),
                        "frac_error_lt_1cm": block.get("frac_error_lt_1cm"),
                        "mean_rdm_corr": block.get("mean_rdm_corr"),
                        "mean_pattern_corr": block.get("mean_pattern_corr"),
                        "mean_weight_corr": block.get("mean_weight_corr"),
                    }
                    for c, block in payload["by_C"].items()
                }
            for key in (
                "auc",
                "tpr_at_low_fpr",
                "methods",
                "redisca_auc",
                "summary",
            ):
                if key in payload:
                    row[key] = payload[key]
            rows.append(row)
    return rows


def _source_rows() -> list[dict[str, Any]]:
    rows = []
    root = RESULTS_ROOT / "source_localization"
    if not root.exists():
        return rows
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("QUICK_NONREPRO_"):
            continue
        payload = read_json(path)
        if not _is_reproduction_payload(payload):
            continue
        rows.append(
            {
                "candidate": payload.get("candidate_id"),
                "seed": payload.get("seed"),
                "peak_vertex": payload.get("peak_vertex"),
                "peak_atlas": payload.get("peak_atlas"),
                "top10": payload.get("top10"),
                "choice_note": payload.get("choice_note"),
                "author_saved": (payload.get("author_saved") or {}).get("peak_atlas"),
                "local_meg_airi": (payload.get("local_meg_airi") or {}).get("peak_atlas"),
            }
        )
    return rows


def write_summary() -> Path:
    payload = {
        "n170": _n170_rows(),
        "meg": _meg_rows(),
        "simulations": _sim_rows(),
        "source_localization": _source_rows(),
        "paper_anchors": {
            "face": {"lambda": 0.87209, "p": 0.0, "rdm_corr": 0.81556, "caption_corr": 0.82},
            "car": {
                "c1": {"lambda": 0.91639, "p": 0.0, "rdm_corr": 0.99074},
                "c2": {"lambda": 0.77036, "p": 0.009, "rdm_corr": 0.93002},
            },
            "meg_timing": {
                "face_c1": "first interval ~65 ms, peak ~160 ms, second ~311 ms",
                "face_c2": "~218 ms",
                "face_c3": "~273 ms",
                "tool_c1": "~210 ms",
                "tool_c3": "~240 ms",
                "meaning_c1": "~160 ms",
                "meaning_c3": "early ~182 ms and late ~675 ms",
                "facevstool": "first major tool-vs-face difference ~202 ms",
            },
        },
    }
    dest = RESULTS_ROOT / "stage_a_summary.json"
    write_json(dest, payload)
    return dest


if __name__ == "__main__":
    print(write_summary())
