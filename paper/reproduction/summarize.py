"""Summarize Stage A JSON results against published anchors.

Never treats ``QUICK_NONREPRO_*`` or leftover ``n_mc < 100`` files as
reproduction results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper.reproduction.common.constants import MASTER_SEEDS
from paper.reproduction.common.hashing import read_json, write_json
from paper.reproduction.common.paths import RESULTS_ROOT
from paper.reproduction.simulations.config import REVIEW_ADDED_SIM_CANDIDATES


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
            companions = _load_temporal_companion(candidate_dir, payload["seed"])
            for rdm_key, rdm in payload.get("rdms", {}).items():
                companion = (companions.get("rdms") or {}).get(rdm_key) or {}
                lit = rdm.get("temporal_airi_literal") or companion.get("temporal_airi_literal")
                cor = (
                    rdm.get("temporal_airi_corrected")
                    or companion.get("temporal_airi_corrected")
                    or rdm.get("temporal_airi")
                )
                legacy_only = (
                    rdm.get("temporal_airi") is not None
                    and rdm.get("temporal_airi_literal") is None
                    and companion.get("temporal_airi_literal") is None
                )
                paper = rdm.get("temporal_paper_fwer") or companion.get("temporal_paper_fwer") or {}
                n_sig = rdm.get("n_significant_p05")
                rows.append(
                    {
                        "candidate": payload["candidate_id"],
                        "seed": payload["seed"],
                        "rdm": rdm_key,
                        "n_samples": payload.get("n_samples"),
                        "window_ms": payload.get("window_ms"),
                        "lambdas": rdm.get("eigenvalues", [])[:6],
                        "p_random_phase": rdm.get("p_random_phase", [])[:6],
                        "n_sig": n_sig,
                        "n_sig_ge_3_compatible_with_first_three_significant": (
                            None if n_sig is None else int(n_sig) >= 3
                        ),
                        "rdm_corr": [c.get("rdm_corr") for c in rdm.get("components", [])[:4]],
                        "secondary_p_maxabs": (rdm.get("secondary_condition_labels") or {}).get(
                            "p_maxabs", []
                        )[:4],
                        "temporal_airi_in_seed_json_is_corrected_not_literal": legacy_only,
                        **_temporal_block("airi_corrected", cor),
                        **_temporal_block("airi_literal", lit),
                        "paper_c1": _first_intervals((paper.get("intervals") or [None])[0]),
                        "paper_c2": _first_intervals(
                            (paper.get("intervals") or [None, None])[1]
                            if len(paper.get("intervals") or []) > 1
                            else None
                        ),
                        "paper_c3": _first_intervals(
                            (paper.get("intervals") or [None, None, None])[2]
                            if len(paper.get("intervals") or []) > 2
                            else None
                        ),
                        "paper_qualitative": rdm.get("paper_qualitative"),
                    }
                )
    return rows


def _temporal_block(prefix: str, block: dict[str, Any] | None) -> dict[str, Any]:
    block = block or {}
    pplus = block.get("intervals_pplus") or []
    pminus = block.get("intervals_pminus") or []
    return {
        f"{prefix}_indexing": block.get("indexing"),
        f"{prefix}_role": block.get("role"),
        f"{prefix}_time_axis": block.get("time_axis"),
        f"{prefix}_Nmc": block.get("Nmc"),
        f"{prefix}_pplus_c1": _first_intervals(pplus[0] if pplus else None),
        f"{prefix}_pminus_c1": _first_intervals(pminus[0] if pminus else None),
        f"{prefix}_pplus_c2": _first_intervals(pplus[1] if len(pplus) > 1 else None),
        f"{prefix}_pminus_c2": _first_intervals(pminus[1] if len(pminus) > 1 else None),
        f"{prefix}_pplus_c3": _first_intervals(pplus[2] if len(pplus) > 2 else None),
        f"{prefix}_pminus_c3": _first_intervals(pminus[2] if len(pminus) > 2 else None),
    }


def _load_temporal_companion(candidate_dir: Path, seed: int) -> dict[str, Any]:
    path = candidate_dir / f"temporal_airi_seed{seed}.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    if payload.get("quick_non_reproduction"):
        return {}
    return payload


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
                "added_after_stage_a_review": payload.get("candidate_id")
                in REVIEW_ADDED_SIM_CANDIDATES,
                "generation_modes": payload.get("generation_modes"),
                "generated_C": payload.get("generated_C"),
                "median_loc_error_cm": payload.get("median_loc_error_cm"),
                "mean_loc_error_cm": payload.get("mean_loc_error_cm"),
                "redisca_roc": payload.get("redisca_roc"),
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
                "selection_rule": payload.get("selection_rule"),
                "meg_candidate": payload.get("meg_candidate"),
                "seed": payload.get("seed"),
                "chosen_components": payload.get("chosen_components"),
                "p_values_selected": payload.get("p_values_selected"),
                "n_significant_p05": payload.get("n_significant_p05"),
                "subspace_dim": payload.get("subspace_dim"),
                "patterns_hash": payload.get("patterns_hash"),
                "component_selection_identical_across_registered_seeds": payload.get(
                    "component_selection_identical_across_registered_seeds"
                ),
                "patterns_deterministic_across_registered_seeds": payload.get(
                    "patterns_deterministic_across_registered_seeds"
                ),
                "peak_vertex": payload.get("peak_vertex"),
                "peak_subcorr": payload.get("peak_subcorr"),
                "peak_atlas": payload.get("peak_atlas"),
                "top10": payload.get("top10"),
                "choice_note": payload.get("choice_note"),
                "paper_claimed": payload.get("paper_claimed"),
                "author_saved": (payload.get("author_saved") or {}).get("peak_atlas"),
                "local_meg_airi": (payload.get("local_meg_airi") or {}).get("peak_atlas"),
            }
        )
    return rows


def _required_sim_jobs() -> list[tuple[str, str, float, int]]:
    """(candidate, experiment_key, snr, seed) that must exist before a verdict."""
    seeds = list(MASTER_SEEDS)
    jobs: list[tuple[str, str, float, int]] = []
    for seed in seeds:
        for snr in (0.2, 0.1):
            jobs.append(("SIM-P1", "fig4", snr, seed))
        for snr in (0.4, 0.2):
            jobs.append(("SIM-P1", "fig5_fig6", snr, seed))
        for snr in (0.2, 0.1):
            jobs.append(("SIM-P2", "fig4", snr, seed))
        jobs.append(("SIM-P3", "fig4", 0.1, seed))
        for cand in ("SIM-P4", "SIM-P5", "SIM-P6", "SIM-P7", "SIM-R1"):
            for snr in (0.2, 0.1):
                jobs.append((cand, "fig4", snr, seed))
            for snr in (0.4, 0.2):
                jobs.append((cand, "fig5_fig6", snr, seed))
        for snr in (0.4, 0.2):
            jobs.append(("SIM-P8", "fig5_fig6", snr, seed))
    return jobs


def _completeness(sim_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], meg_rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = set()
    for row in sim_rows:
        exp = "fig4" if (row.get("experiment") or "").startswith("fig4") else (
            "fig5_fig6" if (row.get("experiment") or "").startswith("fig5") else None
        )
        if exp is None:
            continue
        present.add((row.get("candidate"), exp, float(row.get("snr")), int(row.get("seed"))))
    missing_sim = [
        {"candidate": c, "experiment": e, "snr": s, "seed": seed}
        for c, e, s, seed in _required_sim_jobs()
        if (c, e, s, seed) not in present
    ]
    lowestp = [
        r for r in source_rows
        if r.get("candidate") == "FIG18-MUSIC-LOWESTP" and int(r.get("subspace_dim") or 0) == 3
    ]
    missing_fig18 = []
    for meg in ("MEG-PAPER-1501", "MEG-AIRI"):
        for seed in MASTER_SEEDS:
            if not any(r.get("meg_candidate") == meg and r.get("seed") == seed for r in lowestp):
                missing_fig18.append({"meg_candidate": meg, "seed": seed})
    temporal_literal_done = any(r.get("airi_literal_indexing") == "literal" for r in meg_rows)
    temporal_missing = []
    for cand in ("MEG-AIRI", "MEG-PAPER-1501", "MEG-PAPER-1500"):
        for seed in MASTER_SEEDS:
            rows = [r for r in meg_rows if r.get("candidate") == cand and r.get("seed") == seed]
            if not rows or not any(r.get("airi_literal_indexing") == "literal" for r in rows):
                temporal_missing.append({"candidate": cand, "seed": seed})
    complete = not missing_sim and not missing_fig18 and not temporal_missing
    return {
        "stage_a_complete": complete,
        "final_verdict_allowed": complete,
        "status": (
            "complete"
            if complete
            else "Stage A incomplete / no final verdict yet"
        ),
        "n_required_sim_jobs": len(_required_sim_jobs()),
        "n_completed_sim_jobs": len(_required_sim_jobs()) - len(missing_sim),
        "missing_simulations": missing_sim,
        "missing_fig18_lowestp": missing_fig18,
        "missing_airi_literal_temporal": temporal_missing,
        "temporal_literal_present": temporal_literal_done,
        "note": (
            "Do not declare a reproducing pipeline or failure until the "
            "required candidate runs exist. --quick files are excluded."
        ),
    }


def write_summary() -> Path:
    n170 = _n170_rows()
    meg = _meg_rows()
    sims = _sim_rows()
    source = _source_rows()
    payload = {
        "stage_a_status": _completeness(sims, source, meg),
        "n170": n170,
        "meg": meg,
        "simulations": sims,
        "source_localization": source,
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
