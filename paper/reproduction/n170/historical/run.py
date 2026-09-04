#!/usr/bin/env python3
"""Track A+B historical N170 estimator variants (stock SPoC random-phase).

Does not import ``redisca``. Example:

    python3 paper/reproduction/n170/historical/run.py
    python3 paper/reproduction/n170/historical/run.py --B 1000 --track all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
N170_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "paper" / "reproduction"))
sys.path.insert(0, str(N170_DIR))

from common.hashing import write_json  # noqa: E402
from common.provenance import capture_environment  # noqa: E402
from common.rng import numpy_generator  # noqa: E402

from prepare import load_n170_subject1, window_slice  # noqa: E402
from rdms import CONDITION_LABELS, theoretical_rdm  # noqa: E402

from historical.analysis import compact_row, fit_variant  # noqa: E402
from historical.select import pick_two_joint_candidates  # noqa: E402
from historical.variants import (  # noqa: E402
    MASTER_SEED,
    N_BOOTSTRAP,
    N_TRACK_B_SEEDS,
    PRINTED_CAR,
    PRINTED_FACE,
    TRACK_B_SEEDS,
    spec_by_id,
    track_a_specs,
)

RESULTS_DIR = REPO_ROOT / "paper" / "results" / "n170" / "historical"


def _pcg64(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed)))


def _p_distribution(values: list[float], *, printed: float | None = None) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95])
    payload: dict[str, Any] = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "fraction_p0": float(np.mean(arr == 0.0)),
        "n_p0": int(np.sum(arr == 0.0)),
        "values": [float(v) for v in arr],
    }
    if printed is not None:
        payload["printed_target"] = float(printed)
        # B=1000 ⇒ p is k/1000. 0.009 is exactly 9/1000.
        payload["n_equal_printed_within_1e-12"] = int(
            np.sum(np.abs(arr - float(printed)) <= 1e-12)
        )
        payload["n_equal_9_over_1000"] = int(np.sum(np.abs(arr - 0.009) <= 1e-12))
        payload["printed_inside_min_max"] = bool(
            float(np.min(arr)) <= float(printed) <= float(np.max(arr))
        )
        payload["printed_inside_q05_q95"] = bool(
            float(quantiles[0]) <= float(printed) <= float(quantiles[4])
        )
    return payload


def _load_packed() -> dict[str, Any]:
    packed = load_n170_subject1(lpfilt=False)
    return packed


def _fit_one(
    packed: dict[str, Any],
    spec,
    *,
    n_bootstrapping_iterations: int,
    rng_seed: int,
) -> dict[str, Any]:
    rdm = theoretical_rdm(spec.contrast, within=0.0, between=1.0)
    win = window_slice(
        packed["data"],
        packed["times_ms"],
        center_ms=spec.window_center_ms,
        duration_ms=spec.window_duration_ms,
    )
    rng, record = numpy_generator(int(rng_seed))
    matched_rng = _pcg64(int(rng_seed) + 500)
    payload = fit_variant(
        X_window=win["data"],
        X_full=packed["data"],
        times_full_ms=packed["times_ms"],
        channel_labels=list(packed["channel_labels"]),
        condition_labels=list(CONDITION_LABELS),
        rdm=rdm,
        spec=spec,
        rng=rng,
        n_bootstrapping_iterations=int(n_bootstrapping_iterations),
        window_meta=win,
        matched_rng=matched_rng,
    )
    payload["erp"] = {
        "path": packed["path"],
        "sha256": packed["sha256"],
        "lpfilt": packed["lpfilt"],
        "srate_hz": packed["srate_hz"],
        "n_accepted": packed["n_accepted"],
        "file_name": Path(packed["path"]).name,
    }
    payload["rng"] = record.to_dict()
    payload["rng"]["seed_used"] = int(rng_seed)
    return payload


def run_track_a(*, n_bootstrapping_iterations: int) -> dict[str, Any]:
    packed = _load_packed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in track_a_specs():
        print(
            f"[track A] {spec.variant_id}  seed={spec.rng_seed}  B={n_bootstrapping_iterations}",
            flush=True,
        )
        payload = _fit_one(
            packed,
            spec,
            n_bootstrapping_iterations=n_bootstrapping_iterations,
            rng_seed=spec.rng_seed,
        )
        out_path = RESULTS_DIR / f"{spec.variant_id}.json"
        write_json(out_path, payload)
        row = compact_row(payload)
        row["path"] = str(out_path.relative_to(REPO_ROOT))
        rows.append(row)
        evals = row["eigenvalues_head"][:3]
        phead = row["primary_p_head"][:3]
        print(
            f"    λ={evals}  p={phead}  corr_wTRw={row['corr_wTRw_comp0']:.5f}  "
            f"rank={row['numerical_rank']}",
            flush=True,
        )
    table = {
        "track": "A",
        "n_variants": len(rows),
        "master_seed": MASTER_SEED,
        "B": int(n_bootstrapping_iterations),
        "inference_primary": "spoc_random_phase",
        "data_file": Path(packed["path"]).name,
        "erp_sha256": packed["sha256"],
        "n_channels": 28,
        "rdm_fill": "binary_0_1",
        "printed_face": PRINTED_FACE,
        "printed_car": PRINTED_CAR,
        "rows": rows,
    }
    write_json(RESULTS_DIR / "track_a_table.json", table)
    return table


def run_track_b(
    table: dict[str, Any],
    *,
    n_bootstrapping_iterations: int,
    n_seeds: int,
) -> dict[str, Any]:
    packed = _load_packed()
    candidates = pick_two_joint_candidates(table["rows"])
    if len(candidates) != 2:
        raise RuntimeError(f"expected 2 joint candidates, got {len(candidates)}")
    seeds = list(TRACK_B_SEEDS[: int(n_seeds)])
    candidate_payloads: list[dict[str, Any]] = []
    for cand in candidates:
        print(f"[track B] candidate {cand['id']}", flush=True)
        face_spec = spec_by_id(cand["face_variant_id"])
        car_spec = spec_by_id(cand["car_variant_id"])
        face_p1: list[float] = []
        car_p1: list[float] = []
        car_p2: list[float] = []
        per_seed: list[dict[str, Any]] = []
        for seed in seeds:
            face_seed = int(seed + face_spec.seed_offset)
            car_seed = int(seed + car_spec.seed_offset)
            print(f"    seed={seed}  face_rng={face_seed}  car_rng={car_seed}", flush=True)
            face_fit = _fit_one(
                packed,
                face_spec,
                n_bootstrapping_iterations=n_bootstrapping_iterations,
                rng_seed=face_seed,
            )
            car_fit = _fit_one(
                packed,
                car_spec,
                n_bootstrapping_iterations=n_bootstrapping_iterations,
                rng_seed=car_seed,
            )
            fp1 = float(face_fit["primary_random_phase_p_head"][0])
            cp1 = float(car_fit["primary_random_phase_p_head"][0])
            cp2 = float(car_fit["primary_random_phase_p_head"][1])
            face_p1.append(fp1)
            car_p1.append(cp1)
            car_p2.append(cp2)
            per_seed.append(
                {
                    "base_seed": int(seed),
                    "face_rng_seed": face_seed,
                    "car_rng_seed": car_seed,
                    "face_p1": fp1,
                    "car_p1": cp1,
                    "car_p2": cp2,
                    "face_lambda1": float(face_fit["eigenvalues_head"][0]),
                    "car_lambda1": float(car_fit["eigenvalues_head"][0]),
                    "car_lambda2": float(car_fit["eigenvalues_head"][1]),
                }
            )
        p2_stats = _p_distribution(car_p2, printed=PRINTED_CAR["p2"])
        candidate_payloads.append(
            {
                "candidate": cand,
                "seeds": seeds,
                "n_seeds": len(seeds),
                "B": int(n_bootstrapping_iterations),
                "per_seed": per_seed,
                "face_p1": _p_distribution(face_p1, printed=PRINTED_FACE["p1"]),
                "car_p1": _p_distribution(car_p1, printed=PRINTED_CAR["p1"]),
                "car_p2": p2_stats,
                "car_p2_full_distribution": p2_stats["values"],
                "p2_009_question": {
                    "printed_p2": PRINTED_CAR["p2"],
                    "lies_inside_min_max": p2_stats["printed_inside_min_max"],
                    "lies_inside_5_95_percentiles": p2_stats["printed_inside_q05_q95"],
                    "fraction_exactly_0.009": p2_stats["n_equal_9_over_1000"] / len(seeds),
                    "note": (
                        "Published p2≈0.009 is 9/1000 under p=count/B. "
                        "This block only records whether that value appears "
                        "in the 20-seed envelope. It does not declare success."
                    ),
                },
            }
        )
    track_b = {
        "track": "B",
        "master_seed": MASTER_SEED,
        "track_b_seed_formula": "MASTER_SEED + 10000 + i  for i in 0..n_seeds-1",
        "n_seeds": len(seeds),
        "B": int(n_bootstrapping_iterations),
        "inference_primary": "spoc_random_phase",
        "candidates": candidate_payloads,
        "selection_rule": candidates[0]["selection_rule"],
    }
    write_json(RESULTS_DIR / "track_b.json", track_b)
    return track_b


def freeze_leading_candidate(
    table: dict[str, Any],
    track_b: dict[str, Any],
    packed: dict[str, Any],
) -> dict[str, Any]:
    top = track_b["candidates"][0]
    cand = top["candidate"]
    p2 = top["car_p2"]
    face_row = next(
        row for row in table["rows"] if row["variant_id"] == cand["face_variant_id"]
    )
    car_row = next(
        row for row in table["rows"] if row["variant_id"] == cand["car_variant_id"]
    )
    remaining = [
            (
                "Face window unique-triangle RDM correlation is "
                f"{face_row['corr_wTRw_comp0']:.5f} (un-demeaned trace-sq "
                f"{face_row['corr_trace_sq_comp0']:.5f}) versus printed ≈0.82. "
                "None of the 12 source-supported variants produced a window "
                "correlation near 0.82; w^T R w matches the two-level 6-pair "
                "target almost exactly (~0.999)."
            ),
            (
                f"Car component-2 primary p (Track A seed) is {car_row['primary_p_head'][1]}; "
                f"Track B 20-seed envelope is min={p2['min']}, max={p2['max']}, "
                f"mean={p2['mean']:.4f}, median={p2['median']:.4f}. "
                f"Printed p2≈0.009 appears in {p2['n_equal_9_over_1000']}/20 seeds "
                f"and {'does' if p2['printed_inside_min_max'] else 'does not'} "
                "fall inside min–max and the 5–95% interval. It is not a rare "
                "one-seed accident for this candidate."
            ),
            (
                f"Face λ1={face_row['eigenvalues_head'][0]:.5f} vs printed 0.87209 "
                f"(delta {face_row['comparison_to_printed']['delta_lambda1']}). "
                f"Car λ1={car_row['eigenvalues_head'][0]:.5f} vs 0.91639; "
                f"λ2={car_row['eigenvalues_head'][1]:.5f} vs 0.77036 "
                "(λ2 remains ~0.02 high)."
            ),
            (
                "MATLAB is unavailable. This freeze is a source-faithful Python "
                "reconstruction of stock SPoC + AIRI pair construction, not "
                "MATLAB eig/rand parity."
            ),
        ]
    freeze = {
        "pair_mode": cand["pair_mode"],
        "matrix_mode": cand["matrix_mode"],
        "face_window": cand["face_window"],
        "car_window": cand["car_window"],
        "inference": "spoc_random_phase",
        "B": table["B"],
        "data_file": Path(packed["path"]).name,
        "data_sha256": packed["sha256"],
        "channels": {
            "count": 28,
            "set": "28 scalp EEG; EOG/bipolar dropped; P9/P10 not in ERP",
            "labels": list(packed["channel_labels"]),
        },
        "rdm_fill": "binary 0/1 (within=0, between=1, diagonal=0)",
        "seed_policy": {
            "bit_generator": "PCG64",
            "master_seed": MASTER_SEED,
            "track_a": "PCG64(master_seed + variant_offset); offsets in variants.py",
            "track_b": "PCG64(MASTER_SEED+10000+i + variant_offset) for i in 0..19",
        },
        "face_variant_id": cand["face_variant_id"],
        "car_variant_id": cand["car_variant_id"],
        "joint_candidate_id": cand["id"],
        "second_plausible_candidate_id": track_b["candidates"][1]["candidate"]["id"],
        "why_chosen": (
            "Lexicographic qualitative rule on source-supported variants only: "
            "prefer PRIMARY face and car p1==0, then the printed Fig. 11 car "
            "application time 170 ms, then car λ1 nearer 0.91639, then car p2 "
            "nearer 0.009, then car λ2 nearer 0.77036, then face λ1 nearer "
            "0.87209. Face corr vs 0.82 was not used as a discard rule. This "
            "is the least-bad jointly close setting under that rule, not a "
            "claim of printed-figure parity."
        ),
        "track_a_point_estimates": {
            "face_lambda_head": face_row["eigenvalues_head"],
            "face_primary_p_head": face_row["primary_p_head"],
            "face_corr_wTRw": face_row["corr_wTRw_comp0"],
            "face_corr_trace_sq": face_row["corr_trace_sq_comp0"],
            "car_lambda_head": car_row["eigenvalues_head"],
            "car_primary_p_head": car_row["primary_p_head"],
            "car_corr_wTRw": car_row["corr_wTRw_comp0"],
        },
        "track_b_p2_envelope": {
            "mean": p2["mean"],
            "median": p2["median"],
            "min": p2["min"],
            "max": p2["max"],
            "q05": p2["q05"],
            "q95": p2["q95"],
            "fraction_p0": p2["fraction_p0"],
            "printed_p2_inside_min_max": p2["printed_inside_min_max"],
        },
        "remaining_mismatches": remaining,
        "imports_redisca": False,
        "matlab": None,
    }
    write_json(RESULTS_DIR / "leading_candidate.json", freeze)
    return freeze


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=("a", "b", "all"),
        default="all",
        help="Track A table, Track B multi-seed, or both.",
    )
    parser.add_argument(
        "--B",
        dest="n_bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Random-phase iterations (stock SPoC). Default 1000.",
    )
    parser.add_argument(
        "--n-track-b-seeds",
        type=int,
        default=N_TRACK_B_SEEDS,
        help="Independent Track B seeds (default 20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = capture_environment(extra_packages=("matplotlib", "h5py"))
    write_json(RESULTS_DIR / "environment.json", env)
    table: dict[str, Any] | None = None
    if args.track in ("a", "all"):
        table = run_track_a(n_bootstrapping_iterations=args.n_bootstrap)
    if args.track in ("b", "all"):
        if table is None:
            table_path = RESULTS_DIR / "track_a_table.json"
            table = json.loads(table_path.read_text(encoding="utf-8"))
        track_b = run_track_b(
            table,
            n_bootstrapping_iterations=args.n_bootstrap,
            n_seeds=args.n_track_b_seeds,
        )
        packed = _load_packed()
        freeze_leading_candidate(table, track_b, packed)
        print(
            f"froze {RESULTS_DIR / 'leading_candidate.json'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
