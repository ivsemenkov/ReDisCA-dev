#!/usr/bin/env python3
"""Apply the frozen N170 historical candidate to Figs 7 and 8.

Does not import ``redisca``. Example::

    python3 paper/reproduction/n170/historical_apply/run.py
    python3 paper/reproduction/n170/historical_apply/run.py --B 1000 \\
        --freeze paper/results/n170/historical/leading_candidate.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
N170_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "paper" / "reproduction"))
sys.path.insert(0, str(N170_DIR))

from common.hashing import write_json  # noqa: E402
from common.provenance import capture_environment  # noqa: E402

from historical_apply.analysis import run_fig8_windows, run_meaning_scan  # noqa: E402
from historical_apply.freeze import (  # noqa: E402
    DEFAULT_FREEZE_PATH,
    FROZEN_B,
    load_and_validate_freeze,
)
from prepare import DEFAULT_SLIDING_STEP_MS, load_n170_subject1  # noqa: E402

RESULTS_DIR = REPO_ROOT / "paper" / "results" / "n170" / "historical_apply"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=DEFAULT_FREEZE_PATH,
        help="Path to leading_candidate.json (must match the frozen semantics).",
    )
    parser.add_argument(
        "--B",
        dest="n_bootstrap",
        type=int,
        default=None,
        help="Random-phase iterations. Default: freeze B=1000. Smaller B is labeled reduced_B.",
    )
    parser.add_argument(
        "--step-ms",
        type=float,
        default=DEFAULT_SLIDING_STEP_MS,
        help="Sliding step (documented 25 ms; not a paper value; do not tune).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: B=0. Do not treat as the reported reproduction.",
    )
    return parser.parse_args(argv)


def _fig7_verdict(scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "random_phase_p1_lt_0.05_near_400ms": scan[
            "random_phase_recovers_p1_lt_0.05_near_400ms"
        ],
        "nearest_400ms_primary_p1": scan["nearest_400ms_primary_p1"],
        "nearest_400ms_center_ms": scan["nearest_400ms_center_ms"],
        "segments_p1_lt_0.05": scan["p_lt_0.05_segments_comp1_primary"],
        "continuous_segment_covers_400ms": scan[
            "continuous_p1_lt_0.05_segment_covers_400ms"
        ],
        "any_p1_lt_0.05_in_350_450ms": scan["any_primary_p1_lt_0.05_in_350_450ms"],
        "paper_claim": scan["paper_claim"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    freeze = load_and_validate_freeze(args.freeze)
    used_B = 0 if args.quick else (FROZEN_B if args.n_bootstrap is None else int(args.n_bootstrap))
    if args.quick:
        print("[historical_apply] --quick: B=0; not the reported run", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = capture_environment(extra_packages=("matplotlib", "h5py"))
    write_json(RESULTS_DIR / "environment.json", env)

    print("[historical_apply] loading 1_N170_erp_ar.erp (28 scalp)", flush=True)
    packed = load_n170_subject1(lpfilt=False)
    print(
        f"[historical_apply] freeze {freeze.get('joint_candidate_id')} "
        f"pair={freeze['pair_mode']} matrix={freeze['matrix_mode']} "
        f"inference={freeze['inference']} used_B={used_B} step={args.step_ms} ms",
        flush=True,
    )

    print("[historical_apply] Fig. 7 meaning sliding T=150 ms", flush=True)
    fig7 = run_meaning_scan(
        packed,
        freeze,
        n_bootstrapping_iterations=used_B,
        step_ms=float(args.step_ms),
    )
    fig7["erp"] = {
        "path": packed["path"],
        "sha256": packed["sha256"],
        "file_name": Path(packed["path"]).name,
        "srate_hz": packed["srate_hz"],
        "n_accepted": packed["n_accepted"],
        "n_channels": len(packed["channel_labels"]),
        "channel_labels": list(packed["channel_labels"]),
    }
    write_json(RESULTS_DIR / "fig07_meaning_pmap.json", fig7)
    verdict = _fig7_verdict(fig7)
    print(
        f"    nearest-400 p1={verdict['nearest_400ms_primary_p1']}  "
        f"recovers p<0.05 near 400={verdict['random_phase_p1_lt_0.05_near_400ms']}  "
        f"segments={verdict['segments_p1_lt_0.05']}",
        flush=True,
    )

    print("[historical_apply] Fig. 8 windows 375/400/425 ms", flush=True)
    fig8 = run_fig8_windows(
        packed, freeze, n_bootstrapping_iterations=used_B
    )
    write_json(RESULTS_DIR / "fig08_meaning_windows.json", fig8)
    for row in fig8["windows"]:
        print(
            f"    {row['center_ms']:g} ms  λ1={row['lambda1']:.5f}  "
            f"p1={row['primary_p1']}  corr={row['corr_wTRw']:.5f}  "
            f"ch={row['pattern_max_abs_channel']}  "
            f"exact24={row['secondary_exact24_p1_signed_ge']:.3f}",
            flush=True,
        )

    summary = {
        "path_label": "historical_apply",
        "freeze": fig7["estimator"],
        "fig07": verdict,
        "fig08": {
            "centers_ms": fig8["centers_ms"],
            "lambda1": [row["lambda1"] for row in fig8["windows"]],
            "primary_p1": [row["primary_p1"] for row in fig8["windows"]],
            "corr_wTRw": [row["corr_wTRw"] for row in fig8["windows"]],
            "pattern_max_abs_channel": [
                row["pattern_max_abs_channel"] for row in fig8["windows"]
            ],
            "secondary_exact24_p1_signed_ge": [
                row["secondary_exact24_p1_signed_ge"] for row in fig8["windows"]
            ],
        },
        "matlab": None,
        "imports_redisca": False,
        "quick": bool(args.quick),
    }
    write_json(RESULTS_DIR / "summary.json", summary)
    print("[historical_apply] wrote", RESULTS_DIR / "summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
