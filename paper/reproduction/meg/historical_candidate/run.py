#!/usr/bin/env python3
"""MEG historical-candidate applications of the frozen N170 estimator.

Two labeled applications only:

  A. Reuse existing ``paper/results/meg/airi_executable/`` (do not rerun B=1000
     if JSON already matches directed+cov+random-phase B=1000).
  B. One new run: same freeze on the paper MEG epoch (full −500…+1000 ms,
     204 planars, no AIRI bandpass).

Does not import ``redisca``. Example::

    python3 paper/reproduction/meg/historical_candidate/run.py reuse
    python3 paper/reproduction/meg/historical_candidate/run.py paper-epoch
    python3 paper/reproduction/meg/historical_candidate/run.py all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
REPRO = REPO_ROOT / "paper" / "reproduction"
SRC = REPO_ROOT / "src"
for _p in (SRC, REPRO):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from common.hashing import read_json, write_json  # noqa: E402
from common.provenance import capture_environment, capture_run  # noqa: E402
from common.rng import numpy_generator  # noqa: E402

from meg.historical_candidate.analysis import run_paper_epoch  # noqa: E402
from meg.historical_candidate.freeze import (  # noqa: E402
    FROZEN_B,
    MASTER_SEED,
    PAPER_EPOCH_RANDOM_PHASE_SEED,
    PAPER_FAITHFUL_DIR,
    RDM_ORDER,
    RESULTS_DIR,
    frozen_estimator_record,
    load_n170_freeze,
    meg_seed_policy,
)
from meg.historical_candidate.reuse import reuse_airi_executable  # noqa: E402
from meg.prepare import condition_averages, load_meg_bundle, prepare_provenance  # noqa: E402

_ITEM = {
    "face": "fig13-meg-face",
    "tool": "fig14-meg-tool",
    "meaning": "fig15-meg-meaning",
    "facevstool": "fig17-meg-nonbinary-components",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("reuse", "paper-epoch", "all"),
        help="A reuse only, B paper-epoch run, or both plus comparison.",
    )
    parser.add_argument(
        "--B",
        dest="n_bootstrap",
        type=int,
        default=None,
        help="Random-phase iterations for path B. Default: freeze B=1000. Do not reduce silently.",
    )
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument(
        "--rdms",
        nargs="*",
        default=None,
        help="Subset of RDM names (default: face tool meaning facevstool).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: B=0 on path B. Do not treat as the reported reproduction.",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=None,
        help="N170 leading_candidate.json (must match the freeze).",
    )
    return parser.parse_args(argv)


def _used_B(args: argparse.Namespace) -> int:
    if args.quick:
        return 0
    if args.n_bootstrap is None:
        return FROZEN_B
    return int(args.n_bootstrap)


def _compact_paper_epoch_rdm(payload: dict[str, Any]) -> dict[str, Any]:
    p_head = payload["primary_random_phase_p_head"]
    peaks0 = (payload.get("peaks") or [{}])[0]
    emp0 = (payload.get("empirical_rdm") or [{}])[0]
    secondary = payload.get("extras", {}).get("condition_label_permutation", {})
    return {
        "item_id": payload["item_id"],
        "rank": payload["rank"],
        "eigenvalues_head": payload["eigenvalues_head"][:4],
        "primary_p_head": p_head[:4],
        "n_components_p_lt_0.05_all_rank": payload["n_components_p_lt_0.05_all_rank"],
        "n_first3_p_lt_0.05": payload["n_first3_p_lt_0.05"],
        "empirical_rdm_pearson_comp1": emp0.get("pearson_wTRw_unique_triangle"),
        "contrast_peak_ms_comp1": peaks0.get("contrast_peak_ms_paper_convention"),
        "class1_peak_ms_comp1": peaks0.get("class1_peak_ms_paper_convention"),
        "mean_of_six_peak_ms_comp1": peaks0.get("mean_of_six_conditions_peak_ms"),
        "secondary_labelperm_p_maxabs_head": (secondary.get("p_maxabs_familywise") or [])[:4],
        "secondary_labelperm_p_signed_ge_head": (
            secondary.get("p_signed_greater_equal") or []
        )[:4],
    }


def _paper_faithful_row(name: str) -> dict[str, Any] | None:
    summary_path = PAPER_FAITHFUL_DIR / "summary.json"
    if not summary_path.exists():
        return None
    summary = read_json(summary_path)
    row = (summary.get("rdms") or {}).get(name)
    if not row:
        return None
    p = [float(v) for v in row.get("p_maxabs_head") or []]
    return {
        "path_label": "paper_faithful",
        "estimator": "redisca.ReDisCA unique_unordered + unscaled_gram",
        "inference": "condition_label_permutation max|lambda| (not historical primary)",
        "rank": row.get("rank"),
        "eigenvalues_head": row.get("eigenvalues_head"),
        "p_head": p,
        "n_first3_p_lt_0.05": int(sum(x < 0.05 for x in p[:3])),
        "contrast_peak_ms_comp1": row.get("contrast_peak_ms_comp1"),
        "empirical_rdm_pearson_comp1": row.get("empirical_rdm_pearson_comp1"),
    }


def build_comparison(
    reuse: dict[str, Any] | None,
    paper_epoch: dict[str, Any] | None,
) -> dict[str, Any]:
    published = {
        "n_significant_components": 3,
        "face_peak_ms": 160.0,
        "figures": "Figs 13–15, 17",
        "note": (
            "Published component *count* is three significant components per "
            "figure. Numeric p-values are figure-panel fingerprints, not a "
            "printed table. Face peak ~160 ms (Fig. 13 / Fig. 17c)."
        ),
    }
    rdms: dict[str, Any] = {}
    for name in RDM_ORDER:
        airi = None if reuse is None else reuse["rdms"].get(name)
        epoch = None
        if paper_epoch is not None:
            raw = paper_epoch["rdms"].get(name)
            epoch = None if raw is None else _compact_paper_epoch_rdm(raw)
        paper = _paper_faithful_row(name)
        n_epoch = None if epoch is None else epoch["n_first3_p_lt_0.05"]
        n_airi = None if airi is None else airi["n_first3_p_lt_0.05"]
        n_paper = None if paper is None else paper["n_first3_p_lt_0.05"]
        rdms[name] = {
            "item_id": _ITEM[name],
            "published": published,
            "paper_faithful_unique_gram_labelperm": paper,
            "airi_executable_reuse": None
            if airi is None
            else {
                "rank": airi["rank"],
                "eigenvalues_head": airi["eigenvalues_head"][:4],
                "p_spoc_head": airi["p_spoc_head"][:4],
                "n_first3_p_lt_0.05": n_airi,
                "n_components_p_lt_0.05_in_head": airi["n_components_p_lt_0.05_in_head"],
                "contrast_peak_ms_comp1": airi["contrast_peak_ms_comp1"],
                "window": "99–999 ms + butter 0.25–20 Hz",
                "recomputed": False,
            },
            "frozen_on_paper_epoch": epoch,
            "matches_published_three_components": {
                "paper_faithful": n_paper == 3,
                "airi_executable_reuse": n_airi == 3,
                "frozen_on_paper_epoch": n_epoch == 3,
            },
        }
    freeze_without_airi = None
    if paper_epoch is not None:
        freeze_without_airi = all(
            (paper_epoch["rdms"][name]["n_first3_p_lt_0.05"] == 3)
            for name in RDM_ORDER
            if name in paper_epoch["rdms"]
        )
    return {
        "note": (
            "A = frozen estimator PLUS AIRI extras (reused). "
            "B = frozen estimator on the paper epoch, no AIRI window/filter. "
            "paper_faithful = unique+Gram + condition-label permutation. "
            "Do not mix paths in one untagged file. MATLAB not used."
        ),
        "published": published,
        "n170_directed_cov_freeze_reproduces_published_meg_counts_without_airi_extras": freeze_without_airi,
        "rdms": rdms,
        "imports_redisca": False,
        "matlab": None,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    freeze = load_n170_freeze(args.freeze)
    used_B = _used_B(args)
    if args.quick:
        print("[historical_candidate] --quick: B=0; not the reported run", flush=True)
    if used_B < FROZEN_B and not args.quick:
        print(
            f"[historical_candidate] WARNING: used_B={used_B} < freeze B={FROZEN_B}; "
            "labeled reduced_B. Not a silent reduction.",
            flush=True,
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = capture_environment()
    write_json(RESULTS_DIR / "environment.json", env)
    estimator = frozen_estimator_record(freeze, used_B=used_B)
    write_json(RESULTS_DIR / "freeze.json", estimator)

    reuse_payload: dict[str, Any] | None = None
    epoch_payload: dict[str, Any] | None = None

    if args.command in {"reuse", "all"}:
        print("[historical_candidate] A: reuse airi_executable JSON (no B=1000 rerun)", flush=True)
        reuse_payload = reuse_airi_executable()
        write_json(RESULTS_DIR / "airi_executable_reuse.json", reuse_payload)
        print(
            f"    verified={reuse_payload['verified_directed_cov_random_phase_B1000']}",
            flush=True,
        )
        for name, row in reuse_payload["rdms"].items():
            print(
                f"    {name}: rank={row['rank']} λ={row['eigenvalues_head'][:4]} "
                f"p={row['p_spoc_head'][:4]} n_sig_head={row['n_components_p_lt_0.05_in_head']}",
                flush=True,
            )

    if args.command in {"paper-epoch", "all"}:
        names = tuple(args.rdms) if args.rdms else RDM_ORDER
        print("[historical_candidate] B: paper MEG epoch, no AIRI bandpass", flush=True)
        bundle = load_meg_bundle()
        averages = condition_averages(bundle.planars, bundle.indices)
        rng, rec = numpy_generator(PAPER_EPOCH_RANDOM_PHASE_SEED)
        if int(args.seed) != MASTER_SEED:
            rng, rec = numpy_generator(int(args.seed) + 20)
            print(
                f"[historical_candidate] non-default CLI seed {args.seed}; "
                f"paper-epoch stream is seed+20={int(args.seed) + 20}",
                flush=True,
            )
        epoch_payload = run_paper_epoch(
            averages,
            bundle.time_ms,
            n_bootstrapping_iterations=used_B,
            rng=rng,
            rdm_names=names,
        )
        epoch_payload["estimator"] = estimator
        epoch_payload["seed_policy"] = meg_seed_policy()
        epoch_payload["rng"] = rec.to_dict()
        epoch_payload["prepare"] = prepare_provenance(bundle)
        write_json(RESULTS_DIR / "paper_epoch.json", epoch_payload)
        for name, payload in epoch_payload["rdms"].items():
            write_json(RESULTS_DIR / f"paper_epoch_{name}.json", payload)
        provenance = capture_run(
            track="meg_historical_candidate",
            path_label="historical_candidate_paper_epoch",
            seed_record=rec,
            extra={
                "prepare": prepare_provenance(bundle),
                "used_B": used_B,
                "freeze_B": FROZEN_B,
                "rdms": list(names),
                "does_not_call_redisca_fit": True,
                "bandpass": None,
                "window": "full -500..+1000 ms, 1501 samples",
            },
        )
        write_json(RESULTS_DIR / "paper_epoch_provenance.json", provenance)

    if reuse_payload is None:
        reuse_path = RESULTS_DIR / "airi_executable_reuse.json"
        if reuse_path.exists():
            reuse_payload = read_json(reuse_path)
    if epoch_payload is None:
        epoch_path = RESULTS_DIR / "paper_epoch.json"
        if epoch_path.exists():
            epoch_payload = read_json(epoch_path)

    comparison = build_comparison(reuse_payload, epoch_payload)
    write_json(RESULTS_DIR / "comparison.json", comparison)

    summary = {
        "path_label": "historical_candidate",
        "command": args.command,
        "estimator": estimator,
        "seed_policy": meg_seed_policy(),
        "airi_executable_reuse": None
        if reuse_payload is None
        else {
            "verified": reuse_payload["verified_directed_cov_random_phase_B1000"],
            "recomputed": False,
            "rdms": {
                name: {
                    "p_spoc_head": row["p_spoc_head"][:4],
                    "eigenvalues_head": row["eigenvalues_head"][:4],
                    "rank": row["rank"],
                    "n_first3_p_lt_0.05": row["n_first3_p_lt_0.05"],
                }
                for name, row in reuse_payload["rdms"].items()
            },
        },
        "paper_epoch": None
        if epoch_payload is None
        else {
            name: _compact_paper_epoch_rdm(payload)
            for name, payload in epoch_payload["rdms"].items()
        },
        "n170_directed_cov_freeze_reproduces_published_meg_counts_without_airi_extras": comparison[
            "n170_directed_cov_freeze_reproduces_published_meg_counts_without_airi_extras"
        ],
        "matlab": None,
        "imports_redisca": False,
        "quick": bool(args.quick),
        "reduced_B": bool(used_B < FROZEN_B),
    }
    write_json(RESULTS_DIR / "summary.json", summary)
    print("[historical_candidate] wrote", RESULTS_DIR / "summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
