"""Run Stage A MEG experiments (Figs 12–17)."""

from __future__ import annotations

import argparse

from paper.reproduction.common.constants import MASTER_SEEDS, RANDOM_PHASE_B
from paper.reproduction.common.hashing import write_json
from paper.reproduction.common.paths import RESULTS_ROOT
from paper.reproduction.common.provenance import capture_run
from paper.reproduction.meg.analysis import analyze_candidate
from paper.reproduction.meg.prepare import load_meg_bundle, prepare_candidate


def run_meg(
    *,
    candidates: tuple[str, ...],
    seeds: tuple[int, ...],
    quick: bool = False,
) -> dict[str, list[str]]:
    bundle = load_meg_bundle()
    n_surrogates = 16 if quick else RANDOM_PHASE_B
    written = []
    for candidate_id in candidates:
        prepared = prepare_candidate(bundle, candidate_id)
        for seed in seeds:
            payload = analyze_candidate(
                prepared, seed=seed, n_surrogates=n_surrogates, quick=quick
            )
            payload["quick_non_reproduction"] = quick
            payload["provenance"] = capture_run(
                track="meg", candidate_id=candidate_id, seed=seed
            )
            name = f"{'QUICK_NONREPRO_' if quick else ''}seed{seed}.json"
            dest = RESULTS_ROOT / "meg" / candidate_id / name
            write_json(dest, payload)
            written.append(str(dest))
    return {"written": written, "quick": quick}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage A MEG (Figs 12–17).")
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=["MEG-AIRI", "MEG-PAPER-1501", "MEG-PAPER-1500"],
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=list(MASTER_SEEDS))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        print("WARNING: --quick is NON-REPRODUCTION.")
    print(run_meg(candidates=tuple(args.candidates), seeds=tuple(args.seeds), quick=args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
