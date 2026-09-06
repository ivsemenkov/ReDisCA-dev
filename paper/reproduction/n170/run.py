"""Run Stage A N170 experiments (Figs 7–11)."""

from __future__ import annotations

import argparse

from paper.reproduction.common.constants import MASTER_SEEDS, RANDOM_PHASE_B
from paper.reproduction.common.hashing import write_json
from paper.reproduction.common.paths import RESULTS_ROOT
from paper.reproduction.common.provenance import capture_run
from paper.reproduction.n170.analysis import analyze_candidate
from paper.reproduction.n170.prepare import load_n170_subject1


def run_n170(
    *,
    seeds: tuple[int, ...],
    quick: bool = False,
) -> dict[str, list[str]]:
    n_surrogates = 32 if quick else RANDOM_PHASE_B
    meaning_steps = (25.0,) if quick else (25.0, 1000.0 / 256.0)
    written: list[str] = []
    for lpfilt, candidate_id in ((False, "N170-UNFILT"), (True, "N170-LP20")):
        bundle = load_n170_subject1(lpfilt=lpfilt)
        for seed in seeds:
            payload = analyze_candidate(
                bundle,
                candidate_id=candidate_id,
                seed=seed,
                n_surrogates=n_surrogates,
                meaning_steps_ms=meaning_steps,
            )
            payload["quick_non_reproduction"] = quick
            payload["provenance"] = capture_run(
                track="n170", candidate_id=candidate_id, seed=seed
            )
            dest = RESULTS_ROOT / "n170" / candidate_id / f"seed{seed}.json"
            if quick:
                dest = RESULTS_ROOT / "n170" / candidate_id / f"QUICK_NONREPRO_seed{seed}.json"
            write_json(dest, payload)
            written.append(str(dest))
    return {"written": written, "quick": quick}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage A N170 (Figs 7–11).")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(MASTER_SEEDS))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        print("WARNING: --quick is NON-REPRODUCTION.")
    print(run_n170(seeds=tuple(args.seeds), quick=args.quick))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
