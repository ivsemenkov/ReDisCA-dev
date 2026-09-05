"""Re-run AIRI temporal tests only (literal + corrected). Does not refit p-values."""

from __future__ import annotations

import argparse

from paper.reproduction.common.constants import AIRI_NMC_TEMPORAL, MASTER_SEEDS
from paper.reproduction.common.hashing import read_json, write_json
from paper.reproduction.common.paths import RESULTS_ROOT
from paper.reproduction.common.provenance import capture_run
from paper.reproduction.meg.analysis import analyze_candidate
from paper.reproduction.meg.prepare import load_meg_bundle, prepare_candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AIRI literal vs corrected temporal tests.")
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=["MEG-AIRI", "MEG-PAPER-1501", "MEG-PAPER-1500"],
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=list(MASTER_SEEDS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args(argv)
    bundle = load_meg_bundle()
    written = []
    skipped = []
    n_surr = 16 if args.quick else 8
    # Random-phase is not the product of this runner; keep it tiny.
    for candidate_id in args.candidates:
        prepared = prepare_candidate(bundle, candidate_id)
        for seed in args.seeds:
            name = f"{'QUICK_NONREPRO_' if args.quick else ''}temporal_airi_seed{seed}.json"
            dest = RESULTS_ROOT / "meg" / candidate_id / name
            if (
                not args.quick
                and not args.no_skip_existing
                and dest.exists()
            ):
                existing = read_json(dest)
                rdms = existing.get("rdms") or {}
                has_both = all(
                    (block or {}).get("temporal_airi_literal")
                    and (block or {}).get("temporal_airi_corrected")
                    for block in rdms.values()
                ) if rdms else False
                nmc_ok = False
                if rdms:
                    first = next(iter(rdms.values()))
                    lit = (first or {}).get("temporal_airi_literal") or {}
                    nmc_ok = int(lit.get("Nmc") or 0) >= 100
                if has_both and nmc_ok and not existing.get("quick_non_reproduction"):
                    skipped.append(str(dest))
                    continue
            payload = analyze_candidate(
                prepared,
                seed=seed,
                n_surrogates=n_surr,
                quick=args.quick,
                run_secondary_perm=False,
                run_temporal=True,
            )
            compact = {
                "candidate_id": candidate_id,
                "seed": seed,
                "n_samples": payload["n_samples"],
                "window_ms": payload["window_ms"],
                "filter": payload["filter"],
                "quick_non_reproduction": bool(args.quick),
                "note": (
                    "Temporal-only companion. Component p-values here used a "
                    "reduced B and must not replace seed*.json random-phase results."
                ),
                "rdms": {},
                "provenance": capture_run(
                    track="meg_temporal", candidate_id=candidate_id, seed=seed
                ),
            }
            for key, rdm in payload["rdms"].items():
                compact["rdms"][key] = {
                    "eigenvalues": rdm.get("eigenvalues", [])[:6],
                    "p_random_phase_reduced_B_DO_NOT_REPORT": rdm.get("p_random_phase", [])[:6],
                    "temporal_airi_literal": rdm.get("temporal_airi_literal"),
                    "temporal_airi_corrected": rdm.get("temporal_airi_corrected"),
                    "temporal_paper_fwer": rdm.get("temporal_paper_fwer"),
                    "paper_qualitative": rdm.get("paper_qualitative"),
                }
            write_json(dest, compact)
            written.append(str(dest))
    print({
        "written": written,
        "skipped_existing": skipped,
        "quick": args.quick,
        "Nmc": 8 if args.quick else AIRI_NMC_TEMPORAL,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
