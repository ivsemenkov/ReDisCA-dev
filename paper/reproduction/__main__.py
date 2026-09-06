"""One command for Stage A tests and full reproduction.

Examples
--------
python -m paper.reproduction test
python -m paper.reproduction stage-a --quick
python -m paper.reproduction stage-a
python -m paper.reproduction download all
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from paper.reproduction.common.constants import MASTER_SEEDS
from paper.reproduction.common.paths import REPO_ROOT


def _run_pytest(extra: list[str] | None = None) -> int:
    cmd = [sys.executable, "-m", "pytest", "tests", "paper/reproduction", "-q"]
    if extra:
        cmd.extend(extra)
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT)


def _cmd_download(subset: str) -> int:
    from paper.reproduction.common.download_erpcore import download_ica_xlsx, download_processed_subject
    from paper.reproduction.common.download_osf import (
        download_meg_sensor_assets,
        download_source_model_assets,
    )

    if subset in {"n170", "all"}:
        print(download_processed_subject("1"))
        try:
            print(download_ica_xlsx())
        except FileNotFoundError as exc:
            print(f"ICA xlsx not downloaded ({exc}); subject-1 ERP already has ICA applied.")
    if subset in {"meg", "all"}:
        print(download_meg_sensor_assets())
    if subset in {"source-models", "simulations", "all"}:
        print(download_source_model_assets())
    return 0


def _cmd_stage_a(*, quick: bool, tracks: list[str], seeds: list[int]) -> int:
    if quick:
        print("WARNING: --quick is NON-REPRODUCTION. Do not report these as paper results.")
    if "validation" in tracks:
        code = _run_pytest(["paper/reproduction/validation", "paper/reproduction/tests"])
        if code != 0:
            return code
    if "n170" in tracks:
        from paper.reproduction.n170.run import run_n170

        print(run_n170(seeds=tuple(seeds), quick=quick))
    if "meg" in tracks:
        from paper.reproduction.meg.run import run_meg

        print(run_meg(seeds=tuple(seeds), quick=quick))
    if "simulations" in tracks:
        from paper.reproduction.simulations.run import run_candidate

        for candidate in ("SIM-P1",) if not quick else ("SIM-P1",):
            print(run_candidate(candidate, seeds=tuple(seeds), quick=quick, include_rsa=not quick))
    if "source-localization" in tracks:
        from paper.reproduction.source_localization.run import main as sl_main

        sl_main(["--seeds", *[str(s) for s in seeds], *(["--quick"] if quick else [])])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage A ReDisCA paper reproduction.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("test", help="Unit and smoke tests (library + Stage A).")
    dl = sub.add_parser("download", help="Download public datasets into .reproduction_data/")
    dl.add_argument("subset", choices=["n170", "meg", "source-models", "simulations", "all"])
    sa = sub.add_parser("stage-a", help="Full Stage A suite, or --quick NON-REPRODUCTION.")
    sa.add_argument("--quick", action="store_true")
    sa.add_argument(
        "--tracks",
        nargs="*",
        default=["validation", "n170", "meg", "simulations", "source-localization"],
    )
    sa.add_argument("--seeds", nargs="*", type=int, default=list(MASTER_SEEDS))
    args = parser.parse_args(argv)
    if args.command == "test":
        return _run_pytest()
    if args.command == "download":
        return _cmd_download(args.subset)
    if args.command == "stage-a":
        return _cmd_stage_a(quick=args.quick, tracks=args.tracks, seeds=args.seeds)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
