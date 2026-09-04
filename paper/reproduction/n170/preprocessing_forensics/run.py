"""CLI for Track C N170 preprocessing forensics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
N170_DIR = HERE.parent
REPRO_DIR = N170_DIR.parent
REPO_ROOT = REPRO_DIR.parent.parent
for path in (HERE, N170_DIR, REPRO_DIR, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forensics import build_bundle, write_results  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "paper" / "results" / "n170" / "preprocessing"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for compact JSON (default: paper/results/n170/preprocessing)",
    )
    args = parser.parse_args()
    bundle = build_bundle()
    written = write_results(bundle, args.out)
    print(json.dumps({"written": written, "answers": bundle["answers"]}, indent=2))


if __name__ == "__main__":
    main()
