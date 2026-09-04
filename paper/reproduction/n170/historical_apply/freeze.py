"""Load and refuse to retune the N170 Tracks A+B leading freeze.

The freeze is ``paper/results/n170/historical/leading_candidate.json``.
This module does not import ``redisca``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.paths import PAPER_ROOT

FROZEN_PAIR_MODE = "airi_directed"
FROZEN_MATRIX_MODE = "matlab_cov"
FROZEN_INFERENCE = "spoc_random_phase"
FROZEN_B = 1000
MASTER_SEED = 20240904

# Disjoint from historical Track A (offsets < 200) and Track B (MASTER+10000+i).
FIG7_WINDOW_SEED_BASE = MASTER_SEED + 2000
# Fig. 8 reuses FIG7_WINDOW_SEED_BASE + sliding-grid index for the same centers.

DEFAULT_FREEZE_PATH = (
    PAPER_ROOT / "results" / "n170" / "historical" / "leading_candidate.json"
)

_REQUIRED = {
    "pair_mode": FROZEN_PAIR_MODE,
    "matrix_mode": FROZEN_MATRIX_MODE,
    "inference": FROZEN_INFERENCE,
    "B": FROZEN_B,
}


def load_and_validate_freeze(path: Path | str | None = None) -> dict[str, Any]:
    """Load the leading-candidate JSON and assert frozen estimator semantics.

    Raises ``ValueError`` if pair/matrix/inference/B were changed. Callers may
    still override the *used* B for smoke tests; that override is labeled
    ``reduced_B`` and is not a retune of the freeze file.
    """
    freeze_path = Path(path) if path is not None else DEFAULT_FREEZE_PATH
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for key, expected in _REQUIRED.items():
        got = payload.get(key)
        if got != expected:
            mismatches.append(f"{key}: freeze has {got!r}, required {expected!r}")
    if mismatches:
        raise ValueError(
            "Freeze JSON does not match the N170 Tracks A+B leading freeze "
            "(do not retune pair order, matrix, inference, or B): "
            + "; ".join(mismatches)
        )
    payload["_freeze_path"] = str(freeze_path)
    return payload


def frozen_estimator_record(freeze: dict[str, Any], *, used_B: int) -> dict[str, Any]:
    """Compact provenance block copied into every apply result file."""
    return {
        "joint_candidate_id": freeze.get("joint_candidate_id"),
        "face_variant_id": freeze.get("face_variant_id"),
        "car_variant_id": freeze.get("car_variant_id"),
        "pair_mode": FROZEN_PAIR_MODE,
        "matrix_mode": FROZEN_MATRIX_MODE,
        "inference": FROZEN_INFERENCE,
        "p_formula": "p = count(max|lambda_surr| >= |lambda_obs|) / B",
        "executable": "common.source_faithful.fit_condition_averages",
        "freeze_B": FROZEN_B,
        "used_B": int(used_B),
        "reduced_B": bool(int(used_B) < FROZEN_B),
        "freeze_path": freeze.get("_freeze_path"),
        "imports_redisca": False,
        "matlab": None,
        "rdm_fill": freeze.get("rdm_fill"),
        "data_file": freeze.get("data_file"),
        "seed_policy_from_freeze": freeze.get("seed_policy"),
    }
