"""Frozen N170 Tracks A+B estimator applied to MEG.

Does not import ``redisca``. Pair order / matrix / B are not retuned.
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

# Same CLI seed as paper/reproduction/meg/run.py. Existing splits:
#   seed+1  paper_faithful condition-label permutation
#   seed+2  paper_faithful time-series FWER
#   seed+3  airi_executable random-phase (reused JSON; not re-drawn here)
#   seed+10 airi pair-order diagnostic
# This path:
PAPER_EPOCH_RANDOM_PHASE_SEED = MASTER_SEED + 20
PAPER_EPOCH_SECONDARY_PERM_SEED = MASTER_SEED + 21

N170_FREEZE_PATH = PAPER_ROOT / "results" / "n170" / "historical" / "leading_candidate.json"
AIRI_EXECUTABLE_DIR = PAPER_ROOT / "results" / "meg" / "airi_executable"
PAPER_FAITHFUL_DIR = PAPER_ROOT / "results" / "meg" / "paper_faithful"
RESULTS_DIR = PAPER_ROOT / "results" / "meg" / "historical_candidate"

RDM_ORDER = ("face", "tool", "meaning", "facevstool")
N_REPORT = 8


def load_n170_freeze(path: Path | str | None = None) -> dict[str, Any]:
    freeze_path = Path(path) if path is not None else N170_FREEZE_PATH
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "pair_mode": FROZEN_PAIR_MODE,
        "matrix_mode": FROZEN_MATRIX_MODE,
        "inference": FROZEN_INFERENCE,
        "B": FROZEN_B,
    }
    mismatches = [
        f"{key}: freeze has {payload.get(key)!r}, required {expected!r}"
        for key, expected in required.items()
        if payload.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "N170 freeze JSON does not match directed+cov+random-phase B=1000: "
            + "; ".join(mismatches)
        )
    payload["_freeze_path"] = str(freeze_path)
    return payload


def frozen_estimator_record(freeze: dict[str, Any], *, used_B: int) -> dict[str, Any]:
    return {
        "source": "N170 Tracks A+B leading freeze, unmodified",
        "joint_candidate_id": freeze.get("joint_candidate_id"),
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
        "note": (
            "This is the N170 freeze applied to MEG. It is not a retune. "
            "AIRI 99–999 ms crop and 0.25–20 Hz filtfilt are NOT applied on "
            "the paper-epoch path."
        ),
    }


def meg_seed_policy() -> dict[str, Any]:
    return {
        "bit_generator": "PCG64",
        "master_seed": MASTER_SEED,
        "api": "numpy.random.default_rng / PCG64, same CLI seed as meg/run.py",
        "stream_splits": {
            "paper_faithful_perm": "seed+1 (existing meg/run.py; not used here)",
            "paper_faithful_fwer": "seed+2 (existing; not used here)",
            "airi_executable_random_phase": "seed+3 (existing; JSON reused, not re-drawn)",
            "airi_pair_order": "seed+10 (existing; not used here)",
            "historical_candidate_paper_epoch_random_phase": (
                f"seed+20 = {PAPER_EPOCH_RANDOM_PHASE_SEED}; one Generator "
                "consumed in RDM order face, tool, meaning, facevstool"
            ),
            "historical_candidate_secondary_label_perm": (
                f"seed+21 = {PAPER_EPOCH_SECONDARY_PERM_SEED} reserved; "
                "secondary test enumerates all 6! = 720 permutations (no RNG)"
            ),
        },
    }
