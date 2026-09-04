"""Unit tests for MEG historical-candidate application of the N170 freeze.

Do not require B=1000. New modules must not import redisca.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from common.source_faithful import directed_pairs, pair_indices, pair_stack_from_condition_averages

from meg.historical_candidate.analysis import fit_one_rdm, run_paper_epoch
from meg.historical_candidate.freeze import (
    FROZEN_B,
    FROZEN_INFERENCE,
    FROZEN_MATRIX_MODE,
    FROZEN_PAIR_MODE,
    N170_FREEZE_PATH,
    load_n170_freeze,
)
from meg.historical_candidate.reuse import (
    reuse_airi_executable,
    verify_airi_executable_payload,
)
from meg.historical_candidate.run import parse_args

CANDIDATE_DIR = Path(__file__).resolve().parents[1]


def _imported_top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_historical_candidate_modules_do_not_import_redisca() -> None:
    py_files = [
        path for path in CANDIDATE_DIR.rglob("*.py") if path.name != "__pycache__"
    ]
    offenders: list[str] = []
    for path in py_files:
        if "redisca" in _imported_top_level_names(path):
            offenders.append(str(path.relative_to(CANDIDATE_DIR)))
    assert offenders == [], f"historical_candidate modules import redisca: {offenders}"


def test_n170_freeze_json_is_accepted() -> None:
    freeze = load_n170_freeze(N170_FREEZE_PATH)
    assert freeze["pair_mode"] == FROZEN_PAIR_MODE == "airi_directed"
    assert freeze["matrix_mode"] == FROZEN_MATRIX_MODE == "matlab_cov"
    assert freeze["inference"] == FROZEN_INFERENCE == "spoc_random_phase"
    assert freeze["B"] == FROZEN_B == 1000
    args = parse_args(["reuse"])
    assert args.command == "reuse"
    args_b = parse_args(["paper-epoch", "--B", "0"])
    assert args_b.n_bootstrap == 0


def test_airi_executable_reuse_matches_freeze_without_recompute() -> None:
    payload = reuse_airi_executable()
    assert payload["recomputed"] is False
    assert payload["verified_directed_cov_random_phase_B1000"] is True
    for name in ("face", "tool", "meaning", "facevstool"):
        row = payload["rdms"][name]
        assert row["recomputed_B1000"] is False
        assert row["verification"]["ok"] is True
        assert row["rank"] == 68
        assert len(row["p_spoc_head"]) >= 3
        assert row["verification"]["checks"]["B_1000"] is True


def test_verify_rejects_wrong_B() -> None:
    fake = {
        "path_label": "airi_executable",
        "pairs": "airi_directed_i_neq_j",
        "pair_matrix": "matlab_cov",
        "inference_component": {
            "name": "spoc_random_phase",
            "B": 50,
            "used_B": 50,
            "reduced_B": True,
        },
        "n_planars": 204,
        "n_samples": 901,
        "bandpass": {"low_hz": 0.25, "high_hz": 20.0, "butter_order": 3},
    }
    report = verify_airi_executable_payload(fake)
    assert report["ok"] is False
    assert report["checks"]["B_1000"] is False


def test_smoke_fit_b0_synthetic() -> None:
    rng = np.random.default_rng(7)
    n_conditions, n_channels, n_times = 6, 8, 16
    X = rng.standard_normal((n_conditions, n_channels, n_times))
    X[0] += 1.5
    X[1] += 1.5
    pairs = pair_indices(n_conditions, "airi_directed")
    stack = pair_stack_from_condition_averages(X, pairs, matrix_mode="matlab_cov")
    times = np.linspace(-500.0, 1000.0, n_times)
    payload = fit_one_rdm(
        pair_stack=stack,
        averages=X,
        time_ms=times,
        rdm_name="face",
        n_bootstrapping_iterations=0,
        rng=np.random.Generator(np.random.PCG64(8)),
        n_report=3,
    )
    assert payload["pairs"] == "airi_directed_i_neq_j"
    assert payload["pair_matrix"] == "matlab_cov"
    assert payload["bandpass"] is None
    assert payload["n_pairs"] == len(directed_pairs(6)) == 30
    assert payload["primary_random_phase_p_head"][0] is None
    assert payload["extras"]["condition_label_permutation"]["n_permutations"] == 720
    assert payload["imports_redisca"] is False


def test_run_paper_epoch_rejects_wrong_shape() -> None:
    X = np.zeros((6, 10, 20))
    times = np.linspace(-500.0, 1000.0, 20)
    with pytest.raises(ValueError, match="204 planars"):
        run_paper_epoch(
            X,
            times,
            n_bootstrapping_iterations=0,
            rng=np.random.Generator(np.random.PCG64(1)),
            rdm_names=("face",),
        )
