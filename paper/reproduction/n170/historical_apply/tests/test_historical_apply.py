"""Unit tests for the N170 historical-apply track.

Do not require B=1000. New modules must not import redisca.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from common.source_faithful import directed_pairs

from historical_apply.analysis import (
    fit_frozen_window,
    p_lt_segments,
    run_meaning_scan,
)
from historical_apply.freeze import (
    DEFAULT_FREEZE_PATH,
    FROZEN_B,
    FROZEN_INFERENCE,
    FROZEN_MATRIX_MODE,
    FROZEN_PAIR_MODE,
    load_and_validate_freeze,
)
from historical_apply.run import parse_args

APPLY_DIR = Path(__file__).resolve().parents[1]


def _imported_top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_historical_apply_modules_do_not_import_redisca() -> None:
    py_files = [
        path for path in APPLY_DIR.rglob("*.py") if path.name != "__pycache__"
    ]
    offenders: list[str] = []
    for path in py_files:
        if "redisca" in _imported_top_level_names(path):
            offenders.append(str(path.relative_to(APPLY_DIR)))
    assert offenders == [], f"historical_apply modules import redisca: {offenders}"


def test_fig7_runner_accepts_freeze_json() -> None:
    freeze = load_and_validate_freeze(DEFAULT_FREEZE_PATH)
    assert freeze["pair_mode"] == FROZEN_PAIR_MODE == "airi_directed"
    assert freeze["matrix_mode"] == FROZEN_MATRIX_MODE == "matlab_cov"
    assert freeze["inference"] == FROZEN_INFERENCE == "spoc_random_phase"
    assert freeze["B"] == FROZEN_B == 1000
    args = parse_args(["--freeze", str(DEFAULT_FREEZE_PATH), "--B", "0"])
    assert args.freeze == DEFAULT_FREEZE_PATH
    assert args.n_bootstrap == 0
    rng = np.random.default_rng(0)
    n_times = 48
    times = np.linspace(-200.0, 800.0, n_times)
    packed = {
        "data": rng.standard_normal((4, 6, n_times)),
        "times_ms": times,
        "channel_labels": [f"ch{i}" for i in range(6)],
    }
    scan = run_meaning_scan(
        packed,
        freeze,
        n_bootstrapping_iterations=0,
        step_ms=200.0,
        n_report=2,
    )
    assert scan["estimator"]["pair_mode"] == "airi_directed"
    assert scan["estimator"]["matrix_mode"] == "matlab_cov"
    assert scan["estimator"]["inference"] == "spoc_random_phase"
    assert scan["estimator"]["freeze_B"] == 1000
    assert scan["estimator"]["used_B"] == 0
    assert scan["estimator"]["reduced_B"] is True
    assert scan["step_ms"] == 200.0
    assert scan["duration_ms"] == 150.0
    assert scan["n_windows"] >= 1
    assert scan["windows"][0]["primary_p1"] is None
    assert scan["rdm_fill"] == "binary_0_1"
    assert scan["imports_redisca"] is False


def test_rejects_retuned_freeze(tmp_path: Path) -> None:
    freeze = json.loads(Path(DEFAULT_FREEZE_PATH).read_text(encoding="utf-8"))
    freeze["pair_mode"] = "unique_unordered"
    bad = tmp_path / "bad_freeze.json"
    bad.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="pair_mode"):
        load_and_validate_freeze(bad)


def test_smoke_fit_b0() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((4, 5, 8))
    rdm = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    payload = fit_frozen_window(
        X_window=X,
        rdm=rdm,
        channel_labels=[f"ch{i}" for i in range(5)],
        n_bootstrapping_iterations=0,
        rng=np.random.Generator(np.random.PCG64(4)),
        n_report=2,
    )
    assert payload["pair_mode"] == "airi_directed"
    assert payload["matrix_mode"] == "matlab_cov"
    assert payload["pair_sequence"] == [list(p) for p in directed_pairs(4)]
    assert payload["primary_random_phase_p_head"][0] is None
    assert payload["n_pairs"] == 12
    assert payload["extras"]["condition_label_permutation"]["n_permutations"] == 24
    assert payload["components"][0]["pattern_max_abs_channel"].startswith("ch")


def test_p_lt_segments_finds_contiguous_run() -> None:
    centers = np.array([350.0, 375.0, 400.0, 425.0, 450.0])
    p = np.array([0.2, 0.01, 0.02, 0.04, 0.9])
    segs = p_lt_segments(centers, p, alpha=0.05)
    assert len(segs) == 1
    assert segs[0]["center_ms_start"] == 375.0
    assert segs[0]["center_ms_end"] == 425.0
    assert segs[0]["n_windows"] == 3
