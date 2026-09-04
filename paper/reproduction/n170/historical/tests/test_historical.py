"""Unit tests for the N170 historical track.

Do not require B=1000. Historical modules must not import redisca.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from common.source_faithful import directed_pairs, unique_unordered_pairs

from historical.analysis import fit_variant
from historical.variants import (
    MASTER_SEED,
    make_spec,
    track_a_specs,
    variant_id_for,
)
from historical.select import pick_two_joint_candidates

HISTORICAL_DIR = Path(__file__).resolve().parents[1]


def _imported_top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_historical_modules_do_not_import_redisca() -> None:
    py_files = [
        path
        for path in HISTORICAL_DIR.rglob("*.py")
        if path.name != "__pycache__"
    ]
    offenders: list[str] = []
    for path in py_files:
        if "redisca" in _imported_top_level_names(path):
            offenders.append(str(path.relative_to(HISTORICAL_DIR)))
    assert offenders == [], f"historical modules import redisca: {offenders}"


def test_track_a_catalog_is_twelve_source_supported_variants() -> None:
    specs = track_a_specs()
    assert len(specs) == 12
    faces = [s for s in specs if s.contrast == "face"]
    cars = [s for s in specs if s.contrast == "car"]
    assert len(faces) == 4
    assert len(cars) == 8
    assert {s.window_center_ms for s in faces} == {200.0}
    assert {s.window_duration_ms for s in specs} == {100.0}
    assert {s.window_center_ms for s in cars} == {170.0, 200.0}
    ids = [s.variant_id for s in specs]
    assert len(ids) == len(set(ids))
    assert all(s.rng_seed == MASTER_SEED + s.seed_offset for s in specs)


def test_pair_sequences_match_source_faithful_helpers() -> None:
    n_conditions = 4
    unique_spec = make_spec("face", 200.0, "unique_unordered", "unscaled_gram")
    directed_spec = make_spec("car", 170.0, "airi_directed", "matlab_cov")
    from common.source_faithful import pair_indices

    unique_pairs = pair_indices(n_conditions, unique_spec.pair_mode)
    directed = pair_indices(n_conditions, directed_spec.pair_mode)
    assert unique_pairs == unique_unordered_pairs(n_conditions)
    assert directed == directed_pairs(n_conditions)
    assert unique_pairs == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    assert directed[0] == (0, 1) and directed[3] == (1, 0)
    assert (0, 1) in directed and (1, 0) in directed
    assert variant_id_for("face", 200.0, "unique_unordered", "unscaled_gram") == (
        "face_c200_d100_unique_unordered_unscaled_gram"
    )


def test_smoke_fit_tiny_synthetic_b2() -> None:
    rng = np.random.default_rng(0)
    n_conditions, n_channels, n_times = 4, 8, 16
    X = rng.standard_normal((n_conditions, n_channels, n_times))
    # Inject a face-like amplitude contrast so the GEP is non-degenerate.
    X[0] += 2.5 * rng.standard_normal((n_channels, n_times))
    rdm = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    times_ms = np.linspace(-200.0, 800.0, n_times)
    spec = make_spec("face", 200.0, "unique_unordered", "unscaled_gram")
    window_meta = {
        "center_ms": 200.0,
        "duration_ms": 100.0,
        "t_start_ms": 150.0,
        "t_end_ms": 250.0,
        "n_samples": n_times,
        "index_start": 0,
        "index_end_inclusive": n_times - 1,
    }
    payload = fit_variant(
        X_window=X,
        X_full=X,
        times_full_ms=times_ms,
        channel_labels=[f"ch{i}" for i in range(n_channels)],
        condition_labels=["Faces", "Cars", "Scrambled Faces", "Scrambled Cars"],
        rdm=rdm,
        spec=spec,
        rng=np.random.Generator(np.random.PCG64(1)),
        n_bootstrapping_iterations=2,
        window_meta=window_meta,
        n_report=3,
        matched_rng=np.random.Generator(np.random.PCG64(2)),
    )
    assert payload["variant_id"] == spec.variant_id
    assert payload["n_channels"] == n_channels
    assert payload["n_samples"] == n_times
    assert payload["pair_sequence"] == [list(p) for p in unique_unordered_pairs(4)]
    assert len(payload["z_after_matlab_zscore"]) == 6
    assert len(payload["eigenvalues_head"]) >= 3
    assert payload["primary_random_phase_p_head"][0] in {0.0, 0.5, 1.0}
    assert payload["numerical_rank_whitening_rows"] >= 1
    assert payload["extras"]["condition_label_permutation"]["n_permutations"] == 24


def test_smoke_fit_b0_skips_primary_p() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((4, 5, 8))
    rdm = np.zeros((4, 4))
    rdm[0, 1] = rdm[1, 0] = 1.0
    rdm[0, 2] = rdm[2, 0] = 1.0
    rdm[0, 3] = rdm[3, 0] = 1.0
    spec = make_spec("car", 170.0, "airi_directed", "matlab_cov")
    window_meta = {
        "center_ms": 170.0,
        "duration_ms": 100.0,
        "t_start_ms": 120.0,
        "t_end_ms": 220.0,
        "n_samples": 8,
        "index_start": 0,
        "index_end_inclusive": 7,
    }
    payload = fit_variant(
        X_window=X,
        X_full=X,
        times_full_ms=np.linspace(0.0, 100.0, 8),
        channel_labels=[f"ch{i}" for i in range(5)],
        condition_labels=["Faces", "Cars", "Scrambled Faces", "Scrambled Cars"],
        rdm=rdm,
        spec=spec,
        rng=np.random.Generator(np.random.PCG64(4)),
        n_bootstrapping_iterations=0,
        window_meta=window_meta,
        n_report=2,
    )
    assert payload["pair_sequence"] == [list(p) for p in directed_pairs(4)]
    assert payload["primary_random_phase_p_head"][0] is None
    assert payload["inference_primary"]["B"] == 0


def test_pick_two_joint_candidates_lexicographic() -> None:
    def _row(
        *,
        contrast: str,
        pair_mode: str,
        matrix_mode: str,
        center: float,
        lam: list[float],
        p: list[float],
        corr: float,
    ) -> dict:
        vid = variant_id_for(contrast, center, pair_mode, matrix_mode)  # type: ignore[arg-type]
        return {
            "variant_id": vid,
            "contrast": contrast,
            "pair_mode": pair_mode,
            "matrix_mode": matrix_mode,
            "window_center_ms": center,
            "window_duration_ms": 100.0,
            "eigenvalues_head": lam,
            "primary_p_head": p,
            "corr_wTRw_comp0": corr,
            "corr_trace_sq_comp0": corr,
            "faces_peak_ms": 170.0,
            "cars_peak_ms": 150.0,
        }

    rows = []
    for pair_mode in ("unique_unordered", "airi_directed"):
        for matrix_mode in ("unscaled_gram", "matlab_cov"):
            rows.append(
                _row(
                    contrast="face",
                    pair_mode=pair_mode,
                    matrix_mode=matrix_mode,
                    center=200.0,
                    lam=[0.87, 0.5],
                    p=[0.0, 0.2],
                    corr=0.99,
                )
            )
            rows.append(
                _row(
                    contrast="car",
                    pair_mode=pair_mode,
                    matrix_mode=matrix_mode,
                    center=170.0,
                    lam=[0.91, 0.77],
                    p=[0.0, 0.02],
                    corr=0.99,
                )
            )
            rows.append(
                _row(
                    contrast="car",
                    pair_mode=pair_mode,
                    matrix_mode=matrix_mode,
                    center=200.0,
                    lam=[0.80, 0.60],
                    p=[0.1, 0.4],
                    corr=0.99,
                )
            )
    picked = pick_two_joint_candidates(rows)
    assert len(picked) == 2
    assert picked[0]["car_window"]["center_ms"] == 170.0
    assert picked[0]["face_p1_is_zero"] is True
    assert picked[1]["id"] != picked[0]["id"]
