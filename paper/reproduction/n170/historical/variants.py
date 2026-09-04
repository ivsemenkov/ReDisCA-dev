"""Source-supported N170 historical variants (Track A catalog).

Only the two source-defined pair orders are enumerated. Random-phase
inference is pair-order-sensitive; arbitrary pair permutations are out of
scope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from common.source_faithful import PairMatrixMode, PairMode

ContrastName = Literal["face", "car"]

MASTER_SEED = 20240904
WINDOW_DURATION_MS = 100.0
FACE_CENTER_MS = 200.0
CAR_CENTERS_MS: tuple[float, ...] = (170.0, 200.0)
N_BOOTSTRAP = 1000
N_TRACK_B_SEEDS = 20
N_REPORT_COMPONENTS = 8

# Printed figure-panel fingerprints (not tuning targets).
PRINTED_FACE = {
    "lambda1": 0.87209,
    "p1": 0.0,
    "corr": 0.82,
    "burst_ms": 170.0,
}
PRINTED_CAR = {
    "lambda1": 0.91639,
    "p1": 0.0,
    "lambda2": 0.77036,
    "p2": 0.009,
    "corr_gt": 0.99,
}

# Existing unique+unscaled_gram library numbers (fingerprints.json).
# Used only as a labeled sanity check, not as an oracle and not via redisca.
LIBRARY_UNIQUE_GRAM = {
    "source": "paper/results/n170/fingerprints.json unique+Gram ReDisCA(demean_time=False)",
    "face": {
        "lambda1": 0.8800598487297501,
        "corr_window": 0.9998814542043535,
    },
    "car": {
        "lambda1": 0.8869090680968623,
        "lambda2": 0.791703535919934,
        "corr_window": 0.9999190189565106,
        "corr_window_comp1": 0.9996781244822817,
    },
}

PAIR_MODES: tuple[PairMode, ...] = ("unique_unordered", "airi_directed")
MATRIX_MODES: tuple[PairMatrixMode, ...] = ("unscaled_gram", "matlab_cov")

_PAIR_OFFSET = {"unique_unordered": 0, "airi_directed": 10}
_MATRIX_OFFSET = {"unscaled_gram": 0, "matlab_cov": 1}
_CONTRAST_OFFSET = {"face": 0, "car": 100}
_CENTER_OFFSET = {170.0: 0, 200.0: 20}

# Track B: 20 independent PCG64 seeds, disjoint from Track A variant offsets
# (Track A uses MASTER_SEED + offset with offset < 200).
TRACK_B_SEEDS: tuple[int, ...] = tuple(
    MASTER_SEED + 10_000 + index for index in range(N_TRACK_B_SEEDS)
)


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    contrast: ContrastName
    pair_mode: PairMode
    matrix_mode: PairMatrixMode
    window_center_ms: float
    window_duration_ms: float
    seed_offset: int
    rng_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_offset_for(
    contrast: ContrastName,
    window_center_ms: float,
    pair_mode: PairMode,
    matrix_mode: PairMatrixMode,
) -> int:
    return (
        _CONTRAST_OFFSET[contrast]
        + _CENTER_OFFSET[float(window_center_ms)]
        + _PAIR_OFFSET[pair_mode]
        + _MATRIX_OFFSET[matrix_mode]
    )


def variant_id_for(
    contrast: ContrastName,
    window_center_ms: float,
    pair_mode: PairMode,
    matrix_mode: PairMatrixMode,
) -> str:
    return (
        f"{contrast}_c{int(window_center_ms)}_d{int(WINDOW_DURATION_MS)}"
        f"_{pair_mode}_{matrix_mode}"
    )


def make_spec(
    contrast: ContrastName,
    window_center_ms: float,
    pair_mode: PairMode,
    matrix_mode: PairMatrixMode,
) -> VariantSpec:
    offset = seed_offset_for(contrast, window_center_ms, pair_mode, matrix_mode)
    return VariantSpec(
        variant_id=variant_id_for(
            contrast, window_center_ms, pair_mode, matrix_mode
        ),
        contrast=contrast,
        pair_mode=pair_mode,
        matrix_mode=matrix_mode,
        window_center_ms=float(window_center_ms),
        window_duration_ms=float(WINDOW_DURATION_MS),
        seed_offset=int(offset),
        rng_seed=int(MASTER_SEED + offset),
    )


def track_a_specs() -> list[VariantSpec]:
    """Four face variants (paper window) and eight car variants."""
    specs: list[VariantSpec] = []
    for pair_mode in PAIR_MODES:
        for matrix_mode in MATRIX_MODES:
            specs.append(
                make_spec("face", FACE_CENTER_MS, pair_mode, matrix_mode)
            )
    for center in CAR_CENTERS_MS:
        for pair_mode in PAIR_MODES:
            for matrix_mode in MATRIX_MODES:
                specs.append(make_spec("car", center, pair_mode, matrix_mode))
    if len(specs) != 12:
        raise RuntimeError(f"expected 12 Track A variants, got {len(specs)}")
    return specs


def spec_by_id(variant_id: str) -> VariantSpec:
    for spec in track_a_specs():
        if spec.variant_id == variant_id:
            return spec
    raise KeyError(f"unknown variant id {variant_id!r}")
