"""Theoretical 4x4 RDMs for Ossadtchi et al. 2024 N170 figures 7a and 9a,b.

Paper figures are images; no numeric matrix is printed. Condition order matches
ERP CORE / ERPLAB bins of ``1_N170_erp_ar.erp`` (correct-response only):

    0 Faces, 1 Cars, 2 Scrambled Faces, 3 Scrambled Cars

Prose (NeuroImage §4.2.1):

- Fig. 7a meaning: faces+cars similar, scrambled similar, groups dissimilar.
- Fig. 9a face: face unlike the other three, the other three similar.
- Fig. 9b car: car unlike the other three, the other three similar.

Default fill is binary 0/1 (within=0, between=1, diagonal=0). An optional
within=0.1 variant is emitted as a labeled extra (AIRI MEG style). For these
two-level partition RDMs, z-scoring the unique-pair vector (library
``standardize_target``, sample SD) makes within=0 and within=0.1 identical
up to floating-point error, so ReDisCA filters/eigenvalues match.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

ConditionName = Literal[
    "faces",
    "cars",
    "scrambled_faces",
    "scrambled_cars",
]

CONDITION_ORDER: tuple[ConditionName, ...] = (
    "faces",
    "cars",
    "scrambled_faces",
    "scrambled_cars",
)

CONDITION_LABELS: tuple[str, ...] = (
    "Faces",
    "Cars",
    "Scrambled Faces",
    "Scrambled Cars",
)

MEANINGFUL_INDICES: tuple[int, ...] = (0, 1)  # faces, cars
MEANINGLESS_INDICES: tuple[int, ...] = (2, 3)  # scrambled faces, scrambled cars
FACE_INDEX = 0
CAR_INDEX = 1

RdmKind = Literal["meaning", "face", "car"]


def _empty_rdm(within: float) -> NDArray[np.float64]:
    matrix = np.full((4, 4), float(within), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def meaning_rdm(*, within: float = 0.0, between: float = 1.0) -> NDArray[np.float64]:
    """Fig. 7a: meaningful (face, car) vs meaningless (scrambled)."""
    matrix = _empty_rdm(within)
    for i in MEANINGFUL_INDICES:
        for j in MEANINGLESS_INDICES:
            matrix[i, j] = float(between)
            matrix[j, i] = float(between)
    return matrix


def face_rdm(*, within: float = 0.0, between: float = 1.0) -> NDArray[np.float64]:
    """Fig. 9a: face-specific detector (face distinct, other three similar)."""
    matrix = _empty_rdm(within)
    for j in range(4):
        if j == FACE_INDEX:
            continue
        matrix[FACE_INDEX, j] = float(between)
        matrix[j, FACE_INDEX] = float(between)
    return matrix


def car_rdm(*, within: float = 0.0, between: float = 1.0) -> NDArray[np.float64]:
    """Fig. 9b: car-specific detector (car distinct, other three similar)."""
    matrix = _empty_rdm(within)
    for j in range(4):
        if j == CAR_INDEX:
            continue
        matrix[CAR_INDEX, j] = float(between)
        matrix[j, CAR_INDEX] = float(between)
    return matrix


def theoretical_rdm(
    kind: RdmKind,
    *,
    within: float = 0.0,
    between: float = 1.0,
) -> NDArray[np.float64]:
    builders = {
        "meaning": meaning_rdm,
        "face": face_rdm,
        "car": car_rdm,
    }
    if kind not in builders:
        raise ValueError(f"Unknown RDM kind {kind!r}")
    matrix = builders[kind](within=within, between=between)
    if not np.allclose(matrix, matrix.T):
        raise RuntimeError(f"{kind} RDM is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0):
        raise RuntimeError(f"{kind} RDM diagonal is not zero")
    return matrix


def rdm_catalog() -> dict[str, NDArray[np.float64]]:
    """Binary 0/1 RDMs plus labeled 0.1-within extras."""
    catalog: dict[str, NDArray[np.float64]] = {}
    for kind in ("meaning", "face", "car"):
        catalog[kind] = theoretical_rdm(kind, within=0.0, between=1.0)
        catalog[f"{kind}_within0.1"] = theoretical_rdm(kind, within=0.1, between=1.0)
    return catalog


def unique_pair_values(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    """Upper-triangle ``i < j`` entries in row-major pair order."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def zscored_unique_pairs(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    """Match library ``standardize_target`` (sample SD, ``ddof=1``)."""
    values = unique_pair_values(matrix)
    centered = values - np.mean(values)
    scale = float(np.std(centered, ddof=1))
    if scale == 0.0:
        raise ValueError("RDM unique-pair vector has zero sample SD")
    return centered / scale
