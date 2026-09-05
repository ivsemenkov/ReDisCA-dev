"""Theoretical 4x4 RDMs for Ossadtchi et al. 2024 N170 figures 7a and 9a,b."""

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

MEANINGFUL_INDICES: tuple[int, ...] = (0, 1)
MEANINGLESS_INDICES: tuple[int, ...] = (2, 3)
FACE_INDEX = 0
CAR_INDEX = 1
RdmKind = Literal["meaning", "face", "car"]


def _empty_rdm(within: float) -> NDArray[np.float64]:
    matrix = np.full((4, 4), float(within), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def meaning_rdm(*, within: float = 0.0, between: float = 1.0) -> NDArray[np.float64]:
    matrix = _empty_rdm(within)
    for i in MEANINGFUL_INDICES:
        for j in MEANINGLESS_INDICES:
            matrix[i, j] = float(between)
            matrix[j, i] = float(between)
    return matrix


def face_rdm(*, within: float = 0.0, between: float = 1.0) -> NDArray[np.float64]:
    matrix = _empty_rdm(within)
    for j in range(4):
        if j == FACE_INDEX:
            continue
        matrix[FACE_INDEX, j] = float(between)
        matrix[j, FACE_INDEX] = float(between)
    return matrix


def car_rdm(*, within: float = 0.0, between: float = 1.0) -> NDArray[np.float64]:
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
    builders = {"meaning": meaning_rdm, "face": face_rdm, "car": car_rdm}
    if kind not in builders:
        raise ValueError(f"Unknown RDM kind {kind!r}")
    matrix = builders[kind](within=within, between=between)
    if not np.allclose(matrix, matrix.T):
        raise RuntimeError(f"{kind} RDM is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0):
        raise RuntimeError(f"{kind} RDM diagonal is not zero")
    return matrix
