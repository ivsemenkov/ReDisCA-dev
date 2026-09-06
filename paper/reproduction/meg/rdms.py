"""MEG theoretical RDMs: AIRI numeric matrices and binary variants."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

CONDITION_NAMES: tuple[str, ...] = (
    "face1",
    "face2",
    "tool1",
    "tool2",
    "nons1",
    "nons2",
)
AIRI_RDM_NAMES: tuple[str, ...] = ("face", "tool", "meaning", "facevstool")
BINARY_RDM_NAMES: tuple[str, ...] = ("face", "tool", "meaning")
ContrastConvention = Literal["paper", "airi"]

PAPER_QUALITATIVE_ONSETS: dict[str, dict[str, str]] = {
    "face": {
        "comp1": "differential from 65 ms; peak 160 ms; rises again from 311 ms",
        "comp2": "first significance 218 ms",
        "comp3": "late sustained from 273 ms",
    },
    "tool": {
        "comp1": "210 ms",
        "comp3": "from ~240 ms",
    },
    "meaning": {
        "comp1": "from 160 ms",
        "comp3": "early 182 ms and late 675 ms",
    },
    "facevstool": {
        "comp1": "tools vs faces from 202 ms",
    },
}


def airi_rdm(name: str) -> NDArray[np.float64]:
    """6x6 AIRI theoretical RDM copied from the pinned MATLAB script."""
    d = np.zeros((6, 6), dtype=np.float64)
    if name == "face":
        d[0, 1] = 0.1
        d[0, 2] = d[0, 3] = d[0, 4] = d[0, 5] = 1
        d[1, 2] = d[1, 3] = d[1, 4] = d[1, 5] = 1
        d[2, 3] = d[2, 4] = d[2, 5] = 0.1
        d[3, 4] = d[3, 5] = 0.1
        d[4, 5] = 0.1
    elif name == "facevstool":
        d[0, 1] = 0.1
        d[0, 2] = d[0, 3] = 1
        d[0, 4] = d[0, 5] = 0.5
        d[1, 2] = d[1, 3] = 1
        d[1, 4] = d[1, 5] = 0.5
        d[2, 3] = 0.1
        d[2, 4] = d[2, 5] = 0.5
        d[3, 4] = d[3, 5] = 0.5
        d[4, 5] = 0.1
    elif name == "tool":
        d[0, 1] = 0.1
        d[0, 2] = d[0, 3] = 1
        d[0, 4] = d[0, 5] = 0.1
        d[1, 2] = d[1, 3] = 1
        d[1, 4] = d[1, 5] = 0.1
        d[2, 3] = 0.1
        d[2, 4] = d[2, 5] = 1
        d[3, 4] = d[3, 5] = 1
        d[4, 5] = 0.1
    elif name == "meaning":
        d[0, 1] = d[0, 2] = d[0, 3] = 0.1
        d[0, 4] = d[0, 5] = 1
        d[1, 2] = d[1, 3] = 0.1
        d[1, 4] = d[1, 5] = 1
        d[2, 3] = 0.1
        d[2, 4] = d[2, 5] = 1
        d[3, 4] = d[3, 5] = 1
        d[4, 5] = 0.1
    else:
        raise ValueError(f"Unknown AIRI RDM name {name!r}")
    return d + d.T


def theoretical_rdm(name: str, *, fill: Literal["airi", "binary"]) -> NDArray[np.float64]:
    numeric = airi_rdm(name)
    if fill == "airi":
        return numeric
    if name == "facevstool":
        raise ValueError("facevstool has no 0/1 binary analogue")
    binary = np.array(numeric, dtype=np.float64, copy=True)
    binary[np.isclose(binary, 0.1)] = 0.0
    np.fill_diagonal(binary, 0.0)
    return binary


def class_labels(name: str, *, convention: ContrastConvention) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if convention == "airi":
        mapping = {
            "face": ((0, 1), (4, 5)),
            "facevstool": ((0, 1), (2, 3)),
            "tool": ((2, 3), (4, 5)),
            "meaning": ((0, 1, 2, 3), (4, 5)),
        }
        return mapping[name]
    mapping = {
        "face": ((0, 1), (2, 3, 4, 5)),
        "tool": ((2, 3), (0, 1, 4, 5)),
        "meaning": ((0, 1, 2, 3), (4, 5)),
        "facevstool": ((0, 1), (2, 3)),
    }
    return mapping[name]
