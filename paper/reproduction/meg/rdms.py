"""MEG theoretical RDMs: AIRI numeric matrices and 0/1 binary variants.

Condition order (AIRI and this track): face1, face2, tool1, tool2, nons1, nons2.

Paper Figs 12a–c are images; numeric entries are not printed. AIRI uses 0.1
within-category and 1 between (D7). This module *emits both*. Component fits
choose which fill to use; they must not silently mix them.

Fig. 16 / AIRI default ``facevstool`` is non-binary (0.1 / 0.5 / 1). It is not
a 0/1 matrix and is not Fig. 12a.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from common.source_faithful import airi_rdm

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

# Paper §4.2.2 qualitative onsets / peaks. Not numeric targets to force-match.
PAPER_QUALITATIVE_ONSETS: dict[str, dict[str, str]] = {
    "face": {
        "comp1": "differential from 65 ms; peak 160 ms; rises again from 311 ms; central/right occipital",
        "comp2": "parietal; first significance 218 ms",
        "comp3": "late sustained from 273 ms; bilateral occipital / some frontal",
    },
    "tool": {
        "comp1": "210 ms; central occipito-parietal, left (and compact right) central sulcus",
        "comp2": "later; mid-occipital and left parietal",
        "comp3": "from ~240 ms; right and left sensory-motor / temporal, right-prevalent",
    },
    "meaning": {
        "comp1": "ventral; meaning vs nonsense from 160 ms occipito-parietal",
        "comp2": "dorsal; meaning vs nonsense from 160 ms",
        "comp3": "early 182 ms and late 675 ms; focal mid-parietal",
    },
    "facevstool": {
        "comp1": "tools vs faces from 202 ms; left parietal / sensory-motor",
        "comp2": "visual / FG-like late ~260–350 ms for tools",
        "comp3": "face-specific peak ~160 ms",
    },
}

ContrastConvention = Literal["paper", "airi"]


def theoretical_rdm(name: str, *, fill: Literal["airi", "binary"]) -> NDArray[np.float64]:
    """Return a 6×6 theoretical RDM.

    ``fill='airi'`` is ``source_faithful.airi_rdm``.
    ``fill='binary'`` maps AIRI within-category 0.1 → 0 and keeps 1s.
    ``facevstool`` has no 0/1 binary analogue; ``fill='binary'`` is rejected.
    """
    if name not in AIRI_RDM_NAMES:
        raise ValueError(f"Unknown MEG RDM name {name!r}; expected one of {AIRI_RDM_NAMES}")
    numeric = airi_rdm(name)
    if fill == "airi":
        return numeric
    if fill != "binary":
        raise ValueError(f"Unknown fill {fill!r}")
    if name == "facevstool":
        raise ValueError(
            "facevstool is the Fig. 16 non-binary geometry (0.1/0.5/1); "
            "there is no 0/1 binary variant. Use fill='airi'."
        )
    binary = np.array(numeric, dtype=np.float64, copy=True)
    within = np.isclose(binary, 0.1)
    binary[within] = 0.0
    off = ~np.eye(binary.shape[0], dtype=bool)
    allowed = np.isclose(binary, 0.0) | np.isclose(binary, 1.0)
    if not np.all(allowed | ~off):
        raise RuntimeError(f"binary conversion of {name!r} produced values other than 0/1")
    np.fill_diagonal(binary, 0.0)
    return binary


def rdm_catalog() -> dict[str, dict[str, NDArray[np.float64]]]:
    """Emit every matrix this track is required to write out."""
    catalog: dict[str, dict[str, NDArray[np.float64]]] = {
        "airi_numeric": {name: airi_rdm(name) for name in AIRI_RDM_NAMES},
        "binary_0_1": {name: theoretical_rdm(name, fill="binary") for name in BINARY_RDM_NAMES},
    }
    return catalog


def class_labels(name: str, *, convention: ContrastConvention) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """0-based class1 / class2 indices used for time-series contrasts.

    Paper convention follows the prose (faces vs *others*, tools vs others,
    meaning vs nonsense, non-binary faces vs tools).

    AIRI convention copies ``Class1Label`` / ``Class2Label`` in
    ``Redisca_tools_faces_3_random_norm_correct.m`` (face is faces vs *nons*,
    not faces vs others).
    """
    if name not in AIRI_RDM_NAMES:
        raise ValueError(f"Unknown MEG RDM name {name!r}")
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


def unique_upper_entries(rdm: NDArray[np.floating]) -> NDArray[np.float64]:
    """Paper Eq. 2 vector: unique unordered upper triangle, row-major i<j."""
    rdm = np.asarray(rdm, dtype=np.float64)
    n = rdm.shape[0]
    return np.array([rdm[i, j] for i in range(n) for j in range(i + 1, n)], dtype=np.float64)
