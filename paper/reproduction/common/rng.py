"""Deterministic seed and RNG provenance helpers.

These helpers record *how* randomness was requested. They do not choose
scientific null models; those live in the individual reproduction tracks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RNGRecord:
    library: str
    bit_generator: str
    seed: int | None
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def numpy_generator(seed: int) -> tuple[np.random.Generator, RNGRecord]:
    """Return a ``numpy.random.Generator`` with PCG64 and a provenance record."""
    generator = np.random.default_rng(seed)
    record = RNGRecord(
        library="numpy",
        bit_generator="PCG64",
        seed=int(seed),
        extra={"api": "numpy.random.default_rng"},
    )
    return generator, record
