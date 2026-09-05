"""Deterministic seed and RNG provenance helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

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


def spawned_generator(
    master_seed: int,
    *keys: int | str,
) -> tuple[np.random.Generator, int, RNGRecord]:
    """Derive a child Generator from a master seed and integer/string keys.

    The integer ``child_seed`` is a stable 32-bit digest of the SeedSequence
    and can be stored so future ablations regenerate the same stream.
    """
    entropy: list[int] = [int(master_seed)]
    extra_keys: list[Any] = []
    for key in keys:
        if isinstance(key, str):
            extra_keys.append(key)
            entropy.append(int.from_bytes(key.encode("utf-8"), "little") % (2**32))
        else:
            extra_keys.append(int(key))
            entropy.append(int(key))
    sequence = np.random.SeedSequence(entropy)
    child_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
    generator = np.random.default_rng(sequence)
    record = RNGRecord(
        library="numpy",
        bit_generator="PCG64",
        seed=child_seed,
        extra={
            "api": "numpy.random.SeedSequence",
            "master_seed": int(master_seed),
            "keys": extra_keys,
        },
    )
    return generator, child_seed, record
