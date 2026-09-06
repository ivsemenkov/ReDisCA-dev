"""Stage A architecture guards: one ReDisCA factory, no duplicate core."""

from __future__ import annotations

from pathlib import Path

from paper.reproduction.common.constants import MASTER_SEEDS
from paper.reproduction.common.method import AIRI_SPOC_KWARGS, make_redisca


ROOT = Path(__file__).resolve().parents[1]


def test_master_seeds_are_frozen():
    assert MASTER_SEEDS == (20240904, 20240905, 20240906, 20240907, 20240908)


def test_airi_spoc_kwargs_are_literal():
    assert AIRI_SPOC_KWARGS == dict(
        n_components=None,
        demean_time=True,
        divide_by_t_minus_1=True,
        directed_pairs=True,
        aggregation="mean",
        solver="whitening",
        rank=None,
        rank_tol=1e-6,
    )
    model = make_redisca()
    for key, value in AIRI_SPOC_KWARGS.items():
        assert getattr(model, key) == value


def _runtime_modules():
    for path in ROOT.rglob("*.py"):
        if path.name.startswith("test_") or path.name == "conftest.py":
            continue
        if "tests" in path.parts or "validation" in path.parts:
            continue
        yield path


def test_no_source_faithful_or_git_show_dependency():
    offenders = []
    for path in _runtime_modules():
        text = path.read_text(encoding="utf-8")
        for token in ("source_faithful", "git show", "git_show"):
            if token in text:
                offenders.append(f"{path}: {token}")
    assert not offenders, offenders


def test_no_direct_redisca_constructor_outside_factory():
    allowed = {ROOT / "common" / "method.py"}
    offenders = []
    for path in _runtime_modules():
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "ReDisCA(" in text:
            offenders.append(str(path))
    assert not offenders, offenders
