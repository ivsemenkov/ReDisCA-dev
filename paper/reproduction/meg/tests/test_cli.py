"""CLI smoke test (no MEG array load)."""

from __future__ import annotations

from meg.run import parse_args


def test_parse_all_and_quick() -> None:
    args = parse_args(["all", "--quick"])
    assert args.command == "all"
    assert args.quick
    assert args.airi_B <= 32
    assert args.paper_B <= 32
    assert args.airi_nmc <= 20


def test_parse_paper_defaults() -> None:
    args = parse_args(["paper"])
    assert args.paper_B == 500
    assert args.airi_B == 1000
    assert args.paper_nmc == 200
    assert args.airi_nmc == 100
