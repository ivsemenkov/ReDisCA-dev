"""Original freeze vs review addendum must stay distinguishable."""

from __future__ import annotations

import json
from pathlib import Path

from paper.reproduction.common.paths import REPO_ROOT


def test_original_manifest_is_not_rewritten_as_review():
    original = json.loads((REPO_ROOT / "paper" / "reproduction_manifest.json").read_text())
    assert original["frozen_before_full_results"] is True
    ids = [c["id"] for c in original["candidates"]["simulations"]]
    assert ids == ["SIM-P1", "SIM-P2", "SIM-P3"]
    assert "SIM-P4" not in ids
    assert "added_after_stage_a_review" not in original


def test_addendum_marks_review_candidates():
    addendum = json.loads(
        (REPO_ROOT / "paper" / "reproduction_manifest_addendum.json").read_text()
    )
    assert addendum["added_after_stage_a_review"] is True
    assert addendum["does_not_replace"] == "paper/reproduction_manifest.json"
    added = {c["id"]: c for c in addendum["candidates_added"]}
    assert added["SIM-P4"]["not_pre_registered"] is True
    assert added["SIM-P5"]["not_paper_faithful"] is True
    assert added["AIRI-LITERAL-INDEXING"]["category"] == "B_literal_historical_executable"
    assert added["FIG18-MUSIC-LOWESTP"]["category"] == "A_directly_specified_by_paper"
    assert added["SIM-P6"]["category"] == "C_source_supported_ambiguity"
    assert added["SIM-R1"]["category"] == "D_post_review_forensic_because_literal_is_pathological"
    expansions = {c["id"]: c for c in addendum["coverage_expansions_of_original_candidates"]}
    assert expansions["SIM-P3"]["added_after_stage_a_review"] is True
    assert "Fig. 5/6" in expansions["SIM-P3"]["what"]


def test_summarize_refuses_a_final_verdict_while_matrix_is_open():
    from paper.reproduction.summarize import _completeness

    status = _completeness([], [], [])
    assert status["stage_a_complete"] is False
    assert status["final_verdict_allowed"] is False
    assert "incomplete" in status["status"]
