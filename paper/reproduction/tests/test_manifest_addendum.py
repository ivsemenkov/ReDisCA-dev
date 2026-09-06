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


def test_review2_addendum_does_not_rewrite_original_or_first_addendum():
    original = json.loads((REPO_ROOT / "paper" / "reproduction_manifest.json").read_text())
    first = json.loads(
        (REPO_ROOT / "paper" / "reproduction_manifest_addendum.json").read_text()
    )
    review2 = json.loads(
        (REPO_ROOT / "paper" / "reproduction_manifest_addendum_review2.json").read_text()
    )
    assert original["frozen_before_full_results"] is True
    assert "SIM-C1" not in [c["id"] for c in original["candidates"]["simulations"]]
    first_ids = [c["id"] for c in first["candidates_added"]]
    assert "SIM-C1" not in first_ids
    assert "EQ16-CAUSAL" not in first_ids
    assert review2["added_after_stage_a_review"] is True
    assert review2["not_pre_registered"] is True
    added = {c["id"]: c for c in review2["candidates_added"]}
    for cid in ("SIM-C1", "SIM-C2", "SIM-C3", "SIM-C4", "SIM-CR1", "EQ16-CAUSAL", "EQ16-CAUSAL-D"):
        assert added[cid]["added_after_stage_a_review"] is True
        assert added[cid]["not_pre_registered"] is True
    assert added["SIM-C1"]["inherits"] == "SIM-P3"
    assert added["EQ16-CAUSAL"]["values_changed_vs_SIM-P3"]["eq16_single_matrix"] is True
    assert added["EQ16-CAUSAL"]["fig4"] == "not applicable; Eq. (15) is trial-indexed"
    assert review2["fig18_forward_model_reclassification"]["anatomical_reproduction_testable"] is False


def test_completeness_blocks_verdict_while_review2_jobs_missing():
    from paper.reproduction.summarize import _completeness, _required_sim_jobs

    fake_old = []
    for cand, exp, snr, seed in _required_sim_jobs():
        if cand in {
            "SIM-P1",
            "SIM-P2",
            "SIM-P3",
            "SIM-P4",
            "SIM-P5",
            "SIM-P6",
            "SIM-P7",
            "SIM-P8",
            "SIM-R1",
        }:
            fake_old.append(
                {
                    "candidate": cand,
                    "experiment": "fig4_single_source" if exp == "fig4" else "fig5_fig6_multi_source",
                    "snr": snr,
                    "seed": seed,
                    "n_mc": 100,
                }
            )
    # Pretend MEG/Fig18 already exist so only sim jobs can block.
    meg = [
        {"candidate": cand, "seed": seed, "airi_literal_indexing": "literal"}
        for cand in ("MEG-AIRI", "MEG-PAPER-1501", "MEG-PAPER-1500")
        for seed in (20240904, 20240905, 20240906, 20240907, 20240908)
    ]
    fig18 = [
        {
            "candidate": "FIG18-MUSIC-LOWESTP",
            "meg_candidate": meg_c,
            "seed": seed,
            "subspace_dim": 3,
        }
        for meg_c in ("MEG-PAPER-1501", "MEG-AIRI")
        for seed in (20240904, 20240905, 20240906, 20240907, 20240908)
    ]
    status = _completeness(fake_old, fig18, meg)
    assert status["n_completed_sim_jobs"] == 160
    assert status["n_required_sim_jobs"] == 280
    assert status["stage_a_complete"] is False
    assert status["final_verdict_allowed"] is False
    assert any(m["candidate"] == "SIM-C2" for m in status["missing_simulations"])
    assert any(m["candidate"] == "EQ16-CAUSAL" for m in status["missing_simulations"])
