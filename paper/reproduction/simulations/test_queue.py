"""Job-queue provenance: required matrix, priority, and stale-lock steal."""

from __future__ import annotations

from paper.reproduction.simulations import queue as queue_mod
from paper.reproduction.summarize import _required_sim_jobs


def test_required_matrix_has_review_branches_and_five_seeds():
    jobs = _required_sim_jobs()
    old = {
        job
        for job in jobs
        if job[0] in {
            "SIM-P1",
            "SIM-P2",
            "SIM-P3",
            "SIM-P4",
            "SIM-P5",
            "SIM-P6",
            "SIM-P7",
            "SIM-P8",
            "SIM-R1",
        }
    }
    assert len(old) == 160
    assert len(jobs) == 280
    cands = {job[0] for job in jobs}
    assert {
        "SIM-P1",
        "SIM-P2",
        "SIM-P3",
        "SIM-P4",
        "SIM-P5",
        "SIM-P6",
        "SIM-P7",
        "SIM-P8",
        "SIM-R1",
        "SIM-C1",
        "SIM-C2",
        "SIM-C3",
        "SIM-C4",
        "SIM-CR1",
        "EQ16-CAUSAL",
        "EQ16-CAUSAL-D",
    } <= cands
    seeds = {job[3] for job in jobs}
    assert seeds == {20240904, 20240905, 20240906, 20240907, 20240908}
    assert ("SIM-P8", "fig4", 0.2, 20240904) not in jobs
    assert ("EQ16-CAUSAL", "fig4", 0.2, 20240904) not in jobs
    assert ("SIM-P3", "fig4", 0.2, 20240904) in jobs
    assert ("SIM-P3", "fig4", 0.1, 20240904) in jobs
    assert ("SIM-P3", "fig5_fig6", 0.4, 20240904) in jobs
    assert ("SIM-P4", "fig4", 0.2, 20240904) in jobs
    assert ("SIM-P6", "fig5_fig6", 0.4, 20240908) in jobs
    assert ("SIM-C2", "fig5_fig6", 0.4, 20240904) in jobs
    assert ("SIM-CR1", "fig4", 0.1, 20240908) in jobs
    assert ("EQ16-CAUSAL", "fig5_fig6", 0.2, 20240904) in jobs
    assert ("EQ16-CAUSAL-D", "fig5_fig6", 0.4, 20240908) in jobs


def test_queue_prefers_fig4_before_fig5_and_review2_causal_first():
    jobs = queue_mod.prioritized_jobs(prefer="fig4")
    assert jobs[0][1] == "fig4"
    assert jobs[0][0] == "SIM-C2"
    fig4 = [job for job in jobs if job[1] == "fig4"]
    fig5 = [job for job in jobs if job[1] == "fig5_fig6"]
    assert fig4
    assert fig5
    assert jobs.index(fig4[-1]) < jobs.index(fig5[0])


def test_queue_prefers_fig5_of_sim_c2_and_cr1_first():
    jobs = queue_mod.prioritized_jobs(prefer="fig5")
    assert jobs[0][1] == "fig5_fig6"
    assert jobs[0][0] == "SIM-C2"
    fig5 = [job for job in jobs if job[1] == "fig5_fig6"]
    cr1 = [job for job in fig5 if job[0] == "SIM-CR1"]
    p3 = [job for job in fig5 if job[0] == "SIM-P3"]
    assert jobs.index(cr1[0]) < jobs.index(p3[0])


def test_stale_lock_is_stolen_dead_pid(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(queue_mod, "LOCK_DIR", tmp_path)
    dead = subprocess.Popen(["true"])
    dead.wait()
    job = ("SIM-P6", "fig4", 0.2, 20240904)
    stale = queue_mod.lock_path(job)
    stale.mkdir(parents=True)
    (stale / "meta.json").write_text(f'{{"pid": {dead.pid}}}\n', encoding="utf-8")
    claimed = queue_mod.acquire_lock(job)
    assert claimed is not None
    queue_mod.release_lock(claimed)
    assert not stale.exists()


def test_live_lock_is_not_stolen(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_mod, "LOCK_DIR", tmp_path)
    job = ("SIM-P4", "fig4", 0.1, 20240905)
    first = queue_mod.acquire_lock(job)
    assert first is not None
    second = queue_mod.acquire_lock(job)
    assert second is None
    queue_mod.release_lock(first)
