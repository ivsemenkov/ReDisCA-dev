"""Claim missing required simulation jobs without write races.

In-flight dedicated workers that imported an older ``run.py`` do not
participate in this lock protocol. Pass ``--skip-candidates`` for any
candidate those processes still own.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from paper.reproduction.common.hashing import write_json
from paper.reproduction.common.paths import RESULTS_ROOT
from paper.reproduction.simulations.run import run_candidate
from paper.reproduction.summarize import _required_sim_jobs

LOCK_DIR = RESULTS_ROOT / "simulations" / "_locks"

FIG4_PRIORITY = (
    "SIM-C2",
    "SIM-CR1",
    "SIM-C1",
    "SIM-C3",
    "SIM-C4",
    "SIM-P3",
    "SIM-P6",
    "SIM-P4",
    "SIM-P7",
    "SIM-R1",
    "SIM-P1",
    "SIM-P5",
    "SIM-P2",
)
FIG5_PRIORITY = (
    "SIM-C2",
    "SIM-CR1",
    "EQ16-CAUSAL",
    "EQ16-CAUSAL-D",
    "SIM-C1",
    "SIM-C3",
    "SIM-C4",
    "SIM-P3",
    "SIM-P6",
    "SIM-P4",
    "SIM-P7",
    "SIM-R1",
    "SIM-P1",
    "SIM-P8",
    "SIM-P5",
    "SIM-P2",
)

Job = tuple[str, str, float, int]


def lock_path(job: Job) -> Path:
    candidate, experiment, snr, seed = job
    name = f"{candidate}_{experiment}_snr{snr}_seed{seed}.lock"
    return LOCK_DIR / name


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_lock(path: Path) -> dict[str, Any] | None:
    meta = path / "meta.json"
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def acquire_lock(job: Job) -> Path | None:
    path = lock_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        meta = _read_lock(path)
        pid = int((meta or {}).get("pid") or 0)
        if _pid_is_alive(pid):
            return None
        # Stale lock from a dead worker.
        try:
            for child in path.iterdir():
                child.unlink()
            path.rmdir()
            path.mkdir()
        except OSError:
            return None
    write_json(
        path / "meta.json",
        {
            "pid": os.getpid(),
            "job": {
                "candidate": job[0],
                "experiment": job[1],
                "snr": job[2],
                "seed": job[3],
            },
            "started_unix": time.time(),
        },
    )
    return path


def release_lock(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        try:
            child.unlink()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass


def result_path(job: Job) -> Path:
    candidate, experiment, snr, seed = job
    if experiment == "fig4":
        name = f"fig4_snr{snr}_seed{seed}.json"
    else:
        name = f"fig5_fig6_snr{snr}_seed{seed}.json"
    return RESULTS_ROOT / "simulations" / candidate / name


def job_is_complete(job: Job) -> bool:
    from paper.reproduction.simulations.run import _is_complete_reproduction

    return _is_complete_reproduction(result_path(job))


def prioritized_jobs(*, prefer: str = "fig4") -> list[Job]:
    def key(job: Job) -> tuple[int, int, int, float]:
        candidate, experiment, snr, seed = job
        exp_rank = 0 if experiment == "fig4" else 1
        if prefer == "fig5":
            exp_rank = 1 - exp_rank
        table = FIG4_PRIORITY if experiment == "fig4" else FIG5_PRIORITY
        try:
            cand_rank = table.index(candidate)
        except ValueError:
            cand_rank = 99
        return (exp_rank, cand_rank, seed, -float(snr))

    return sorted(_required_sim_jobs(), key=key)


def claim_next_job(
    *,
    prefer: str = "fig4",
    skip_candidates: tuple[str, ...] = (),
    only_candidates: tuple[str, ...] = (),
) -> tuple[Job, Path] | None:
    for job in prioritized_jobs(prefer=prefer):
        if skip_candidates and job[0] in skip_candidates:
            continue
        if only_candidates and job[0] not in only_candidates:
            continue
        if job_is_complete(job):
            continue
        lock = acquire_lock(job)
        if lock is None:
            continue
        if job_is_complete(job):
            release_lock(lock)
            continue
        return job, lock
    return None


def run_one_job(job: Job, *, include_rsa: bool = False) -> dict[str, Any]:
    candidate, experiment, snr, seed = job
    experiments = ("fig4",) if experiment == "fig4" else ("fig5",)
    return run_candidate(
        candidate,
        seeds=(seed,),
        quick=False,
        include_rsa=include_rsa and candidate in {"SIM-P1", "SIM-R1"},
        experiments=experiments,
        skip_existing=True,
        snrs=(float(snr),),
    )


def worker_loop(
    *,
    prefer: str = "fig4",
    skip_candidates: tuple[str, ...] = (),
    only_candidates: tuple[str, ...] = (),
    include_rsa: bool = False,
    idle_sleep_s: float = 30.0,
    max_jobs: int | None = None,
) -> dict[str, Any]:
    written: list[str] = []
    claimed: list[dict[str, Any]] = []
    n_done = 0
    while max_jobs is None or n_done < max_jobs:
        claimed_job = claim_next_job(
            prefer=prefer,
            skip_candidates=skip_candidates,
            only_candidates=only_candidates,
        )
        if claimed_job is None:
            remaining = [
                job
                for job in prioritized_jobs(prefer=prefer)
                if not job_is_complete(job)
                and (not skip_candidates or job[0] not in skip_candidates)
                and (not only_candidates or job[0] in only_candidates)
            ]
            if not remaining:
                break
            print(f"queue idle; {len(remaining)} jobs locked by others; sleep {idle_sleep_s}s", flush=True)
            time.sleep(idle_sleep_s)
            continue
        job, lock = claimed_job
        print(f"queue claimed {job}", flush=True)
        try:
            result = run_one_job(job, include_rsa=include_rsa)
            written.extend(result.get("written") or [])
            claimed.append({"job": job, "result": result})
        finally:
            release_lock(lock)
        n_done += 1
    return {"written": written, "claimed": claimed, "n_done": n_done}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run missing required Stage A simulation jobs.")
    parser.add_argument("--prefer", choices=["fig4", "fig5"], default="fig4")
    parser.add_argument("--skip-candidates", nargs="*", default=[])
    parser.add_argument("--only-candidates", nargs="*", default=[])
    parser.add_argument("--include-rsa", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--idle-sleep", type=float, default=30.0)
    args = parser.parse_args(argv)
    result = worker_loop(
        prefer=args.prefer,
        skip_candidates=tuple(args.skip_candidates),
        only_candidates=tuple(args.only_candidates),
        include_rsa=args.include_rsa,
        idle_sleep_s=args.idle_sleep,
        max_jobs=args.max_jobs,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
