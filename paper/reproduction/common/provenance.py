"""Environment and run provenance capture."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from .method import AIRI_SHA, AIRI_SPOC_KWARGS, ERP_CORE_SHA, LIBRARY_MAIN_SHA, SPOC_SHA
from .rng import RNGRecord


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_head(repo_root: str | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def capture_environment(*, extra_packages: tuple[str, ...] = ()) -> dict[str, Any]:
    packages = ["numpy", "scipy", "scikit-learn", "redisca", *extra_packages]
    versions = {name: _package_version(name) for name in packages}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "matlab": None,
        "library_main_sha_expected": LIBRARY_MAIN_SHA,
        "library_main_sha_observed": git_head(),
        "airi_sha": AIRI_SHA,
        "spoc_sha": SPOC_SHA,
        "erp_core_sha": ERP_CORE_SHA,
        "redisca_kwargs": dict(AIRI_SPOC_KWARGS),
        "note": (
            "Stage A uses current-main redisca.ReDisCA with the frozen "
            "AIRI-SPoC kwargs. MATLAB is not assumed; NumPy RNG is not "
            "MATLAB rand bitwise parity."
        ),
    }


def capture_run(
    *,
    track: str,
    candidate_id: str,
    seed: int | None = None,
    seed_record: RNGRecord | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "track": track,
        "candidate_id": candidate_id,
        "seed": seed,
        "environment": capture_environment(),
        "redisca_kwargs": dict(AIRI_SPOC_KWARGS),
    }
    if seed_record is not None:
        payload["rng"] = seed_record.to_dict()
    if extra:
        payload["extra"] = extra
    return payload
