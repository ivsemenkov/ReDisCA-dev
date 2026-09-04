"""Environment and run provenance capture."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from .rng import RNGRecord


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


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
        "note": (
            "This is a source-faithful Python reconstruction environment. "
            "MATLAB is not assumed to be available; results are not MATLAB parity."
        ),
    }


def capture_run(
    *,
    track: str,
    path_label: str,
    seed_record: RNGRecord | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "track": track,
        "path_label": path_label,
        "environment": capture_environment(),
    }
    if seed_record is not None:
        payload["rng"] = seed_record.to_dict()
    if extra:
        payload["extra"] = extra
    return payload
