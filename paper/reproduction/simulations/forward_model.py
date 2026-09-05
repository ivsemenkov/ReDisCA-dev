"""Load the public AD overlapping-spheres forward model.

This is a documented hypothesis, not a paper statement. The paper never names
the simulation mesh. fsaverage is never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from .config import FORWARD_CANDIDATE, FORWARD_STATUS


@dataclass(frozen=True)
class ForwardModel:
    """Constrained planar-gradiometer leadfield on tess_cortex_pial_low."""

    gain: NDArray[np.float64]  # (n_channels, n_vertices)
    vertices: NDArray[np.float64]  # (n_vertices, 3) meters
    normals: NDArray[np.float64]  # (n_vertices, 3)
    channel_index: NDArray[np.intp]
    n_channels: int
    n_vertices: int
    tess_path: str
    headmodel_path: str
    tess_sha256: str | None
    headmodel_sha256: str | None
    surface_file: str
    meg_method: str
    notes: tuple[str, ...]

    def distances_from(self, vertex: int) -> NDArray[np.float64]:
        delta = self.vertices - self.vertices[int(vertex)]
        return np.linalg.norm(delta, axis=1)

    def provenance(self) -> dict[str, Any]:
        return {
            "status": FORWARD_STATUS,
            "candidate": FORWARD_CANDIDATE,
            "tess_path": self.tess_path,
            "headmodel_path": self.headmodel_path,
            "tess_sha256": self.tess_sha256,
            "headmodel_sha256": self.headmodel_sha256,
            "surface_file": self.surface_file,
            "meg_method": self.meg_method,
            "gain_shape_used": [int(self.n_channels), int(self.n_vertices)],
            "n_vertices": int(self.n_vertices),
            "n_channels": int(self.n_channels),
            "channel_index_head": [int(i) for i in self.channel_index[:12]],
            "channel_index_rule": "Gain[:306] MEG; drop index%3==2 (MAG of GRAD-GRAD-MAG)",
            "orientation": "constrained GridOrient",
            "units_vertices": "meters",
            "fsaverage_used": False,
            "notes": list(self.notes),
        }


def _planar_indices_306() -> NDArray[np.intp]:
    """204 planar rows in a 306-channel GRAD-GRAD-MAG layout."""
    return np.asarray([i for i in range(306) if i % 3 != 2], dtype=np.intp)


def load_ad_forward(
    source_model_dir: Path,
    *,
    sha256_file=None,
) -> ForwardModel:
    """Load tess + Gain from ``.reproduction_data/source_models/``."""
    tess_path = source_model_dir / "tess_cortex_pial_low.mat"
    hm_path = source_model_dir / "headmodel_surf_os_meg.mat"
    if not tess_path.exists() or not hm_path.exists():
        raise FileNotFoundError(
            "AD source-model files are missing. Expected "
            f"{tess_path} and {hm_path}. "
            "Run: python paper/reproduction/common/download_osf.py source-models"
        )

    tess = loadmat(tess_path, squeeze_me=True, struct_as_record=False)
    hm = loadmat(hm_path, squeeze_me=True, struct_as_record=False)

    vertices = np.asarray(tess["Vertices"], dtype=np.float64)
    normals = np.asarray(tess["VertNormals"], dtype=np.float64)
    gain_full = np.asarray(hm["Gain"], dtype=np.float64)
    grid_orient = np.asarray(hm["GridOrient"], dtype=np.float64)
    grid_loc = np.asarray(hm["GridLoc"], dtype=np.float64)

    if vertices.shape != (5002, 3):
        raise ValueError(f"Unexpected tess Vertices shape {vertices.shape}")
    if gain_full.shape != (322, 15006):
        raise ValueError(f"Unexpected Gain shape {gain_full.shape}")
    if not np.allclose(vertices, grid_loc):
        raise ValueError("GridLoc does not match tess Vertices.")
    if not np.allclose(normals, grid_orient):
        raise ValueError("GridOrient does not match tess VertNormals.")

    meg = gain_full[:306]
    if not np.isfinite(meg).all():
        raise ValueError("First 306 Gain rows are not all finite; channel map is not as assumed.")
    if np.isfinite(gain_full[306:]).any():
        extra = "some finite extra rows"
    else:
        extra = "rows 306:322 are all-NaN (dropped)"

    planar = _planar_indices_306()
    g3 = meg[planar].reshape(planar.size, 5002, 3)
    ori = grid_orient / np.clip(np.linalg.norm(grid_orient, axis=1, keepdims=True), 1e-12, None)
    gain = np.einsum("mvd,vd->mv", g3, ori)
    if not np.isfinite(gain).all():
        raise ValueError("Constrained planar leadfield contains non-finite values.")
    if np.any(np.linalg.norm(gain, axis=0) == 0.0):
        raise ValueError("Constrained leadfield has a zero column.")

    tess_hash = sha256_file(tess_path) if sha256_file is not None else None
    hm_hash = sha256_file(hm_path) if sha256_file is not None else None
    notes = (
        "NOT the paper-named simulation mesh (unnamed).",
        extra,
        "Row RMS of MEG triplets is large/large/small => assumed GRAD-GRAD-MAG.",
        "Never fsaverage.",
    )
    return ForwardModel(
        gain=gain,
        vertices=vertices,
        normals=ori,
        channel_index=planar,
        n_channels=int(gain.shape[0]),
        n_vertices=int(gain.shape[1]),
        tess_path=str(tess_path),
        headmodel_path=str(hm_path),
        tess_sha256=tess_hash,
        headmodel_sha256=hm_hash,
        surface_file=str(hm["SurfaceFile"]),
        meg_method=str(hm["MEGMethod"]),
        notes=notes,
    )
