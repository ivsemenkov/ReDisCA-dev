"""Subject-AD overlapping-spheres forward model and Neuromag index sets.

Do **not** substitute fsaverage. The public OSF Gain is the Brainstorm
overlapping-spheres operator for Kozunov subject AD, ``tess_cortex_pial_low``
(5002 vertices, 3 orientations). Individual T1 is not released; this Gain is
the scan operator used here.

D15 (MAG vs GRAD index hazard)
------------------------------
AIRI MATLAB::

    megplanarbst = sort([1:3:304, 2:3:305])   % 1-based, length 204

On a Neuromag triplet this is channels ``(1,2)`` of each ``(1,2,3)`` group,
i.e. 0-based indices ``0,1, 3,4, …, 303,304``.

Whether that is “both planars” or “MAG + first GRAD” depends on the 306-channel
order:

- **GRAD, GRAD, MAG** (this OSF kernel’s ``Options.ChannelTypes``): AIRI
  ``megplanarbst`` **is** the 204 planar gradiometers. Verified on the cached
  sLORETA file: 204× ``MEG GRAD``.
- **MAG, GRAD, GRAD**: the same index vector would mix 102 magnetometers with
  102 gradiometers and would **not** match sensor-space ``d(1:204)`` (already
  both planars, no mags).

Gain has 322 rows: 306 MEG + 16 trailing all-NaN extras. ``ImagingKernel`` is
``(5002, 306)`` with ``GoodChannel = 1…306``. MUSIC must use the 204 planar
rows of Gain, not the 16 NaN rows and not the 102 magnetometers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

N_VERTICES = 5002
N_ORIENT = 3
N_MEG = 306
N_GAIN_ROWS = 322
N_PLANAR = 204
N_EXTRA_NAN_ROWS = 16


def matlab_megplanarbst_1based() -> NDArray[np.int64]:
    """Literal AIRI ``sort([1:3:304, 2:3:305])`` (1-based MATLAB)."""
    return np.sort(
        np.concatenate([np.arange(1, 305, 3), np.arange(2, 306, 3)])
    ).astype(np.int64)


def megplanarbst_0based() -> NDArray[np.int64]:
    """0-based equivalent of AIRI ``megplanarbst`` into a 306-channel MEG block."""
    return matlab_megplanarbst_1based() - 1


def mag_plus_first_grad_if_mag_first_0based() -> NDArray[np.int64]:
    """D15: the AIRI index vector *interpreted as* MAG + first GRAD.

    Those integers are identical to ``megplanarbst_0based()``. The hazard is
    not a second MATLAB formula — it is that the same ``1:3:304, 2:3:305``
    list is MAG+GRAD if the triplet is MAG, GRAD, GRAD, and both planars if
    the triplet is GRAD, GRAD, MAG (this OSF kernel).
    """
    return megplanarbst_0based()


def mag_plus_grad1_on_grad_grad_mag_0based() -> NDArray[np.int64]:
    """102 MAG + 102 GRAD1 on *this* file's GRAD, GRAD, MAG order.

    Negative-control mix: 0-based ``0,2, 3,5, …`` (GRAD1 + MAG). Not Fig. 18.
    """
    grad1 = np.arange(0, N_MEG, 3)
    mag = np.arange(2, N_MEG, 3)
    return np.sort(np.concatenate([grad1, mag])).astype(np.int64)


def both_planars_grad_grad_mag_0based() -> NDArray[np.int64]:
    """Both planars under GRAD, GRAD, MAG (identical to AIRI ``megplanarbst``)."""
    return megplanarbst_0based()


def magnetometer_0based() -> NDArray[np.int64]:
    """Magnetometer index of each triplet under GRAD, GRAD, MAG (0-based)."""
    return np.arange(2, N_MEG, 3, dtype=np.int64)


@dataclass(frozen=True)
class SourceModel:
    gain: NDArray[np.float64]  # (322, 15006)
    grid_loc: NDArray[np.float64]  # (5002, 3)
    grid_orient: NDArray[np.float64]  # (5002, 3)
    vertices: NDArray[np.float64]  # (5002, 3)
    faces_1based: NDArray[np.int64]  # (9974, 3) MATLAB 1-based
    imaging_kernel: NDArray[np.float64]  # (5002, 306)
    good_channel_1based: NDArray[np.int64]
    channel_types: tuple[str, ...]
    atlas_labels: dict[str, tuple[str, ...]]
    hemisphere: tuple[str, ...]
    meg_method: str
    headmodel_comment: str
    kernel_comment: str
    kernel_function: str
    surface_comment: str
    paths: dict[str, str]


def _iter_scouts(atlas: Any) -> Iterator[Any]:
    scouts = getattr(atlas, "Scouts", None)
    if scouts is None:
        return
    if isinstance(scouts, np.ndarray):
        if scouts.size == 0:
            return
        for scout in scouts.ravel():
            yield scout
        return
    yield scouts


def _vertex_labels_from_atlas(atlas: Any, n_vertices: int = N_VERTICES) -> list[str]:
    labels = ["unlabeled"] * n_vertices
    for scout in _iter_scouts(atlas):
        name = str(getattr(scout, "Label", "unknown"))
        raw = np.asarray(getattr(scout, "Vertices"), dtype=np.int64).ravel()
        if raw.size == 0:
            continue
        # Brainstorm scout vertex indices are 1-based.
        idx = raw - 1
        if idx.min() < 0 or idx.max() >= n_vertices:
            raise ValueError(
                f"Scout {name!r} vertex indices out of range: "
                f"min={int(raw.min())} max={int(raw.max())}"
            )
        for vertex in idx:
            labels[int(vertex)] = name
    return labels


def load_source_model(
    source_model_dir: Path,
    *,
    require_not_fsaverage: bool = True,
) -> SourceModel:
    """Load the three OSF source-model files from the gitignored cache."""
    source_model_dir = Path(source_model_dir)
    head_path = source_model_dir / "headmodel_surf_os_meg.mat"
    tess_path = source_model_dir / "tess_cortex_pial_low.mat"
    kernel_path = (
        source_model_dir / "results_sLORETA_MEG_GRAD_MEG_MAG_KERNEL_150924_1824.mat"
    )
    for path in (head_path, tess_path, kernel_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing source-model asset: {path}")

    head = loadmat(head_path, squeeze_me=True, struct_as_record=False)
    tess = loadmat(
        tess_path,
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["Vertices", "Faces", "Atlas", "Comment"],
    )
    kernel = loadmat(kernel_path, squeeze_me=True, struct_as_record=False)

    surface_file = str(head.get("SurfaceFile", ""))
    if require_not_fsaverage:
        blob = " ".join(
            [
                surface_file,
                str(head.get("Comment", "")),
                str(tess.get("Comment", "")),
                str(kernel.get("SurfaceFile", "")),
            ]
        ).lower()
        if "fsaverage" in blob:
            raise RuntimeError(
                "Refusing to use an fsaverage forward model for this track."
            )

    gain = np.asarray(head["Gain"], dtype=np.float64)
    if gain.shape != (N_GAIN_ROWS, N_VERTICES * N_ORIENT):
        raise ValueError(f"Unexpected Gain shape {gain.shape}")
    grid_loc = np.asarray(head["GridLoc"], dtype=np.float64)
    vertices = np.asarray(tess["Vertices"], dtype=np.float64)
    if not np.allclose(grid_loc, vertices):
        raise ValueError("GridLoc and tess Vertices differ; expected a surface model.")

    channel_types = tuple(str(x) for x in kernel["Options"].ChannelTypes)
    if len(channel_types) != N_MEG:
        raise ValueError(f"Expected 306 ChannelTypes, got {len(channel_types)}")

    atlas_labels: dict[str, tuple[str, ...]] = {}
    hemisphere = ["unknown"] * N_VERTICES
    for atlas in np.atleast_1d(tess["Atlas"]):
        name = str(getattr(atlas, "Name", "unnamed"))
        labels = _vertex_labels_from_atlas(atlas)
        atlas_labels[name] = tuple(labels)
        if name == "Structures":
            hemisphere = labels

    return SourceModel(
        gain=gain,
        grid_loc=grid_loc,
        grid_orient=np.asarray(head["GridOrient"], dtype=np.float64),
        vertices=vertices,
        faces_1based=np.asarray(tess["Faces"], dtype=np.int64),
        imaging_kernel=np.asarray(kernel["ImagingKernel"], dtype=np.float64),
        good_channel_1based=np.asarray(kernel["GoodChannel"], dtype=np.int64).ravel(),
        channel_types=channel_types,
        atlas_labels=atlas_labels,
        hemisphere=tuple(hemisphere),
        meg_method=str(head.get("MEGMethod", "")),
        headmodel_comment=str(head.get("Comment", "")),
        kernel_comment=str(kernel.get("Comment", "")),
        kernel_function=str(kernel.get("Function", "")),
        surface_comment=str(tess.get("Comment", "")),
        paths={
            "headmodel": str(head_path),
            "tess": str(tess_path),
            "kernel": str(kernel_path),
        },
    )


def leadfield_blocks(
    gain: NDArray[np.floating],
    row_index: NDArray[np.integer],
    *,
    n_vertices: int = N_VERTICES,
    n_orient: int = N_ORIENT,
) -> NDArray[np.float64]:
    """Return Gain as ``(n_sensors, n_vertices, 3)`` for the selected rows."""
    rows = np.asarray(gain, dtype=np.float64)[np.asarray(row_index, dtype=np.int64)]
    if rows.shape[1] != n_vertices * n_orient:
        raise ValueError(
            f"Gain has {rows.shape[1]} columns; expected {n_vertices * n_orient}"
        )
    if not np.isfinite(rows).all():
        raise ValueError("Selected Gain rows contain NaN/Inf (not the 306 MEG block).")
    return rows.reshape(rows.shape[0], n_vertices, n_orient)


def constrained_topographies(
    blocks: NDArray[np.floating],
    grid_orient: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Normal-oriented one-column topography per vertex: ``G_m @ n_m``."""
    blocks = np.asarray(blocks, dtype=np.float64)
    grid_orient = np.asarray(grid_orient, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[2] != 3:
        raise ValueError(f"blocks must be (n_sensors, n_src, 3), got {blocks.shape}")
    if grid_orient.shape != (blocks.shape[1], 3):
        raise ValueError(
            f"grid_orient shape {grid_orient.shape} != ({blocks.shape[1]}, 3)"
        )
    return np.einsum("cvo,vo->cv", blocks, grid_orient)


def index_audit(channel_types: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Document what AIRI ``megplanarbst`` actually selects on this kernel."""
    types = [str(t) for t in channel_types]
    airi = megplanarbst_0based()
    mixed = mag_plus_grad1_on_grad_grad_mag_0based()
    airi_types = [types[i] for i in airi]
    mixed_types = [types[i] for i in mixed]
    n_grad_airi = sum(t == "MEG GRAD" for t in airi_types)
    n_mag_airi = sum(t == "MEG MAG" for t in airi_types)
    same_as_airi = bool(np.array_equal(mag_plus_first_grad_if_mag_first_0based(), airi))
    return {
        "triplet_order_verified": types[:3],
        "airi_megplanarbst_n": int(airi.size),
        "airi_megplanarbst_n_grad": n_grad_airi,
        "airi_megplanarbst_n_mag": n_mag_airi,
        "airi_megplanarbst_is_both_planars_on_this_file": n_grad_airi == N_PLANAR
        and n_mag_airi == 0,
        "d15_same_integers_as_megplanarbst_if_one_assumes_MAG_first": same_as_airi,
        "d15_this_file_grad1_plus_mag_counts": {
            "MEG GRAD": sum(t == "MEG GRAD" for t in mixed_types),
            "MEG MAG": sum(t == "MEG MAG" for t in mixed_types),
        },
        "note": (
            "D15 is an ordering hazard on the same MATLAB index vector. On this "
            "OSF kernel ChannelTypes are GRAD, GRAD, MAG, so AIRI megplanarbst "
            "selects 204 planars. A MAG+GRAD mix on this file uses a different "
            "index set (GRAD1+MAG) and is a labeled negative control, not Fig. 18."
        ),
    }


def atlas_payload_for_vertex(
    model: SourceModel,
    vertex_0based: int,
    *,
    atlas_names: tuple[str, ...] = ("Destrieux", "Mindboggle", "Brodmann"),
) -> dict[str, str]:
    out = {
        "hemisphere": model.hemisphere[vertex_0based],
    }
    for name in atlas_names:
        labels = model.atlas_labels.get(name)
        if labels is not None:
            out[name] = labels[vertex_0based]
    return out
