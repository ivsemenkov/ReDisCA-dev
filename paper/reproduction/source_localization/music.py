"""Eq. 13 cosine-similarity and Eq. 14 MUSIC / subspace-correlation scanners.

Paper (NeuroImage 301:120868, §2.3):

- Eq. 13: ``ρ_CS_m = (g_m^T a_k) / (||g_m|| ||a_k||)``. The printed
  denominator uses ``||a_1||`` for every k (typesetting slip). This module
  defaults to ``||a_k||`` and can emit the literal-slip variant.
- Eq. 14: ``ρ_MUSIC_m = subcorr(g_m, A_K)`` with ``g_m`` the pair of
  topographies of a freely oriented dipole (two leading left singular vectors
  of the 3-column Gain block) and ``A_K`` the K-column significant-component
  subspace. ``subcorr`` is the cosine of the first principal angle.

Fig. 18 caption: cortical map of that first-principal-angle cosine between the
Fig. 17 dissimilarity subspace and the free-orientation dipole subspace at
each vertex.

AIRI ``method='music'`` (not Fig. 18) uses the same SVD scan but:

- default topography is the single column ``A1(:,4)``;
- ``P = eye(size(Nsns,1))`` is ``eye(1)`` — a dimension bug. The obvious
  fix is ``P = eye(Nsns)``. With ``nRAP=1`` the fix is ordinary MUSIC
  (the RAP projector is applied only after the first scan).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

ProjectorVariant = Literal["literal_bug", "eye_nsns_fix"]


class AiriMusicDimensionError(ValueError):
    """AIRI ``P = eye(size(Nsns,1))`` is ``eye(1)``; ``P*G`` is non-executable."""


def _as_2d_patterns(patterns: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(patterns, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"patterns must be 1-D or 2-D, got {array.shape}")
    return array


def orthonormal_basis(columns: NDArray[np.floating]) -> NDArray[np.float64]:
    """Left singular vectors spanning the column space (thin SVD)."""
    columns = np.asarray(columns, dtype=np.float64)
    if columns.ndim != 2:
        raise ValueError(f"columns must be 2-D, got {columns.shape}")
    if columns.size == 0:
        raise ValueError("empty basis")
    u, s, _vt = np.linalg.svd(columns, full_matrices=False)
    rank = int(np.sum(s > (s[0] * 1e-12 if s[0] > 0.0 else 0.0)))
    rank = max(rank, 1)
    return u[:, :rank]


def dipole_tangent_bases(blocks: NDArray[np.floating]) -> NDArray[np.float64]:
    """Two leading left singular vectors of each 3-orientation Gain block.

    Parameters
    ----------
    blocks :
        ``(n_sensors, n_sources, 3)``.

    Returns
    -------
    bases :
        ``(n_sources, n_sensors, 2)``. MEG lead fields are numerically rank-2.
    """
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[2] != 3:
        raise ValueError(f"blocks must be (n_sensors, n_src, 3), got {blocks.shape}")
    # Batched SVD over sources: (n_src, n_sensors, 3)
    stacked = np.moveaxis(blocks, 1, 0)
    u, _s, _vt = np.linalg.svd(stacked, full_matrices=False)
    return u[:, :, :2]


def cosine_similarity_scan(
    leadfield: NDArray[np.floating],
    pattern: NDArray[np.floating],
    *,
    denominator: Literal["a_k", "a_1_typesetting_slip"] = "a_k",
    a_1: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Paper Eq. 13 against one-column topographies ``g_m``.

    ``leadfield`` has shape ``(n_sensors, n_sources)``. The default denominator
    is ``||a_k||`` (the obvious correction of the printed ``||a_1||`` slip).
    """
    leadfield = np.asarray(leadfield, dtype=np.float64)
    pattern = np.asarray(pattern, dtype=np.float64).ravel()
    if leadfield.ndim != 2:
        raise ValueError(f"leadfield must be 2-D, got {leadfield.shape}")
    if leadfield.shape[0] != pattern.size:
        raise ValueError(
            f"sensor mismatch: leadfield {leadfield.shape} vs pattern {pattern.shape}"
        )
    g_norm = np.linalg.norm(leadfield, axis=0)
    if denominator == "a_k":
        a_norm = float(np.linalg.norm(pattern))
    elif denominator == "a_1_typesetting_slip":
        if a_1 is None:
            raise ValueError("a_1 is required for the typesetting-slip denominator")
        a_norm = float(np.linalg.norm(np.asarray(a_1, dtype=np.float64).ravel()))
    else:
        raise ValueError(f"Unknown denominator {denominator!r}")
    denom = g_norm * a_norm
    num = leadfield.T @ pattern
    out = np.full(leadfield.shape[1], np.nan, dtype=np.float64)
    ok = denom > 0.0
    out[ok] = num[ok] / denom[ok]
    return out


def subspace_correlation_scan(
    dipole_bases: NDArray[np.floating],
    patterns: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Eq. 14: cosine of the first principal angle, one value per source.

    Parameters
    ----------
    dipole_bases :
        ``(n_sources, n_sensors, n_dipole_dim)`` (typically dim=2).
    patterns :
        ``(n_sensors, K)`` or ``(n_sensors,)``.
    """
    dipole_bases = np.asarray(dipole_bases, dtype=np.float64)
    if dipole_bases.ndim != 3:
        raise ValueError(f"dipole_bases must be 3-D, got {dipole_bases.shape}")
    a_k = _as_2d_patterns(patterns)
    n_src, n_sensors, _n_dip = dipole_bases.shape
    if a_k.shape[0] != n_sensors:
        raise ValueError(
            f"sensor mismatch: bases {dipole_bases.shape} vs patterns {a_k.shape}"
        )
    u_a = orthonormal_basis(a_k)
    # M[s] = U_A.T @ bases[s]  -> (K, n_dip) per source
    mixed = np.tensordot(u_a.T, dipole_bases, axes=([1], [1]))  # (K, n_src, n_dip)
    stacked = np.moveaxis(mixed, 1, 0)  # (n_src, K, n_dip)
    singular_values = np.linalg.svd(stacked, compute_uv=False)
    return np.clip(singular_values[:, 0], 0.0, 1.0)


def music_scan(
    blocks: NDArray[np.floating],
    patterns: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Paper Fig. 18 scanner: free-orientation MUSIC of subspace ``A_K``."""
    return subspace_correlation_scan(dipole_tangent_bases(blocks), patterns)


def airi_initial_projector(
    n_sensors: int,
    *,
    variant: ProjectorVariant,
) -> NDArray[np.float64]:
    """AIRI ``P`` before the RAP loop.

    ``literal_bug`` reconstructs ``eye(size(Nsns,1))`` = ``eye(1)``.
    ``eye_nsns_fix`` is ``eye(Nsns)``.
    """
    if variant == "literal_bug":
        return np.eye(1, dtype=np.float64)
    if variant == "eye_nsns_fix":
        return np.eye(int(n_sensors), dtype=np.float64)
    raise ValueError(f"Unknown projector variant {variant!r}")


def _require_projector(P: NDArray[np.floating], n_sensors: int) -> NDArray[np.float64]:
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (n_sensors, n_sensors):
        raise AiriMusicDimensionError(
            "AIRI sets P = eye(size(Nsns,1)), which is eye(1) because Nsns is a "
            f"scalar. P has shape {tuple(P.shape)}; G has {n_sensors} rows. "
            "P*G is non-executable in MATLAB and NumPy. This is not Fig. 18. "
            "The obvious fix is P = eye(Nsns)."
        )
    return P


def airi_music_scan(
    gain_sensors_by_orient: NDArray[np.floating],
    topos: NDArray[np.floating],
    *,
    n_rap: int = 1,
    projector_variant: ProjectorVariant = "eye_nsns_fix",
    projector: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """AIRI ``method='music'`` branch (not paper Fig. 18).

    Parameters
    ----------
    gain_sensors_by_orient :
        Planar Gain block ``(Nsns, 3*Nsrc)`` as in ``G = hm.Gain(megplanarbst,:)``.
    topos :
        ``(Nsns, K)``. AIRI default is the single column ``A1(:,4)``.
    n_rap :
        AIRI ``nRAP`` (default 1). With ``nRAP=1`` and ``P = I`` this equals
        Eq. 14 on whatever columns ``topos`` contains.
    projector_variant :
        ``literal_bug`` or ``eye_nsns_fix``. Ignored if ``projector`` is given.

    Returns
    -------
    ctx_map :
        ``(Nsrc, n_rap)`` as in AIRI ``ctx_map(:,rap) = scn``.
    """
    G = np.asarray(gain_sensors_by_orient, dtype=np.float64)
    topos_2d = _as_2d_patterns(topos)
    if G.ndim != 2:
        raise ValueError(f"G must be 2-D, got {G.shape}")
    n_sensors, n_cols = G.shape
    if n_cols % 3 != 0:
        raise ValueError(f"G has {n_cols} columns; not divisible by 3")
    n_src = n_cols // 3
    if topos_2d.shape[0] != n_sensors:
        raise ValueError(f"topos sensors {topos_2d.shape[0]} != G rows {n_sensors}")
    if n_rap < 1:
        raise ValueError("n_rap must be >= 1")

    if projector is None:
        projector = airi_initial_projector(n_sensors, variant=projector_variant)
    P = _require_projector(projector, n_sensors)

    ctx_map = np.zeros((n_src, n_rap), dtype=np.float64)
    Gsrc = np.zeros((n_sensors, 0), dtype=np.float64)
    for rap in range(n_rap):
        Gp = P @ G
        topos_p = P @ topos_2d
        blocks = Gp.reshape(n_sensors, n_src, 3)
        scan = music_scan(blocks, topos_p)
        ctx_map[:, rap] = scan
        imax = int(np.argmax(scan))
        bases = dipole_tangent_bases(blocks)
        Gsrc = np.concatenate([Gsrc, bases[imax]], axis=1)
        P = np.eye(n_sensors) - Gsrc @ np.linalg.pinv(Gsrc)
    return ctx_map


def first_principal_angle_rad(subcorr: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert Eq. 14 cosines to principal angles (radians)."""
    return np.arccos(np.clip(np.asarray(subcorr, dtype=np.float64), 0.0, 1.0))
