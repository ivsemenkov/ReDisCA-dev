"""Run Fig. 18 MUSIC and AIRI source-loc variants; write compact JSON metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

_REPRO = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"
for _path in (_REPRO, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.hashing import sha256_file, write_json
from common.paths import MEG_DIR, SOURCE_MODEL_DIR
from common.provenance import capture_run
from common.rng import numpy_generator
from common.serialize import array_fingerprint
from source_localization.forward import (
    SourceModel,
    atlas_payload_for_vertex,
    constrained_topographies,
    index_audit,
    leadfield_blocks,
    load_source_model,
    mag_plus_grad1_on_grad_grad_mag_0based,
    megplanarbst_0based,
)
from source_localization.meg_patterns import (
    PatternFit,
    fit_airi_executable,
    fit_paper_faithful,
    prepare_condition_averages,
    theoretical_rdm,
    three_lowest_p_indices,
)
from source_localization.music import (
    AiriMusicDimensionError,
    airi_music_scan,
    cosine_similarity_scan,
    first_principal_angle_rad,
    music_scan,
)
from source_localization.plot_cortex import save_cortex_scatter
from source_localization.sloreta import precomp_abs_kernel_map

N_TOP = 10
FIG18_K = 3
AUTHOR_COMP4_0BASED = 3  # MATLAB A1(:,4)


def _load_author_topo(path: Path) -> dict[str, Any]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    a1 = np.asarray(payload["A1"], dtype=np.float64)
    comps = np.asarray(payload["comps_order"], dtype=np.int64).ravel()
    return {
        "A1": a1,
        "comps_order": comps,
        "path": str(path),
        "sha256": sha256_file(path),
        "note": (
            "Author-saved OSF topo_face_vs_tool_correct_filt15.mat (D17). "
            "The committed AIRI main script returns before save. This is not a "
            "local SPoC run and is not a Fig. 18 subspace by itself."
        ),
    }


def _top_peaks(
    scan: NDArray[np.floating],
    model: SourceModel,
    *,
    n_top: int = N_TOP,
) -> list[dict[str, Any]]:
    scan = np.asarray(scan, dtype=np.float64).ravel()
    order = np.argsort(scan)[::-1]
    peaks = []
    for rank, vertex in enumerate(order[:n_top], start=1):
        vertex = int(vertex)
        peaks.append(
            {
                "rank": rank,
                "vertex_index_0based": vertex,
                "vertex_index_1based_matlab": vertex + 1,
                "xyz_m": [float(x) for x in model.vertices[vertex]],
                "scan_value": float(scan[vertex]),
                **atlas_payload_for_vertex(model, vertex),
            }
        )
    return peaks


def _scan_record(
    *,
    scan_id: str,
    figure: str | None,
    status: str,
    algorithm: str,
    path_label: str,
    scan: NDArray[np.floating],
    model: SourceModel,
    extra: dict[str, Any],
) -> dict[str, Any]:
    peaks = _top_peaks(scan, model)
    peak = peaks[0]
    angle = first_principal_angle_rad(scan)
    return {
        "id": scan_id,
        "figure": figure,
        "status": status,
        "algorithm": algorithm,
        "path_label": path_label,
        "n_vertices": int(scan.size),
        "peak": peak,
        "top_peaks": peaks,
        "scan_fingerprint": array_fingerprint(scan),
        "principal_angle_rad_at_peak": float(angle[peak["vertex_index_0based"]]),
        "forward": {
            "headmodel": "headmodel_surf_os_meg.mat",
            "tess": "tess_cortex_pial_low.mat",
            "n_vertices": 5002,
            "not_fsaverage": True,
            "meg_method": model.meg_method,
            "surface_comment": model.surface_comment,
            "gain_rows": "204 MEG GRAD via AIRI megplanarbst on this file",
        },
        **extra,
    }


def _write_map(
    out_dir: Path,
    stem: str,
    scan: NDArray[np.floating],
    vertices: NDArray[np.floating],
    *,
    title: str,
    peak_index: int,
) -> dict[str, str]:
    npz_path = out_dir / "maps" / f"{stem}.npz"
    png_path = out_dir / "figures" / f"{stem}.png"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, scan=np.asarray(scan, dtype=np.float64), vertices=vertices)
    save_cortex_scatter(
        png_path,
        vertices,
        scan,
        title=title,
        peak_index=peak_index,
    )
    return {"npz": str(npz_path), "png": str(png_path)}


def _pattern_json(fit: PatternFit, k_index: NDArray[np.integer]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": fit.label,
        "estimator": fit.estimator,
        "pair_mode": fit.pair_mode,
        "matrix_mode": fit.matrix_mode,
        "inference": fit.inference,
        "permutation_B": fit.permutation_B,
        "rdm_name": fit.rdm_name,
        "n_times": fit.n_times,
        "time_window_s": list(fit.time_window_s),
        "preprocessing": fit.preprocessing,
        "n_patterns": int(fit.patterns.shape[0]),
        "n_channels": int(fit.patterns.shape[1]),
        "selected_component_indices_0based": [int(i) for i in np.asarray(k_index).ravel()],
        "eigenvalues_head": [float(v) for v in fit.eigenvalues[:8]],
        "patterns_fingerprint": array_fingerprint(fit.patterns),
        "haufe_patterns": True,
        "d9_note": (
            "MEG GEP rank is ~67 < 204, so paper A=W^{-1} is undefined. "
            "Haufe/SPoC patterns are used."
        ),
    }
    if fit.p_values is not None:
        payload["p_values_head"] = [float(v) for v in fit.p_values[:8]]
        payload["selected_p_values"] = [
            float(fit.p_values[int(i)]) for i in np.asarray(k_index).ravel()
        ]
    return payload


def _try_local_fits(
    *,
    meg_mat: Path,
    spm_mat: Path,
    permutation_b: int,
    rng: np.random.Generator,
) -> dict[str, PatternFit | None | str]:
    out: dict[str, PatternFit | None | str] = {
        "paper_faithful": None,
        "airi_executable": None,
        "error": None,
    }
    try:
        rdm = theoretical_rdm("facevstool")
        X_paper, _meta_p = prepare_condition_averages(
            meg_mat, spm_mat, path="paper_faithful"
        )
        out["paper_faithful"] = fit_paper_faithful(
            X_paper,
            rdm,
            demean_time=False,
            permutation_B=permutation_b,
            rng=rng,
            rdm_name="facevstool",
        )
        del X_paper
        X_airi, _meta_a = prepare_condition_averages(
            meg_mat, spm_mat, path="airi_executable"
        )
        out["airi_executable"] = fit_airi_executable(X_airi, rdm, rdm_name="facevstool")
        del X_airi
    except Exception as exc:  # noqa: BLE001 — record and continue with author-saved topo
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def run(
    *,
    permutation_b: int = 200,
    skip_meg_fit: bool = False,
    k: int = FIG18_K,
    seed: int = 20240915,
) -> dict[str, Any]:
    out_dir = _REPO / "paper" / "results" / "source_localization"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_source_model(SOURCE_MODEL_DIR)
    planar_idx = megplanarbst_0based()
    mixed_idx = mag_plus_grad1_on_grad_grad_mag_0based()
    audit = index_audit(model.channel_types)
    blocks = leadfield_blocks(model.gain, planar_idx)
    gain_planar = model.gain[planar_idx]

    topo_path = MEG_DIR / "topo_face_vs_tool_correct_filt15.mat"
    author = _load_author_topo(topo_path)
    a1: NDArray[np.float64] = author["A1"]
    a1_comp4 = a1[:, AUTHOR_COMP4_0BASED]
    a1_k3 = a1[:, :k]
    a1_k4 = a1[:, :4]

    records: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}

    # --- AIRI precomp (not Fig. 18) ---
    precomp = precomp_abs_kernel_map(model.imaging_kernel, a1_comp4, planar_idx)
    rec = _scan_record(
        scan_id="airi-source-loc-precomp",
        figure="AIRI-executable source loc (not Fig. 18)",
        status="approximate",
        algorithm="constrained_sloreta_precomp_abs_W_times_A1_col4",
        path_label="airi_executable",
        scan=precomp,
        model=model,
        extra={
            "numeric_map": "reproduced",
            "visual_screenshot": "blocked",
            "blocked_visual_reason": (
                "show_on_cortex / prepare4topoNMG / FieldTrip are not in the AIRI repo."
            ),
            "d17": author["note"],
            "topography": {
                "source": "OSF_author_saved_A1",
                "matlab_column": 4,
                "zero_based_column": AUTHOR_COMP4_0BASED,
                "comps_order": [int(x) for x in author["comps_order"]],
            },
            "kernel": {
                "comment": model.kernel_comment,
                "function": model.kernel_function,
                "shape": list(model.imaging_kernel.shape),
                "columns": "ImagingKernel[:, megplanarbst] (0-based)",
            },
            "index_audit": audit,
            "paper_claimed_regions": [
                "right fusiform",
                "right insula",
                "left IPS",
                "anterior central gyrus",
            ],
            "region_note": (
                "Those regions are the paper Fig. 18 claim. This map is AIRI "
                "sLORETA of component 4, not that claim."
            ),
        },
    )
    artifacts["airi-source-loc-precomp"] = _write_map(
        out_dir,
        "airi_precomp_A1_col4",
        precomp,
        model.vertices,
        title="AIRI precomp |W A1(:,4)| (not Fig. 18; not FieldTrip)",
        peak_index=rec["peak"]["vertex_index_0based"],
    )
    rec["artifacts"] = artifacts["airi-source-loc-precomp"]
    records["airi-source-loc-precomp"] = rec
    write_json(out_dir / "airi_source_loc_precomp.json", rec)

    # D15 counterfactual kernel columns
    precomp_mixed = precomp_abs_kernel_map(model.imaging_kernel, a1_comp4, mixed_idx)
    rec_mix = _scan_record(
        scan_id="airi-precomp-d15-mag-plus-first-grad",
        figure=None,
        status="approximate",
        algorithm="constrained_sloreta_precomp_wrong_MAG_GRAD_mix",
        path_label="d15_negative_control",
        scan=precomp_mixed,
        model=model,
        extra={
            "note": (
                "Negative control: ImagingKernel columns MAG+first-GRAD under a "
                "MAG,GRAD,GRAD assumption. On this file the true triplet is "
                "GRAD,GRAD,MAG, so this mix is 102 GRAD + 102 MAG. Not Fig. 18."
            ),
            "index_audit": audit,
        },
    )
    records["airi-precomp-d15-mag-plus-first-grad"] = rec_mix
    write_json(out_dir / "d15_index_hazard.json", rec_mix)

    # --- AIRI music literal bug ---
    bug_payload: dict[str, Any] = {
        "id": "airi-music-literal-P-eye1",
        "figure": None,
        "status": "blocked",
        "path_label": "airi_executable_literal_bug",
        "algorithm": "AIRI method=music with P=eye(size(Nsns,1))=eye(1)",
        "is_fig18": False,
        "nRAP": 1,
        "topography": "A1(:,4)",
        "executable": False,
    }
    try:
        airi_music_scan(
            gain_planar,
            a1_comp4,
            n_rap=1,
            projector_variant="literal_bug",
        )
        bug_payload["error"] = "expected AiriMusicDimensionError was not raised"
    except AiriMusicDimensionError as exc:
        bug_payload["error"] = str(exc)
        bug_payload["status"] = "blocked"
        bug_payload["blocked_reason"] = (
            "MATLAB P = eye(size(Nsns,1)) is eye(1); P*G is a dimension error. "
            "The committed music branch is non-executable as written. Not Fig. 18."
        )
    records["airi-music-literal-bug"] = bug_payload
    write_json(out_dir / "airi_music_literal_bug.json", bug_payload)

    # --- AIRI music eye(Nsns) fix, single column A1(:,4) ---
    airi_music_fix = airi_music_scan(
        gain_planar,
        a1_comp4,
        n_rap=1,
        projector_variant="eye_nsns_fix",
    )[:, 0]
    rec_fix = _scan_record(
        scan_id="airi-music-eye-Nsns-fix-A1-col4",
        figure=None,
        status="approximate",
        algorithm="AIRI_music_SVD_scan_P_eq_eye_Nsns_nRAP1",
        path_label="airi_executable_music_fix",
        scan=airi_music_fix,
        model=model,
        extra={
            "is_fig18": False,
            "note": (
                "Obvious eye(Nsns) fix of the AIRI music branch, still a single "
                "topography A1(:,4), nRAP=1 so RAP never deflates the scan. "
                "Equals Eq. 14 with K=1 on the author-saved fourth component. "
                "Not Fig. 18 (Fig. 18 is the multi-component Fig. 17 subspace)."
            ),
            "nRAP": 1,
            "topography_matlab_column": 4,
        },
    )
    artifacts["airi-music-fix-comp4"] = _write_map(
        out_dir,
        "airi_music_eyeNsns_A1_col4",
        airi_music_fix,
        model.vertices,
        title="AIRI music P=I fix, A1(:,4) (not Fig. 18)",
        peak_index=rec_fix["peak"]["vertex_index_0based"],
    )
    rec_fix["artifacts"] = artifacts["airi-music-fix-comp4"]
    records["airi-music-eye-Nsns-fix-A1-col4"] = rec_fix
    write_json(out_dir / "airi_music_eye_nsns_fix.json", rec_fix)

    # Author-saved 3-column subspace MUSIC — labeled NOT Fig. 18 (D17)
    author_sub = music_scan(blocks, a1_k3)
    rec_auth = _scan_record(
        scan_id="author-saved-A1-cols1-3-music",
        figure="not Fig. 18",
        status="approximate",
        algorithm="paper_eq14_music_free_orientation",
        path_label="author_saved_subspace_not_fig18",
        scan=author_sub,
        model=model,
        extra={
            "is_fig18": False,
            "fig18_status_if_used_as_substitute": "blocked",
            "reason": (
                "OSF A1 columns 1–K are an author-saved subspace (D17), not a "
                "local Fig. 17 fit. Using them as Fig. 18 is not reproduction."
            ),
            "K": k,
            "matlab_columns": list(range(1, k + 1)),
        },
    )
    artifacts["author-saved-k3"] = _write_map(
        out_dir,
        "author_saved_A1_cols1to3_music",
        author_sub,
        model.vertices,
        title="Eq.14 MUSIC of author-saved A1(:,1:3) — NOT Fig. 18 (D17)",
        peak_index=rec_auth["peak"]["vertex_index_0based"],
    )
    rec_auth["artifacts"] = artifacts["author-saved-k3"]
    records["author-saved-A1-cols1-3-music"] = rec_auth
    write_json(out_dir / "author_saved_subspace_music.json", rec_auth)

    author_k4 = music_scan(blocks, a1_k4)
    rec_auth4 = _scan_record(
        scan_id="author-saved-A1-cols1-4-music",
        figure="not Fig. 18",
        status="approximate",
        algorithm="paper_eq14_music_free_orientation",
        path_label="author_saved_subspace_K4_not_fig18",
        scan=author_k4,
        model=model,
        extra={"is_fig18": False, "K": 4, "matlab_columns": [1, 2, 3, 4]},
    )
    records["author-saved-A1-cols1-4-music"] = rec_auth4

    # Eq. 13 cosine of author-saved component 4 vs constrained normals (sim-style scanner demo)
    g_con = constrained_topographies(blocks, model.grid_orient)
    eq13 = cosine_similarity_scan(g_con, a1_comp4)
    rec_eq13 = _scan_record(
        scan_id="eq13-cosine-author-A1-col4-constrained",
        figure=None,
        status="approximate",
        algorithm="paper_eq13_cosine_constrained_orientation",
        path_label="scanner_demo_not_fig18",
        scan=np.abs(eq13),
        model=model,
        extra={
            "is_fig18": False,
            "note": (
                "Eq. 13 scanner demonstration on the author-saved fourth pattern "
                "against normal-oriented Gain columns. Simulation localization "
                "belongs to the simulations track; this only ships the scanner."
            ),
            "signed_peak": float(eq13[int(np.argmax(np.abs(eq13)))]),
        },
    )
    records["eq13-cosine-author-A1-col4-constrained"] = rec_eq13
    write_json(out_dir / "eq13_scanner_demo.json", rec_eq13)

    rng, rng_record = numpy_generator(seed)
    fig18_status = "blocked"
    fig18_reason = (
        "Local Fig. 17 fit was not run or failed; author-saved A1 is not Fig. 18 (D17)."
    )
    fits: dict[str, Any]
    if skip_meg_fit:
        fits = {"paper_faithful": None, "airi_executable": None, "error": "skipped by flag"}
    else:
        meg_mat = MEG_DIR / "MEG_AD_run1.mat"
        spm_mat = MEG_DIR / "ibfctfprespm8_AD_run1_raw_tsss_mc.mat"
        if not (meg_mat.is_file() and spm_mat.is_file()):
            fits = {
                "paper_faithful": None,
                "airi_executable": None,
                "error": "MEG OSF cache missing",
            }
        else:
            fits = _try_local_fits(
                meg_mat=meg_mat,
                spm_mat=spm_mat,
                permutation_b=permutation_b,
                rng=rng,
            )

    paper_fit = fits.get("paper_faithful")
    airi_fit = fits.get("airi_executable")

    if isinstance(paper_fit, PatternFit):
        k_index = three_lowest_p_indices(paper_fit.eigenvalues, paper_fit.p_values, k=k)
        a_k = paper_fit.patterns[k_index].T  # (n_channels, K)
        fig18_scan = music_scan(blocks, a_k)
        fig18_status = "approximate"
        fig18_reason = (
            "Eq. 14 MUSIC of a locally fitted Fig. 16/17-style subspace on the "
            "public AD overlapping-spheres Gain (5002 vtx). Approximate because: "
            "(1) individual T1 is not released so Gain cannot be rebuilt; "
            "(2) no show_on_cortex/FieldTrip screenshot parity; "
            "(3) paper permutation B is unspecified — using documented "
            f"B={permutation_b} and the three lowest-p components as the Fig. 17 analog; "
            "(4) Haufe patterns (D9), unique unscaled Gram, full 1501-sample epoch."
        )
        rec18 = _scan_record(
            scan_id="fig18-meg-music",
            figure="Figure 18",
            status=fig18_status,
            algorithm="paper_eq14_music_free_orientation_2_left_SV",
            path_label="paper_faithful",
            scan=fig18_scan,
            model=model,
            extra={
                "is_fig18": True,
                "status_detail": fig18_reason,
                "subspace": _pattern_json(paper_fit, k_index),
                "K": int(k_index.size),
                "paper_claimed_regions": [
                    "right fusiform gyrus",
                    "right insula",
                    "left intraparietal sulcus",
                    "anterior central gyrus",
                ],
                "atlas_at_peak_is_not_screenshot_parity": True,
            },
        )
        artifacts["fig18"] = _write_map(
            out_dir,
            "fig18_music_paper_faithful",
            fig18_scan,
            model.vertices,
            title="Fig.18 analog: Eq.14 MUSIC of local paper_faithful A_K (not FieldTrip)",
            peak_index=rec18["peak"]["vertex_index_0based"],
        )
        rec18["artifacts"] = artifacts["fig18"]
        records["fig18-meg-music"] = rec18
        write_json(out_dir / "fig18_meg_music.json", rec18)
        write_json(out_dir / "paper_faithful_patterns.json", _pattern_json(paper_fit, k_index))
        np.savez_compressed(
            out_dir / "maps" / "paper_faithful_patterns.npz",
            patterns=paper_fit.patterns,
            filters=paper_fit.filters,
            eigenvalues=paper_fit.eigenvalues,
            p_values=(
                paper_fit.p_values
                if paper_fit.p_values is not None
                else np.array([], dtype=np.float64)
            ),
            selected=k_index.astype(np.int64),
            rdm=paper_fit.rdm,
        )
    else:
        rec18_blocked = {
            "id": "fig18-meg-music",
            "figure": "Figure 18",
            "status": "blocked",
            "is_fig18": True,
            "path_label": "paper_faithful",
            "blocked_reason": fig18_reason,
            "meg_fit_error": fits.get("error"),
            "note": (
                "OSF A1 columns 1–K were scanned separately as "
                "author-saved-A1-cols1-3-music; that file is not Fig. 18."
            ),
        }
        records["fig18-meg-music"] = rec18_blocked
        write_json(out_dir / "fig18_meg_music.json", rec18_blocked)

    if isinstance(airi_fit, PatternFit):
        k_airi = np.arange(min(k, airi_fit.patterns.shape[0]), dtype=np.int64)
        airi_local = music_scan(blocks, airi_fit.patterns[k_airi].T)
        rec_al = _scan_record(
            scan_id="airi-executable-local-fit-music-k3",
            figure="not Fig. 18",
            status="approximate",
            algorithm="paper_eq14_music_free_orientation",
            path_label="airi_executable",
            scan=airi_local,
            model=model,
            extra={
                "is_fig18": False,
                "note": (
                    "MUSIC of locally fitted AIRI-executable facevstool patterns "
                    "(directed pairs, MATLAB cov, 99–999 ms, 0.25–20 Hz). "
                    "Not paper Fig. 18."
                ),
                "subspace": _pattern_json(airi_fit, k_airi),
            },
        )
        artifacts["airi-local"] = _write_map(
            out_dir,
            "airi_executable_local_music_k3",
            airi_local,
            model.vertices,
            title="MUSIC of local airi_executable K=3 (not Fig. 18)",
            peak_index=rec_al["peak"]["vertex_index_0based"],
        )
        rec_al["artifacts"] = artifacts["airi-local"]
        records["airi-executable-local-fit-music-k3"] = rec_al
        write_json(out_dir / "airi_executable_local_fit_music.json", rec_al)
        write_json(out_dir / "airi_executable_patterns.json", _pattern_json(airi_fit, k_airi))

    asset_hashes = {
        name: sha256_file(SOURCE_MODEL_DIR / name)
        for name in (
            "headmodel_surf_os_meg.mat",
            "tess_cortex_pial_low.mat",
            "results_sLORETA_MEG_GRAD_MEG_MAG_KERNEL_150924_1824.mat",
        )
    }
    asset_hashes["topo_face_vs_tool_correct_filt15.mat"] = author["sha256"]

    summary = {
        "track": "source_localization",
        "fig18_status": records["fig18-meg-music"]["status"],
        "fig18_peak": records["fig18-meg-music"].get("peak"),
        "airi_precomp_status": records["airi-source-loc-precomp"]["status"],
        "airi_precomp_numeric_map": records["airi-source-loc-precomp"].get("numeric_map"),
        "airi_precomp_peak": records["airi-source-loc-precomp"]["peak"],
        "airi_music_literal_bug_status": records["airi-music-literal-bug"]["status"],
        "airi_music_fix_peak": records["airi-music-eye-Nsns-fix-A1-col4"]["peak"],
        "index_audit": audit,
        "meg_fit_error": fits.get("error"),
        "permutation_B": permutation_b if isinstance(paper_fit, PatternFit) else None,
        "rng": rng_record.to_dict(),
        "asset_sha256": asset_hashes,
        "commands": [
            "PYTHONPATH=src:paper/reproduction python paper/reproduction/source_localization/run.py",
            "python -m pytest paper/reproduction/source_localization/tests -q",
        ],
        "provenance": capture_run(
            track="source_localization",
            path_label="paper_faithful+airi_executable",
            seed_record=rng_record,
            extra={"skip_meg_fit": skip_meg_fit, "permutation_b": permutation_b},
        ),
        "ids": sorted(records),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "index_audit.json", audit)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--permutation-b",
        type=int,
        default=200,
        help="Paper-style condition-label permutations (paper B unspecified).",
    )
    parser.add_argument(
        "--skip-meg-fit",
        action="store_true",
        help="Skip local Fig. 16/17 refit (Fig. 18 becomes blocked).",
    )
    parser.add_argument("--k", type=int, default=FIG18_K)
    parser.add_argument("--seed", type=int, default=20240915)
    args = parser.parse_args()
    summary = run(
        permutation_b=args.permutation_b,
        skip_meg_fit=args.skip_meg_fit,
        k=args.k,
        seed=args.seed,
    )
    print(
        "fig18_status={0} airi_precomp_status={1}".format(
            summary["fig18_status"], summary["airi_precomp_status"]
        )
    )
    if summary.get("fig18_peak"):
        peak = summary["fig18_peak"]
        print(
            "fig18_peak vertex={0} xyz={1} value={2} {3}/{4}".format(
                peak["vertex_index_0based"],
                peak["xyz_m"],
                peak["scan_value"],
                peak.get("hemisphere"),
                peak.get("Mindboggle"),
            )
        )
    pre = summary["airi_precomp_peak"]
    print(
        "airi_precomp_peak vertex={0} xyz={1} value={2} {3}/{4}".format(
            pre["vertex_index_0based"],
            pre["xyz_m"],
            pre["scan_value"],
            pre.get("hemisphere"),
            pre.get("Mindboggle"),
        )
    )


if __name__ == "__main__":
    main()
