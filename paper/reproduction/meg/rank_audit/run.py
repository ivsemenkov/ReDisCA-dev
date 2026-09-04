#!/usr/bin/env python3
"""Track F MEG rank audit: Cxx / Rbar spectra at the SPoC 1e-6 cutoff.

Does not run SPoC bootstrap, paper permutation, time-course Monte Carlo,
or source localization.

    PYTHONPATH=src:paper/reproduction python3 paper/reproduction/meg/rank_audit/run.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

_REPO = Path(__file__).resolve().parents[4]
_SRC = _REPO / "src"
_REPRO = _REPO / "paper" / "reproduction"
for _p in (_SRC, _REPRO):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from common.hashing import sha256_file  # noqa: E402
from common.paths import AIRI_CLONE, MEG_DIR, PAPER_ROOT, SOURCE_MODEL_DIR, SPOC_CLONE  # noqa: E402
from common.provenance import capture_environment  # noqa: E402
from common.serialize import save_metrics  # noqa: E402
from common.source_faithful import (  # noqa: E402
    AIRI_FILTER,
    airi_bandpass_trials,
    airi_rdm,
    pair_indices,
    pair_stack_from_condition_averages,
    spoc_from_pair_stack,
    theoretical_rdm_vector,
)
from meg.prepare import (  # noqa: E402
    AIRI_SLICE,
    airi_time_ms,
    bandpass_airi,
    condition_averages,
    extract_used_trials,
    load_meg_bundle,
)
from meg.rank_audit.report import render_rank_audit_md  # noqa: E402
from meg.rank_audit.spectrum import (  # noqa: E402
    RANK_TOL,
    author_a1_payload,
    mean_pair_matrix,
    spectrum_payload,
)

RESULTS_DIR = PAPER_ROOT / "results" / "meg" / "rank_audit"
REPORT_PATH = PAPER_ROOT / "reproduction" / "meg" / "RANK_AUDIT.md"
AUTHOR_TOPO_CANDIDATES = (
    MEG_DIR / "topo_face_vs_tool_correct_filt15.mat",
    SOURCE_MODEL_DIR / "topo_face_vs_tool_correct_filt15.mat",
)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float64, copy=False).tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def inspect_airi_spoc_sources() -> dict[str, Any]:
    """Read pinned AIRI/SPoC clones. No MATLAB execution."""
    main_path = AIRI_CLONE / "Redisca_tools_faces_3_random_norm_correct.m"
    sloc_path = AIRI_CLONE / "Redisca_source_loc_for_tools_faces_3_random_.m"
    spoc_path = SPOC_CLONE / "SPoC" / "spoc.m"
    whiten_path = SPOC_CLONE / "utils" / "whiten_data.m"
    main = main_path.read_text(encoding="utf-8", errors="replace") if main_path.exists() else ""
    sloc = sloc_path.read_text(encoding="utf-8", errors="replace") if sloc_path.exists() else ""
    spoc = spoc_path.read_text(encoding="utf-8", errors="replace") if spoc_path.exists() else ""
    whiten = whiten_path.read_text(encoding="utf-8", errors="replace") if whiten_path.exists() else ""

    spoc_calls = re.findall(r"(?<![\w])spoc\s*\(([^;]+)\)\s*;", main, flags=re.S)
    pca_hits = []
    for label, text in (("main", main), ("source_loc", sloc)):
        for pat in ("pca_X_var_explained", "pca_var", "filt15", "highCutOff"):
            n = text.count(pat)
            if n:
                pca_hits.append({"file": label, "pattern": pat, "n": n})
    pca_hits_txt = (
        "; ".join(f"{h['file']}: {h['pattern']} ×{h['n']}" for h in pca_hits)
        if pca_hits
        else "none"
    )

    high = re.search(r"highCutOff\s*=\s*([0-9.]+)", main)
    low = re.search(r"lowCutOff\s*=\s*([0-9.]+)", main)
    default_pca = bool(re.search(r"'pca_X_var_explained'\s*,\s*1", spoc))
    tol_line = ""
    for line in whiten.splitlines():
        if "10^-6" in line or "1e-6" in line.lower() or "10^-6" in line.replace(" ", ""):
            tol_line = line.strip()
            break

    explicit = "none in AIRI scripts; SPoC default pca_X_var_explained=1 (numerical rank only)"
    if any("pca_X_var_explained" in h["pattern"] and h["file"] == "main" for h in pca_hits):
        explicit = "AIRI main script mentions pca_X_var_explained"
    return {
        "airi_commit": "15bc19cdc76989da202714b257f6de4d26a42c51",
        "spoc_commit": "18e4754aec1411160fd5b7ef0db852f1e0a87d90",
        "airi_main_sha256": sha256_file(main_path) if main_path.exists() else None,
        "airi_source_loc_sha256": sha256_file(sloc_path) if sloc_path.exists() else None,
        "spoc_call": spoc_calls[0].replace("\n", " ").strip() if spoc_calls else None,
        "n_spoc_calls_in_main": len(spoc_calls),
        "low_cutoff_hz": float(low.group(1)) if low else None,
        "high_cutoff_hz": float(high.group(1)) if high else None,
        "spoc_default_pca_is_1": default_pca,
        "whiten_tol_line": tol_line,
        "pca_rank_mentions": pca_hits,
        "pca_rank_mentions_text": pca_hits_txt,
        "explicit_pca_or_rank_setting": explicit,
        "source_loc_topo_file": "topo_face_vs_tool_correct_filt15",
        "main_saves_after_return": bool(re.search(r"^\s*return\s*;", main, flags=re.M))
        and "save topo_face_vs_tool_correct" in main,
        "plot_ylim_15_20": ("cfg.ylim = [15 20]" in main) or ("cfg.ylim = [15 20]" in sloc),
        "filt15_in_main_script": "filt15" in main,
        "filt15_in_source_loc": "filt15" in sloc,
        "note": (
            "Committed AIRI MEG call is spoc(..., n_bootstrapping_iterations=1000) "
            "with no pca_X_var_explained. Filename filt15 is not highCutOff=20. "
            "D17: return precedes save topo_*."
        ),
    }


def _haufe_patterns(
    averages: np.ndarray,
    *,
    pair_mode: str,
    matrix_mode: str,
) -> np.ndarray:
    """Deterministic SPoC GEP, B=0. Used only to interpret A1's column space."""
    rdm = airi_rdm("facevstool")
    pairs = pair_indices(averages.shape[0], pair_mode)  # type: ignore[arg-type]
    stack = pair_stack_from_condition_averages(
        averages, pairs, matrix_mode=matrix_mode  # type: ignore[arg-type]
    )
    z_raw = theoretical_rdm_vector(rdm, pairs)
    result = spoc_from_pair_stack(
        stack,
        z_raw,
        n_bootstrapping_iterations=0,
        pair_mode=pair_mode,  # type: ignore[arg-type]
        matrix_mode=matrix_mode,  # type: ignore[arg-type]
        inference="none",
    )
    return np.asarray(result.patterns, dtype=np.float64)


def _cxx_spec(
    averages: np.ndarray,
    *,
    label: str,
    pair_mode: str,
    matrix_mode: str,
    window_ms: np.ndarray,
    bandpass: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cxx, n_pairs = mean_pair_matrix(
        averages, pair_mode=pair_mode, matrix_mode=matrix_mode
    )
    other_mode = "airi_directed" if pair_mode == "unique_unordered" else "unique_unordered"
    cxx_other, n_other = mean_pair_matrix(
        averages, pair_mode=other_mode, matrix_mode=matrix_mode
    )
    extra_payload = {
        "cxx_directed_vs_unique_max_abs": float(np.max(np.abs(cxx - cxx_other))),
        "n_pairs_alternate_mode": n_other,
        "alternate_pair_mode": other_mode,
        "note_pairs": (
            "For a symmetric pair matrix (Gram or cov of ±Δ), directed duplication "
            "does not change Cxx. Rank is a property of Cxx, not of z or bootstrap."
        ),
    }
    if extra:
        extra_payload.update(extra)
    t = np.asarray(window_ms, dtype=np.float64)
    return spectrum_payload(
        cxx,
        label=label,
        pair_mode=pair_mode,
        matrix_mode=matrix_mode,
        n_times=int(averages.shape[2]),
        n_pairs=n_pairs,
        bandpass=bandpass,
        window_ms=[float(t[0]), float(t[-1])],
        extra=extra_payload,
    )


def _load_author_a1() -> tuple[np.ndarray, np.ndarray, Path]:
    path = next((p for p in AUTHOR_TOPO_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Author-saved A1 missing. Expected "
            + " or ".join(str(p) for p in AUTHOR_TOPO_CANDIDATES)
        )
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    a1 = np.asarray(payload["A1"], dtype=np.float64)
    comps = np.asarray(payload["comps_order"], dtype=np.int64).ravel()
    return a1, comps, path


def _averages_from_used(
    used: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    n_times = used.shape[1]
    averages = np.empty((6, used.shape[0], n_times), dtype=np.float64)
    for condition in range(6):
        averages[condition] = used[:, :, labels == condition].mean(axis=2)
    return averages


def build_conclusion(
    paths: dict[str, Any],
    diagnostics: dict[str, Any],
    a1: dict[str, Any],
    airi_src: dict[str, Any],
) -> dict[str, Any]:
    paper = paths["paper_faithful"]
    airi = paths["airi_executable"]
    r_paper = int(paper["numerical_rank_1e-6"])
    r_airi = int(airi["numerical_rank_1e-6"])
    n_a1 = int(a1["n_columns"])
    flip_p = paper["matlab_eig_flip"]
    flip_a = airi["matlab_eig_flip"]
    filt15 = diagnostics.get("airi_window_filt15_matlab_cov")
    unfilt = diagnostics.get("airi_window_unfiltered_matlab_cov")
    airi_overlap = (a1.get("vs_airi_executable_haufe") or {}).get("subspace_first_n_take") or {}
    paper_overlap = (a1.get("vs_paper_faithful_haufe") or {}).get("subspace_first_n_take") or {}
    min_cos_airi = airi_overlap.get("min_cosine")
    min_cos_paper = paper_overlap.get("min_cosine")

    disagree_a1 = not (r_paper == n_a1 and r_airi == n_a1)
    both_68 = r_paper == 68 and r_airi == 68
    solver_only = (
        flip_p["plausible_that_matlab_eig_alone_flips_68_to_67"]
        or flip_a["plausible_that_matlab_eig_alone_flips_68_to_67"]
    )
    filt15_explains = filt15 is not None and int(filt15["numerical_rank_1e-6"]) == n_a1
    unfilt_rank = None if unfilt is None else int(unfilt["numerical_rank_1e-6"])

    paragraphs = []
    paragraphs.append(
        f"Paper-faithful unique-unordered unscaled Gram on the full 1501-sample "
        f"unfiltered averages has rank **{r_paper}** at `tol = λ_max·1e-6`. "
        f"AIRI-executable directed MATLAB-cov on the 99–999 ms window after "
        f"0.25–20 Hz scipy filtfilt has rank **{r_airi}**. Author-saved `A1` has "
        f"**{n_a1}** columns."
    )
    if disagree_a1:
        paragraphs.append(
            "These numbers **disagree**. This audit does **not** truncate local "
            "whitening to 67 merely because the OSF file has 67 columns."
        )
    if both_68 and n_a1 == 67:
        paragraphs.append(
            f"On both owned Cxx paths the 68th relative eigenvalue is "
            f"`{_ratio(paper, 68)}` (paper) and `{_ratio(airi, 68)}` (AIRI), "
            f"versus cutoff `1e-6`. The 69th is `{_ratio(paper, 69)}` and "
            f"`{_ratio(airi, 69)}`. MATLAB-eig-only flip verdicts: paper "
            f"**{flip_p['verdict']}**, AIRI **{flip_a['verdict']}**."
        )
    if solver_only:
        paragraphs.append(
            "The cutoff sits close enough to λ₆₈ that a MATLAB `eig` vs SciPy "
            "`eigh` discrepancy *could* move the count. That is a gap statement, "
            "not a measured MATLAB spectrum."
        )
    else:
        paragraphs.append(
            "The gap from λ₆₈ down to the cutoff, and from the cutoff down to λ₆₉, "
            "is large compared with dense-Hermitian solver noise (~n·ε ≈ 4×10⁻¹⁴, "
            "padded to 10⁻¹²). A SciPy-vs-MATLAB eig difference **alone** is not "
            "a plausible explanation of 67 vs 68 on *these* Cxx matrices."
        )
    paragraphs.append(
        f"AIRI MATLAB does not pass `pca_X_var_explained` "
        f"(`{airi_src.get('spoc_call')}`). SPoC default pca=1 keeps the numerical "
        f"rank, so an explicit PCA setting is **not** present in the committed "
        f"scripts. The `pca` interval that would yield exactly 67 on the AIRI "
        f"Cxx is `{airi['pca_interval_for_exactly_67'].get('interval_note')}`."
    )
    if filt15 is not None:
        paragraphs.append(
            f"Labeled extra: AIRI window + butter(3) **0.25–15 Hz** (filename "
            f"`filt15` hypothesis) has rank **{filt15['numerical_rank_1e-6']}**. "
            + (
                "That matches A1's column count, so a different Cxx (different "
                "filter) remains a live hypothesis for the saved file — not a "
                "reason to change the 0.25–20 Hz AIRI-executable path."
                if filt15_explains
                else "That does **not** match 67, so the filename alone does not "
                "explain A1's column count on this reconstruction."
            )
        )
    if unfilt_rank is not None:
        paragraphs.append(
            f"AIRI window **without** bandpass, MATLAB cov: rank **{unfilt_rank}**."
        )
    if min_cos_airi is not None:
        lead = (a1.get("vs_airi_executable_haufe") or {}).get("leading_column_abs_pearson") or []
        lead_txt = ", ".join(
            f"c{row['component_1based']}|r|={row['abs_pearson']:.3f}" for row in lead
        )
        airi_ang = (a1.get("vs_airi_executable_haufe") or {}).get("subspace_first_n_take") or {}
        paragraphs.append(
            f"A1 is **AIRI-like in the leading columns** of a local facevstool "
            f"Haufe fit (no bootstrap): {lead_txt or 'n/a'}. The 67-D column space "
            f"vs the first 67 of the rank-{r_airi} AIRI-executable patterns has min "
            f"cosine `{min_cos_airi:.4g}` (max principal angle "
            f"`{airi_ang.get('max_angle_rad', float('nan')):.3g}` rad). "
            f"Vs paper-faithful Haufe min cosine is `{min_cos_paper:.4g}` (worse; "
            "D2/D6/D8). Matching leading topographies does **not** license truncating "
            "local Cxx to 67 columns."
        )
    diagnostic_ranks = {
        name: int(spec["numerical_rank_1e-6"]) for name, spec in diagnostics.items()
    }
    if diagnostic_ranks and all(rank == 68 for rank in diagnostic_ranks.values()) and both_68:
        paragraphs.append(
            "Every labeled extra Cxx we formed is also rank **68**, with a cliff "
            "between λ₆₈ (few ×10⁻⁶ λ_max) and λ₆₉ (~10⁻⁸ λ_max) then a numerical "
            "floor (~10⁻¹⁴). Window, Gram-vs-cov, 0.25–20 Hz, 0.25–15 Hz, and no "
            "bandpass do not move the count. The rank looks like a property of the "
            "tSSS planar data, not of those analysis knobs."
        )
    paragraphs.append(
        "Remaining untested path to 67: MATLAB Signal Processing Toolbox `filtfilt` "
        "plus MATLAB `eig` on **that** Cxx (D8 + D10 together), or some other "
        "unsaved MATLAB run (D17). That is a *different matrix*, not a solver-only "
        "flip of the SciPy Cxx computed here. It is not an explicit "
        "`pca_X_var_explained` setting in the committed AIRI scripts."
    )
    cause = "different_cxx_or_unsaved_matlab_run"
    if solver_only and both_68 and n_a1 == 67:
        cause = "possibly_solver_borderline_or_different_cxx"
    elif both_68 and n_a1 == 67 and not solver_only:
        if filt15_explains:
            cause = "not_solver_borderline;_A1_may_be_a_different_Cxx_or_MATLAB_run_(filt15_live)"
        elif min_cos_airi is not None and float(min_cos_airi) >= 0.8:
            cause = (
                "not_solver_borderline;_A1_leading_columns_are_AIRI-like_but_"
                "saved_whitening_size_is_67_vs_local_68_(D17)"
            )
        else:
            cause = "not_solver_borderline;_A1_is_a_different_saved_run_(D17)"
    summary_md = "\n\n".join(paragraphs)
    return {
        "paper_faithful_rank": r_paper,
        "airi_executable_rank": r_airi,
        "author_a1_columns": n_a1,
        "do_not_force_rank_67": True,
        "matlab_eig_flip_plausible_paper": flip_p[
            "plausible_that_matlab_eig_alone_flips_68_to_67"
        ],
        "matlab_eig_flip_plausible_airi": flip_a[
            "plausible_that_matlab_eig_alone_flips_68_to_67"
        ],
        "explicit_airi_pca_setting": airi_src["explicit_pca_or_rank_setting"],
        "filt15_rank": None if filt15 is None else int(filt15["numerical_rank_1e-6"]),
        "airi_window_unfiltered_rank": unfilt_rank,
        "a1_vs_airi_min_cosine": min_cos_airi,
        "a1_vs_paper_min_cosine": min_cos_paper,
        "primary_cause_label": cause,
        "disagreement_with_A1": disagree_a1,
        "diagnostic_ranks": diagnostic_ranks,
        "summary": " ".join(paragraphs),
        "summary_markdown": summary_md,
    }


def _ratio(spec: dict[str, Any], index_1based: int) -> str:
    row = next(r for r in spec["focus_67_69"] if r["index_1based"] == index_1based)
    return f"{row['ratio_to_max']:.8e}"


def run_audit(*, skip_filt15: bool = False) -> dict[str, Any]:
    print("[rank_audit] inspecting AIRI/SPoC clones", flush=True)
    airi_src = inspect_airi_spoc_sources()

    print("[rank_audit] loading MEG_AD_run1.mat (204 planars)", flush=True)
    bundle = load_meg_bundle()
    time_ms = bundle.time_ms
    paper_avg = condition_averages(bundle.planars, bundle.indices)

    print(
        "[rank_audit] paper_faithful: unique unordered + unscaled Gram, "
        f"T={paper_avg.shape[2]}, no bandpass",
        flush=True,
    )
    paper_spec = _cxx_spec(
        paper_avg,
        label="paper_faithful",
        pair_mode="unique_unordered",
        matrix_mode="unscaled_gram",
        window_ms=time_ms,
        bandpass=None,
    )

    print(
        "[rank_audit] AIRI bandpass butter(3) 0.25–20 Hz scipy filtfilt "
        "(not MATLAB parity; D8)",
        flush=True,
    )
    filtered = bandpass_airi(bundle.planars)
    airi_avg = condition_averages(filtered, bundle.indices)[:, :, AIRI_SLICE]
    airi_t = airi_time_ms(time_ms)
    airi_band = {
        "order": int(AIRI_FILTER["order"]),
        "low_hz": float(AIRI_FILTER["low_hz"]),
        "high_hz": float(AIRI_FILTER["high_hz"]),
        "fs": float(AIRI_FILTER["fs"]),
        "note": "scipy filtfilt; AIRI MATLAB Signal Processing Toolbox filtfilt is not bit-exact",
    }
    print(
        "[rank_audit] airi_executable: directed + matlab_cov, "
        f"T={airi_avg.shape[2]} (99–999 ms)",
        flush=True,
    )
    airi_spec = _cxx_spec(
        airi_avg,
        label="airi_executable",
        pair_mode="airi_directed",
        matrix_mode="matlab_cov",
        window_ms=airi_t,
        bandpass=airi_band,
    )

    diagnostics: dict[str, Any] = {}
    print("[rank_audit] diagnostic: AIRI window, unfiltered, matlab_cov", flush=True)
    unfilt_avg = condition_averages(bundle.planars, bundle.indices)[:, :, AIRI_SLICE]
    diagnostics["airi_window_unfiltered_matlab_cov"] = _cxx_spec(
        unfilt_avg,
        label="airi_window_unfiltered_matlab_cov",
        pair_mode="airi_directed",
        matrix_mode="matlab_cov",
        window_ms=airi_t,
        bandpass=None,
        extra={"role": "isolate D8 (bandpass) from D6 (window)"},
    )
    print("[rank_audit] diagnostic: full epoch, unfiltered, matlab_cov unique", flush=True)
    diagnostics["paper_window_matlab_cov"] = _cxx_spec(
        paper_avg,
        label="paper_window_matlab_cov",
        pair_mode="unique_unordered",
        matrix_mode="matlab_cov",
        window_ms=time_ms,
        bandpass=None,
        extra={"role": "isolate D2 (Gram vs cov) on the paper window"},
    )
    print("[rank_audit] diagnostic: AIRI-filtered window, unique Gram", flush=True)
    diagnostics["airi_window_filtered_unscaled_gram"] = _cxx_spec(
        airi_avg,
        label="airi_window_filtered_unscaled_gram",
        pair_mode="unique_unordered",
        matrix_mode="unscaled_gram",
        window_ms=airi_t,
        bandpass=airi_band,
        extra={"role": "isolate D2 on the AIRI window+filter"},
    )

    if not skip_filt15:
        print(
            "[rank_audit] diagnostic: AIRI window, butter(3) 0.25–15 Hz on used trials "
            "(filt15 filename hypothesis)",
            flush=True,
        )
        used, labels = extract_used_trials(bundle.planars, bundle.indices)
        used15 = airi_bandpass_trials(
            used,
            low_hz=0.25,
            high_hz=15.0,
            fs=float(AIRI_FILTER["fs"]),
            order=int(AIRI_FILTER["order"]),
        )
        avg15 = _averages_from_used(used15, labels)[:, :, AIRI_SLICE]
        diagnostics["airi_window_filt15_matlab_cov"] = _cxx_spec(
            avg15,
            label="airi_window_filt15_matlab_cov",
            pair_mode="airi_directed",
            matrix_mode="matlab_cov",
            window_ms=airi_t,
            bandpass={
                "order": int(AIRI_FILTER["order"]),
                "low_hz": 0.25,
                "high_hz": 15.0,
                "fs": float(AIRI_FILTER["fs"]),
                "note": (
                    "Labeled extra for the OSF filename filt15. Not the committed "
                    "AIRI highCutOff=20 path. Applied to the 480 used trials only "
                    "(unused trials never enter condition averages)."
                ),
            },
            extra={"role": "filename filt15 hypothesis; do not replace airi_executable"},
        )

    print("[rank_audit] GEP B=0 for A1 subspace comparison (facevstool)", flush=True)
    paper_patterns = _haufe_patterns(
        paper_avg, pair_mode="unique_unordered", matrix_mode="unscaled_gram"
    )
    airi_patterns = _haufe_patterns(
        airi_avg, pair_mode="airi_directed", matrix_mode="matlab_cov"
    )
    a1, comps, a1_path = _load_author_a1()
    try:
        # Do not Path.resolve(): .reproduction_data is a symlink into the cache.
        a1_path_display = str(a1_path.relative_to(_REPO))
    except ValueError:
        a1_path_display = f".reproduction_data/meg/{a1_path.name}"
    a1_payload = author_a1_payload(
        a1,
        comps_order=comps,
        path=a1_path_display,
        sha256=sha256_file(a1_path),
        airi_patterns=airi_patterns,
        paper_patterns=paper_patterns,
    )
    a1_payload["local_airi_n_patterns"] = int(airi_patterns.shape[1])
    a1_payload["local_paper_n_patterns"] = int(paper_patterns.shape[1])

    paths = {"paper_faithful": paper_spec, "airi_executable": airi_spec}
    conclusion = build_conclusion(paths, diagnostics, a1_payload, airi_src)
    environment = capture_environment(extra_packages=("h5py", "matplotlib"))
    payload = {
        "track": "meg_rank_audit",
        "rank_tol": RANK_TOL,
        "no_bootstrap": True,
        "no_source_localization": True,
        "environment": environment,
        "airi_spoc_inspection": airi_src,
        "paths": paths,
        "diagnostic_paths": diagnostics,
        "author_saved_a1": a1_payload,
        "conclusion": conclusion,
        "meg_source_mat": bundle.source_mat,
        "labels_mat": bundle.labels_mat,
    }
    return _jsonable(payload)


def write_outputs(payload: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(RESULTS_DIR / "summary.json", payload)
    save_metrics(RESULTS_DIR / "paper_faithful_rbar.json", payload["paths"]["paper_faithful"])
    save_metrics(
        RESULTS_DIR / "airi_executable_cxx.json", payload["paths"]["airi_executable"]
    )
    save_metrics(
        RESULTS_DIR / "cutoff_window_60_75.json",
        {
            "paper_faithful": payload["paths"]["paper_faithful"]["window_60_75"],
            "airi_executable": payload["paths"]["airi_executable"]["window_60_75"],
            "diagnostics": {
                name: spec["window_60_75"]
                for name, spec in payload["diagnostic_paths"].items()
            },
        },
    )
    save_metrics(RESULTS_DIR / "author_saved_a1.json", payload["author_saved_a1"])
    save_metrics(
        RESULTS_DIR / "pca_and_solvers.json",
        {
            name: {
                "pca_interval_for_exactly_67": spec["pca_interval_for_exactly_67"],
                "pca_n_at_default_1": spec["pca_n_at_default_1"],
                "solver_comparison": spec["solver_comparison"],
                "matlab_eig_flip": spec["matlab_eig_flip"],
                "numerical_rank_1e-6": spec["numerical_rank_1e-6"],
            }
            for name, spec in {**payload["paths"], **payload["diagnostic_paths"]}.items()
        },
    )
    REPORT_PATH.write_text(render_rank_audit_md(payload), encoding="utf-8")
    print(f"[rank_audit] wrote {RESULTS_DIR}", flush=True)
    print(f"[rank_audit] wrote {REPORT_PATH}", flush=True)
    print(
        "[rank_audit] conclusion:",
        payload["conclusion"]["primary_cause_label"],
        "paper",
        payload["conclusion"]["paper_faithful_rank"],
        "airi",
        payload["conclusion"]["airi_executable_rank"],
        "A1",
        payload["conclusion"]["author_a1_columns"],
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MEG Cxx rank 67 vs 68 audit")
    parser.add_argument(
        "--skip-filt15",
        action="store_true",
        help="Skip the extra 0.25–15 Hz bandpass (filename hypothesis).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_audit(skip_filt15=bool(args.skip_filt15))
    write_outputs(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
