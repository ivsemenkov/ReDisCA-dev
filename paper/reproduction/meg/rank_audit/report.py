"""Render RANK_AUDIT.md from computed JSON (no MEG I/O)."""

from __future__ import annotations

from typing import Any


def _fmt(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np_isfinite(number):
        return "nan"
    abs_n = abs(number)
    if abs_n != 0.0 and (abs_n < 1e-3 or abs_n >= 1e4):
        return f"{number:.6e}"
    return f"{number:.{digits}g}"


def np_isfinite(number: float) -> bool:
    return number == number and number not in (float("inf"), float("-inf"))


def _window_table_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| i (1-based) | eigenvalue | λᵢ / λ_max | λᵢ > λ_max·1e-6 | margin (ratio − 1e-6) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {i} | {eig} | {ratio} | {above} | {margin} |".format(
                i=row["index_1based"],
                eig=_fmt(row["eigenvalue"]),
                ratio=_fmt(row["ratio_to_max"], digits=8),
                above="yes" if row["above_cutoff"] else "**no**",
                margin=_fmt(row["margin_ratio"], digits=8),
            )
        )
    return "\n".join(lines)


def _path_section(spec: dict[str, Any]) -> str:
    flip = spec["matlab_eig_flip"]
    pca67 = spec["pca_interval_for_exactly_67"]
    band = spec.get("bandpass")
    if band is None:
        band_txt = "none"
    else:
        band_txt = (
            f"butter({band.get('order')}) {band.get('low_hz')}–{band.get('high_hz')} Hz"
        )
        if band.get("note"):
            band_txt += f". Note: {band['note']}"
    win = spec.get("window_ms", [None, None])
    solvers = spec["solver_comparison"]
    ranks = solvers["ranks"]
    lines = [
        f"### `{spec['label']}`",
        "",
        f"- Pairs: `{spec['pair_mode']}` ({spec['n_pairs']} pair matrices).",
        f"- Pair matrix: `{spec['matrix_mode']}`.",
        f"- Window: {win[0]}…{win[1]} ms, T={spec['n_times']}.",
        f"- Bandpass: {band_txt}.",
        f"- λ_max = `{_fmt(spec['eig_max'])}`, cutoff = λ_max·1e-6 = `{_fmt(spec['cutoff'])}`.",
        f"- **Numerical rank at 1e-6 (numpy eigh): {spec['numerical_rank_1e-6']}**.",
        f"- `whiten_from_covariance(..., pca_var_explained=1)` rows: "
        f"{spec['whitening_n_components_pca1']}.",
        f"- Solver ranks: numpy_eigh={ranks['numpy_eigh']}, "
        f"scipy_eigh={ranks['scipy_eigh']}, scipy_eig_real={ranks['scipy_eig_real']}"
        f"{' (agree)' if solvers['all_solvers_agree_on_rank'] else ' (**disagree**)'}.",
        f"- max |numpy eigh − scipy eigh| = `{_fmt(solvers['max_abs_numpy_vs_scipy_eigh'])}`.",
        f"- Cxx asymmetry ‖C−Cᵀ‖_F / ‖C‖_F = `{_fmt(solvers['asymmetry_fro_over_fro'])}`.",
        "",
        "Indices 60–75 (1-based, descending λ):",
        "",
        _window_table_md(spec["window_60_75"]),
        "",
        f"Focus: λ₆₇/λ_max=`{_fmt(flip['ratio_67'], digits=8)}`, "
        f"λ₆₈/λ_max=`{_fmt(flip['ratio_68'], digits=8)}`, "
        f"λ₆₉/λ_max=`{_fmt(flip['ratio_69'], digits=8)}`.",
        f"Margin of 68 above cutoff = `{_fmt(flip['margin_68_above_cutoff'], digits=8)}`; "
        f"margin of 69 below = `{_fmt(flip['margin_69_below_cutoff'], digits=8)}`; "
        f"gap 68−69 = `{_fmt(flip['gap_ratio_68_minus_69'], digits=8)}`.",
        f"Relative drop of λ₆₈ needed to cross the cutoff: "
        f"`{_fmt(flip['relative_drop_of_eig68_to_cross_cutoff'], digits=6)}`.",
        f"MATLAB-eig-flip verdict (gap argument, not parity): **{flip['verdict']}** "
        f"(plausible solver-only 68→67: {flip['plausible_that_matlab_eig_alone_flips_68_to_67']}).",
        "",
        "SPoC `pca_X_var_explained` (AIRI does not pass it; default 1):",
        "",
        f"- default pca=1 selects **{spec['pca_n_at_default_1']}** components "
        f"(= numerical rank after `min(n, r)`).",
        f"- interval that would yield *exactly* 67 before `min(., r)`: "
        f"{pca67.get('interval_note', pca67.get('reason'))}.",
        f"- cumulative variance at 66 / 67: "
        f"`{_fmt(pca67.get('cumulative_at_n_minus_1'), digits=12)}` / "
        f"`{_fmt(pca67.get('cumulative_at_n'), digits=12)}`.",
    ]
    extra = spec.get("extra")
    if extra:
        lines.extend(["", "Extra:", ""])
        for key, value in extra.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _overlap_md(title: str, overlap: dict[str, Any] | None) -> str:
    if not overlap:
        return f"**{title}:** not computed."
    sub = overlap["subspace_first_n_take"]
    lead = overlap.get("leading_column_abs_pearson") or []
    lead_txt = ", ".join(
        f"c{row['component_1based']} |r|={_fmt(row['abs_pearson'], digits=4)}"
        for row in lead
    )
    return (
        f"**{title}** (local n={overlap['local_n_patterns']}, "
        f"compared dim={overlap['compared_dim']}): "
        f"min cosine `{_fmt(sub['min_cosine'], digits=4)}`, "
        f"max principal angle `{_fmt(sub['max_angle_rad'], digits=4)}` rad. "
        f"Leading-column |Pearson|: {lead_txt or '—'}."
    )


def render_rank_audit_md(payload: dict[str, Any]) -> str:
    paths = payload["paths"]
    a1 = payload["author_saved_a1"]
    airi_src = payload["airi_spoc_inspection"]
    conclusion = payload["conclusion"]
    extras = payload.get("diagnostic_paths") or {}

    lines = [
        "# MEG rank audit: 67 vs 68",
        "",
        "Track F. Owner: `paper/reproduction/meg/rank_audit/`. "
        "Does **not** rerun MEG Monte Carlo, SPoC B=1000, or source localization.",
        "",
        "Question: local reconstructions report whitening rank **68**; author-saved "
        "AIRI `A1` is **204 × 67**. Is that a SciPy-vs-MATLAB eig borderline at "
        "`tol = λ_max · 1e-6`, or a different Cxx / an explicit PCA setting?",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src:paper/reproduction python3 paper/reproduction/meg/rank_audit/run.py",
        "PYTHONPATH=src:paper/reproduction python3 -m pytest paper/reproduction/meg/rank_audit -q",
        "```",
        "",
        "No bootstrap. Pair matrices only. Bandpass is the AIRI 0.25–20 Hz "
        "`scipy.signal.filtfilt` reconstruction (D8: not MATLAB parity).",
        "",
        "## Source evidence (not tuned)",
        "",
        "| Claim | Source |",
        "| --- | --- |",
        "| Rank cutoff `tol = ev(1)*1e-6`, `r = sum(ev > tol)` | stock SPoC `utils/whiten_data.m` @ `18e4754` |",
        "| Optional PCA `pca_X_var_explained` default **1** | `SPoC/spoc.m` `set_defaults` |",
        "| AIRI call does **not** pass PCA / rank | `spoc(Xspoc, z, 'n_bootstrapping_iterations',1000)` |",
        "| AIRI bandpass 0.25–20 Hz, `trange=600:1500` | `Redisca_tools_faces_3_random_norm_correct.m` |",
        "| `A1` is 204×67 Haufe patterns after whitening | `stock_spoc.md`; OSF `topo_face_vs_tool_correct_filt15.mat` |",
        "| Committed script `return`s before `save topo_*` (D17) | same AIRI main script |",
        "| Filename `filt15` is **not** `highCutOff` | committed `highCutOff=20`; `cfg.ylim=[15 20]` is a plot setting |",
        "",
        "## Verdict (computed)",
        "",
        conclusion.get("summary_markdown", conclusion.get("summary", "")),
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| paper_faithful rank | **{paths['paper_faithful']['numerical_rank_1e-6']}** |",
        f"| airi_executable rank | **{paths['airi_executable']['numerical_rank_1e-6']}** |",
        f"| author-saved A1 columns | **{a1['n_columns']}** |",
        f"| Force local rank to 67 because A1 has 67 columns? | **no** |",
        f"| MATLAB-eig-only flip (paper path) | {paths['paper_faithful']['matlab_eig_flip']['verdict']} |",
        f"| MATLAB-eig-only flip (AIRI path) | {paths['airi_executable']['matlab_eig_flip']['verdict']} |",
        f"| Explicit AIRI PCA setting for 67? | {airi_src['explicit_pca_or_rank_setting']} |",
    ]
    extra_rank_lines = [
        f"| `{name}` | {spec['numerical_rank_1e-6']} |" for name, spec in extras.items()
    ]
    if extra_rank_lines:
        lines.extend(
            [
                "",
                "Labeled extras (still not a reason to force rank 67):",
                "",
                "| extra Cxx | rank at 1e-6 |",
                "| --- | --- |",
                *extra_rank_lines,
            ]
        )
    lines.extend(
        [
            "",
            "## Cxx / Rbar spectra",
            "",
            _path_section(paths["paper_faithful"]),
            _path_section(paths["airi_executable"]),
        ]
    )

    if extras:
        lines.extend(["## Diagnostic Cxx (labeled extras; not the two owned paths)", ""])
        for spec in extras.values():
            lines.append(_path_section(spec))

    lines.extend(
        [
            "## Author-saved A1",
            "",
            f"- File: `{a1['path']}`",
            f"- SHA-256: `{a1['sha256']}`",
            f"- Shape: **{a1['shape'][0]} × {a1['shape'][1]}** (`comps_order`={a1.get('comps_order')}).",
            f"- Column ‖·‖₂: min `{_fmt(a1['column_norm_min'])}`, "
            f"median `{_fmt(a1['column_norm_median'])}`, "
            f"max `{_fmt(a1['column_norm_max'])}`.",
            f"- SVD numerical rank of A1 itself at 1e-6 relative: "
            f"{a1['svd_numerical_rank_1e-6_of_singular_values']}.",
            "",
            _overlap_md(
                "Subspace vs local AIRI-executable Haufe patterns (facevstool, no bootstrap)",
                a1.get("vs_airi_executable_haufe"),
            ),
            "",
            _overlap_md(
                "Subspace vs local paper-faithful Haufe patterns (facevstool Gram, no bootstrap)",
                a1.get("vs_paper_faithful_haufe"),
            ),
            "",
            a1.get("note", ""),
            "",
            "## AIRI / SPoC clone inspection",
            "",
            f"- AIRI pin: `{airi_src.get('airi_commit')}`.",
            f"- SPoC pin: `{airi_src.get('spoc_commit')}`.",
            f"- Main-script `spoc(...)` kwargs: `{airi_src.get('spoc_call')}`.",
            f"- `highCutOff` in main script: `{airi_src.get('high_cutoff_hz')}`.",
            f"- `pca_X_var_explained` / `filt15` / `highCutOff` mentions in AIRI `.m`: "
            f"{airi_src.get('pca_rank_mentions_text') or airi_src.get('pca_rank_mentions')}.",
            f"- Source-loc loads `{airi_src.get('source_loc_topo_file')}` "
            f"(not produced by a vanilla run of the committed main script).",
            f"- Plot `cfg.ylim=[15 20]` present: {airi_src.get('plot_ylim_15_20')} "
            "(FieldTrip display limits, not a 15 Hz Butterworth).",
            "",
            "## What this does not claim",
            "",
            "- MATLAB `eig` / `filtfilt` bit-exact parity (MATLAB is not in this environment).",
            "- That 67 is “the correct” rank to impose on the Python reconstruction.",
            "- Any change to MEG GEP *p*-values (those tracks are frozen).",
            "",
            "## Environment",
            "",
            f"- Python packages: `{payload.get('environment', {}).get('packages')}`.",
            f"- MATLAB: `{payload.get('environment', {}).get('matlab')}`.",
            f"- Captured: `{payload.get('environment', {}).get('captured_at_utc')}`.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
