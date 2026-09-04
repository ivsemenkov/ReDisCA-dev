#!/usr/bin/env python3
"""MEG sensor-space reproduction: labeled paper_faithful and airi_executable paths.

Never mix the two paths in one figure or one untagged metric file.

Subcommands::

    python paper/reproduction/meg/run.py paper
    python paper/reproduction/meg/run.py airi
    python paper/reproduction/meg/run.py all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"
_REPRO = _REPO / "paper" / "reproduction"
for _p in (_SRC, _REPRO):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from common.metrics import peak_latency_and_amplitude, pearson, sign_align_vectors, subspace_similarity  # noqa: E402
from common.paths import PAPER_ROOT  # noqa: E402
from common.provenance import capture_run  # noqa: E402
from common.rng import numpy_generator  # noqa: E402
from common.serialize import array_fingerprint, save_metrics, save_npz  # noqa: E402
from common.source_faithful import (  # noqa: E402
    AIRI_N_BOOTSTRAP,
    AIRI_N_MC_TIMECourse,
    fit_condition_averages,
    pair_stack_from_condition_averages,
    unique_unordered_pairs,
)

from meg.figures import plot_component_panel, plot_planar_rms_row, plot_rdm  # noqa: E402
from meg.inference import (  # noqa: E402
    airi_halfsplit_timecourse,
    empirical_rdm_from_traces,
    pair_order_sensitivity,
    paper_component_permutation,
    paper_timeseries_fwer,
    unique_pair_pearson,
)
from meg.prepare import (  # noqa: E402
    AIRI_SLICE,
    MegBundle,
    airi_channel_time_std,
    airi_time_ms,
    bandpass_airi,
    condition_averages,
    extract_used_trials,
    load_meg_bundle,
    prepare_provenance,
)
from meg.rdms import (  # noqa: E402
    AIRI_RDM_NAMES,
    BINARY_RDM_NAMES,
    CONDITION_NAMES,
    PAPER_QUALITATIVE_ONSETS,
    class_labels,
    rdm_catalog,
    theoretical_rdm,
)

RESULTS_ROOT = PAPER_ROOT / "results" / "meg"
PRIMARY_PAPER_RDMS = ("face", "tool", "meaning", "facevstool")
N_REPORT = 8
N_PLOT_PAPER = 3
N_PLOT_AIRI = 4


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.bool_ or obj.dtype == bool:
            return obj.astype(bool).tolist()
        return obj.astype(np.float64, copy=False).tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    save_metrics(path, _jsonable(payload))


def _sign_align_display(
    traces: NDArray[np.float64],
    filters_rows: NDArray[np.float64],
    patterns_rows: NDArray[np.float64],
    *,
    class1: tuple[int, ...],
    time_ms: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Flip each component so mean(class1) is positive at its post-stimulus peak."""
    traces = np.array(traces, dtype=np.float64, copy=True)
    filters_rows = np.array(filters_rows, dtype=np.float64, copy=True)
    patterns_rows = np.array(patterns_rows, dtype=np.float64, copy=True)
    post = np.asarray(time_ms) >= 0.0
    n_comp = traces.shape[1]
    for k in range(n_comp):
        series = traces[list(class1), k][:, post].mean(axis=0)
        if series.size == 0:
            continue
        peak = series[int(np.argmax(np.abs(series)))]
        if peak < 0.0:
            traces[:, k] *= -1.0
            filters_rows[k] *= -1.0
            patterns_rows[k] *= -1.0
    return traces, filters_rows, patterns_rows


def _onset_ms(time_ms: NDArray[np.float64], significant: NDArray[np.bool_], *, min_run: int = 10) -> dict[str, float | None]:
    t = np.asarray(time_ms, dtype=np.float64)
    sig = np.asarray(significant, dtype=bool) & (t >= 0.0)
    idx = np.flatnonzero(sig)
    first = float(t[idx[0]]) if idx.size else None
    cluster = None
    run = 0
    start = None
    for i, flag in enumerate(sig):
        if flag:
            if run == 0:
                start = i
            run += 1
            if run >= min_run and cluster is None and start is not None:
                cluster = float(t[start])
                break
        else:
            run = 0
            start = None
    return {"first_significant_ms": first, "first_cluster10_ms": cluster}


def _peak_ms(time_ms: NDArray[np.float64], series: NDArray[np.float64]) -> dict[str, float | None]:
    t = np.asarray(time_ms, dtype=np.float64)
    y = np.asarray(series, dtype=np.float64)
    post = t >= 0.0
    if not np.any(post):
        return {"peak_ms": None, "peak_amp": None}
    helper = peak_latency_and_amplitude(t[post] / 1000.0, y[post])
    return {
        "peak_ms": float(helper["peak_time_s"] * 1000.0),
        "peak_amp": float(helper["peak_amplitude"]),
        "peak_index_post": int(helper["peak_index"]),
    }


def _component_peaks(
    time_ms: NDArray[np.float64],
    traces: NDArray[np.float64],
    contrast: NDArray[np.float64],
    class1: tuple[int, ...],
    class2: tuple[int, ...],
    n_plot: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in range(n_plot):
        c1 = traces[list(class1), k].mean(axis=0)
        c2 = traces[list(class2), k].mean(axis=0)
        peak_c = _peak_ms(time_ms, contrast[k])
        peak_1 = _peak_ms(time_ms, c1)
        peak_2 = _peak_ms(time_ms, c2)
        out.append(
            {
                "component": k + 1,
                "contrast_peak_ms": peak_c["peak_ms"],
                "contrast_peak_amp": peak_c["peak_amp"],
                "class1_peak_ms": peak_1["peak_ms"],
                "class1_peak_amp": peak_1["peak_amp"],
                "class2_peak_ms": peak_2["peak_ms"],
                "class2_peak_amp": peak_2["peak_amp"],
            }
        )
    return out


def _empirical_for_components(
    traces: NDArray[np.float64],
    rdm: NDArray[np.float64],
    *,
    demean_time: bool,
    n_plot: int,
) -> list[dict[str, Any]]:
    rows = []
    for k in range(n_plot):
        dhat = empirical_rdm_from_traces(traces[:, k, :], demean_time=demean_time)
        rows.append(
            {
                "component": k + 1,
                "pearson_unique_triangle": unique_pair_pearson(dhat, rdm),
                "empirical_fingerprint": array_fingerprint(dhat),
            }
        )
    return rows


def _write_fig12(out_dir: Path, path_label: str) -> None:
    catalog = rdm_catalog()
    fig_dir = out_dir / "figures"
    payload = {
        "path_label": path_label,
        "item_id": "fig12-meg-theoretical-rdms",
        "condition_order": list(CONDITION_NAMES),
        "airi_numeric_0p1_1_or_facevstool": {
            name: catalog["airi_numeric"][name].tolist() for name in AIRI_RDM_NAMES
        },
        "binary_0_1": {name: catalog["binary_0_1"][name].tolist() for name in BINARY_RDM_NAMES},
        "note": (
            "Paper Figs 12a–c are images; numeric entries are not printed. "
            "AIRI within-category fill is 0.1 not 0 (D7). Both fills are emitted. "
            "facevstool is Fig. 16 geometry (0.1/0.5/1), not a binary detector."
        ),
    }
    _save_json(out_dir / "fig12_theoretical_rdms.json", payload)
    if path_label == "paper_faithful":
        for name in BINARY_RDM_NAMES:
            plot_rdm(
                catalog["binary_0_1"][name],
                fig_dir / f"fig12_{name}_binary.png",
                title=f"Fig. 12 {name} binary 0/1",
                path_label=path_label,
                vmin=0.0,
                vmax=1.0,
            )
        plot_rdm(
            catalog["airi_numeric"]["facevstool"],
            fig_dir / "fig16_facevstool_airi_numeric.png",
            title="Fig. 16 / AIRI facevstool 0.1/0.5/1",
            path_label=path_label,
            vmin=0.0,
            vmax=1.0,
        )
    else:
        for name in AIRI_RDM_NAMES:
            plot_rdm(
                catalog["airi_numeric"][name],
                fig_dir / f"airi_rdm_{name}.png",
                title=f"AIRI numeric RDM {name}",
                path_label=path_label,
                vmin=0.0,
                vmax=1.0,
            )


def run_paper(args: argparse.Namespace, bundle: MegBundle | None = None) -> dict[str, Any]:
    """Paper-faithful path: unique pairs, printed Gram, full epoch, no AIRI bandpass."""
    from redisca import ReDisCA

    out_dir = RESULTS_ROOT / "paper_faithful"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    arr_dir = out_dir / "arrays"
    if bundle is None:
        print("[paper_faithful] loading MEG_AD_run1.mat (204 planars, unfiltered)", flush=True)
        bundle = load_meg_bundle()
    averages = condition_averages(bundle.planars, bundle.indices)
    used, labels = extract_used_trials(bundle.planars, bundle.indices)
    time_ms = bundle.time_ms
    rng_perm, rec_perm = numpy_generator(args.seed + 1)
    rng_fwer, rec_fwer = numpy_generator(args.seed + 2)
    _write_fig12(out_dir, "paper_faithful")
    names = list(args.rdms) if args.rdms else list(PRIMARY_PAPER_RDMS)
    summary: dict[str, Any] = {"path_label": "paper_faithful", "rdms": {}}
    extra_demean: dict[str, Any] = {}
    extra_fill: dict[str, Any] = {}

    for name in names:
        fill = "airi" if name == "facevstool" else "binary"
        rdm = theoretical_rdm(name, fill=fill)
        print(
            f"[paper_faithful] ReDisCA(demean_time=False) {name} fill={fill} "
            f"X={averages.shape}",
            flush=True,
        )
        model = ReDisCA(demean_time=False).fit(averages, rdm)
        model_demean = ReDisCA(demean_time=True).fit(averages, rdm)
        n_rank = int(model.rank_)
        n_plot = min(args.n_plot_paper, n_rank)
        filters = np.asarray(model.filters_, dtype=np.float64)
        patterns = np.asarray(model.patterns_, dtype=np.float64)
        evals = np.asarray(model.eigenvalues_, dtype=np.float64)
        traces = np.einsum("kc,nct->nkt", filters, averages)

        class1, class2 = class_labels(name, convention="paper")
        traces, filters, patterns = _sign_align_display(
            traces, filters, patterns, class1=class1, time_ms=time_ms
        )

        pairs = unique_unordered_pairs(averages.shape[0])
        stack = pair_stack_from_condition_averages(averages, pairs, matrix_mode="unscaled_gram")
        print(f"[paper_faithful] condition-label permutation B={args.paper_B} ({name})", flush=True)
        perm = paper_component_permutation(
            stack,
            rdm,
            evals[:N_REPORT],
            n_permutations=args.paper_B,
            rng=rng_perm,
            pair_mode="unique_unordered",
            matrix_mode="unscaled_gram",
        )
        # Spatially filter used trials then FWER (linear ≡ average-then-filter).
        u_trials = np.einsum("kc,cti->kti", filters[:n_plot], used)
        print(f"[paper_faithful] time-series FWER Nmc={args.paper_nmc} ({name})", flush=True)
        fwer = paper_timeseries_fwer(
            u_trials,
            labels,
            class1=class1,
            class2=class2,
            nmc=args.paper_nmc,
            rng=rng_fwer,
        )
        contrast = fwer["observed_contrast"]
        emp = _empirical_for_components(traces, rdm, demean_time=False, n_plot=n_plot)
        peaks = _component_peaks(time_ms, traces, contrast, class1, class2, n_plot)
        onsets = [
            _onset_ms(time_ms, fwer["significant"][k]) for k in range(n_plot)
        ]
        p_max = np.asarray(perm["p_maxabs"], dtype=np.float64)

        demean_filters = np.asarray(model_demean.filters_, dtype=np.float64)
        n_cmp = min(n_plot, demean_filters.shape[0], filters.shape[0])
        aligned_demean = sign_align_vectors(filters[:n_cmp], demean_filters[:n_cmp])
        demean_corrs = [
            float(np.corrcoef(filters[k], aligned_demean[k])[0, 1]) for k in range(n_cmp)
        ]
        extra_demean[name] = {
            "eigenvalues_demean_time_false": evals[:N_REPORT].tolist(),
            "eigenvalues_demean_time_true": np.asarray(model_demean.eigenvalues_[:N_REPORT]).tolist(),
            "sign_aligned_filter_pearson": demean_corrs,
            "subspace_first_n": subspace_similarity(filters[:n_cmp], demean_filters[:n_cmp]),
            "note": "Labeled extra: ReDisCA(demean_time=True). Primary paper Gram is demean_time=False.",
        }

        if name in BINARY_RDM_NAMES:
            rdm_airi = theoretical_rdm(name, fill="airi")
            model_fill = ReDisCA(demean_time=False).fit(averages, rdm_airi)
            fill_f = np.asarray(model_fill.filters_, dtype=np.float64)
            n_f = min(n_plot, fill_f.shape[0], filters.shape[0])
            aligned_fill = sign_align_vectors(filters[:n_f], fill_f[:n_f])
            extra_fill[name] = {
                "primary_fill": "binary_0_1",
                "extra_fill": "airi_0.1_1",
                "eigenvalues_airi_fill": np.asarray(model_fill.eigenvalues_[:N_REPORT]).tolist(),
                "sign_aligned_filter_pearson": [
                    float(np.corrcoef(filters[k], aligned_fill[k])[0, 1]) for k in range(n_f)
                ],
                "subspace_first_n": subspace_similarity(filters[:n_f], fill_f[:n_f]),
            }

        item_id = {
            "face": "fig13-meg-face",
            "tool": "fig14-meg-tool",
            "meaning": "fig15-meg-meaning",
            "facevstool": "fig17-meg-nonbinary-components",
        }[name]
        payload = {
            "path_label": "paper_faithful",
            "item_id": item_id,
            "rdm_name": name,
            "rdm_fill": fill,
            "window_ms": [float(time_ms[0]), float(time_ms[-1])],
            "n_samples": int(time_ms.size),
            "n_planars": 204,
            "bandpass": None,
            "pairs": "unique_unordered",
            "pair_matrix": "unscaled_gram_ReDisCA_demean_time_False",
            "estimator": "redisca.ReDisCA",
            "patterns": "Haufe (library); invert-W does not apply at MEG rank < 204 (D9)",
            "rank": n_rank,
            "n_plot": n_plot,
            "eigenvalues": evals[:N_REPORT].tolist(),
            "eigenvalues_paper_eq7_sum_scale": (evals[:N_REPORT] * len(pairs)).tolist(),
            "d4_note": "Library aggregates with a mean (D4). Sum scale is n_pairs * mean lambda; filter rays unchanged.",
            "p_component_maxabs": p_max[:N_REPORT].tolist(),
            "p_component_matched_exploratory": np.asarray(perm["p_matched_exploratory"]).tolist(),
            "inference_component": {k: perm[k] for k in perm if k not in ("p_maxabs", "p_matched_exploratory")},
            "inference_time": {
                "name": fwer["inference"],
                "Nmc": fwer["Nmc"],
                "alpha": fwer["alpha"],
                "class1": list(class1),
                "class2": list(class2),
                "class_convention": "paper",
                "fwer_threshold": fwer["fwer_threshold"].tolist(),
            },
            "empirical_rdm_pearson": emp,
            "peaks": peaks,
            "onsets": onsets,
            "paper_qualitative_onsets": PAPER_QUALITATIVE_ONSETS.get(name),
            "filter_fingerprints": [array_fingerprint(filters[k]) for k in range(n_plot)],
            "pattern_fingerprints": [array_fingerprint(patterns[k]) for k in range(n_plot)],
            "n_significant_time_samples": [int(fwer["significant"][k].sum()) for k in range(n_plot)],
        }
        fname = {
            "face": "fig13_face.json",
            "tool": "fig14_tool.json",
            "meaning": "fig15_meaning.json",
            "facevstool": "fig17_nonbinary_components.json",
        }[name]
        _save_json(out_dir / fname, payload)
        if name == "facevstool":
            _save_json(
                out_dir / "fig16_nonbinary_rdm.json",
                {
                    "path_label": "paper_faithful",
                    "item_id": "fig16-meg-nonbinary-rdm",
                    "rdm": rdm.tolist(),
                    "source": "AIRI facevstool 0.1/0.5/1 (paper figure values not printed)",
                },
            )
        save_npz(
            arr_dir / f"{name}.npz",
            filters=filters[:n_plot],
            patterns=patterns[:n_plot],
            eigenvalues=evals,
            traces=traces[:, :n_plot],
            time_ms=time_ms,
            contrast=contrast,
            significant=fwer["significant"].astype(np.uint8),
            p_maxabs=p_max,
        )
        plot_component_panel(
            fig_dir / f"{item_id}_traces_patterns.png",
            path_label="paper_faithful",
            rdm_name=name,
            time_ms=time_ms,
            traces=traces,
            patterns=patterns,
            eigenvalues=evals,
            p_values=p_max,
            asterisk_hi=contrast > fwer["fwer_threshold"][:, np.newaxis],
            asterisk_lo=contrast < -fwer["fwer_threshold"][:, np.newaxis],
            n_plot=n_plot,
            extra_title="full epoch −500…+1000 ms; no AIRI bandpass; paper FWER asterisks",
        )
        plot_planar_rms_row(
            fig_dir / f"{item_id}_planar_rms.png",
            path_label="paper_faithful",
            rdm_name=name,
            patterns=patterns,
            n_plot=n_plot,
        )
        summary["rdms"][name] = {
            "item_id": item_id,
            "rank": n_rank,
            "eigenvalues_head": evals[:n_plot].tolist(),
            "p_maxabs_head": p_max[:n_plot].tolist(),
            "empirical_rdm_pearson_comp1": emp[0]["pearson_unique_triangle"] if emp else None,
            "contrast_peak_ms_comp1": peaks[0]["contrast_peak_ms"] if peaks else None,
            "onset_comp1": onsets[0] if onsets else None,
        }
        print(
            f"[paper_faithful] {name}: rank={n_rank} λ={evals[:n_plot]} "
            f"p_maxabs={p_max[:n_plot]} peak1={peaks[0]['contrast_peak_ms']} ms",
            flush=True,
        )

    provenance = capture_run(
        track="meg",
        path_label="paper_faithful",
        seed_record=rec_perm,
        extra={
            "prepare": prepare_provenance(bundle),
            "paper_B": args.paper_B,
            "paper_Nmc": args.paper_nmc,
            "fwer_rng": rec_fwer.to_dict(),
            "rdms": names,
        },
    )
    _save_json(out_dir / "provenance.json", provenance)
    _save_json(out_dir / "demean_time_extra.json", extra_demean)
    _save_json(out_dir / "rdm_fill_extra.json", extra_fill)
    _save_json(out_dir / "summary.json", summary)
    return summary


def run_airi(args: argparse.Namespace, bundle: MegBundle | None = None) -> dict[str, Any]:
    """AIRI-executable path: directed pairs, MATLAB cov, 99–999 ms, butter(3)."""
    out_dir = RESULTS_ROOT / "airi_executable"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    arr_dir = out_dir / "arrays"
    if bundle is None:
        print("[airi_executable] loading MEG_AD_run1.mat", flush=True)
        bundle = load_meg_bundle()
    print(
        "[airi_executable] butter(3) 0.25–20 Hz scipy filtfilt on 204×880 trials "
        "(AIRI MATLAB filtfilt is not bit-exact; D8)",
        flush=True,
    )
    filtered = bandpass_airi(bundle.planars)
    averages_full = condition_averages(filtered, bundle.indices)
    averages = averages_full[:, :, AIRI_SLICE]
    time_ms = airi_time_ms(bundle.time_ms)
    vr = airi_channel_time_std(filtered)
    vr_safe = np.where(vr > 0.0, vr, 1.0)
    std_planars = filtered / vr_safe[:, :, np.newaxis]
    rng, rec = numpy_generator(args.seed + 3)
    rng_diag, rec_diag = numpy_generator(args.seed + 10)
    _write_fig12(out_dir, "airi_executable")
    names = list(args.rdms) if args.rdms else list(AIRI_RDM_NAMES)
    summary: dict[str, Any] = {"path_label": "airi_executable", "rdms": {}}

    for name in names:
        rdm = theoretical_rdm(name, fill="airi")
        print(
            f"[airi_executable] source_faithful directed matlab_cov {name} "
            f"B={args.airi_B} X={averages.shape} (no redisca.fit)",
            flush=True,
        )
        result = fit_condition_averages(
            averages,
            rdm,
            pair_mode="airi_directed",
            matrix_mode="matlab_cov",
            n_bootstrapping_iterations=args.airi_B,
            rng=rng,
            inference="spoc_random_phase",
        )
        n_rank = int(result.filters.shape[1])
        n_plot = min(args.n_plot_airi, n_rank)
        filters_cols = np.asarray(result.filters, dtype=np.float64)
        patterns_cols = np.asarray(result.patterns, dtype=np.float64)
        evals = np.asarray(result.eigenvalues, dtype=np.float64)
        p_spoc = (
            np.asarray(result.p_values, dtype=np.float64)
            if result.p_values is not None
            else np.full(n_rank, np.nan)
        )
        filters = filters_cols.T
        patterns = patterns_cols.T
        # GEP uses 99–999 ms. Sign-align on that window, then project the same W.
        traces_fit = np.einsum("kc,nct->nkt", filters, averages)
        class1, class2 = class_labels(name, convention="airi")
        traces_fit, filters, patterns = _sign_align_display(
            traces_fit, filters, patterns, class1=class1, time_ms=time_ms
        )
        traces_full = np.einsum("kc,nct->nkt", filters, averages_full)
        filters_cols = filters.T
        idx_map = {cname: bundle.indices[cname] for cname in CONDITION_NAMES}
        class1_idx = np.concatenate([idx_map[CONDITION_NAMES[i]] for i in class1])
        class2_idx = np.concatenate([idx_map[CONDITION_NAMES[i]] for i in class2])
        print(f"[airi_executable] half-split Nmc={args.airi_nmc} ({name}) — AIRI test, not paper", flush=True)
        half = airi_halfsplit_timecourse(
            std_planars,
            class1_idx,
            class2_idx,
            filters_cols,
            nmc=args.airi_nmc,
            rng=rng,
            n_components=n_plot,
        )
        emp = _empirical_for_components(traces_fit, rdm, demean_time=True, n_plot=n_plot)
        contrast_fit = traces_fit[list(class2)].mean(axis=0) - traces_fit[list(class1)].mean(axis=0)
        contrast_full = traces_full[list(class2)].mean(axis=0) - traces_full[list(class1)].mean(axis=0)
        peaks = _component_peaks(time_ms, traces_fit, contrast_fit, class1, class2, n_plot)
        ast_hi = half["asterisk_positive"]
        ast_lo = half["asterisk_negative"]
        ast_hi_fit = ast_hi[:, AIRI_SLICE]
        ast_lo_fit = ast_lo[:, AIRI_SLICE]
        onsets = [
            _onset_ms(time_ms, ast_hi_fit[k] | ast_lo_fit[k])
            for k in range(n_plot)
        ]
        item_id = {
            "face": "fig13-meg-face",
            "tool": "fig14-meg-tool",
            "meaning": "fig15-meg-meaning",
            "facevstool": "airi-executable-meg-facevstool",
        }[name]
        payload = {
            "path_label": "airi_executable",
            "item_id": item_id,
            "rdm_name": name,
            "rdm_fill": "airi_numeric",
            "window_ms": [float(time_ms[0]), float(time_ms[-1])],
            "n_samples": int(time_ms.size),
            "n_planars": 204,
            "bandpass": {
                "butter_order": 3,
                "low_hz": 0.25,
                "high_hz": 20.0,
                "filtfilt": "scipy.signal.filtfilt",
                "matlab_parity": False,
            },
            "pairs": "airi_directed_i_neq_j",
            "n_pairs": 30,
            "pair_matrix": "matlab_cov",
            "estimator": "common.source_faithful.fit_condition_averages (does not import redisca)",
            "patterns": "Haufe / SPoC",
            "rank": n_rank,
            "n_plot": n_plot,
            "eigenvalues": evals[:N_REPORT].tolist(),
            "d1_d4_note": (
                "Directed pairs (D1) scale z via sample SD N-1; aggregation is SPoC mean (D4). "
                "Do not compare raw lambda to paper_faithful without those labels."
            ),
            "p_component_spoc_random_phase": p_spoc[:N_REPORT].tolist(),
            "inference_component": {
                "name": "spoc_random_phase",
                "B": int(args.airi_B),
                "p_formula": "count(max|lambda_s| >= |lambda|)/B ; p=0 possible",
                "requested_B": AIRI_N_BOOTSTRAP,
                "used_B": int(args.airi_B),
                "reduced_B": bool(args.airi_B < AIRI_N_BOOTSTRAP),
            },
            "inference_time": {
                "name": half["inference"],
                "Nmc": half["Nmc"],
                "requested_Nmc": AIRI_N_MC_TIMECourse,
                "class1": list(class1),
                "class2": list(class2),
                "class_convention": "airi",
                "matlab_indexing_hazard": half["matlab_indexing_hazard"],
                "not_the_paper_test": True,
            },
            "empirical_rdm_pearson": emp,
            "peaks": peaks,
            "onsets_airi_asterisks": onsets,
            "paper_qualitative_onsets": PAPER_QUALITATIVE_ONSETS.get(name),
            "filter_fingerprints": [array_fingerprint(filters[k]) for k in range(n_plot)],
            "pattern_fingerprints": [array_fingerprint(patterns[k]) for k in range(n_plot)],
            "n_airi_asterisk_samples_fit_window": [
                int((ast_hi_fit[k] | ast_lo_fit[k]).sum()) for k in range(n_plot)
            ],
            "airi_matlab_plots_full_mx_with_wrong_linspace": (
                "AIRI plots W'*mx on all 1501 samples against linspace(-536,964). "
                "This path plots the fit window with the true 99–999 ms axis."
            ),
            "peaks_full_epoch_projection": _component_peaks(
                bundle.time_ms, traces_full, contrast_full, class1, class2, n_plot
            ),
        }
        fname = {
            "face": "fig13_face.json",
            "tool": "fig14_tool.json",
            "meaning": "fig15_meaning.json",
            "facevstool": "airi_executable_meg_facevstool.json",
        }[name]
        _save_json(out_dir / fname, payload)
        if name == "facevstool":
            _save_json(
                out_dir / "fig16_17_facevstool.json",
                {
                    "path_label": "airi_executable",
                    "item_id": "fig17-meg-nonbinary-components",
                    "note": "Closest unmodified AIRI run to Fig. 16/17; still not paper (D1,D2,D5,D6,D8).",
                    "metrics_file": fname,
                    "eigenvalues": evals[:N_REPORT].tolist(),
                    "p_spoc": p_spoc[:N_REPORT].tolist(),
                },
            )
        save_npz(
            arr_dir / f"{name}.npz",
            filters=filters[:n_plot],
            patterns=patterns[:n_plot],
            eigenvalues=evals,
            traces=traces_fit[:, :n_plot],
            time_ms=time_ms,
            p_spoc=p_spoc,
            asterisk_hi=ast_hi_fit.astype(np.uint8),
            asterisk_lo=ast_lo_fit.astype(np.uint8),
        )
        plot_component_panel(
            fig_dir / f"{item_id}_traces_patterns.png",
            path_label="airi_executable",
            rdm_name=name,
            time_ms=time_ms,
            traces=traces_fit,
            patterns=patterns,
            eigenvalues=evals,
            p_values=p_spoc,
            asterisk_hi=ast_hi_fit,
            asterisk_lo=ast_lo_fit,
            n_plot=n_plot,
            extra_title="99–999 ms; 0.25–20 Hz; SPoC random-phase p; AIRI half-split asterisks",
        )
        plot_planar_rms_row(
            fig_dir / f"{item_id}_planar_rms.png",
            path_label="airi_executable",
            rdm_name=name,
            patterns=patterns,
            n_plot=n_plot,
        )
        summary["rdms"][name] = {
            "item_id": item_id,
            "rank": n_rank,
            "eigenvalues_head": evals[:n_plot].tolist(),
            "p_spoc_head": p_spoc[:n_plot].tolist(),
            "empirical_rdm_pearson_comp1": emp[0]["pearson_unique_triangle"] if emp else None,
            "contrast_peak_ms_comp1": peaks[0]["contrast_peak_ms"] if peaks else None,
        }
        print(
            f"[airi_executable] {name}: rank={n_rank} λ={evals[:n_plot]} "
            f"p_spoc={p_spoc[:n_plot]} peak1={peaks[0]['contrast_peak_ms']} ms",
            flush=True,
        )

    if "facevstool" in names:
        print(
            f"[airi_executable] pair-order diagnostic n_shuffles={args.pair_order_shuffles} "
            f"B={args.pair_order_B} (not a replacement test)",
            flush=True,
        )
        diag = pair_order_sensitivity(
            averages,
            theoretical_rdm("facevstool", fill="airi"),
            n_shuffles=args.pair_order_shuffles,
            n_bootstrapping_iterations=args.pair_order_B,
            rng=rng_diag,
            n_report=N_PLOT_AIRI,
        )
        _save_json(out_dir / "pair_order_diagnostic.json", diag)

    provenance = capture_run(
        track="meg",
        path_label="airi_executable",
        seed_record=rec,
        extra={
            "prepare": prepare_provenance(bundle),
            "airi_B": args.airi_B,
            "airi_Nmc": args.airi_nmc,
            "pair_order_rng": rec_diag.to_dict(),
            "rdms": names,
            "does_not_call_redisca_fit": True,
        },
    )
    _save_json(out_dir / "provenance.json", provenance)
    _save_json(out_dir / "summary.json", summary)
    return summary


def _overlap_traces(
    time_a: NDArray[np.float64],
    traces_a: NDArray[np.float64],
    time_b: NDArray[np.float64],
    traces_b: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Restrict both traces to the AIRI window using nearest true samples."""
    t0, t1 = float(time_b[0]), float(time_b[-1])
    mask = (time_a >= t0 - 0.5) & (time_a <= t1 + 0.5)
    t_overlap = time_a[mask]
    a = traces_a[..., mask]
    # AIRI time is a subset of the 1 kHz grid.
    idx = np.searchsorted(time_b, t_overlap)
    idx = np.clip(idx, 0, time_b.size - 1)
    b = traces_b[..., idx]
    return t_overlap, a, b


def run_compare() -> dict[str, Any]:
    """Numeric comparison of the two labeled paths. Not a mixed figure."""
    out_dir = RESULTS_ROOT / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison: dict[str, Any] = {
        "note": (
            "Paths differ by D1 (pairs), D2 (Gram vs cov demean), D4 (lambda scale), "
            "D5 (inference), D5b (time test), D6 (window), D7 (default RDM), D8 (bandpass). "
            "Eigenvector signs are aligned before correlation. Do not compare asterisks."
        ),
        "rdms": {},
    }
    for name in PRIMARY_PAPER_RDMS:
        paper_npz = RESULTS_ROOT / "paper_faithful" / "arrays" / f"{name}.npz"
        airi_npz = RESULTS_ROOT / "airi_executable" / "arrays" / f"{name}.npz"
        if not paper_npz.exists() or not airi_npz.exists():
            comparison["rdms"][name] = {"status": "missing_arrays"}
            continue
        paper = np.load(paper_npz)
        airi = np.load(airi_npz)
        n = min(paper["filters"].shape[0], airi["filters"].shape[0], N_PLOT_PAPER)
        pf = paper["filters"][:n]
        af = sign_align_vectors(pf, airi["filters"][:n])
        pp = paper["patterns"][:n]
        ap = sign_align_vectors(pp, airi["patterns"][:n])
        filter_corr = [float(np.corrcoef(pf[k], af[k])[0, 1]) for k in range(n)]
        pattern_corr = [float(np.corrcoef(pp[k], ap[k])[0, 1]) for k in range(n)]
        t_ov, tr_p, tr_a = _overlap_traces(
            paper["time_ms"], paper["traces"][:, :n], airi["time_ms"], airi["traces"][:, :n]
        )
        # Align trace signs to paper class-1 mean on overlap (already display-aligned separately).
        tc_corr = []
        peak_ms = []
        for k in range(n):
            # mean over conditions of |corr|; use condition 0 as sign already aligned in each path
            corrs = [
                pearson(tr_p[c, k], sign_align_vectors(tr_p[c, k], tr_a[c, k]))
                for c in range(tr_p.shape[0])
            ]
            tc_corr.append(float(np.nanmean(corrs)))
            p_peak = peak_latency_and_amplitude(t_ov / 1000.0, tr_p[:, k].mean(0))
            a_peak = peak_latency_and_amplitude(t_ov / 1000.0, tr_a[:, k].mean(0))
            peak_ms.append(
                {
                    "paper_mean_trace_peak_ms": float(p_peak["peak_time_s"] * 1000.0),
                    "airi_mean_trace_peak_ms": float(a_peak["peak_time_s"] * 1000.0),
                }
            )
        lam_p = np.asarray(paper["eigenvalues"][:n], dtype=np.float64)
        lam_a = np.asarray(airi["eigenvalues"][:n], dtype=np.float64)
        n_unique, n_directed = 15, 30
        std_ratio = float(np.sqrt((2.0 * (n_unique - 1)) / (n_directed - 1)))
        comparison["rdms"][name] = {
            "n_components_compared": n,
            "eigenvalues_paper_mean_aggregation": lam_p.tolist(),
            "eigenvalues_airi_mean_aggregation": lam_a.tolist(),
            "d1_std_ratio_directed_vs_unique": std_ratio,
            "d4_note": "Both executable estimators use mean aggregation; paper Eq.7 sum would scale lambda only.",
            "d2_note": "Paper path is uncentered Gram; AIRI path demeans (MATLAB cov). 1/(T-1) cancels in the GEP.",
            "sign_aligned_filter_pearson": filter_corr,
            "sign_aligned_pattern_pearson": pattern_corr,
            "subspace_filters": subspace_similarity(pf, airi["filters"][:n]),
            "subspace_patterns": subspace_similarity(pp, airi["patterns"][:n]),
            "timecourse_mean_pearson_on_airi_window": tc_corr,
            "peak_latency_on_overlap": peak_ms,
            "overlap_window_ms": [float(t_ov[0]), float(t_ov[-1])] if t_ov.size else None,
        }
    _save_json(out_dir / "paper_vs_airi.json", comparison)
    return comparison


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["paper", "airi", "all"])
    parser.add_argument("--paper-B", type=int, default=500, help="Paper condition-label permutations (unspecified in paper; documented here).")
    parser.add_argument("--airi-B", type=int, default=1000, help="SPoC random-phase iterations (AIRI default 1000).")
    parser.add_argument("--paper-nmc", type=int, default=200, help="Paper time-series FWER Monte Carlo (unspecified in paper).")
    parser.add_argument("--airi-nmc", type=int, default=100, help="AIRI half-split Nmc (MATLAB 100).")
    parser.add_argument("--pair-order-shuffles", type=int, default=5)
    parser.add_argument("--pair-order-B", type=int, default=200)
    parser.add_argument("--n-plot-paper", type=int, default=N_PLOT_PAPER)
    parser.add_argument("--n-plot-airi", type=int, default=N_PLOT_AIRI)
    parser.add_argument("--seed", type=int, default=20240904)
    parser.add_argument("--rdms", nargs="*", default=None, help="Subset of RDM names.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced B/Nmc for smoke tests. Do not treat as the reported reproduction.",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.paper_B = min(args.paper_B, 32)
        args.airi_B = min(args.airi_B, 32)
        args.paper_nmc = min(args.paper_nmc, 20)
        args.airi_nmc = min(args.airi_nmc, 20)
        args.pair_order_shuffles = min(args.pair_order_shuffles, 1)
        args.pair_order_B = min(args.pair_order_B, 32)
        print("[meg] --quick: reduced Monte Carlo; not the reported run", flush=True)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    bundle = None
    need_data = args.command in {"paper", "airi", "all"}
    if need_data:
        print("[meg] loading cached MEG (planars only; not the .dat)", flush=True)
        bundle = load_meg_bundle()
        print(
            f"[meg] planars {bundle.planars.shape}; used {bundle.n_used_trials}/"
            f"{bundle.n_file_trials}; t=[{bundle.time_ms[0]}, {bundle.time_ms[-1]}] ms",
            flush=True,
        )
    paper_summary = airi_summary = comparison = None
    if args.command in {"paper", "all"}:
        paper_summary = run_paper(args, bundle=bundle)
    if args.command in {"airi", "all"}:
        airi_summary = run_airi(args, bundle=bundle)
    if args.command == "all" or (
        (RESULTS_ROOT / "paper_faithful" / "arrays").exists()
        and (RESULTS_ROOT / "airi_executable" / "arrays").exists()
    ):
        comparison = run_compare()
    top = {
        "command": args.command,
        "paper_B": args.paper_B,
        "airi_B": args.airi_B,
        "paper_Nmc": args.paper_nmc,
        "airi_Nmc": args.airi_nmc,
        "quick": bool(args.quick),
        "paper_faithful": paper_summary,
        "airi_executable": airi_summary,
        "comparison_written": comparison is not None,
    }
    _save_json(RESULTS_ROOT / "summary.json", top)
    print("[meg] wrote", RESULTS_ROOT / "summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
