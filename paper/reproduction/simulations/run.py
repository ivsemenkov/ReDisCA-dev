"""Run Ossadtchi et al. 2024 simulation reconstructions (Figs 3–6).

Canonical estimator: ``from redisca import ReDisCA`` with unique pairs and
``demean_time=False`` (printed Gram). ``demean_time=True`` is a labeled extra
on Fig. 4.

All numeric outputs are ``approximate`` because the paper does not name the
forward model, I_c, f_s, or Υ_d. The public AD Gain is a documented hypothesis.
Never fsaverage. Do not tune to the ~85% hit-rate claim.

Usage (from the repository root)::

    python paper/reproduction/simulations/run.py
    python paper/reproduction/simulations/run.py --quick
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_REPRO = REPO_ROOT / "paper" / "reproduction"
for path in (REPO_ROOT / "src", PAPER_REPRO, Path(__file__).resolve().parent):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.hashing import sha256_file, write_json  # noqa: E402
from common.paths import SOURCE_MODEL_DIR  # noqa: E402
from common.provenance import capture_environment  # noqa: E402

from redisca import ReDisCA  # noqa: E402

from config import (  # noqa: E402
    ASSUMED_FIG6_SNR,
    FORWARD_STATUS,
    PAPER_FIG6_C,
    PAPER_MULTI_SOURCE_C_METHODS,
    PAPER_MULTI_SOURCE_P,
    PAPER_SINGLE_SOURCE_C,
    PAPER_SNR_FIG4_HIGH,
    PAPER_SNR_FIG4_LOW,
    PAPER_SNR_FIG5_HIGH,
    PAPER_SNR_FIG5_LOW,
    SimulationConfig,
    assumed_value_table,
)
from forward_model import load_ad_forward  # noqa: E402
from generate import (  # noqa: E402
    simulate_multi_source,
    simulate_single_source,
    squared_euclidean_rdm,
    subset_conditions_multi,
)
from metrics_roc import (  # noqa: E402
    cosine_abs_scan,
    default_thresholds,
    downsample_roc,
    localization_error_m,
    pearson,
    roc_from_mc,
    sign_align_to_reference,
    sphere_mask,
    summarize_roc,
)
from rsa_baselines import (  # noqa: E402
    build_mne_kernel,
    four_rsa_scans,
    rsa_scans_for_targets,
    unique_pairs,
)

RESULTS_DIR = REPO_ROOT / "paper" / "results" / "simulations"
STATUS = "approximate"
STATUS_TAGS = [
    "approximate",
    "blocked by missing source asset",
    "source-model-dependent",
    "stochastic",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return value


def _write(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    write_json(path, _jsonable(payload))
    return path


def _common_header(config: SimulationConfig, forward_prov: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    header = {
        "status": STATUS,
        "status_tags": STATUS_TAGS,
        "forward_status": FORWARD_STATUS,
        "do_not_claim_exact_published_numbers": True,
        "tuned_to_85pct_hit_rate": False,
        "config": config.to_dict(),
        "assumed_values": assumed_value_table(),
        "forward_model": forward_prov,
        "environment": capture_environment(extra_packages=("h5py", "mne")),
        "canonical_estimator": {
            "import": "from redisca import ReDisCA",
            "demean_time": False,
            "pairs": "unique i<j",
            "patterns": "Haufe (library compute_patterns)",
            "n_components_used_for_localization": 1,
        },
    }
    if extra:
        header.update(extra)
    return header


def _fit_redisca(averages: np.ndarray, target: np.ndarray, *, demean_time: bool):
    try:
        return ReDisCA(n_components=1, demean_time=demean_time).fit(averages, target)
    except Exception:
        return None


def empirical_rdm(filter_row: np.ndarray, averages: np.ndarray) -> np.ndarray:
    filtered = np.einsum("n,cnt->ct", filter_row, averages)
    return squared_euclidean_rdm(filtered)


def run_fig3_exemplar(
    forward,
    config: SimulationConfig,
    rng: np.random.Generator,
    mixings: np.ndarray,
) -> dict[str, Any]:
    draw = simulate_multi_source(
        forward,
        rng,
        mixings,
        config=config,
        snr=PAPER_SNR_FIG5_LOW,
        n_conditions=PAPER_MULTI_SOURCE_C_METHODS,
    )
    return {
        "id": "fig03-sim-four-source-rdms",
        "figure": "Figure 3",
        "status": STATUS,
        "note": "Exemplar C=6 theoretical RDMs (D = D0 + assumed Upsilon_d). Not a fixed matrix to copy.",
        "C": PAPER_MULTI_SOURCE_C_METHODS,
        "P": PAPER_MULTI_SOURCE_P,
        "vertices": [int(v) for v in draw.vertices],
        "min_pairwise_distance_m": float(
            min(
                np.linalg.norm(
                    forward.vertices[int(draw.vertices[i])]
                    - forward.vertices[int(draw.vertices[j])]
                )
                for i, j in unique_pairs(PAPER_MULTI_SOURCE_P)
            )
        ),
        "D0": [draw.d0[p].tolist() for p in range(PAPER_MULTI_SOURCE_P)],
        "D_target": [draw.d_target[p].tolist() for p in range(PAPER_MULTI_SOURCE_P)],
        "source_fingerprints": [
            {
                "row_rms": [float(x) for x in np.sqrt(np.mean(draw.sources[p] ** 2, axis=1))],
            }
            for p in range(PAPER_MULTI_SOURCE_P)
        ],
    }


def run_single_source_roc(
    forward,
    config: SimulationConfig,
    *,
    mixing: np.ndarray,
    snr: float,
    seed: int,
    n_mc: int,
    with_rsa: bool,
    rsa_methods: tuple[str, ...],
    extra_demean: bool,
    label: str,
) -> dict[str, Any]:
    ss = np.random.SeedSequence(seed)
    child = ss.spawn(n_mc + 1)
    mne_kernel = build_mne_kernel(forward.gain, snr=config.mne_snr) if with_rsa else None
    methods = ["redisca_demean_false"]
    if extra_demean:
        methods.append("redisca_demean_true")
    if with_rsa:
        methods.extend(rsa_methods)

    scores: dict[str, list[np.ndarray]] = {name: [] for name in methods}
    inside_masks: list[np.ndarray] = []
    errors: dict[str, list[float]] = {name: [] for name in methods}
    failed = 0
    exemplars: list[dict[str, Any]] = []

    for mc in range(n_mc):
        rng = np.random.default_rng(child[mc])
        draw = simulate_single_source(forward, rng, mixing, config=config, snr=snr)
        model = _fit_redisca(draw.averages, draw.d_target, demean_time=False)
        if model is None:
            failed += 1
            continue
        scan_cs = cosine_abs_scan(model.patterns_[0], forward.gain)
        dist = forward.distances_from(draw.vertex)
        inside = sphere_mask(dist, config.r_max_m)
        inside_masks.append(inside)
        scores["redisca_demean_false"].append(scan_cs)
        errors["redisca_demean_false"].append(
            localization_error_m(scan_cs, draw.vertex, forward.vertices)["error_m"]
        )
        if extra_demean:
            model_d = _fit_redisca(draw.averages, draw.d_target, demean_time=True)
            if model_d is not None:
                scan_d = cosine_abs_scan(model_d.patterns_[0], forward.gain)
                scores["redisca_demean_true"].append(scan_d)
                errors["redisca_demean_true"].append(
                    localization_error_m(scan_d, draw.vertex, forward.vertices)["error_m"]
                )
            else:
                scores["redisca_demean_true"].append(np.zeros(forward.n_vertices))
                errors["redisca_demean_true"].append(float("nan"))
        if with_rsa:
            rsa = four_rsa_scans(
                forward.gain,
                draw.averages,
                draw.trials,
                draw.d_target,
                config=config,
                mne_kernel=mne_kernel,
                methods=rsa_methods,
            )
            for name, scan in rsa.items():
                scores[name].append(scan)
                errors[name].append(
                    localization_error_m(scan, draw.vertex, forward.vertices)["error_m"]
                )
        if mc == 0:
            ch_mean_noisy = draw.averages.mean(axis=1)
            ch_mean_clean = draw.noiseless_average.mean(axis=1)
            exemplars.append(
                {
                    "vertex": int(draw.vertex),
                    "D_target": draw.d_target.tolist(),
                    "D0": draw.d0.tolist(),
                    "channel_mean_noisy": ch_mean_noisy.tolist(),
                    "channel_mean_clean": ch_mean_clean.tolist(),
                    "gamma_mean": float(np.mean(draw.gammas)),
                    "n_inside_sphere": int(inside.sum()),
                }
            )
        if (mc + 1) % 10 == 0 or mc == 0:
            print(f"  [{label}] MC {mc + 1}/{n_mc} failed={failed}", flush=True)

    n_kept = len(inside_masks)
    inside_arr = np.stack(inside_masks, axis=0) if n_kept else np.zeros((0, forward.n_vertices), dtype=bool)
    out_methods: dict[str, Any] = {}
    for name, series in scores.items():
        if not series:
            continue
        stacked = np.stack(series[:n_kept], axis=0)
        kind = "cosine_abs" if name.startswith("redisca") else "pearson"
        roc = roc_from_mc(stacked, inside_arr, default_thresholds(kind))
        err = np.asarray(errors[name][:n_kept], dtype=np.float64)
        out_methods[name] = {
            "n_mc_kept": int(n_kept),
            "roc": summarize_roc(roc),
            "roc_curve": downsample_roc(roc),
            "median_error_m": float(np.nanmedian(err)),
            "mean_error_m": float(np.nanmean(err)),
            "frac_error_lt_1cm": float(np.mean(err < 0.01)),
            "frac_error_lt_2cm": float(np.mean(err < 0.02)),
        }
    return {
        "label": label,
        "snr": float(snr),
        "seed": int(seed),
        "n_mc_requested": int(n_mc),
        "n_mc_kept": int(n_kept),
        "n_mc_failed_fit": int(failed),
        "rsa_included": bool(with_rsa),
        "rsa_methods": list(rsa_methods) if with_rsa else [],
        "methods": out_methods,
        "exemplars": exemplars,
        "seed_sequence": str(ss),
    }


def _redisca_metrics_for_source(
    averages: np.ndarray,
    target: np.ndarray,
    g_true: np.ndarray,
    vertex: int,
    vertices: np.ndarray,
    gain: np.ndarray,
) -> dict[str, Any] | None:
    model = _fit_redisca(averages, target, demean_time=False)
    if model is None:
        return None
    pattern = model.patterns_[0]
    weights = model.filters_[0]
    pattern_a = sign_align_to_reference(pattern, g_true)
    # flip weights with the same sign as the pattern vs g
    if float(np.dot(pattern, g_true)) < 0.0:
        weights_a = -weights
    else:
        weights_a = weights
    scan = cosine_abs_scan(pattern, gain)
    loc = localization_error_m(scan, vertex, vertices)
    d_hat = empirical_rdm(weights, averages)
    return {
        "error_m": loc["error_m"],
        "corr_pattern_g": pearson(pattern_a, g_true),
        "corr_weight_g": pearson(weights_a, g_true),
        "corr_rdm": pearson(
            d_hat[np.triu_indices(d_hat.shape[0], k=1)],
            np.asarray(target)[np.triu_indices(target.shape[0], k=1)],
        ),
        "est_vertex": loc["est_vertex"],
        "scan": scan,
    }


def run_four_source(
    forward,
    config: SimulationConfig,
    *,
    mixings: np.ndarray,
    snr: float,
    seed: int,
    n_mc: int,
    with_rsa: bool,
    rsa_methods: tuple[str, ...],
    eval_C: tuple[int, ...],
    label: str,
) -> dict[str, Any]:
    ss = np.random.SeedSequence(seed)
    child = ss.spawn(n_mc)
    mne_kernel = build_mne_kernel(forward.gain, snr=config.mne_snr) if with_rsa else None
    per_c: dict[int, dict[str, Any]] = {
        c: {
            "redisca_errors_m": [],
            "redisca_corr_a": [],
            "redisca_corr_w": [],
            "redisca_corr_rdm": [],
            "rsa_errors": {m: [] for m in rsa_methods} if with_rsa else {},
            "failed": 0,
        }
        for c in eval_C
    }

    for mc in range(n_mc):
        rng = np.random.default_rng(child[mc])
        draw = simulate_multi_source(
            forward,
            rng,
            mixings,
            config=config,
            snr=snr,
            n_conditions=PAPER_MULTI_SOURCE_C_METHODS,
        )
        for c in eval_C:
            sliced = subset_conditions_multi(draw, c)
            bucket = per_c[c]
            for p in range(PAPER_MULTI_SOURCE_P):
                metrics = _redisca_metrics_for_source(
                    sliced["averages"],
                    sliced["d_target"][p],
                    sliced["g_true"][p],
                    int(sliced["vertices"][p]),
                    forward.vertices,
                    forward.gain,
                )
                if metrics is None:
                    bucket["failed"] += 1
                    continue
                bucket["redisca_errors_m"].append(metrics["error_m"])
                bucket["redisca_corr_a"].append(metrics["corr_pattern_g"])
                bucket["redisca_corr_w"].append(metrics["corr_weight_g"])
                bucket["redisca_corr_rdm"].append(metrics["corr_rdm"])
            if with_rsa:
                rsa_per_source = rsa_scans_for_targets(
                    forward.gain,
                    sliced["averages"],
                    sliced["trials"],
                    sliced["d_target"],
                    config=config,
                    mne_kernel=mne_kernel,
                    methods=rsa_methods,
                )
                for p, rsa in enumerate(rsa_per_source):
                    for name, scan in rsa.items():
                        loc = localization_error_m(
                            scan, int(sliced["vertices"][p]), forward.vertices
                        )
                        bucket["rsa_errors"][name].append(loc["error_m"])
        if (mc + 1) % 10 == 0 or mc == 0:
            print(f"  [{label}] MC {mc + 1}/{n_mc}", flush=True)

    summary: dict[str, Any] = {}
    for c, bucket in per_c.items():
        errors = np.asarray(bucket["redisca_errors_m"], dtype=np.float64)
        # median over the 4 sources within each MC, then mean of those medians
        if errors.size and errors.size % PAPER_MULTI_SOURCE_P == 0:
            by_mc = errors.reshape(-1, PAPER_MULTI_SOURCE_P)
            median_per_mc = np.median(by_mc, axis=1)
            mean_median = float(np.mean(median_per_mc))
        else:
            median_per_mc = np.asarray([], dtype=np.float64)
            mean_median = float("nan")
        rsa_sum: dict[str, Any] = {}
        for name, vals in bucket["rsa_errors"].items():
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size and arr.size % PAPER_MULTI_SOURCE_P == 0:
                med = np.median(arr.reshape(-1, PAPER_MULTI_SOURCE_P), axis=1)
                rsa_mean_median = float(np.mean(med))
            else:
                rsa_mean_median = float("nan")
                med = arr
            rsa_sum[name] = {
                "mean_median_error_m": rsa_mean_median,
                "median_error_m": float(np.nanmedian(arr)) if arr.size else None,
                "frac_lt_1cm": float(np.mean(arr < 0.01)) if arr.size else None,
                "frac_lt_2cm": float(np.mean(arr < 0.02)) if arr.size else None,
                "n": int(arr.size),
            }
        summary[str(c)] = {
            "C": int(c),
            "n_source_localizations": int(errors.size),
            "n_failed_fits": int(bucket["failed"]),
            "redisca": {
                "mean_median_error_m": mean_median,
                "mean_median_error_cm": None if mean_median != mean_median else 100.0 * mean_median,
                "median_error_m": float(np.nanmedian(errors)) if errors.size else None,
                "frac_lt_1cm": float(np.mean(errors < 0.01)) if errors.size else None,
                "frac_lt_2cm": float(np.mean(errors < 0.02)) if errors.size else None,
                "corr_pattern_g_median": float(np.nanmedian(bucket["redisca_corr_a"])) if bucket["redisca_corr_a"] else None,
                "corr_weight_g_median": float(np.nanmedian(bucket["redisca_corr_w"])) if bucket["redisca_corr_w"] else None,
                "corr_rdm_median": float(np.nanmedian(bucket["redisca_corr_rdm"])) if bucket["redisca_corr_rdm"] else None,
                "errors_m_head": [float(x) for x in errors[:40]],
                "corr_pattern_g_head": [float(x) for x in bucket["redisca_corr_a"][:40]],
                "corr_weight_g_head": [float(x) for x in bucket["redisca_corr_w"][:40]],
                "corr_rdm_head": [float(x) for x in bucket["redisca_corr_rdm"][:40]],
                "median_per_mc_m": [float(x) for x in median_per_mc],
            },
            "rsa": rsa_sum,
        }
    return {
        "label": label,
        "snr": float(snr),
        "seed": int(seed),
        "n_mc": int(n_mc),
        "eval_C": list(eval_C),
        "rsa_included": bool(with_rsa),
        "by_C": summary,
        "seed_sequence": str(ss),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReDisCA paper simulation reconstruction")
    parser.add_argument("--quick", action="store_true", help="Tiny MC for smoke tests")
    parser.add_argument("--n-mc", type=int, default=None)
    parser.add_argument("--i-c", type=int, default=None)
    parser.add_argument("--n-noise", type=int, default=None)
    parser.add_argument("--skip-rsa", action="store_true", help="ReDisCA only (RSA marked skipped)")
    parser.add_argument(
        "--rsa-n-mc",
        type=int,
        default=None,
        help="If set, RSA uses this many MC (ReDisCA still uses --n-mc). Status stays approximate.",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SimulationConfig:
    n_mc = 100
    i_c = 40
    n_noise = 1000
    if args.quick:
        n_mc = 3
        i_c = 8
        n_noise = 80
    if args.n_mc is not None:
        n_mc = args.n_mc
    if args.i_c is not None:
        i_c = args.i_c
    if args.n_noise is not None:
        n_noise = args.n_noise
    seed = args.seed if args.seed is not None else SimulationConfig().master_seed
    rsa_n = args.rsa_n_mc
    return SimulationConfig(
        n_mc=n_mc,
        i_c=i_c,
        n_noise_sources=n_noise,
        master_seed=seed,
        rsa_n_mc=rsa_n,
        extra_fields={"quick": bool(args.quick), "skip_rsa": bool(args.skip_rsa)},
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    print("Loading AD forward model (documented hypothesis, not paper-named)...", flush=True)
    forward = load_ad_forward(SOURCE_MODEL_DIR, sha256_file=sha256_file)
    print(
        f"  Gain {forward.gain.shape} channels={forward.n_channels} "
        f"vtx={forward.n_vertices} {forward.surface_file}",
        flush=True,
    )
    forward_prov = forward.provenance()
    rng_root = np.random.default_rng(config.master_seed)
    # Fixed mixings across MC (paper §2.4.1)
    mixing_single = rng_root.standard_normal((PAPER_SINGLE_SOURCE_C, PAPER_SINGLE_SOURCE_C))
    mixings_multi = rng_root.standard_normal(
        (PAPER_MULTI_SOURCE_P, PAPER_MULTI_SOURCE_C_METHODS, PAPER_MULTI_SOURCE_C_METHODS)
    )
    skip_rsa = bool(args.skip_rsa)
    rsa_n_mc = config.rsa_n_mc if config.rsa_n_mc is not None else config.n_mc
    # If RSA is too heavy, rsa_n_mc can be smaller; ReDisCA always uses n_mc.
    rsa_full = (not skip_rsa) and (rsa_n_mc >= config.n_mc)
    fig4_rsa_methods = ("mne_av", "mne_st", "bf_av", "bf_st")
    fig5_rsa_methods = ("mne_st", "bf_st")

    header = _common_header(
        config,
        forward_prov,
        extra={
            "mixing_single_seeded_from": config.master_seed,
            "rsa_n_mc": None if skip_rsa else rsa_n_mc,
            "rsa_reduction": None
            if rsa_full or skip_rsa
            else f"RSA MC reduced to {rsa_n_mc} vs ReDisCA {config.n_mc}",
        },
    )

    print("Fig 3 exemplar...", flush=True)
    fig3_rng = np.random.default_rng(np.random.SeedSequence(config.master_seed, spawn_key=(3,)))
    fig3 = run_fig3_exemplar(forward, config, fig3_rng, mixings_multi)
    fig3_path = _write("fig03_rdms.json", {**header, **fig3})
    print(f"  wrote {fig3_path}", flush=True)

    print("Fig 4 single-source ROC...", flush=True)
    fig4_runs = []
    for snr, tag in ((PAPER_SNR_FIG4_HIGH, "snr_0.2_preprint"), (PAPER_SNR_FIG4_LOW, "snr_0.1_published")):
        print(f"  ReDisCA+RSA n_mc={config.n_mc} snr={snr}", flush=True)
        fig4_runs.append(
            run_single_source_roc(
                forward,
                config,
                mixing=mixing_single,
                snr=snr,
                seed=config.master_seed + int(round(1000 * snr)),
                n_mc=config.n_mc,
                with_rsa=not skip_rsa,
                rsa_methods=fig4_rsa_methods,
                extra_demean=True,
                label=f"fig4-{tag}",
            )
        )
    print("  secondary seed ReDisCA-only SNR=0.1...", flush=True)
    secondary = run_single_source_roc(
        forward,
        config,
        mixing=mixing_single,
        snr=PAPER_SNR_FIG4_LOW,
        seed=config.secondary_seed,
        n_mc=min(config.n_mc_secondary, config.n_mc),
        with_rsa=False,
        rsa_methods=(),
        extra_demean=False,
        label="fig4-secondary-snr0.1",
    )
    fig4 = {
        **header,
        "id": "fig04-single-source-roc",
        "figure": "Figure 4",
        "C": PAPER_SINGLE_SOURCE_C,
        "primary_runs": fig4_runs,
        "secondary_seed_run": secondary,
        "note": (
            "High SNR 0.2 is preprint overlay; low SNR 0.1 is published body. "
            "demean_time=True is a labeled extra. Numbers are approximate."
        ),
    }
    fig4_path = _write("fig04_roc.json", fig4)
    print(f"  wrote {fig4_path}", flush=True)

    traces = {
        **header,
        "id": "fig04-single-source-traces",
        "figure": "Figure 4b,d",
        "exemplars_by_snr": [
            {"snr": run["snr"], "exemplars": run["exemplars"]} for run in fig4_runs
        ],
    }
    traces_path = _write("fig04_traces.json", traces)
    print(f"  wrote {traces_path}", flush=True)

    print("Fig 5 four-source MC (C=5 and C=6 from C=6 data)...", flush=True)
    fig5_runs = []
    for snr, tag in ((PAPER_SNR_FIG5_HIGH, "snr_0.4_preprint"), (PAPER_SNR_FIG5_LOW, "snr_0.2_body")):
        print(f"  four-source n_mc={config.n_mc} snr={snr}", flush=True)
        fig5_runs.append(
            run_four_source(
                forward,
                config,
                mixings=mixings_multi,
                snr=snr,
                seed=config.master_seed + 5000 + int(round(1000 * snr)),
                n_mc=config.n_mc,
                with_rsa=not skip_rsa,
                rsa_methods=fig5_rsa_methods,
                eval_C=(5, 6),
                label=f"fig5-{tag}",
            )
        )
    fig5 = {
        **header,
        "id": "fig05-four-source-mc",
        "figure": "Figure 5",
        "D14": "caption C=6 vs body C=5; both evaluated as subset/full of the same C=6 draws",
        "P": PAPER_MULTI_SOURCE_P,
        "runs": fig5_runs,
    }
    fig5_path = _write("fig05_four_source.json", fig5)
    print(f"  wrote {fig5_path}", flush=True)

    print("Fig 6 error vs C (SNR assumed 0.2)...", flush=True)
    fig6_run = run_four_source(
        forward,
        config,
        mixings=mixings_multi,
        snr=ASSUMED_FIG6_SNR,
        seed=config.master_seed + 6000,
        n_mc=config.n_mc,
        with_rsa=not skip_rsa,
        rsa_methods=fig5_rsa_methods,
        eval_C=tuple(PAPER_FIG6_C),
        label="fig6-snr0.2-assumed",
    )
    fig6 = {
        **header,
        "id": "fig06-error-vs-C",
        "figure": "Figure 6",
        "snr_assumed": ASSUMED_FIG6_SNR,
        "paper_claim": "ReDisCA mean median error < 2 cm at C=6",
        "run": fig6_run,
    }
    fig6_path = _write("fig06_error_vs_C.json", fig6)
    print(f"  wrote {fig6_path}", flush=True)

    summary = {
        **header,
        "id": "simulations-summary",
        "artifacts": {
            "fig03": str(fig3_path.relative_to(REPO_ROOT)),
            "fig04_roc": str(fig4_path.relative_to(REPO_ROOT)),
            "fig04_traces": str(traces_path.relative_to(REPO_ROOT)),
            "fig05": str(fig5_path.relative_to(REPO_ROOT)),
            "fig06": str(fig6_path.relative_to(REPO_ROOT)),
        },
        "fig04_auc": {
            run["label"]: {name: m["roc"]["auc"] for name, m in run["methods"].items()}
            for run in fig4_runs
        },
        "fig05_mean_median_error_m": {
            run["label"]: {
                c: run["by_C"][c]["redisca"]["mean_median_error_m"] for c in run["by_C"]
            }
            for run in fig5_runs
        },
        "fig06_mean_median_error_m": {
            c: fig6_run["by_C"][c]["redisca"]["mean_median_error_m"] for c in fig6_run["by_C"]
        },
        "commands": [
            "python paper/reproduction/common/download_osf.py source-models",
            "python paper/reproduction/simulations/run.py",
            "python paper/reproduction/simulations/run.py --quick",
        ],
    }
    summary_path = _write("summary.json", summary)
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
