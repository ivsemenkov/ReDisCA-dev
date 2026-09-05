"""Run Stage A simulation experiments (Figs 3–6).

All ReDisCA fits go through ``make_redisca``. Generated realizations are
hashed so later method ablations can reuse or regenerate the same inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from paper.reproduction.common.constants import MASTER_SEEDS
from paper.reproduction.common.hashing import sha256_array, write_json
from paper.reproduction.common.method import AIRI_SPOC_KWARGS, fit_redisca
from paper.reproduction.common.paths import RESULTS_ROOT, SOURCE_MODEL_DIR, ARRAY_CACHE_DIR
from paper.reproduction.common.provenance import capture_run
from paper.reproduction.common.rng import spawned_generator
from paper.reproduction.simulations.config import (
    DEFAULT_CONFIG,
    PAPER_SNR_FIG4_HIGH,
    PAPER_SNR_FIG4_LOW,
    PAPER_SNR_FIG5_HIGH,
    PAPER_SNR_FIG5_LOW,
    QUICK_CONFIG,
    SimulationConfig,
)
from paper.reproduction.simulations.forward_model import load_ad_forward
from paper.reproduction.simulations.generate import (
    simulate_multi_source,
    simulate_single_source,
    subset_conditions_multi,
)
from paper.reproduction.simulations.metrics_roc import (
    cosine_abs_scan,
    default_thresholds,
    downsample_roc,
    localization_error_m,
    pearson,
    roc_from_mc,
    sign_align_to_reference,
    summarize_roc,
)
from paper.reproduction.simulations.rsa_baselines import (
    four_rsa_scans,
    rsa_scans_for_targets,
    unique_pairs,
)


def _empirical_rdm(traces: np.ndarray) -> np.ndarray:
    n = traces.shape[0]
    rdm = np.zeros((n, n), dtype=np.float64)
    for i, j in unique_pairs(n):
        delta = traces[i] - traces[j]
        val = float(delta @ delta)
        rdm[i, j] = rdm[j, i] = val
    return rdm


def _fit_first_component(averages: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    model = fit_redisca(averages, target)
    traces = model.transform(averages)[:, 0, :]
    return {
        "eigenvalues": model.eigenvalues_,
        "filters": model.filters_,
        "patterns": model.patterns_,
        "pattern0": model.patterns_[0],
        "filter0": model.filters_[0],
        "traces0": traces,
        "rank": int(model.rank_),
        "empirical_rdm": _empirical_rdm(traces),
        "rdm_corr": pearson(_empirical_rdm(traces)[np.triu_indices(traces.shape[0], 1)],
                            target[np.triu_indices(target.shape[0], 1)]),
    }


def _record_draw_hashes(draw, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "averages": sha256_array(draw.averages),
        "d_target": sha256_array(draw.d_target),
        "d0": sha256_array(draw.d0),
        "trials": sha256_array(draw.trials),
        "noiseless_average": sha256_array(draw.noiseless_average),
    }
    if extra:
        payload.update(extra)
    return payload


def run_fig4(
    *,
    candidate_id: str,
    config: SimulationConfig,
    seed: int,
    snr: float,
    n_conditions: int = 5,
    include_rsa: bool = True,
    store_arrays: bool = False,
) -> dict[str, Any]:
    forward = load_ad_forward(SOURCE_MODEL_DIR)
    mix_rng, mix_seed, _ = spawned_generator(seed, "fig4", "mixing", n_conditions)
    mixing = mix_rng.standard_normal((n_conditions, n_conditions))
    scores_redisca = []
    scores_rsa: dict[str, list[np.ndarray]] = {k: [] for k in ("mne_av", "mne_st", "bf_av", "bf_st")}
    inside = []
    loc_errors = []
    realization_records = []
    example = None
    for mc in range(config.n_mc):
        rng, child_seed, _ = spawned_generator(seed, "fig4", int(round(snr * 1000)), mc)
        draw = simulate_single_source(forward, rng, mixing, config=config, snr=snr)
        fit = _fit_first_component(draw.averages, draw.d_target)
        scan = cosine_abs_scan(fit["pattern0"], forward.gain)
        scores_redisca.append(scan)
        mask = forward.distances_from(draw.vertex) <= config.r_max_m
        inside.append(mask)
        loc = localization_error_m(scan, draw.vertex, forward.vertices)
        loc_errors.append(loc)
        rsa = {}
        if include_rsa:
            rsa = four_rsa_scans(
                forward.gain, draw.averages, draw.trials, draw.d_target, config=config
            )
            for name, values in rsa.items():
                scores_rsa[name].append(values)
        record = {
            "mc": mc,
            "child_seed": child_seed,
            "vertex": draw.vertex,
            "snr": snr,
            "hashes": _record_draw_hashes(draw),
            "redisca_lambda0": float(fit["eigenvalues"][0]),
            "rdm_corr": float(fit["rdm_corr"]),
            "loc_error_cm": loc["error_cm"],
        }
        realization_records.append(record)
        if mc == 0:
            example = {
                "noiseless_erp_rms": float(np.sqrt(np.mean(draw.noiseless_average ** 2))),
                "noisy_erp_rms": float(np.sqrt(np.mean(draw.averages ** 2))),
                "d0": draw.d0.tolist(),
                "d_target": draw.d_target.tolist(),
            }
        if store_arrays:
            dest = ARRAY_CACHE_DIR / "simulations" / candidate_id / f"fig4_snr{snr}_seed{seed}_mc{mc}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dest,
                averages=draw.averages,
                d_target=draw.d_target,
                vertex=np.asarray([draw.vertex]),
            )
        del draw
    thresholds = default_thresholds("cosine_abs")
    roc = roc_from_mc(np.stack(scores_redisca), np.stack(inside), thresholds)
    summary = {
        "candidate_id": candidate_id,
        "experiment": "fig4_single_source",
        "seed": seed,
        "snr": snr,
        "n_mc": config.n_mc,
        "n_conditions": n_conditions,
        "mixing_seed": mix_seed,
        "config": config.to_dict(),
        "redisca_kwargs": dict(AIRI_SPOC_KWARGS),
        "redisca_roc": summarize_roc(roc),
        "redisca_roc_curve": downsample_roc(roc),
        "mean_loc_error_cm": float(np.mean([x["error_cm"] for x in loc_errors])),
        "median_loc_error_cm": float(np.median([x["error_cm"] for x in loc_errors])),
        "example": example,
        "realizations": realization_records,
        "provenance": capture_run(track="simulations", candidate_id=candidate_id, seed=seed),
    }
    if include_rsa:
        rsa_summary = {}
        for name, scans in scores_rsa.items():
            roc_m = roc_from_mc(np.stack(scans), np.stack(inside), default_thresholds("pearson"))
            rsa_summary[name] = summarize_roc(roc_m)
        summary["rsa_roc"] = rsa_summary
    return summary


def run_fig5_fig6(
    *,
    candidate_id: str,
    config: SimulationConfig,
    seed: int,
    snr: float,
    eval_conditions: tuple[int, ...],
    include_rsa: bool = True,
) -> dict[str, Any]:
    forward = load_ad_forward(SOURCE_MODEL_DIR)
    n_gen = 6
    mix_rng, mix_seed, _ = spawned_generator(seed, "fig5", "mixing")
    mixings = mix_rng.standard_normal((4, n_gen, n_gen))
    by_c: dict[int, list[dict[str, Any]]] = {c: [] for c in eval_conditions}
    for mc in range(config.n_mc):
        rng, child_seed, _ = spawned_generator(seed, "fig5", int(round(snr * 1000)), mc)
        draw = simulate_multi_source(
            forward, rng, mixings, config=config, snr=snr, n_conditions=n_gen
        )
        for n_cond in eval_conditions:
            sliced = subset_conditions_multi(draw, n_cond)
            source_metrics = []
            for p in range(4):
                fit = _fit_first_component(sliced["averages"], sliced["d_target"][p])
                pattern = sign_align_to_reference(fit["pattern0"], sliced["g_true"][p])
                weight = sign_align_to_reference(fit["filter0"], sliced["g_true"][p])
                scan = cosine_abs_scan(fit["pattern0"], forward.gain)
                loc = localization_error_m(scan, int(sliced["vertices"][p]), forward.vertices)
                source_metrics.append(
                    {
                        "source": p,
                        "lambda0": float(fit["eigenvalues"][0]),
                        "pattern_corr": pearson(pattern, sliced["g_true"][p]),
                        "weight_corr": pearson(weight, sliced["g_true"][p]),
                        "rdm_corr": float(fit["rdm_corr"]),
                        "loc_error_cm": loc["error_cm"],
                    }
                )
            rsa_loc = {}
            if include_rsa:
                scans = rsa_scans_for_targets(
                    forward.gain,
                    sliced["averages"],
                    sliced["trials"],
                    sliced["d_target"],
                    config=config,
                    methods=("mne_st", "bf_st"),
                )
                for p, method_map in enumerate(scans):
                    rsa_loc[str(p)] = {
                        name: localization_error_m(
                            values, int(sliced["vertices"][p]), forward.vertices
                        )["error_cm"]
                        for name, values in method_map.items()
                    }
            by_c[n_cond].append(
                {
                    "mc": mc,
                    "child_seed": child_seed,
                    "vertices": [int(v) for v in sliced["vertices"]],
                    "hashes": {
                        "averages": sha256_array(sliced["averages"]),
                        "d_target": sha256_array(sliced["d_target"]),
                    },
                    "sources": source_metrics,
                    "rsa_loc_error_cm": rsa_loc,
                }
            )
        del draw
    summary = {
        "candidate_id": candidate_id,
        "experiment": "fig5_fig6_multi_source",
        "seed": seed,
        "snr": snr,
        "n_mc": config.n_mc,
        "generated_C": n_gen,
        "eval_conditions": list(eval_conditions),
        "mixing_seed": mix_seed,
        "config": config.to_dict(),
        "redisca_kwargs": dict(AIRI_SPOC_KWARGS),
        "by_C": {},
        "provenance": capture_run(track="simulations", candidate_id=candidate_id, seed=seed),
    }
    for n_cond, records in by_c.items():
        errors = [src["loc_error_cm"] for rec in records for src in rec["sources"]]
        rdm_corrs = [src["rdm_corr"] for rec in records for src in rec["sources"]]
        pattern_corrs = [src["pattern_corr"] for rec in records for src in rec["sources"]]
        weight_corrs = [src["weight_corr"] for rec in records for src in rec["sources"]]
        summary["by_C"][str(n_cond)] = {
            "mean_median_error_cm": float(np.mean([
                np.median([src["loc_error_cm"] for src in rec["sources"]]) for rec in records
            ])),
            "frac_error_lt_1cm": float(np.mean(np.asarray(errors) < 1.0)),
            "mean_rdm_corr": float(np.mean(rdm_corrs)),
            "mean_pattern_corr": float(np.mean(pattern_corrs)),
            "mean_weight_corr": float(np.mean(weight_corrs)),
            "realizations": records,
        }
    return summary


def _out_path(candidate_id: str, name: str, seed: int, *, quick: bool = False) -> Path:
    prefix = "QUICK_NONREPRO_" if quick else ""
    dest = RESULTS_ROOT / "simulations" / candidate_id / f"{prefix}{name}_seed{seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def run_candidate(
    candidate_id: str,
    *,
    seeds: tuple[int, ...],
    quick: bool = False,
    include_rsa: bool = True,
) -> dict[str, Any]:
    config = QUICK_CONFIG if quick else DEFAULT_CONFIG
    if candidate_id == "SIM-P2":
        config = SimulationConfig(n_mc=config.n_mc, i_c=80)
    if candidate_id == "SIM-P3":
        config = SimulationConfig(n_mc=config.n_mc, i_c=config.i_c, butter_zero_phase=False)
    written = []
    for seed in seeds:
        if candidate_id in {"SIM-P1", "SIM-P2", "SIM-P3"}:
            for snr in (PAPER_SNR_FIG4_HIGH, PAPER_SNR_FIG4_LOW):
                if candidate_id == "SIM-P3" and snr != PAPER_SNR_FIG4_LOW:
                    continue
                payload = run_fig4(
                    candidate_id=candidate_id,
                    config=config,
                    seed=seed,
                    snr=snr,
                    include_rsa=include_rsa and candidate_id == "SIM-P1",
                )
                path = _out_path(candidate_id, f"fig4_snr{snr}", seed, quick=quick)
                write_json(path, payload)
                written.append(str(path))
        if candidate_id == "SIM-P1":
            for snr in (PAPER_SNR_FIG5_HIGH, PAPER_SNR_FIG5_LOW):
                payload = run_fig5_fig6(
                    candidate_id=candidate_id,
                    config=config,
                    seed=seed,
                    snr=snr,
                    eval_conditions=(3, 4, 5, 6),
                    include_rsa=include_rsa,
                )
                path = _out_path(candidate_id, f"fig5_fig6_snr{snr}", seed, quick=quick)
                write_json(path, payload)
                written.append(str(path))
    return {"candidate_id": candidate_id, "quick": quick, "written": written}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage A simulations (Figs 3–6).")
    parser.add_argument("--candidate", default="SIM-P1", choices=["SIM-P1", "SIM-P2", "SIM-P3"])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(MASTER_SEEDS))
    parser.add_argument("--quick", action="store_true", help="NON-REPRODUCTION: reduced MC/I_c")
    parser.add_argument("--no-rsa", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        print("WARNING: --quick is NON-REPRODUCTION and must not be reported as a paper result.")
    result = run_candidate(
        args.candidate,
        seeds=tuple(args.seeds),
        quick=args.quick,
        include_rsa=not args.no_rsa,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
