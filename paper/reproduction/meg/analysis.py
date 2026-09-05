"""MEG analyses using the frozen AIRI-SPoC ReDisCA factory."""

from __future__ import annotations

from typing import Any

import numpy as np
from redisca import random_phase_test

from paper.reproduction.common.constants import (
    AIRI_NMC_TEMPORAL,
    PAPER_TEMPORAL_NMC_ASSUMED,
    RANDOM_PHASE_B,
)
from paper.reproduction.common.hashing import sha256_array
from paper.reproduction.common.inference_secondary import (
    airi_halfsplit_timecourse,
    condition_label_permutation,
    empirical_rdm_from_traces,
    paper_timeseries_fwer,
)
from paper.reproduction.common.method import AIRI_SPOC_KWARGS, fit_redisca
from paper.reproduction.common.metrics import rdm_pearson
from paper.reproduction.common.rng import spawned_generator
from paper.reproduction.meg.rdms import PAPER_QUALITATIVE_ONSETS, class_labels, theoretical_rdm


def _intervals_ms(mask: np.ndarray, time_ms: np.ndarray) -> list[dict[str, float]]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    out = []
    for s, e in zip(starts, ends):
        out.append(
            {
                "t_start_ms": float(time_ms[s]),
                "t_end_ms": float(time_ms[e]),
                "peak_ms": float(time_ms[s + int(np.argmax(mask[s : e + 1]))]),
            }
        )
    return out


def analyze_rdm(
    prepared: dict[str, Any],
    *,
    rdm_name: str,
    fill: str,
    seed: int,
    n_surrogates: int = RANDOM_PHASE_B,
    nmc_airi: int = AIRI_NMC_TEMPORAL,
    nmc_paper: int = PAPER_TEMPORAL_NMC_ASSUMED,
    run_secondary_perm: bool = True,
    run_temporal: bool = True,
) -> dict[str, Any]:
    rdm = theoretical_rdm(rdm_name, fill=fill) if fill == "binary" else theoretical_rdm(rdm_name, fill="airi")
    averages = prepared["averages"]
    model = fit_redisca(averages, rdm)
    traces = model.transform(averages)
    primary = random_phase_test(model, n_surrogates=n_surrogates, random_state=seed)
    n_sig = int(np.sum(np.asarray(primary.p_values) < 0.05))
    components = []
    for k in range(min(4, model.rank_)):
        emp = empirical_rdm_from_traces(traces[:, k, :])
        components.append(
            {
                "index": k,
                "lambda": float(model.eigenvalues_[k]),
                "p_random_phase": float(primary.p_values[k]),
                "rdm_corr": rdm_pearson(emp, rdm),
                "empirical_rdm": emp.tolist(),
            }
        )
    payload: dict[str, Any] = {
        "rdm_name": rdm_name,
        "rdm_fill": fill,
        "rdm": rdm.tolist(),
        "input_hash": sha256_array(averages),
        "rank": int(model.rank_),
        "eigenvalues": model.eigenvalues_.tolist(),
        "p_random_phase": primary.p_values.tolist(),
        "n_significant_p05": n_sig,
        "B": int(primary.n_surrogates),
        "seed": seed,
        "redisca_kwargs": dict(AIRI_SPOC_KWARGS),
        "components": components,
        "filters_hash": sha256_array(model.filters_),
        "patterns_hash": sha256_array(model.patterns_),
        "paper_qualitative": PAPER_QUALITATIVE_ONSETS.get(rdm_name, {}),
    }
    if run_secondary_perm:
        payload["secondary_condition_labels"] = condition_label_permutation(
            model, rdm, kind="condition_labels"
        )
    if run_temporal:
        rng_airi, _, _ = spawned_generator(seed, "meg", prepared["candidate_id"], rdm_name, "airi_time")
        rng_paper, _, _ = spawned_generator(seed, "meg", prepared["candidate_id"], rdm_name, "paper_time")
        used, labels = prepared["used_trials"], prepared["trial_labels"]
        # filters_ is (rank, channels); transform used trials
        filtered = np.einsum("rc,ctn->rtn", model.filters_[:4], used)
        class1, class2 = class_labels(rdm_name, convention="airi")
        idx1 = np.concatenate([prepared["indices"][name] for i, name in enumerate(
            ("face1", "face2", "tool1", "tool2", "nons1", "nons2")
        ) if i in class1])
        idx2 = np.concatenate([prepared["indices"][name] for i, name in enumerate(
            ("face1", "face2", "tool1", "tool2", "nons1", "nons2")
        ) if i in class2])
        # AIRI executable applies the half-split to the full filtered epoch
        # even though SPoC is fit on trange 600:1500. Paper-window candidates
        # keep the half-split on the same analysis window as the fit.
        if prepared["candidate_id"] == "MEG-AIRI":
            std_source = prepared["planars_for_std"]
            airi_time_ms = prepared["time_ms_full"]
        else:
            std_source = prepared["planars_windowed"]
            airi_time_ms = prepared["time_ms"]
        std_map = np.std(std_source, axis=2, ddof=1)
        std_planars = std_source / np.maximum(std_map[..., None], 1e-12)
        airi_corrected = airi_halfsplit_timecourse(
            std_planars,
            idx1,
            idx2,
            model.filters_,
            nmc=nmc_airi,
            rng=rng_airi,
            n_components=4,
            indexing="corrected_pooled",
        )
        rng_airi_lit, _, _ = spawned_generator(
            seed, "meg", prepared["candidate_id"], rdm_name, "airi_time_literal"
        )
        airi_literal = airi_halfsplit_timecourse(
            std_planars,
            idx1,
            idx2,
            model.filters_,
            nmc=nmc_airi,
            rng=rng_airi_lit,
            n_components=4,
            indexing="literal",
        )
        paper_c1, paper_c2 = class_labels(rdm_name, convention="paper")
        paper_time = paper_timeseries_fwer(
            filtered,
            labels,
            class1=paper_c1,
            class2=paper_c2,
            nmc=nmc_paper,
            rng=rng_paper,
        )
        time_ms = prepared["time_ms"]
        time_axis = "full_epoch" if prepared["candidate_id"] == "MEG-AIRI" else "analysis_window"
        payload["temporal_airi_corrected"] = {
            "Nmc": nmc_airi,
            "indexing": "corrected_pooled",
            "role": "airi_corrected_pooled_intent",
            "time_axis": time_axis,
            "intervals_pplus": [
                _intervals_ms(airi_corrected["asterisk_positive"][k], airi_time_ms) for k in range(4)
            ],
            "intervals_pminus": [
                _intervals_ms(airi_corrected["asterisk_negative"][k], airi_time_ms) for k in range(4)
            ],
        }
        payload["temporal_airi_literal"] = {
            "Nmc": nmc_airi,
            "indexing": "literal",
            "role": "literal_airi_executable_indexing",
            "time_axis": time_axis,
            "intervals_pplus": [
                _intervals_ms(airi_literal["asterisk_positive"][k], airi_time_ms) for k in range(4)
            ],
            "intervals_pminus": [
                _intervals_ms(airi_literal["asterisk_negative"][k], airi_time_ms) for k in range(4)
            ],
        }
        # Backward-compatible alias: previous files stored the corrected intent
        # under this key and incorrectly labeled it "literal AIRI".
        payload["temporal_airi"] = {
            **payload["temporal_airi_corrected"],
            "note": "Alias of temporal_airi_corrected; not literal MATLAB indexing.",
        }
        payload["temporal_paper_fwer"] = {
            "Nmc": nmc_paper,
            "intervals": [
                _intervals_ms(paper_time["significant"][k], time_ms)
                for k in range(paper_time["significant"].shape[0])
            ],
        }
    return payload


def analyze_candidate(
    prepared: dict[str, Any],
    *,
    seed: int,
    n_surrogates: int = RANDOM_PHASE_B,
    quick: bool = False,
    run_secondary_perm: bool | None = None,
    run_temporal: bool = True,
) -> dict[str, Any]:
    candidate_id = prepared["candidate_id"]
    if candidate_id == "MEG-AIRI":
        jobs = [("face", "airi"), ("tool", "airi"), ("meaning", "airi"), ("facevstool", "airi")]
    elif candidate_id == "MEG-PAPER-1501":
        jobs = [
            ("face", "binary"),
            ("tool", "binary"),
            ("meaning", "binary"),
            ("facevstool", "airi"),
            ("face", "airi"),
            ("tool", "airi"),
            ("meaning", "airi"),
        ]
    else:
        jobs = [("face", "binary"), ("tool", "binary"), ("meaning", "binary"), ("facevstool", "airi")]
    if quick:
        jobs = jobs[:1]
    if run_secondary_perm is None:
        run_secondary_perm = not quick
    results = {}
    for name, fill in jobs:
        key = f"{name}_{fill}"
        results[key] = analyze_rdm(
            prepared,
            rdm_name=name,
            fill=fill,
            seed=seed,
            n_surrogates=n_surrogates,
            nmc_airi=8 if quick else AIRI_NMC_TEMPORAL,
            nmc_paper=8 if quick else PAPER_TEMPORAL_NMC_ASSUMED,
            run_secondary_perm=run_secondary_perm,
            run_temporal=run_temporal,
        )
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "n_samples": prepared["n_samples"],
        "window_ms": prepared["window_ms"],
        "filter": prepared["filter"],
        "source_sha256": prepared["source_sha256"],
        "rdms": results,
    }
