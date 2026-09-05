"""Stage A source localization: paper Fig. 18 MUSIC and AIRI executable branches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from paper.reproduction.common.constants import MASTER_SEEDS, RANDOM_PHASE_B
from paper.reproduction.common.hashing import read_json, sha256_array, sha256_file, write_json
from paper.reproduction.common.method import AIRI_SPOC_KWARGS, fit_redisca
from paper.reproduction.common.paths import MEG_DIR, RESULTS_ROOT, SOURCE_MODEL_DIR
from paper.reproduction.common.provenance import capture_run
from paper.reproduction.meg.prepare import load_meg_bundle, prepare_candidate
from paper.reproduction.meg.rdms import theoretical_rdm
from paper.reproduction.source_localization.forward import (
    atlas_payload_for_vertex,
    index_audit,
    leadfield_blocks,
    load_source_model,
    megplanarbst_0based,
)
from paper.reproduction.source_localization.music import (
    AiriMusicDimensionError,
    airi_music_scan,
    music_scan,
)
from paper.reproduction.source_localization.selection import select_fig17_lowest_p
from paper.reproduction.source_localization.sloreta import precomp_abs_kernel_map
from redisca import random_phase_test


def _topk(map_1d: np.ndarray, model, n: int = 10) -> list[dict[str, Any]]:
    order = np.argsort(map_1d)[::-1][:n]
    out = []
    for rank, vertex in enumerate(order, start=1):
        item = {
            "rank": rank,
            "vertex": int(vertex),
            "value": float(map_1d[vertex]),
        }
        item.update(atlas_payload_for_vertex(model, int(vertex)))
        out.append(item)
    return out


def load_existing_facevstool_pvalues(meg_candidate: str, seed: int) -> np.ndarray | None:
    """Reuse B=1000 random-phase p-values already stored in MEG seed JSON."""
    path = RESULTS_ROOT / "meg" / meg_candidate / f"seed{seed}.json"
    if not path.exists():
        return None
    payload = read_json(path)
    rdm = (payload.get("rdms") or {}).get("facevstool_airi")
    if not rdm:
        return None
    if int(rdm.get("B") or 0) < 1000:
        return None
    return np.asarray(rdm["p_random_phase"], dtype=np.float64)


def _choose_fig18_components(pvals: np.ndarray, rank: int, selection: str) -> tuple[np.ndarray, str]:
    if selection == "lowest_p":
        chosen = select_fig17_lowest_p(pvals, n=3)
        note = (
            "Paper Fig. 17 rule: three components with the lowest p-values, "
            "not a p<0.05 cutoff. This is the Fig. 18 subspace input."
        )
        return chosen, note
    if selection == "p05":
        sig = np.flatnonzero(pvals < 0.05)
        chosen = sig[:3] if sig.size else np.arange(min(3, rank))
        note = "p<0.05 prefix (not the paper Fig. 17 rule); labeled comparison only"
        return np.asarray(chosen, dtype=np.intp), note
    raise ValueError(f"Unknown selection {selection!r}")


def run_fig18(
    seed: int,
    *,
    n_surrogates: int = RANDOM_PHASE_B,
    meg_candidate: str = "MEG-PAPER-1501",
    selection: str = "lowest_p",
    p_values: np.ndarray | None = None,
    fitted_model=None,
    music_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if fitted_model is None:
        bundle = load_meg_bundle()
        prepared = prepare_candidate(bundle, meg_candidate)
        rdm = theoretical_rdm("facevstool", fill="airi")
        fitted_model = fit_redisca(prepared["averages"], rdm)
    p_source = "recomputed_random_phase"
    if p_values is None:
        loaded = load_existing_facevstool_pvalues(meg_candidate, seed)
        if loaded is not None:
            pvals = loaded
            p_source = "reused_meg_seed_json_B1000"
        else:
            pvals = random_phase_test(
                fitted_model, n_surrogates=n_surrogates, random_state=seed
            ).p_values
    else:
        pvals = np.asarray(p_values, dtype=np.float64)
        p_source = "supplied"
    chosen, note = _choose_fig18_components(pvals, int(fitted_model.rank_), selection)
    patterns_hash = sha256_array(fitted_model.patterns_[chosen])
    cache_key = (meg_candidate, selection, tuple(int(i) for i in chosen), patterns_hash)
    reused = False
    if music_cache is not None and cache_key in music_cache:
        scan_payload = music_cache[cache_key]
        reused = True
    else:
        patterns = fitted_model.patterns_[chosen].T
        source = load_source_model(SOURCE_MODEL_DIR)
        rows = megplanarbst_0based()
        blocks = leadfield_blocks(source.gain, rows)
        scan = music_scan(blocks, patterns)
        peak = int(np.argmax(scan))
        scan_payload = {
            "peak_vertex": peak,
            "peak_subcorr": float(scan[peak]),
            "peak_atlas": atlas_payload_for_vertex(source, peak),
            "top10": _topk(scan, source),
            "index_audit": index_audit(source.channel_types),
        }
        if music_cache is not None:
            music_cache[cache_key] = scan_payload
    return {
        "candidate_id": "FIG18-MUSIC-LOWESTP" if selection == "lowest_p" else "FIG18-MUSIC-P05",
        "selection_rule": selection,
        "seed": seed,
        "redisca_kwargs": dict(AIRI_SPOC_KWARGS),
        "meg_candidate": meg_candidate,
        "rdm": "airi_facevstool",
        "p_values": pvals.tolist(),
        "p_values_selected": [float(pvals[i]) for i in chosen],
        "n_significant_p05": int(np.sum(pvals < 0.05)),
        "chosen_components": [int(i) for i in chosen],
        "subspace_dim": int(chosen.size),
        "choice_note": note,
        "p_value_source": p_source,
        "music_reused_identical_patterns": reused,
        "peak_vertex": scan_payload["peak_vertex"],
        "peak_subcorr": scan_payload["peak_subcorr"],
        "peak_atlas": scan_payload["peak_atlas"],
        "top10": scan_payload["top10"],
        "index_audit": scan_payload["index_audit"],
        "paper_claimed": ["right fusiform", "right insula", "left intraparietal", "anterior-central"],
        "patterns_hash": patterns_hash,
    }


def run_airi_precomp() -> dict[str, Any]:
    source = load_source_model(SOURCE_MODEL_DIR)
    topo_path = MEG_DIR / "topo_face_vs_tool_correct_filt15.mat"
    payload = {"author_saved": None, "local_meg_airi": None}
    if topo_path.exists():
        saved = loadmat(topo_path, squeeze_me=True, struct_as_record=False)
        a1 = np.asarray(saved["A1"], dtype=np.float64)
        ctx = precomp_abs_kernel_map(source.imaging_kernel, a1[:, 3], megplanarbst_0based())
        peak = int(np.argmax(ctx))
        payload["author_saved"] = {
            "file": str(topo_path),
            "sha256": sha256_file(topo_path),
            "peak_vertex": peak,
            "peak_value": float(ctx[peak]),
            "peak_atlas": atlas_payload_for_vertex(source, peak),
            "top10": _topk(ctx, source),
            "note": "AIRI default precomp of author-saved A1(:,4); not Fig. 18",
        }
    bundle = load_meg_bundle()
    prepared = prepare_candidate(bundle, "MEG-AIRI")
    rdm = theoretical_rdm("facevstool", fill="airi")
    model = fit_redisca(prepared["averages"], rdm)
    local = precomp_abs_kernel_map(
        source.imaging_kernel, model.patterns_[3] if model.rank_ > 3 else model.patterns_[-1],
        megplanarbst_0based(),
    )
    peak = int(np.argmax(local))
    payload["local_meg_airi"] = {
        "component_used": 3 if model.rank_ > 3 else int(model.rank_ - 1),
        "peak_vertex": peak,
        "peak_value": float(local[peak]),
        "peak_atlas": atlas_payload_for_vertex(source, peak),
        "top10": _topk(local, source),
        "note": "Local MEG-AIRI facevstool pattern through the same sLORETA kernel",
    }
    return {"candidate_id": "AIRI-PRECOMP-SLORETA", **payload}


def run_airi_music() -> dict[str, Any]:
    source = load_source_model(SOURCE_MODEL_DIR)
    rows = megplanarbst_0based()
    g = source.gain[rows]
    bundle = load_meg_bundle()
    prepared = prepare_candidate(bundle, "MEG-AIRI")
    model = fit_redisca(prepared["averages"], theoretical_rdm("facevstool", fill="airi"))
    topos = model.patterns_[:3].T
    literal_error = None
    try:
        airi_music_scan(g, topos, projector_variant="literal_bug")
    except AiriMusicDimensionError as exc:
        literal_error = str(exc)
    scan = airi_music_scan(g, topos, projector_variant="eye_nsns_fix")
    peak = int(np.argmax(scan[:, 0]))
    return {
        "candidate_id": "AIRI-MUSIC-EYE-NSNS",
        "literal_bug": literal_error,
        "peak_vertex": peak,
        "peak_value": float(scan[peak, 0]),
        "peak_atlas": atlas_payload_for_vertex(source, peak),
        "top10": _topk(scan[:, 0], source),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage A source localization.")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(MASTER_SEEDS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--meg-candidates",
        nargs="*",
        default=["MEG-PAPER-1501", "MEG-AIRI"],
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute even if a complete lowest-p Fig. 18 file already exists.",
    )
    parser.add_argument(
        "--skip-airi-aux",
        action="store_true",
        help="Do not rewrite airi_precomp.json / airi_music.json.",
    )
    args = parser.parse_args(argv)
    n_surr = 16 if args.quick else RANDOM_PHASE_B
    written = []
    skipped = []
    prefix = "QUICK_NONREPRO_" if args.quick else ""
    music_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for meg_candidate in args.meg_candidates:
        bundle = load_meg_bundle()
        prepared = prepare_candidate(bundle, meg_candidate)
        fitted = fit_redisca(prepared["averages"], theoretical_rdm("facevstool", fill="airi"))
        dests: list[Path] = []
        chosen_by_seed: list[Any] = []
        hashes: list[Any] = []
        for seed in args.seeds:
            dest = (
                RESULTS_ROOT
                / "source_localization"
                / f"{prefix}fig18_lowestp_{meg_candidate}_seed{seed}.json"
            )
            dests.append(dest)
            if (
                not args.quick
                and not args.no_skip_existing
                and dest.exists()
            ):
                existing = read_json(dest)
                if (
                    existing.get("selection_rule") == "lowest_p"
                    and int(existing.get("subspace_dim") or 0) == 3
                    and not existing.get("quick_non_reproduction")
                ):
                    skipped.append(str(dest))
                    chosen_by_seed.append(existing.get("chosen_components"))
                    hashes.append(existing.get("patterns_hash"))
                    continue
            fig18 = run_fig18(
                seed,
                n_surrogates=n_surr,
                meg_candidate=meg_candidate,
                selection="lowest_p",
                fitted_model=fitted,
                music_cache=music_cache,
            )
            fig18["provenance"] = capture_run(
                track="source_localization",
                candidate_id="FIG18-MUSIC-LOWESTP",
                seed=seed,
            )
            write_json(dest, fig18)
            written.append(str(dest))
            chosen_by_seed.append(fig18["chosen_components"])
            hashes.append(fig18["patterns_hash"])
        selection_stable = bool(chosen_by_seed) and all(
            c == chosen_by_seed[0] for c in chosen_by_seed
        )
        patterns_stable = bool(hashes) and all(h == hashes[0] for h in hashes)
        for dest in dests:
            if not dest.exists():
                continue
            payload = read_json(dest)
            payload["component_selection_identical_across_registered_seeds"] = selection_stable
            payload["patterns_deterministic_across_registered_seeds"] = patterns_stable
            write_json(dest, payload)
    if not args.skip_airi_aux:
        precomp = run_airi_precomp()
        precomp["provenance"] = capture_run(track="source_localization", candidate_id="AIRI-PRECOMP-SLORETA")
        dest = RESULTS_ROOT / "source_localization" / "airi_precomp.json"
        write_json(dest, precomp)
        written.append(str(dest))
        music = run_airi_music()
        music["provenance"] = capture_run(track="source_localization", candidate_id="AIRI-MUSIC-EYE-NSNS")
        dest = RESULTS_ROOT / "source_localization" / "airi_music.json"
        write_json(dest, music)
        written.append(str(dest))
    print({"written": written, "skipped_existing": skipped})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
