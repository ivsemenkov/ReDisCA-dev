"""Stage A source localization: paper Fig. 18 MUSIC and AIRI executable branches."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from scipy.io import loadmat

from paper.reproduction.common.constants import MASTER_SEEDS, RANDOM_PHASE_B
from paper.reproduction.common.hashing import sha256_file, write_json
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


def run_fig18(seed: int, *, n_surrogates: int = RANDOM_PHASE_B) -> dict[str, Any]:
    bundle = load_meg_bundle()
    prepared = prepare_candidate(bundle, "MEG-PAPER-1501")
    rdm = theoretical_rdm("facevstool", fill="airi")
    model = fit_redisca(prepared["averages"], rdm)
    pvals = random_phase_test(model, n_surrogates=n_surrogates, random_state=seed).p_values
    sig = np.flatnonzero(pvals < 0.05)
    if sig.size == 0:
        chosen = np.arange(min(3, model.rank_))
        note = "no p<0.05 components; using first 3 for the scan (labeled fallback)"
    else:
        chosen = sig[:3]
        note = "significant p<0.05 components"
    patterns = model.patterns_[chosen].T
    source = load_source_model(SOURCE_MODEL_DIR)
    rows = megplanarbst_0based()
    blocks = leadfield_blocks(source.gain, rows)
    scan = music_scan(blocks, patterns)
    peak = int(np.argmax(scan))
    return {
        "candidate_id": "FIG18-MUSIC",
        "seed": seed,
        "redisca_kwargs": dict(AIRI_SPOC_KWARGS),
        "meg_candidate": "MEG-PAPER-1501",
        "rdm": "airi_facevstool",
        "p_values": pvals.tolist(),
        "chosen_components": chosen.tolist(),
        "choice_note": note,
        "peak_vertex": peak,
        "peak_subcorr": float(scan[peak]),
        "peak_atlas": atlas_payload_for_vertex(source, peak),
        "top10": _topk(scan, source),
        "index_audit": index_audit(source.channel_types),
        "paper_claimed": ["right fusiform", "right insula", "left intraparietal", "anterior-central"],
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
    parser.add_argument("--seeds", nargs="*", type=int, default=list(MASTER_SEEDS[:1]))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    n_surr = 16 if args.quick else RANDOM_PHASE_B
    written = []
    for seed in args.seeds:
        fig18 = run_fig18(seed, n_surrogates=n_surr)
        fig18["provenance"] = capture_run(track="source_localization", candidate_id="FIG18-MUSIC", seed=seed)
        dest = RESULTS_ROOT / "source_localization" / f"{'QUICK_NONREPRO_' if args.quick else ''}fig18_seed{seed}.json"
        write_json(dest, fig18)
        written.append(str(dest))
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
    print({"written": written})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
