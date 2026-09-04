"""Source-cited N170 preprocessing forensics (Track C).

Does not invent ICA components, interpolate P9/P10, or run a combinatorial
estimator search. Optional historical fits are gated behind a clearly
source-supported alternative data state — none is found.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
for _path in (
    HERE,
    HERE.parent,
    HERE.parents[1],
    REPO_ROOT / "src",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.erp_io import load_erplab_erp
from common.hashing import sha256_file
from common.paths import UPSTREAM_DIR

from prepare import (
    EOG_AND_BIPOLAR_LABELS,
    PREFERRED_ERP_NAME,
    SCALP_LABELS,
    subject1_dir,
)
from eeglab_io import (
    load_eeglab_set,
    ica_mixing,
    removed_ica_components_1based,
)
from ica_xlsx import (
    ICA_XLSX_SHA256,
    ICA_XLSX_URL,
    default_xlsx_path,
    download_ica_components_xlsx,
    xlsx_summary,
)

ERPCORE_COMMIT = "c18b43d70d791ca914d90410afe4ff06d6f7f429"
PREFERRED_ERP_SHA256 = "53e74e931e6f0adaf1e5be4d606d028fcc3e04ee8b066569c2ed2d033d9bbc72"
PAPER_ICA_QUOTE = (
    "were preprocessed and three ICA components corresponding to ocular "
    "and cardiac artifacts were removed from the data. Then, the ERP were "
    "computed by averaging responses within each of the stimulus types."
)
SCRIPT7_LPFILT_COMMENT = (
    "Apply a low-pass filter (non-causal Butterworth impulse response "
    "function, 20 Hz half-amplitude cut-off, 48 dB/oct roll-off) to the ERP waveforms"
)
SCRIPT7_LPFILT_CALL = (
    "ERP = pop_filterp( ERP,  1:35 , 'Cutoff',  20, 'Design', 'butter', "
    "'Filter', 'lowpass', 'Order',  8 );"
)

SET_STAGE_ORDER: tuple[str, ...] = (
    "1_N170.set",
    "1_N170_shifted.set",
    "1_N170_shifted_ds_reref_ucbip.set",
    "1_N170_shifted_ds_reref_ucbip_hpfilt.set",
    "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_prep2_weighted.set",
    "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_weighted.set",
    "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr.set",
    "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip.set",
    "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch.set",
    "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp.set",
    "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
)

ERP_NAMES: tuple[str, ...] = (
    PREFERRED_ERP_NAME,
    "1_N170_erp_ar_lpfilt.erp",
    "1_N170_erp_ar_diff_waves_lpfilt.erp",
)

SCRIPTS: dict[str, str] = {
    "script1": "N170/EEG_ERP_Processing/1_Import_Raw_EEG_Shift_DS_Reref_Hpfilt.m",
    "script3": "N170/EEG_ERP_Processing/3_Run_ICA.m",
    "script4": "N170/EEG_ERP_Processing/4_Remove_ICA_Components.m",
    "script5": "N170/EEG_ERP_Processing/5_Elist_Bin_Epoch.m",
    "script6": "N170/EEG_ERP_Processing/6_Artifact_Rejection.m",
    "script7": "N170/EEG_ERP_Processing/7_Average_ERPs.m",
    "script8": "N170/EEG_ERP_Processing/8_Plot_Individual_Subject_ERPs.m",
    "script12": "N170/EEG_ERP_Processing/ERP_Measurements/12_Measure_ERPs.m",
    "bdf": "N170/EEG_ERP_Processing/BDF_N170.txt",
    "diff_wave": "N170/EEG_ERP_Processing/N170_Diff_Wave.txt",
    "reref": "N170/EEG_ERP_Processing/Rereference_Add_Uncorrected_Bipolars_N170.txt",
    "corr_bipolars": "N170/EEG_ERP_Processing/Add_Corrected_Bipolars_N170.txt",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def inventory_subject_files(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or subject1_dir()
    records = [_file_record(path) for path in sorted(root.iterdir()) if path.is_file()]
    return records


def hash_scripts(scripts_root: Path | None = None) -> dict[str, Any]:
    scripts_root = scripts_root or (UPSTREAM_DIR / "ERP_CORE")
    out: dict[str, Any] = {"commit": ERPCORE_COMMIT, "root": str(scripts_root)}
    files = {}
    for key, rel in SCRIPTS.items():
        path = scripts_root / rel
        files[key] = {**_file_record(path), "relative": rel}
    out["files"] = files
    return out


def inspect_erp_file(path: Path) -> dict[str, Any]:
    payload = load_erplab_erp(path)
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)["ERP"]
    labels = list(payload["channel_labels"])
    bins = [str(x) for x in payload["bin_descriptions"]]
    n_accepted = np.asarray(payload["n_accepted"], dtype=np.int64).tolist()
    n_rejected = np.asarray(getattr(raw.ntrials, "rejected", []), dtype=np.int64).tolist()
    history = str(getattr(raw, "history", ""))
    return {
        **_file_record(path),
        "erpname": str(payload["erpname"]),
        "nchan": int(raw.nchan),
        "nbin": int(raw.nbin),
        "pnts": int(raw.pnts),
        "srate_hz": float(payload["srate"]),
        "xmin_s": float(payload["xmin"]),
        "xmax_s": float(payload["xmax"]),
        "times_ms_first": float(payload["times_ms"][0]),
        "times_ms_last": float(payload["times_ms"][-1]),
        "isfilt": int(getattr(raw, "isfilt", 0)),
        "channel_labels": labels,
        "has_p9": "P9" in labels,
        "has_p10": "P10" in labels,
        "n_scalp_in_file": sum(1 for lab in labels if lab in SCALP_LABELS),
        "eog_or_bipolar_in_file": [lab for lab in labels if lab in EOG_AND_BIPOLAR_LABELS],
        "bin_descriptions": bins,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "history_has_pop_filterp": "pop_filterp" in history,
        "history_has_binoperator": "pop_binoperator" in history,
        "history_has_averager_good": "Criterion', 'good'" in history or 'Criterion", "good"' in history,
    }


def compare_erp_stages(root: Path | None = None) -> dict[str, Any]:
    root = root or subject1_dir()
    stages = {name: inspect_erp_file(root / name) for name in ERP_NAMES if (root / name).exists()}
    ar = load_erplab_erp(root / PREFERRED_ERP_NAME)
    lp = load_erplab_erp(root / "1_N170_erp_ar_lpfilt.erp")
    dw_path = root / "1_N170_erp_ar_diff_waves_lpfilt.erp"
    dw = load_erplab_erp(dw_path) if dw_path.exists() else None
    ar_data = np.asarray(ar["data"], dtype=np.float64)
    lp_data = np.asarray(lp["data"], dtype=np.float64)
    comparison = {
        "ar_vs_lpfilt_identical": bool(np.allclose(ar_data, lp_data)),
        "ar_vs_lpfilt_rms": float(np.sqrt(np.mean((ar_data - lp_data) ** 2))),
        "ar_vs_lpfilt_max_abs": float(np.max(np.abs(ar_data - lp_data))),
        "missing_unfiltered_diff_waves_erp": not (root / "1_N170_erp_ar_diff_waves.erp").exists(),
    }
    if dw is not None:
        dw_data = np.asarray(dw["data"], dtype=np.float64)
        comparison["lpfilt_vs_diff_waves_bins1to4_identical"] = bool(
            np.array_equal(lp_data, dw_data[:4])
        )
        comparison["diff_waves_nbin"] = int(dw_data.shape[0])
        comparison["diff_waves_extra_bins"] = [str(x) for x in dw["bin_descriptions"][4:]]
    return {"stages": stages, "comparison": comparison}


def inspect_set_stages(root: Path | None = None) -> dict[str, Any]:
    root = root or subject1_dir()
    stages = []
    for name in SET_STAGE_ORDER:
        path = root / name
        if not path.exists():
            stages.append({"name": name, "exists": False})
            continue
        meta = load_eeglab_set(path)
        meta["sha256"] = sha256_file(path)
        meta["size_bytes"] = path.stat().st_size
        stages.append(meta)
    return {"stages": stages}


def eventlist_trial_counts(root: Path | None = None) -> dict[str, Any]:
    root = root or subject1_dir()
    path = root / "1_N170_Eventlist_Bins.txt"
    text = path.read_text(encoding="utf-8", errors="replace")
    header_bins: dict[str, int] = {}
    for line in text.splitlines():
        match = re.search(r"bin\s+(\d+),\s+#\s+(\d+),\s+(.*\S)", line)
        if match:
            header_bins[match.group(3).strip()] = int(match.group(2))
    n_stim = 0
    n_unbinned_stim = 0
    responses: Counter[int] = Counter()
    stim_codes = set(range(1, 81)) | set(range(101, 181))
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        try:
            code = int(parts[2])
        except ValueError:
            continue
        bin_field = line[line.rfind("[") :] if "[" in line else ""
        empty_bin = bin_field.strip() in {"[       ]", "[]", "[ ]"}
        if code in stim_codes:
            n_stim += 1
            if empty_bin:
                n_unbinned_stim += 1
        if code in (201, 202):
            responses[code] += 1
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "header_correct_bins": header_bins,
        "n_correct_binned_header": int(sum(header_bins.values())),
        "n_stimulus_events": n_stim,
        "n_unbinned_stimuli": n_unbinned_stim,
        "n_correct_response_201": int(responses[201]),
        "n_incorrect_response_202": int(responses[202]),
        "bdf": "BDF_N170.txt bins 1-4 are correct-response only (stimulus then 201 in 200-1000 ms)",
        "paper_vs_bdf": (
            "Paper: averaging responses within each stimulus type. "
            "ERP CORE BDF and the precomputed .erp are correct-response only. "
            "No public all-trials 4-condition .erp exists in the subject-1 dump."
        ),
    }


def artifact_rejection_counts(root: Path | None = None) -> dict[str, Any]:
    root = root or subject1_dir()
    ar_path = root / (
        "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set"
    )
    eeg = loadmat(ar_path, squeeze_me=True, struct_as_record=False)["EEG"]
    info = np.atleast_1d(eeg.EVENTLIST.eventinfo)
    epoch_flags: dict[int, int] = {}
    for item in info:
        bepoch = np.asarray(getattr(item, "bepoch", 0)).reshape(-1)
        flag = np.asarray(getattr(item, "flag", 0)).reshape(-1)
        if bepoch.size == 0 or int(bepoch[0]) <= 0:
            continue
        epoch = int(bepoch[0])
        value = int(flag[0]) if flag.size else 0
        epoch_flags[epoch] = epoch_flags.get(epoch, 0) | value
    n_good = sum(1 for value in epoch_flags.values() if value == 0)
    n_flagged = sum(1 for value in epoch_flags.values() if value != 0)
    csv_path = root / "1_AR_Percentages_N170.csv"
    return {
        "epoched_trials": int(eeg.trials),
        "eventlist_unique_epochs": len(epoch_flags),
        "n_good_flag0": n_good,
        "n_flagged": n_flagged,
        "flag_histogram": {str(k): v for k, v in sorted(Counter(epoch_flags.values()).items())},
        "averager_criterion": "good (Script 7 pop_averager Criterion='good')",
        "ar_csv": csv_path.read_text(encoding="utf-8") if csv_path.exists() else None,
        "ar_csv_sha256": sha256_file(csv_path) if csv_path.exists() else None,
        "pre_ar_public_erp": False,
        "note": (
            "Pre-AR epoched data exist as a .set (240 correct-response epochs). "
            "The only public 4-condition averaged ERPs are post-AR. "
            "Re-averaging pre-AR trials would reconstruct a new file, not use a "
            "precomputed official ERP; that is not a clearly source-supported "
            "alternative analysis state for the paper."
        ),
    }


def ica_forensics(root: Path | None = None, xlsx_path: Path | None = None) -> dict[str, Any]:
    root = root or subject1_dir()
    xlsx_path = xlsx_path or default_xlsx_path()
    if not xlsx_path.exists() or sha256_file(xlsx_path) != ICA_XLSX_SHA256:
        xlsx_path = download_ica_components_xlsx(xlsx_path)
    weighted = root / "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_weighted.set"
    corrected = root / "1_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr.set"
    removed = removed_ica_components_1based(weighted, corrected)
    mixing = ica_mixing(weighted)
    weighted_meta = load_eeglab_set(weighted)
    ica_labels = [
        lab
        for index, lab in enumerate(weighted_meta["channel_labels"])
        if (index + 1) in set(weighted_meta["icachansind_1based"])
    ]
    eog_idx = [i for i, lab in enumerate(ica_labels) if "EOG" in lab]
    eog_table = []
    if mixing.ndim == 2 and mixing.shape[0] == len(ica_labels):
        for component in range(mixing.shape[1]):
            eog_sum = float(np.abs(mixing[eog_idx, component]).sum()) if eog_idx else None
            max_i = int(np.argmax(np.abs(mixing[:, component])))
            eog_table.append(
                {
                    "component_1based": component + 1,
                    "eog_abs_sum": eog_sum,
                    "max_abs_channel": ica_labels[max_i],
                    "removed_in_official_subcomp": (component + 1) in removed,
                }
            )
    eog_table_sorted = sorted(eog_table, key=lambda row: -(row["eog_abs_sum"] or 0.0))
    return {
        "xlsx": xlsx_summary(xlsx_path),
        "paper_quote": PAPER_ICA_QUOTE,
        "paper_quote_source": ".reproduction_data/paper_text/published_neuroimage.txt (N170 paragraph)",
        "script4_quotes": {
            "xlsx_load": "Load list of ICA component(s) corresponding to ocular artifacts from Excel file ICA_Components_N170.xlsx",
            "subcomp": "Perform ocular correction by removing the ICA component(s) specified above",
            "cardiac_mentioned": False,
        },
        "script3_note": (
            "Script 3 is commented out so original ICA weights are kept. "
            "This forensics track does not re-run ICA."
        ),
        "weighted_set": weighted_meta,
        "corrected_set": load_eeglab_set(corrected),
        "removed_components_1based_from_weights": removed,
        "n_components_before": 31,
        "n_components_after_subcomp": 29,
        "matches_xlsx_subject_1": removed == list(
            xlsx_summary(xlsx_path)["subject_1_components_1based"]
        ),
        "gcompreject_not_used": (
            "gcompreject is all zeros in the weighted file; Script 4 reads the xlsx "
            "rather than EEG.reject.gcompreject."
        ),
        "no_ecg_channel": True,
        "eog_abs_loading_by_ic": eog_table,
        "highest_remaining_eog_ic_observation": {
            "note": (
                "Some remaining ICs have non-zero EOG-channel mixing energy "
                "(IC 18 is the largest remaining EOG-sum in subject 1). "
                "That is not a source-supported 'third component'. Official "
                "xlsx + pop_subcomp remove 2 and 7 only. Do not select IC 18 "
                "or any other leftover IC to chase the paper's 'three'."
            ),
            "highest_remaining_after_official_removal": next(
                (
                    row
                    for row in eog_table_sorted
                    if not row["removed_in_official_subcomp"]
                ),
                None,
            ),
        },
        "third_component_source_supported": False,
        "stop_reason": (
            "No official list, reject mask, or script history marks a third "
            "subject-1 ICA component. The paper sentence is a coarse ERP CORE "
            "gloss (10/40 subjects have exactly 3 listed components; modal count "
            "is 1; mean ≈ 2.03). D11 remains a documented discrepancy."
        ),
        "xlsx_url": ICA_XLSX_URL,
        "xlsx_sha256_pinned": ICA_XLSX_SHA256,
    }


def analysis_channel_set() -> dict[str, Any]:
    return {
        "likely_analysis_stage": PREFERRED_ERP_NAME,
        "channels_present_in_erp": 35,
        "documented_scalp_subset": list(SCALP_LABELS),
        "n_scalp": len(SCALP_LABELS),
        "dropped_eog_or_bipolar": list(EOG_AND_BIPOLAR_LABELS),
        "p9_p10": {
            "in_raw_1_N170_set": True,
            "in_post_script1_and_erp": False,
            "script1_reref": (
                "Rereference_Add_Uncorrected_Bipolars_N170.txt skips original "
                "ch9=P9 and ch27=P10 when building 28 average-referenced scalp "
                "channels (nch9 = ch10 PO7, nch26 = ch28 PO8)."
            ),
            "do_not_interpolate": True,
        },
        "paper_names_channel_list": False,
        "do_not_add_eog_to_change_results": True,
    }


def lpfilt_assessment() -> dict[str, Any]:
    return {
        "paper_mentions_lowpass_on_erps": False,
        "paper_n170_preprocessing_sentence": PAPER_ICA_QUOTE,
        "paper_only_lowpass_in_n170_neighborhood": (
            "The published N170 section does not mention a 20 Hz (or any) "
            "low-pass on the ERPs. The word 'low-pass' in the paper extract "
            "near this section is the simulation source-generation Butterworth "
            "at 2 Hz, not EEG preprocessing."
        ),
        "script7_optional_after_averaging": True,
        "script7_comment": SCRIPT7_LPFILT_COMMENT,
        "script7_call": SCRIPT7_LPFILT_CALL,
        "script8_plots_use": "1_N170_erp_ar_diff_waves_lpfilt.erp (parents + difference bins)",
        "script12_measures": (
            "mean amplitude / 50% area latency on unfiltered diff-waves; "
            "peaks/onset on lpfilt diff-waves"
        ),
        "scientifically_plausible_as_erpcore_extra": True,
        "scientifically_required_by_paper": False,
        "silent_default": False,
        "current_track_preference": (
            "1_N170_erp_ar.erp is primary; lpfilt is a labeled ERP CORE extra"
        ),
    }


def track_a_variants_spec() -> dict[str, Any]:
    return {
        "ran": False,
        "reason_not_run": (
            "No alternative PUBLIC preprocessing state is clearly source-supported "
            "as a better match to the paper than 1_N170_erp_ar.erp. Track C does "
            "not run a combinatorial preprocessing search."
        ),
        "would_have_been": {
            "estimator": "common.source_faithful.fit_condition_averages",
            "inference": "spoc_random_phase",
            "B": 1000,
            "face": {
                "centers_ms": [200.0],
                "duration_ms": 100.0,
                "pair_modes": ["unique_unordered", "airi_directed"],
                "matrix_modes": ["unscaled_gram", "matlab_cov"],
                "n": 4,
            },
            "car": {
                "centers_ms": [170.0, 200.0],
                "duration_ms": 100.0,
                "pair_modes": ["unique_unordered", "airi_directed"],
                "matrix_modes": ["unscaled_gram", "matlab_cov"],
                "n": 8,
                "center_motivation": (
                    "170 ms treats the paper's 'applied at t=170 ms' as a window "
                    "center; 200 ms uses the same centering as the face analysis "
                    "(still containing 170 ms). Documented, not executed."
                ),
            },
            "n_total": 12,
        },
    }


def answers(bundle: dict[str, Any]) -> dict[str, Any]:
    ica = bundle["ica"]
    stages = bundle["erp_stages"]["comparison"]
    return {
        "q1_better_matching_erp": {
            "answer": (
                "No. The 4-condition official average closest to the paper "
                "('averaging responses within each of the stimulus types') is "
                "1_N170_erp_ar.erp. lpfilt is the same averages after Script 7's "
                "optional 20 Hz filter. Diff-waves_lpfilt bins 1-4 are bit-identical "
                "to lpfilt; bins 5-9 are derived contrasts, not the 4-condition "
                "ReDisCA input."
            ),
            "preferred": PREFERRED_ERP_NAME,
            "preferred_sha256": bundle["erp_stages"]["stages"][PREFERRED_ERP_NAME]["sha256"],
            "lpfilt_identical_to_diff_waves_parents": stages.get(
                "lpfilt_vs_diff_waves_bins1to4_identical"
            ),
            "missing_unfiltered_diff_waves": stages.get("missing_unfiltered_diff_waves_erp"),
        },
        "q2_lpfilt_paper_plausible": {
            "answer": (
                "Plausible as a documented ERP CORE extra (Script 7), not as the "
                "paper-stated analysis. The paper does not mention low-pass "
                "filtering ERPs. Do not use lpfilt as a silent default."
            ),
            "paper_mentions_lowpass_on_erps": False,
            "script7_applies_20hz_after_averaging": True,
            "estimator_run_on_lpfilt": False,
        },
        "q3_channels_at_analysis_stage": {
            "answer": (
                "The averaged ERP has 35 labels: 28 scalp EEG (FP1…O2) plus 7 "
                "EOG/bipolar channels. P9/P10 are in the raw 33-channel 1_N170.set "
                "and are dropped by Script 1; they are absent from every later "
                ".set and from the .erp. Do not interpolate them. Do not add EOG "
                "merely to change results. Documented analysis subset remains the "
                "28 scalp channels."
            ),
            "erp_nchan": 35,
            "scalp_n": 28,
            "p9_p10_in_erp": False,
            "p9_p10_in_raw_set": True,
        },
        "q4_third_ica_component": {
            "answer": (
                "No source-supported third subject-1 component. xlsx subject 1 = "
                "[2, 7]; icaweights 31×31 -> 29×31 after pop_subcomp, and the "
                "dropped rows match components 2 and 7. Script 4 calls them ocular, "
                "not cardiac. Across 40 subjects the modal count is 1 (15 subjects); "
                "exactly 3 components occur in 10/40 subjects, not including '1'. "
                "STOP: do not invent a third IC."
            ),
            "subject_1_components_1based": ica["xlsx"]["subject_1_components_1based"],
            "removed_from_weights": ica["removed_components_1based_from_weights"],
            "third_component_source_supported": False,
            "invented_component": None,
        },
        "q5_alternative_state_ran": {
            "answer": "No alternative estimator battery was run.",
            "ran_track_a_12_variants": False,
            "ran_lpfilt_estimators": False,
            "ran_diff_waves_estimators": False,
            "ran_pre_ar_estimators": False,
            "ran_all_trials_estimators": False,
        },
        "d11_status": {
            "id": "D11",
            "status": "documented_unresolved",
            "paper": "three ICA components ocular+cardiac",
            "erpcore_subject_1": [2, 7],
            "action": (
                "Keep official precomputed averages with components 2 and 7 "
                "already removed. Do not invent a third component. MATLAB was "
                "not used; no MATLAB parity claimed."
            ),
        },
    }


def build_bundle() -> dict[str, Any]:
    root = subject1_dir()
    erp_stages = compare_erp_stages(root)
    ica = ica_forensics(root)
    averaging = {
        "eventlist": eventlist_trial_counts(root),
        "artifact_rejection": artifact_rejection_counts(root),
        "correct_response_only": True,
        "diff_waves_are_not_4condition_redisca_input": True,
    }
    bundle = {
        "track": "C",
        "branch": "cursor/paper-n170-preproc-f368",
        "matlab": None,
        "erp_core_scripts_commit": ERPCORE_COMMIT,
        "files": inventory_subject_files(root),
        "scripts": hash_scripts(),
        "channels": {
            "analysis": analysis_channel_set(),
            "set_stages": inspect_set_stages(root),
            "erp_labels": {
                name: erp_stages["stages"][name]["channel_labels"]
                for name in erp_stages["stages"]
            },
        },
        "erp_stages": erp_stages,
        "ica": ica,
        "averaging": averaging,
        "lpfilt": lpfilt_assessment(),
        "track_a_12_variants": track_a_variants_spec(),
        "preferred_erp_sha256_expected": PREFERRED_ERP_SHA256,
        "preferred_erp_sha256_observed": erp_stages["stages"][PREFERRED_ERP_NAME]["sha256"],
    }
    bundle["answers"] = answers(bundle)
    return bundle


def write_results(bundle: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def dump(name: str, payload: Any) -> None:
        path = out_dir / name
        path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[name] = str(path)

    dump(
        "summary.json",
        {
            "track": bundle["track"],
            "matlab": bundle["matlab"],
            "erp_core_scripts_commit": bundle["erp_core_scripts_commit"],
            "preferred_erp": PREFERRED_ERP_NAME,
            "preferred_erp_sha256": bundle["preferred_erp_sha256_observed"],
            "answers": bundle["answers"],
            "d11": bundle["answers"]["d11_status"],
            "alternative_estimators_run": False,
            "lpfilt_used_as_silent_default": False,
            "invented_third_ica_component": False,
        },
    )
    dump(
        "files.json",
        {
            "subject_1": bundle["files"],
            "scripts": bundle["scripts"],
            "xlsx": {
                "sha256": bundle["ica"]["xlsx"]["sha256"],
                "url": ICA_XLSX_URL,
                "pinned_sha256": ICA_XLSX_SHA256,
                "path": bundle["ica"]["xlsx"]["path"],
            },
        },
    )
    dump("channels.json", bundle["channels"])
    dump("ica.json", bundle["ica"])
    dump("erp_stages.json", bundle["erp_stages"])
    dump("averaging.json", bundle["averaging"])
    dump("answers.json", bundle["answers"])
    dump("lpfilt.json", bundle["lpfilt"])
    dump("track_a_variants.json", bundle["track_a_12_variants"])
    return written
