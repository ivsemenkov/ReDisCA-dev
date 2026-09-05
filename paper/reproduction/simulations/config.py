"""Paper-stated and assumed simulation parameters.

Every value that is *not* printed in Ossadtchi et al. 2024 is marked
``assumed``. ReDisCA constructor settings are not stored here; they live
only in ``paper.reproduction.common.method``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DeltaMode = Literal["literal_covariance", "norm_15pct"]
SnrGammaMode = Literal["per_trial", "global"]
NoiseLociMode = Literal["per_epoch", "fixed"]

PAPER_T_MS = 200
PAPER_BUTTER_ORDER = 6
PAPER_BUTTER_CUTOFF_HZ = 2.0
PAPER_SINGLE_SOURCE_C = 5
PAPER_MULTI_SOURCE_P = 4
PAPER_MULTI_SOURCE_C_METHODS = 6
PAPER_FIG6_C = (3, 4, 5, 6)
PAPER_N_MC = 100
PAPER_R_MAX_M = 0.01
PAPER_SIGMA_DELTA_REL = 0.15
PAPER_N_ONE_OVER_F = 1000
PAPER_MIN_SEP_M = 0.02
PAPER_SNR_FIG4_HIGH = 0.2
PAPER_SNR_FIG4_LOW = 0.1
PAPER_SNR_FIG5_HIGH = 0.4
PAPER_SNR_FIG5_LOW = 0.2

ASSUMED_FS_HZ = 1000.0
ASSUMED_N_TIMES = 200
ASSUMED_I_C = 40
ASSUMED_I_C_MEG_MATCHED = 80
ASSUMED_UPSILON_D_REL_STD = 0.05
ASSUMED_UPSILON_D_MODEL = (
    "symmetric additive Gaussian on unique i<j entries; "
    "sigma = 0.05 * std(D0_upper, ddof=1); clip to >=0; diag=0"
)
ASSUMED_BUTTER_ZERO_PHASE = True
ASSUMED_PINK_PSD_EXPONENT = 1.0
ASSUMED_MNE_SNR = 3.0
ASSUMED_LCMV_REG_FRAC = 0.05
ASSUMED_CHANNEL_SET = "204_planar"
ASSUMED_ORIENTATION = "constrained_gridorient"
FORWARD_STATUS = "approximate / paper does not name the simulation mesh"
FORWARD_CANDIDATE = (
    "OSF 8rk67 subject-AD tess_cortex_pial_low (5002 vtx) + "
    "headmodel_surf_os_meg overlapping-spheres Gain (322 x 15006, 3 ori). "
    "Paper does not name this mesh."
)


@dataclass(frozen=True)
class SimulationConfig:
    fs_hz: float = ASSUMED_FS_HZ
    n_times: int = ASSUMED_N_TIMES
    i_c: int = ASSUMED_I_C
    n_mc: int = PAPER_N_MC
    n_noise_sources: int = PAPER_N_ONE_OVER_F
    r_max_m: float = PAPER_R_MAX_M
    min_sep_m: float = PAPER_MIN_SEP_M
    sigma_delta_rel: float = PAPER_SIGMA_DELTA_REL
    upsilon_d_rel_std: float = ASSUMED_UPSILON_D_REL_STD
    butter_order: int = PAPER_BUTTER_ORDER
    butter_cutoff_hz: float = PAPER_BUTTER_CUTOFF_HZ
    butter_zero_phase: bool = ASSUMED_BUTTER_ZERO_PHASE
    pink_psd_exponent: float = ASSUMED_PINK_PSD_EXPONENT
    mne_snr: float = ASSUMED_MNE_SNR
    lcmv_reg_frac: float = ASSUMED_LCMV_REG_FRAC
    delta_mode: DeltaMode = "literal_covariance"
    snr_gamma_mode: SnrGammaMode = "per_trial"
    noise_loci_mode: NoiseLociMode = "per_epoch"
    fig5_generate_c: int = 6
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["paper_stated"] = {
            "T_ms": PAPER_T_MS,
            "butter_order": PAPER_BUTTER_ORDER,
            "butter_cutoff_hz": PAPER_BUTTER_CUTOFF_HZ,
            "single_source_C": PAPER_SINGLE_SOURCE_C,
            "multi_source_P": PAPER_MULTI_SOURCE_P,
            "fig6_C": list(PAPER_FIG6_C),
            "n_monte_carlo": PAPER_N_MC,
            "r_max_m": PAPER_R_MAX_M,
            "sigma_delta": "0.15 ||g||",
            "n_one_over_f": PAPER_N_ONE_OVER_F,
            "min_sep_m": PAPER_MIN_SEP_M,
            "snr_fig4": {"high_preprint": PAPER_SNR_FIG4_HIGH, "low_published": PAPER_SNR_FIG4_LOW},
            "snr_fig5": {"high_preprint": PAPER_SNR_FIG5_HIGH, "low_body": PAPER_SNR_FIG5_LOW},
            "I_c": None,
            "fs_hz": None,
            "upsilon_d": None,
            "fig6_snr": None,
            "forward_model": "unspecified",
        }
        payload["forward_status"] = FORWARD_STATUS
        payload["forward_candidate"] = FORWARD_CANDIDATE
        payload["redisca"] = "paper.reproduction.common.method.make_redisca (AIRI-SPoC only)"
        return payload


DEFAULT_CONFIG = SimulationConfig()
QUICK_CONFIG = SimulationConfig(n_mc=2, i_c=8)


def config_for_candidate(candidate_id: str, *, n_mc: int | None = None) -> SimulationConfig:
    """Return the generation config for a labeled simulation candidate.

    Defaults match the original frozen SIM-P1 reconstruction. Review-added
    candidates change only documented external generation knobs.
    """
    mc = PAPER_N_MC if n_mc is None else int(n_mc)
    specs: dict[str, dict[str, Any]] = {
        "SIM-P1": {},
        "SIM-P2": {"i_c": 80},
        "SIM-P3": {"butter_zero_phase": False},
        "SIM-P4": {"i_c": 100},
        "SIM-P5": {"delta_mode": "norm_15pct"},
        "SIM-P6": {"snr_gamma_mode": "global"},
        "SIM-P7": {"noise_loci_mode": "fixed"},
        "SIM-P8": {"fig5_generate_c": 5},
        "SIM-R1": {
            "i_c": 100,
            "delta_mode": "norm_15pct",
            "snr_gamma_mode": "global",
            "noise_loci_mode": "fixed",
        },
    }
    if candidate_id not in specs:
        raise ValueError(f"Unknown simulation candidate {candidate_id!r}")
    return SimulationConfig(n_mc=mc, **specs[candidate_id])


REVIEW_ADDED_SIM_CANDIDATES: tuple[str, ...] = (
    "SIM-P4",
    "SIM-P5",
    "SIM-P6",
    "SIM-P7",
    "SIM-P8",
    "SIM-R1",
)
ORIGINAL_SIM_CANDIDATES: tuple[str, ...] = ("SIM-P1", "SIM-P2", "SIM-P3")
