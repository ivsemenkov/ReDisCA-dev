"""Assumed and paper-stated simulation parameters.

Every value that is *not* printed in Ossadtchi et al. 2024 is marked
``assumed``. Do not treat this module as a paper methods dump.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# --- printed in the paper ---
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
PAPER_SNR_FIG4_HIGH = 0.2  # bioRxiv overlay (preprint-supported)
PAPER_SNR_FIG4_LOW = 0.1  # published body for panels c,d
PAPER_SNR_FIG5_HIGH = 0.4  # bioRxiv overlay
PAPER_SNR_FIG5_LOW = 0.2  # published body

# --- not printed; documented assumptions ---
ASSUMED_FS_HZ = 1000.0
# 200 ms at 1000 Hz. Alternative considered: 256 Hz (ERP CORE rate) -> 51 samples.
ASSUMED_N_TIMES = 200
ASSUMED_I_C = 40
# Modest trial count. MEG subcategories have 80 epochs; bioRxiv Fig. 4 overlay
# "100 trials" is ambiguous (MC vs I_c). Not tuned to the 85% hit-rate claim.
ASSUMED_UPSILON_D_REL_STD = 0.05
# Symmetric Gaussian on the unique triangle of D0, scale = this * sample SD of
# D0's unique entries; then clip negatives and zero the diagonal.
ASSUMED_UPSILON_D_MODEL = (
    "symmetric additive Gaussian on unique i<j entries; "
    "sigma = 0.05 * std(D0_upper, ddof=1); clip to >=0; diag=0"
)
ASSUMED_BUTTER_ZERO_PHASE = True  # scipy sosfiltfilt; paper does not say filtfilt vs causal
ASSUMED_PINK_PSD_EXPONENT = 1.0  # S(f) ∝ 1/f  => amplitude ∝ 1/sqrt(f); DC bin zeroed
ASSUMED_MNE_SNR = 3.0  # Tikhonov via trace(GG^T)/(N * SNR^2); matches OSF sLORETA kernel SNR, not a paper sim statement
ASSUMED_LCMV_REG_FRAC = 0.05  # ridge = frac * mean eigenvalue of the trial covariance
ASSUMED_CHANNEL_SET = "204_planar"
ASSUMED_ORIENTATION = "constrained_gridorient"
ASSUMED_FIG6_SNR = 0.2  # Fig. 6 SNR is unspecified; reuse Fig. 5 low SNR
MASTER_SEED = 20240904
SECONDARY_SEED = 20240915
FORWARD_STATUS = "approximate / blocked by missing source asset"
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
    master_seed: int = MASTER_SEED
    secondary_seed: int = SECONDARY_SEED
    n_mc_secondary: int = 20
    channel_set: str = ASSUMED_CHANNEL_SET
    rsa_n_mc: int | None = None  # None => same as n_mc
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["paper_stated"] = {
            "T_ms": PAPER_T_MS,
            "butter_order": PAPER_BUTTER_ORDER,
            "butter_cutoff_hz": PAPER_BUTTER_CUTOFF_HZ,
            "single_source_C": PAPER_SINGLE_SOURCE_C,
            "multi_source_P": PAPER_MULTI_SOURCE_P,
            "fig3_C": PAPER_MULTI_SOURCE_C_METHODS,
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
        payload["assumed"] = assumed_value_table()
        payload["forward_status"] = FORWARD_STATUS
        payload["forward_candidate"] = FORWARD_CANDIDATE
        return payload


def assumed_value_table() -> list[dict[str, str]]:
    """Rows for TRACK_REPORT.md. Every non-paper value lives here."""
    return [
        {
            "name": "fs_hz",
            "value": str(ASSUMED_FS_HZ),
            "status": "assumed",
            "rationale": (
                "Paper prints T=200 ms, not fs. 1000 Hz matches Kozunov/OSF MEG. "
                "Gives n_times=200. Alternative 256 Hz (ERP CORE) was not used."
            ),
        },
        {
            "name": "n_times",
            "value": str(ASSUMED_N_TIMES),
            "status": "assumed (implied by fs and T_ms)",
            "rationale": "200 ms * 1000 Hz = 200 samples.",
        },
        {
            "name": "I_c",
            "value": str(ASSUMED_I_C),
            "status": "assumed",
            "rationale": (
                "Unspecified. Modest value 40. MEG has 80 epochs/subcategory; "
                "bioRxiv overlay '100 trials' is ambiguous (MC vs I_c). Not tuned."
            ),
        },
        {
            "name": "Upsilon_d",
            "value": ASSUMED_UPSILON_D_MODEL,
            "status": "assumed",
            "rationale": "Paper adds unspecified C×C noise to D0. Small symmetric Gaussian.",
        },
        {
            "name": "butterworth_implementation",
            "value": "scipy.signal.butter SOS + sosfiltfilt (zero-phase)",
            "status": "assumed",
            "rationale": "Paper: 6th-order Butterworth 2 Hz LPF on rows of Z. filtfilt vs causal not stated.",
        },
        {
            "name": "one_over_f_generator",
            "value": "FFT pink noise, S(f)∝1/f, DC=0, unit RMS per source, random vertices, constrained orientation",
            "status": "assumed reconstruction of Ossadtchi 2018 [43]",
            "rationale": (
                "PSIICOS used 5th-order zero-phase bandpass (theta/alpha/beta/gamma) with 1/f "
                "band weights on a 20000-vertex grid and longer epochs. That grid is not public; "
                "a 200 ms window cannot host well-resolved theta IIR bands. FFT 1/f is the "
                "narrowest '1/f noise' reconstruction on the AD 5002 mesh."
            ),
        },
        {
            "name": "forward_model",
            "value": FORWARD_CANDIDATE,
            "status": FORWARD_STATUS,
            "rationale": "Only public candidate on OSF 8rk67. Never fsaverage. D13.",
        },
        {
            "name": "sensors",
            "value": "204 planar gradiometers (Gain rows 0:306, drop every 3rd of GRAD-GRAD-MAG triplets)",
            "status": "assumed",
            "rationale": (
                "Gain is 322×15006; rows 306:322 are NaN (non-MEG). First 306 are finite MEG. "
                "Row RMS repeats large, large, small => GRAD, GRAD, MAG. Mixing MAG+GRAD without "
                "whitening would ignore magnetometers. Paper MEG analysis used 204 planars. "
                "Channel.mat is not on OSF."
            ),
        },
        {
            "name": "source_orientation",
            "value": "constrained: Gain 3-ori × GridOrient (= tess VertNormals)",
            "status": "assumed",
            "rationale": "Paper says N×1 topography g_m. Constrained normals are the Brainstorm default for this headmodel.",
        },
        {
            "name": "SNR_definition",
            "value": "gamma such that RMS(noiseless stacked conditions) / RMS(gamma * noise) = SNR, per trial",
            "status": "paper Eq. 15–16 discussion (root mean powers); stacking/trial grouping assumed",
            "rationale": "Paper: ratio of root mean powers of noiseless sensor matrix to noise matrix.",
        },
        {
            "name": "MNE_regularization",
            "value": f"lambda^2 = trace(G G.T) / (n_channels * SNR^2) with SNR={ASSUMED_MNE_SNR}",
            "status": "assumed",
            "rationale": "Paper names MNE, not lambda. SNR=3 matches the public constrained sLORETA kernel field, not a simulation statement.",
        },
        {
            "name": "LCMV_regularization",
            "value": f"ridge = {ASSUMED_LCMV_REG_FRAC} * mean eigenvalue of sample covariance of concatenated demeaned trials",
            "status": "assumed",
            "rationale": "Paper names LCMV, not the covariance estimator or loading.",
        },
        {
            "name": "RSA_ST_trial_pairing",
            "value": "pair trials by index l=1..I_c (equal I_c across conditions)",
            "status": "assumed",
            "rationale": "Fig. 1b: inverse per trial, then average squared distances. Pairing rule not printed.",
        },
        {
            "name": "ReDisCA_cosine_abs",
            "value": "Eq. 13 with |g^T a| / (||g|| ||a||)",
            "status": "assumed (sign)",
            "rationale": "Printed cosine is signed; Haufe/GEP pattern sign is free. Absolute cosine is required for localization.",
        },
        {
            "name": "fig6_snr",
            "value": str(ASSUMED_FIG6_SNR),
            "status": "assumed",
            "rationale": "Fig. 6 SNR is not stated. Reuse Fig. 5 body SNR=0.2 (four-source, more challenging).",
        },
        {
            "name": "fig5_C",
            "value": "generate C=6; evaluate C=6 and C=5 (first 5 conditions) — D14",
            "status": "both run; paper-internal conflict",
            "rationale": "Caption C=6, body C=5. Subset language in §2.4.3.",
        },
        {
            "name": "fig6_C_construction",
            "value": "subset of the C=6 condition set (first C conditions), not independent M of size C×C",
            "status": "paper-supported reading of 'subset of C=3,4,5'",
            "rationale": "Section 2.4.3.",
        },
        {
            "name": "mixing_M",
            "value": "C×C i.i.d. N(0,1); one M (single-source) or four M_p (multi-source); held fixed across MC",
            "status": "paper",
            "rationale": "S = M Z with Z in R^{C×T}.",
        },
        {
            "name": "canonical_ReDisCA",
            "value": "redisca.ReDisCA(demean_time=False); unique pairs; Haufe patterns",
            "status": "paper Gram + library pairs",
            "rationale": "demean_time=True is a labeled extra on Fig. 4 only.",
        },
        {
            "name": "rng",
            "value": f"numpy Generator PCG64; master_seed={MASTER_SEED}; secondary_seed={SECONDARY_SEED}",
            "status": "assumed",
            "rationale": "Paper does not give seeds. Secondary seed is a 20-MC ReDisCA-only check, not a tuning loop.",
        },
    ]


DEFAULT_CONFIG = SimulationConfig()
