# Simulations track report (Ossadtchi et al. 2024, Figs 3–6)

**Status:** `approximate` / **blocked for exact published numbers**
(missing named forward model, I_c, f_s, Υ_d; no AIRI simulation script; D13, D14).

This reconstruction uses only paper-stated generative facts plus the **narrowest
documented assumptions** below. It does **not** use student synthetic
benchmarks, does **not** use fsaverage, and was **not** tuned to the ~85%
hit-rate sentence.

Canonical estimator: `from redisca import ReDisCA`, unique pairs,
`demean_time=False` (printed Gram). `demean_time=True` is a labeled extra on
Fig. 4.

## Assumed-value table

| Name | Value used | Status | Rationale |
| --- | --- | --- | --- |
| `fs_hz` | 1000 | assumed | Paper prints T=200 ms, not f_s. 1000 Hz matches Kozunov/OSF MEG → 200 samples. 256 Hz (ERP CORE) was considered and not used. |
| `n_times` | 200 | assumed (implied) | 200 ms × 1000 Hz. |
| `I_c` | 40 | assumed | Unspecified. Modest. MEG has 80 epochs/subcategory; bioRxiv overlay “100 trials” is ambiguous (MC vs I_c). Not tuned. |
| `Υ_d` | symmetric Gaussian on unique i<j; σ = 0.05 × std(D0_upper, ddof=1); clip ≥0; diag=0 | assumed | Paper adds unspecified C×C noise to D0. |
| Butterworth | 6th-order, 2 Hz low-pass, `scipy` SOS + `sosfiltfilt` | cutoff/order paper; zero-phase assumed | Paper does not say filtfilt vs causal. |
| 1/f noise | FFT pink, S(f)∝1/f, DC=0, unit RMS/source, 1000 random vertices, constrained orientation | assumed reconstruction of Ossadtchi et al. 2018 | PSIICOS used 5th-order zero-phase bandpass (θ/α/β/γ) with 1/f band weights on a **20000**-vertex grid and longer epochs. That grid is not public; 200 ms cannot host well-resolved theta IIR bands. |
| Forward model | AD `tess_cortex_pial_low` (5002) + overlapping-spheres `Gain` (322×15006) | **blocked / approximate** | Only public candidate (OSF 8rk67). Paper does not name the mesh. D13. Never fsaverage. |
| Sensors | 204 planars: Gain[:306] MEG; drop index%3==2 of GRAD-GRAD-MAG triplets | assumed | Rows 306:322 are NaN. Row RMS is large/large/small. Channel.mat not on OSF. Mixing MAG+GRAD without whitening would ignore magnetometers. Paper MEG used 204 planars. |
| Orientation | constrained `GridOrient` (= tess `VertNormals`) | assumed | Paper uses an N×1 topography g_m. |
| SNR | per-trial γ so RMS(noiseless stacked conditions)/RMS(γ Υ) = SNR | paper “root mean powers”; stacking assumed | Eq. 15–16 discussion. |
| MNE λ | λ² = tr(GGᵀ)/(N SNR²), SNR=3 | assumed | Paper names MNE, not λ. SNR=3 matches the public sLORETA kernel field, not a simulation statement. |
| LCMV C | sample cov of concatenated demeaned trials; ridge = 0.05 × mean eigenvalue | assumed | Paper names LCMV, not the covariance recipe. |
| RSA S.T. pairing | pair trials by index l=1…I_c | assumed | Fig. 1b does not print the pairing rule. |
| ReDisCA cosine | \|gᵀ a\| / (‖g‖ ‖a‖) | sign assumed | Printed Eq. 13 is signed; GEP/Haufe sign is free. |
| Fig. 6 SNR | 0.2 | assumed | Unspecified in the paper. Reuse Fig. 5 body SNR. |
| Fig. 5 C | generate C=6; evaluate C=6 and C=5 (first 5 conditions) | both; D14 | Caption C=6 vs body C=5. |
| Fig. 6 C | subset of the C=6 condition set (first C) | paper “subset” | §2.4.3. |
| Mixing M | C×C i.i.d. N(0,1); held fixed across MC | paper | Single-source: one M. Multi-source: four M_p. |
| RNG | numpy Generator PCG64; master_seed=20240904; secondary_seed=20240915 (20 MC ReDisCA-only, Fig. 4 SNR=0.1) | assumed | Paper gives no seeds. Secondary seed is a variability check, not a tuning loop. |

## Figure status

| Figure | Numeric status | Why |
| --- | --- | --- |
| Fig. 3 | approximate (visual exemplar) | Stochastic RDM; missing Υ_d law and mesh. |
| Fig. 4 ROC / traces | approximate; blocked for exact AUC / 85% hit | Missing mesh, I_c, f_s, Υ_d, 1/f recipe. High-SNR panel uses preprint overlay 0.2. |
| Fig. 5 | approximate; C=5 **and** C=6 recorded | D14 caption/body conflict. SNR 0.4 preprint / 0.2 body. |
| Fig. 6 | approximate | Same gaps; SNR assumed 0.2. Paper claim: ReDisCA mean median < 2 cm at C=6 — compared, not fitted. |

## Commands

```bash
python paper/reproduction/common/download_osf.py source-models
python -m pytest paper/reproduction/simulations/tests -q
python paper/reproduction/simulations/run.py --quick
python paper/reproduction/simulations/run.py
```

## Numeric results (approximate; 100 MC; seed 20240904)

See `paper/results/simulations/summary.json`. Runtime: `python3 paper/reproduction/simulations/run.py` (~108 min, 4 cores). RSA was **not** reduced (100 MC × 5002 vertices). Zero ReDisCA fit failures. These are **not** published-panel numbers.

### Fig. 4 ROC (C=5)

Canonical path: `redisca_demean_false`. `demean_time=True` is a labeled extra.

| SNR | Method | AUC | TPR@FPR=0 | TPR@FPR≤0.01 | median error (cm) |
| --- | --- | --- | --- | --- | --- |
| 0.2 (preprint overlay) | ReDisCA `demean_time=False` | 0.880 | 0.002 | 0.397 | 0.61 |
| 0.2 | ReDisCA `demean_time=True` (extra) | 0.528 | 0.000 | 0.016 | 7.90 |
| 0.2 | MNE S.T. | 0.695 | 0.000 | 0.000 | 7.28 |
| 0.2 | MNE AV | 0.625 | 0.000 | 0.000 | 8.16 |
| 0.2 | BF S.T. | 0.509 | 0.000 | 0.009 | 7.39 |
| 0.2 | BF AV | 0.528 | 0.000 | 0.020 | 8.32 |
| 0.1 (published c,d) | ReDisCA `demean_time=False` | 0.874 | 0.000 | 0.381 | 0.86 |
| 0.1 | ReDisCA `demean_time=True` (extra) | 0.500 | 0.000 | 0.007 | 6.74 |
| 0.1 | MNE S.T. | 0.693 | 0.000 | 0.000 | 7.44 |
| 0.1 | MNE AV | 0.688 | 0.000 | 0.000 | 8.52 |
| 0.1 | BF S.T. | 0.478 | 0.000 | 0.008 | 7.40 |
| 0.1 | BF AV | 0.516 | 0.000 | 0.014 | 7.52 |

Secondary seed `20240915`, 20 MC, ReDisCA-only, SNR=0.1: AUC **0.845** (primary 100 MC: 0.874). The ReDisCA–RSA gap is large on both seeds. The published ~85% hit @ ~0 FA is **not** reproduced (TPR at FPR≤0.01 ≈ 0.38). Not tuned.

`demean_time=True` is near chance: 6th-order 2 Hz ERPs are near-DC, so temporal centering of pair differences removes most of the signal. That extra is **not** the paper Gram.

Qualitative: printed-Gram ReDisCA dominates all four RSA versions (paper ranking), not the 85% operating point.

### Fig. 5 four-source (P=4; C=5 and C=6 from the same C=6 draws)

Mean of per-MC medians over the four sources.

| SNR | C | ReDisCA mean median (cm) | frac <1 cm | median corr(a,g) | corr(w,g) | corr(RDM) | MNE S.T. (cm) | BF S.T. (cm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 5 | 4.14 | 0.233 | 0.294 | 0.001 | 0.955 | 6.93 | 8.10 |
| 0.4 | 6 | 3.77 | 0.242 | 0.304 | 0.001 | 0.953 | 6.77 | 7.77 |
| 0.2 | 5 | 3.85 | 0.250 | 0.287 | ~0 | 0.959 | 7.02 | 8.04 |
| 0.2 | 6 | 3.56 | 0.263 | 0.311 | 0.000 | 0.954 | 6.83 | 7.73 |

Qualitative matches: ReDisCA better than MNE/BF S.T.; patterns align better than weights; empirical RDMs highly correlated with targets. The paper’s “largest mass <1 cm” is **not** reproduced (≈24–26% <1 cm).

### Fig. 6 mean median error vs C (SNR **assumed 0.2**)

Paper claim: ReDisCA best at all C; mean median **< 2 cm at C=6**.

| C | ReDisCA mean median (cm) | frac <2 cm | MNE S.T. (cm) | BF S.T. (cm) |
| --- | --- | --- | --- | --- |
| 3 | 5.42 | 0.300 | 7.85 | 7.86 |
| 4 | 4.32 | 0.357 | 7.60 | 7.60 |
| 5 | 4.13 | 0.405 | 7.03 | 7.54 |
| 6 | **3.36** | 0.482 | 6.95 | 7.51 |

ReDisCA is best at every C and improves with C. The **< 2 cm at C=6** claim is **not** met (3.36 cm). Approximate; not tuned.

## What would unblock exact numbers

1. Author simulation script, or a named mesh/Gain (individual MRI vs template).
2. Printed I_c, f_s, Υ_d law, Butterworth causal vs filtfilt, 1/f recipe (PSIICOS bands vs FFT pink).
3. Resolution of Fig. 5 C=5 vs C=6 (D14) if only one was plotted.
4. Fig. 6 SNR.
