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

## Numeric results

Filled after `run.py` completes. See `paper/results/simulations/summary.json`.

### Fig. 4 AUC / TPR (placeholder until run)

| Run | Method | AUC | TPR@FPR≤0.01 |
| --- | --- | --- | --- |
| (pending) | | | |

### Fig. 5 mean median error (m)

| SNR | C | ReDisCA | MNE S.T. | BF S.T. |
| --- | --- | --- | --- | --- |
| (pending) | | | | |

### Fig. 6 mean median error (m)

| C | ReDisCA | MNE S.T. | BF S.T. |
| --- | --- | --- | --- |
| (pending) | | | |

## What would unblock exact numbers

1. Author simulation script, or a named mesh/Gain (individual MRI vs template).
2. Printed I_c, f_s, Υ_d law, Butterworth causal vs filtfilt, 1/f recipe (PSIICOS bands vs FFT pink).
3. Resolution of Fig. 5 C=5 vs C=6 (D14) if only one was plotted.
4. Fig. 6 SNR.
