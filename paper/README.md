# Ossadtchi et al. 2024 — paper-branch reproduction

Ossadtchi, A., Semenkov, I., Zhuravleva, A., Kozunov, V., Serikov, O.,
& Voloshina, E. (2024). Representational dissimilarity component analysis
(ReDisCA). *NeuroImage*, 301, 120868.
https://doi.org/10.1016/j.neuroimage.2024.120868

This directory is the scientific reproduction program. It lives on the
permanent `paper` branch. The lightweight public library remains on `main`.
Do not merge `paper` into `main`. Do not change library semantics on this
branch merely to improve figure match.

Overnight integration review snapshot: commit `SNAPSHOT_SHA` (Tracks A–F
merged at `38466de`; this commit records independent numerical review).
Statuses below are from committed JSON and reports, not from TRACK_REPORT
prose.

**No MATLAB parity is claimed.** Historical AIRI/SPoC numerics are a
source-faithful Python reconstruction in
`paper/reproduction/common/source_faithful.py`. SciPy `eigh` / `filtfilt` /
`Generator` are not MATLAB `eig` / `filtfilt` / `rand`.

## Inference hierarchy (historical vs printed)

**PRIMARY historical reproduction** is executable stock-SPoC inference:

```text
random_phase_surrogate(z)
recompute surrogate eigenspectrum
statistic = max(abs(lambda))
p = count / B
B = 1000
```

Implemented by `common.source_faithful.fit_condition_averages(...,
inference="spoc_random_phase")`. Frozen N170 candidate:
`airi_directed` + `matlab_cov` on official `1_N170_erp_ar.erp`, 28 scalp
(`paper/results/n170/historical/leading_candidate.json`).

**SECONDARY:** literal printed paper §2.3 condition-label permutation of
the theoretical RDM. Record it. Do **not** call it the historical
reproduction oracle. The previous integration snapshot (`4813b38`) treated
§2.3 as primary; that priority was wrong for HISTORICAL reproduction.

Canonical library (`from redisca import ReDisCA`) remains a **separate**
path: unique pairs, printed Gram with `demean_time=False`. Do not change
library semantics. `source_faithful.py` must not (and does not) import
`redisca`. Unique+Gram library λ matches `source_faithful` unique+gram to
~1e-15. That is expected GEP identity, **not** a library bug. D1–D9 remain
design differences.

## Scope

Reproduce every paper result that can be faithfully reproduced from:

1. the published NeuroImage text,
2. author-supplied AIRI MATLAB (pinned commit),
3. stock SPoC (pinned commit),
4. public datasets and source-model assets,
5. the current Python `redisca` library (`main` pin `5a5c865`).

The goal is not “make plots look similar at any cost”. Useful outcomes
include exact numerical reproduction, qualitative reproduction, an
approximate reconstruction with documented missing dependencies, or a
classified discrepancy.

Never collapse (A) paper text, (B) AIRI MATLAB, (C) stock SPoC, (D) Python
`redisca`, (E) what actually reproduces a published panel.

Inventory and dual-path rules: `paper/reproduction_manifest.md`.
Reproduction matrix: `paper/results_summary.md` and
`paper/results_summary.json`. Hierarchy notes:
`paper/HISTORICAL_INVESTIGATION.md`.

## Environment

```bash
python3 -m pip install -e ".[dev,paper]"
```

Python ≥ 3.10. MATLAB is not required and was not used in the recorded
runs. Recorded environment (N170 / MEG / simulations / source loc JSON):

| Item | Recorded value |
| --- | --- |
| Python | 3.12.3 |
| numpy | 2.4.4 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.0 |
| redisca | 0.1.0 (`ReDisCA`; library default `demean_time=True`) |
| matplotlib | 3.11.1 (N170) |
| h5py / mne | 3.16.0 / 1.12.1 (simulations) |

`paper/reproduction/common/source_faithful.py` must not import `redisca`.
Integration checked the module AST and grep: it does not. Call it a
source-faithful Python reconstruction, not MATLAB parity.

Data cache (gitignored):

```text
.reproduction_data/
```

Override with `REDISCA_REPRODUCTION_DATA`.

## Data download

```bash
python paper/reproduction/common/download_osf.py meg-sensor
python paper/reproduction/common/download_osf.py source-models
python paper/reproduction/common/download_erpcore.py processed-subject-1
```

- `meg-sensor` — Kozunov/AIRI MEG run-1 planars + SPM labels (~1.2 GB).
  Do not load the companion `.dat`.
- `source-models` — AD `tess_cortex_pial_low` (5002 vtx), overlapping-spheres
  `Gain`, constrained sLORETA kernel, and the author-saved `filt15` topo
  (D17). Needed for simulations and source localization.
- `processed-subject-1` — ERP CORE N170 official averages for folder `"1"`
  from OSF `pfde9` (preferred file `1_N170_erp_ar.erp`).

Source localization also needs the MEG sensor files for the local Fig. 17
subspace fit. Simulations need source models only.

Official sources:

- Paper dataset landing page: https://osf.io/pfde9/ (**N170 only**)
- AIRI MEG + source models: https://osf.io/8rk67/
- ERP CORE N170 (Kappenman et al., 2021): OSF `pfde9` / parent `thsqg`
- AIRI code: https://github.com/AIRI-Institute/ReDisCA
  (`15bc19cdc76989da202714b257f6de4d26a42c51`)
- Stock SPoC: https://github.com/svendaehne/matlab_SPoC
  (`18e4754aec1411160fd5b7ef0db852f1e0a87d90`)
- ERP CORE scripts: https://github.com/lucklab/ERP_CORE
  (`c18b43d70d791ca914d90410afe4ff06d6f7f429`)

## Commands (one per track)

From the repository root, after the downloads above:

```bash
python -m pytest paper/reproduction/common/tests tests -q
python paper/reproduction/n170/run.py
python paper/reproduction/meg/run.py all
python paper/reproduction/simulations/run.py
python paper/reproduction/source_localization/run.py
```

Documented production flags (do not treat `--quick` as the recorded run):

```bash
python paper/reproduction/n170/run.py --B 1000 --step-ms 25 --seed 20240904
python paper/reproduction/meg/run.py all          # paper B=500, Nmc=200; AIRI B=1000, Nmc=100
python paper/reproduction/simulations/run.py      # 100 MC; not --quick
PYTHONPATH=src:paper/reproduction python paper/reproduction/source_localization/run.py --permutation-b 200
```

Overnight historical (already recorded; do not retune):

```bash
PYTHONPATH=src:paper/reproduction:paper/reproduction/n170 \
  python paper/reproduction/n170/historical/run.py --track all --B 1000 --n-track-b-seeds 20
python paper/reproduction/n170/historical_apply/run.py
python paper/reproduction/meg/historical_candidate/run.py paper-epoch
python paper/reproduction/n170/rdm_correlation/run.py
python paper/reproduction/n170/preprocessing_forensics/run.py
PYTHONPATH=src:paper/reproduction python paper/reproduction/meg/rank_audit/run.py
```

MEG **must** be run as separate `paper_faithful` and `airi_executable`
paths (`run.py paper` / `run.py airi` / `run.py all`). Never mix those
settings in one figure. Comparison metrics live in
`paper/results/meg/comparison/paper_vs_airi.json` and are not a mixed plot.

Canonical deterministic fits: `from redisca import ReDisCA` with
`demean_time=False` (printed Gram). Historical / AIRI-executable fits:
`common.source_faithful.fit_condition_averages` (no `redisca` fit).

## Expected runtime / compute

From TRACK_REPORTs of the recorded runs (not re-timed here):

| Track | Recorded compute | Notes |
| --- | --- | --- |
| Unit tests | see this review’s pytest count | `common` + library + per-track tests |
| N170 library path | **~13 s** | exact 24-permutation + B=1000 MC; 28 ch × 256 samples |
| N170 historical A+B | recorded overnight | 12 variants B=1000 + 20-seed Track B |
| MEG | wall time **not printed** | full MC, not `--quick`. MEG file is ~1.24 GB |
| MEG paper-epoch freeze | **~13 s** | directed+cov random-phase B=1000, no bandpass |
| Simulations | **~108 min, 4 cores** | 100 MC; RSA was **not** reduced. `--quick` is smoke only |
| Source localization | wall time **not printed** | local MEG fit + condition-label B=200 + Eq. 14 scan |

RNG: NumPy PCG64. N170 / simulations master seed `20240904`. MEG CLI seed
`20240904` splits streams. Source loc permutation seed `20240915`.
Paper text prints no seeds.

## Result locations

| Track | Code | Compact results |
| --- | --- | --- |
| Manifest | `paper/reproduction_manifest.md` | `paper/reproduction_manifest.json` |
| N170 library | `paper/reproduction/n170/` | `paper/results/n170/` |
| N170 historical A+B | `paper/reproduction/n170/historical/` | `paper/results/n170/historical/` |
| N170 preproc C | `paper/reproduction/n170/preprocessing_forensics/` | `paper/results/n170/preprocessing/` |
| N170 RDM corr D | `paper/reproduction/n170/rdm_correlation/` | `paper/results/n170/rdm_correlation/` |
| N170 Fig. 7/8 apply | `paper/reproduction/n170/historical_apply/` | `paper/results/n170/historical_apply/` |
| MEG library dual path | `paper/reproduction/meg/` | `paper/results/meg/{paper_faithful,airi_executable,comparison}/` |
| MEG historical freeze | `paper/reproduction/meg/historical_candidate/` | `paper/results/meg/historical_candidate/` |
| MEG rank 67/68 | `paper/reproduction/meg/rank_audit/` | `paper/results/meg/rank_audit/` |
| Simulations | `paper/reproduction/simulations/` | `paper/results/simulations/` |
| Source localization | `paper/reproduction/source_localization/` | `paper/results/source_localization/` |
| **This review** | this README | `paper/results_summary.md`, `paper/results_summary.json` |

PNG / NPZ under `paper/results/` are gitignored. Compact JSON is committed.

## Reproduction status

Allowed labels: `reproduced numerically` | `reproduced qualitatively` |
`approximate` | `blocked by missing source asset` | `paper/code discrepancy`
| `stochastic mismatch` | `not yet reproduced`.

**No paper figure is classified `reproduced numerically`.** Do not upgrade
Fig. 10 because λ is close and PRIMARY p1=0: printed corr 0.82 is unmatched,
and p1=0 is the historical SPoC test, not the printed §2.3 test. Do not
call the MEG paper-epoch freeze `reproduced numerically` for three
components: it got two. AIRI extras (99–999 ms, 0.25–20 Hz) are **not**
paper methods. Simulations (Figs 3–6) and Fig. 18 stay as previously
classified.

For N170/MEG figures the notes split: (1) deterministic match (λ, RDM
corr, peaks, patterns); (2) stochastic/inference match under PRIMARY
random-phase; (3) secondary printed-method condition-label result;
(4) preprocessing uncertainty; (5) printed-method vs executable-method
discrepancy.

### Status headlines (Figs 7, 10, 11, 13–15, 17)

| Figure | Status | Headline |
| --- | --- | --- |
| **Fig. 7** N170 meaning p-map | **approximate** | PRIMARY random-phase **p1=0.018 at 400 ms only** (p<0.05 at that single 25 ms-grid center). Neighbors 0.076 / 0.050. Pattern max-abs **Pz**, not occipital (energy 0.22). Secondary exact-24 still **8/24**. Not a continuous interval. |
| **Fig. 10** N170 face | **paper/code discrepancy** | Freeze λ1=**0.88010** vs printed 0.87209. PRIMARY p1=**0 in 20/20** seeds. Window corr still **~0.999 / 0.948**, **not 0.82**. Burst ~168 ms vs ~170. Secondary exact-24 p1=0.50 (floor). Not numerical. |
| **Fig. 11** N170 car | **approximate** | Freeze λ1=**0.91312** vs 0.91639; λ2=**0.79043** vs 0.77036. PRIMARY p1=0 in 20/20. p2 mean **0.0074** (min 0.003, max 0.014); **2/20 exactly 0.009** — printed p2 sits in the envelope. corr **0.999>0.99**. Unique-pair car p2 **~0.12–0.14 does NOT match**. Secondary floor 0.25 cannot produce 0.009. |
| **Fig. 13** MEG face | **paper/code discrepancy** | Paper-epoch freeze (no AIRI extras): **2** comps p<0.05 (p=0.006, 0.019, 0.114), not 3. Contrast peak 309 ms vs paper 160 ms. AIRI-executable extras: first 3 p<0.05 — **not paper methods**. Secondary label-perm: not 3. Rank **68**. |
| **Fig. 14** MEG tool | **paper/code discrepancy** | Paper-epoch freeze: **2** comps p<0.05 (0.006, 0.010, 0.057). AIRI extras: 3. Not numerical for three components. |
| **Fig. 15** MEG meaning | **paper/code discrepancy** | Paper-epoch freeze: **2** comps p<0.05 (0.001, 0.018, 0.061). AIRI extras: 3. |
| **Fig. 17** MEG non-binary | **paper/code discrepancy** | Paper-epoch freeze: **2** comps p<0.05 (0.010, 0.024, 0.104). AIRI extras: first 3 p<0.05 (c4=0.054). |

### Full status table

| Figure / result | Status | One-line evidence |
| --- | --- | --- |
| Table 1 SPoC correspondence | reproduced qualitatively | Methods identity; library uses unique pairs + Haufe, not invert-`W` |
| Fig. 1 source-space RSA diagrams | reproduced qualitatively | Visual/methods; RSA AV/S.T. baselines implemented in simulations |
| Fig. 2 ReDisCA diagram | reproduced qualitatively | Visual/methods; GEP + Haufe patterns on rank-deficient data (D9) |
| Fig. 3 simulated multi-source RDMs | approximate | Visual exemplar; unnamed mesh / Υ_d (D13) |
| Fig. 4 single-source ROC / traces | approximate | ReDisCA ranks above RSA; **~85% hit @ ~0 FA not recovered** (TPR@FPR=0 is 0 at SNR 0.1) |
| Fig. 5 four-source Monte Carlo | approximate | ReDisCA better than MNE/BF S.T.; **“largest mass <1 cm” not recovered** (~23–26% <1 cm) |
| Fig. 6 localization error vs C | approximate | ReDisCA best at every C; **mean median 3.36 cm at C=6, not <2 cm** |
| Fig. 7 N170 meaning p-map | approximate | See headline. JSON: `historical_apply/fig07_meaning_pmap.json` |
| Fig. 8 N170 meaning patterns | reproduced qualitatively | Adjacent 375/400/425 ms. Freeze max-abs Oz/Pz/Pz (400 ms not occipital). Secondary still 8/24. Empirical RDMs still show the 2–2 meaning split |
| Fig. 9 N170 face/car theoretical RDMs | reproduced qualitatively | 0/1 structure encoded from §4.2.1; figures are images, fill not printed |
| Fig. 10 N170 face-specific | paper/code discrepancy | See headline. 0.82 unresolved. Not upgraded to numerical |
| Fig. 11 N170 car-specific | approximate | See headline. Two-comp p<0.01 is reachable under PRIMARY directed-pair random-phase, not under unique-pair or §2.3 |
| Fig. 12 MEG theoretical RDMs | reproduced qualitatively | Binary 0/1 and AIRI 0.1/1 emitted; figures are images (D7) |
| Fig. 13 MEG face-specific | paper/code discrepancy | See headline. JSON: `meg/historical_candidate/summary.json` |
| Fig. 14 MEG tool-specific | paper/code discrepancy | See headline |
| Fig. 15 MEG meaning | paper/code discrepancy | See headline |
| Fig. 16 MEG non-binary RDM | reproduced qualitatively | AIRI `facevstool` 0.1/0.5/1 emitted; not Fig. 12a (D7) |
| Fig. 17 MEG non-binary components | paper/code discrepancy | See headline |
| `airi-executable-meg-facevstool` | approximate | Literal AIRI defaults ran (B=1000); SciPy ≠ MATLAB; **not paper methods** |
| Fig. 18 MEG MUSIC | approximate | Eq. 14 ran on public AD Gain; argmax **left cuneus / V2**, not paper right FG / insula / left IPS |
| `airi-source-loc-precomp` | reproduced numerically | `abs(W @ A1[:,3])` peak vertex 394 / lingual L. **Not Fig. 18.** |
| AIRI `music` `P=eye(1)` | paper/code discrepancy | Non-executable dimension error in committed MATLAB |
| AIRI `music` `P=eye(Nsns)` | approximate | Obvious fix; still `A1(:,4)` only; not Fig. 18 |

## Especially reported findings

### 1. Full N170 variant table (Track A)

Source: `paper/results/n170/historical/track_a_table.json` and
`paper/reproduction/n170/HISTORICAL_REPORT.md`. Official
`1_N170_erp_ar.erp`, 28 scalp, binary 0/1 RDMs, B=1000, PRIMARY =
`spoc_random_phase`. Twelve source-supported variants only.

**Face** (Fig. 10 window: 100 ms @ 200 ms). Printed λ1≈0.87209, p1=0, corr≈0.82.

| variant | pair | matrix | λ1 | PRIMARY p1 | corr wᵀRw | Faces peak |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| unique+Gram | unique | Gram | 0.88006 | 0.000 | 0.99988 | 171.9 ms |
| unique+cov | unique | cov | 0.83915 | 0.000 | 0.99858 | 168.0 ms |
| directed+Gram | directed | Gram | 0.92301 | 0.000 | 0.99988 | 171.9 ms |
| **directed+cov (freeze)** | directed | cov | **0.88010** | **0.000** | **0.99858** | 168.0 ms |

No variant has window corr near 0.82.

**Car @ 170 ms** (printed Fig. 11 application time). Printed λ1≈0.91639,
λ2≈0.77036, p2≈0.009.

| variant | pair | matrix | λ1, λ2 | PRIMARY p1, p2 | corr wᵀRw | Cars peak |
| --- | --- | --- | --- | --- | ---: | ---: |
| unique+Gram | unique | Gram | 0.88691, 0.79170 | 0.000, **0.141** | 0.99992 | 136.7 ms |
| unique+cov | unique | cov | 0.87063, 0.75365 | 0.000, **0.115** | 0.99909 | 140.6 ms |
| directed+Gram | directed | Gram | 0.93020, 0.83035 | 0.000, 0.009 | 0.99992 | 136.7 ms |
| **directed+cov (freeze)** | directed | cov | **0.91312, 0.79043** | 0.000, 0.003 | **0.99909** | 140.6 ms |

Unique-pair car p2 ~0.12–0.14 does **not** match printed 0.009. Directed
pairs change the random-phase null (z length 12 vs 6) even though filters
stay on the same rays.

Car @ 200 ms was enumerated; it is not the printed application time.
Closest car λ1 overall is directed+cov @ 200 ms (0.91489). Full 12-row
table: `track_a_table.json`.

### 2. Face printed-number comparison (freeze)

| Item | Printed | Freeze (directed+cov) |
| --- | ---: | ---: |
| λ1 | 0.87209 | 0.88010 (Δ +0.008) |
| PRIMARY p1 | 0 | **0** (20/20 Track B seeds) |
| window corr wᵀRw | 0.82 | 0.99858 |
| window corr ‖Δu‖² (undemeaned) | 0.82 | 0.94819 |
| Faces peak | ~170 ms | 167.97 ms |
| secondary exact-24 p1 | (implied significant) | 0.50 |

### 3–4. Car printed-number comparison and p2≈0.009 (Track B)

| Item | Printed | Freeze (directed+cov @ 170 ms) |
| --- | ---: | ---: |
| λ1 | 0.91639 | 0.91312 (Δ −0.003) |
| λ2 | 0.77036 | 0.79043 (Δ +0.020) |
| PRIMARY p1 | 0 | **0** (20/20) |
| PRIMARY p2 | 0.009 | Track A seed 0.003; Track B mean **0.0074**, min 0.003, max 0.014, median 0.0075 |
| window corr | >0.99 | 0.99909 |
| secondary exact-24 | (implied p<0.01) | 0.25 / 0.25 (floor; **cannot** equal 0.009) |

Track B 20-seed car p2 distribution
(`paper/results/n170/historical/track_b.json`):
`0.009, 0.009, 0.003, 0.005, 0.008, 0.004, 0.011, 0.006, 0.005, 0.010, 0.007, 0.008, 0.005, 0.005, 0.010, 0.006, 0.010, 0.008, 0.014, 0.005`.

Exactly 0.009: **2/20**. Printed 0.009 lies inside min–max and the 5–95%
interval. It is **not** a one-seed accident for this candidate. It is also
**not** recovered by unique-pair random-phase.

### 5. Fig. 7 p<0.05 near 400 ms

PRIMARY **yes at 400 ms only** (p1=0.018). Neighbors 375/425 ms: 0.076 /
0.050. Isolated single window on the 25 ms grid; step was not narrowed.
Secondary exact-24: still 8/24. Pattern max-abs **Pz** (occipital energy
0.218). JSON: `paper/results/n170/historical_apply/`.

### 6. Empirical-RDM correlation 0.82

**Unresolved.** Track D
(`paper/reproduction/n170/RDM_CORR_REPORT.md`): no paper/Eq-named
correlation on official `1_N170_erp_ar.erp` is 0.82. Unique-triangle
Pearson of the face window is 0.99988 (Gram) / 0.99858 (cov native).
Full-epoch unique Pearson 0.94466 is closer to 0.82 than the window but
still not 0.82, and it would make Fig. 11 worse (car full-epoch 0.486).
Eq. 2 sample-SD inner product is ~0.833 on **both** face and car windows,
so the car control **kills** that reading of 0.82. Flattening the 4×4 /
AIRI grown `corrcoef` stays ~1. Classification:
unresolved historical preprocessing / implementation, **not** a
canonical-library bug.

### 7. Preprocessing provenance

Keep `1_N170_erp_ar.erp` (SHA-256 `53e74e93…9bbc72`). lpfilt is an ERP
CORE plotting extra, **not** a paper analysis statement; no estimator was
run on it. ICA: official subject `"1"` list is components **2 and 7**
only. **No third IC invented** (D11 remains documented/unresolved: paper
says “three ocular+cardiac”). P9/P10 absent from the `.erp`; do not
interpolate. EOG not added back. Report:
`paper/reproduction/n170/PREPROCESSING_REPORT.md`.

### 8. Frozen-candidate MEG (paper-epoch vs AIRI-executable reuse)

Same freeze: directed + matlab_cov + random-phase B=1000. Rank **68**.

**Paper epoch** (−500…+1000 ms, 1501 samples, **no** AIRI bandpass) —
this is the historical application without AIRI extras:

| RDM | λ₁…λ₃ | PRIMARY p (B=1000) | first-3 p<0.05 |
| --- | --- | --- | ---: |
| face | 0.862, 0.796, 0.694 | 0.006, 0.019, 0.114 | **2** |
| tool | 0.818, 0.805, 0.738 | 0.006, 0.010, 0.057 | **2** |
| meaning | 0.845, 0.787, 0.731 | 0.001, 0.018, 0.061 | **2** |
| facevstool | 0.808, 0.766, 0.697 | 0.010, 0.024, 0.104 | **2** |

**AIRI-executable reuse** (same estimator **plus** 99–999 ms and
0.25–20 Hz SciPy filtfilt — **not paper methods**): first three
components p<0.05 on face / tool / meaning / facevstool
(face p=0.006, 0.008, 0.014, 0.032). That **count** matches the published
three-component figures; the extras, not the N170 freeze alone, recover it.

Library `paper_faithful` unique+Gram + §2.3 `max|λ|` still yields 0–1
significant components.

### 9. Rank 67/68

Local Cxx/Rbar is **68** on both paper_faithful and airi_executable paths
at `tol = λ_max·1e-6`. Author-saved A1 is **204×67**. λ₆₈/λ_max is
4.36×10⁻⁶ (paper) and 2.48×10⁻⁶ (AIRI) vs cutoff 1e-6; λ₆₉ is ~10⁻⁸.
MATLAB-eig-only flip: **not_borderline** on these matrices. Do not force
local rank to 67. Report: `paper/reproduction/meg/RANK_AUDIT.md`.

### 10. Canonical-library bug

**None found.** D1–D9 remain design differences. Overnight: unique+Gram
library λ matches `source_faithful` unique+gram to ~1e-15
(`library_unique_gram_sanity` in the unique-Gram variant JSON). Near-perfect
6-entry two-level RDM correlation is the expected GEP optimum, not a bug.

## Paper vs AIRI implementation discrepancies

Verified against the NeuroImage text, AIRI `15bc19c`, stock SPoC `18e4754`,
and library `5a5c865`. Full write-up:
`paper/reference/source_notes/discrepancies.md`. Starting hypotheses 1, 2,
4, 5, 6, 7: **confirmed**. Hypothesis 3 (target SD): **same convention**,
not a bug.

| ID | Topic | Paper | AIRI / SPoC / library | This review |
| --- | --- | --- | --- | --- |
| D1 | Pairs | unique triangle | AIRI `i≠j` directed (30 vs 15; N170 12 vs 6). Library unique | Symmetric D: filters stay on the same rays; λ scales with sample SD. Random-phase **does** change with pair order (car p2 ~0.12 unique vs ~0.009 directed) |
| D2 | Pair matrix | unscaled Gram | MATLAB `cov` (demean + `/(T-1)`). Library: demean optional, no `T-1` | `1/(T-1)` cancels in the GEP; **demeaning does not**. Historical freeze uses `matlab_cov` |
| D3 | Target SD | unspecified | MATLAB `std` = library `ddof=1` | **Not a library bug** |
| D4 | Aggregation | Eq. 7 **sum** (prose: average) | SPoC/library **mean** | Filter rays invariant; quote λ only within one convention |
| D5 | Component p | §2.3 condition-label permutation | SPoC random-phase, `max\|λ\|`, `p=count/B` (0 possible). Library: none | **PRIMARY historical = SPoC random-phase.** §2.3 is **secondary**. MEG “three significant components” on the AIRI path is this test **plus** AIRI extras, not paper methods |
| D5b | MEG time p | subcategory-label shuffle, FWER max-T | `Nmc=100` half-split, pointwise, std-normalized data | Asterisks are **not** compared across paths |
| D6 | MEG window | entire 1500 ms (−500…+1000) | `trange=600:1500` → **99–999 ms** | Dual path. File is **1501** samples (D16). Paper-epoch freeze uses 1501, no crop |
| D7 | Default RDM | Fig. 12 then Fig. 16 | `ThRDMArr(2)='facevstool'` (0.1/0.5/1) | AIRI default is Fig. 16-like, **not** Fig. 12a |
| D8 | MEG filter | none stated | butter(3) 0.25–20 Hz filtfilt | Paper-epoch freeze: no bandpass. AIRI path: SciPy `filtfilt`, not MATLAB |
| D9 | Patterns | `A=W^{-1}` | Haufe; MEG rank 68 < 204 | Invert-`W` undefined. All executable paths use Haufe |
| D11 | N170 ICA | “three ocular+cardiac” | ERP CORE subject 1: comps **2, 7** | Official averages used; a third component is not invented. **Unresolved wording** |
| D12 | N170 code | paper + ERP CORE | no AIRI N170 script | Confirmed |
| D13 | Simulations | unnamed forward model | no AIRI script | Public AD Gain is a **hypothesis**. Never fsaverage |
| D14 | Fig. 5 C | caption C=6 vs body C=5 | — | Both recorded; not silently “fixed” |
| D15 | Fig. 18 | MUSIC of Fig. 17 subspace | default sLORETA of component 4 + index vector | On this kernel `megplanarbst` **is** 204 GRAD |
| D16 | Sample count | 204 × 1500 | file 204 × **1501** | paper_faithful / paper-epoch freeze use 1501 |
| D17 | Saved topo | — | script `return`s before `save`; OSF `filt15` | Precomp is author-saved `A1` (204×67), not a vanilla script output |
| D18 | `toolvsface` | — | named but unimplemented (zero RDM) | Not used |

## Integration QC (did not rewrite track science)

Looked for, and did **not** find: `redisca` imported inside
`source_faithful.py`; student N170/synthetic code used as an oracle;
silent retuning to 0.82 / 85% hit / <2 cm; paper/AIRI settings mixed in
one figure; fsaverage substituted for AD Gain; a canonical-library GEP bug.

Did find, and classified rather than “fixed”:

- PRIMARY random-phase on the directed+cov freeze recovers face/car p1=0
  (20/20) and a car p2 envelope containing 0.009. Unique-pair random-phase
  car p2 ~0.12 does not. §2.3 exact-24 **cannot** produce p=0.009 at C=4.
- Face corr **0.82 is unmatched** on this ERP under every named definition.
- Fig. 7 PRIMARY p<0.05 at 400 ms only; pattern is Pz; secondary still 8/24.
- MEG paper-epoch freeze: **2** p<0.05 components, not 3. AIRI extras get 3
  and are not paper methods.
- Rank 68 locally vs author A1 67: not a SciPy/MATLAB eig flip of these Cxx.
- Simulations: ReDisCA > RSA ranking holds; published operating points do
  not. `tuned_to_85pct_hit_rate: 0`.
- Fig. 18 argmax is left cuneus, not the paper’s qualitative peaks.
- Canonical library: **no bug** identified.

Remaining unresolved: Fig. 10 corr 0.82; MEG 2 vs 3 without AIRI extras;
D11 ICA wording; Fig. 18 anatomy.

See `paper/WORKER_CONTRACT.md` for ownership. See
`paper/results_summary.md` for the per-item matrix.
