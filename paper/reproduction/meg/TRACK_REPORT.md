# MEG track report (Ossadtchi et al. 2024, Figs 12–17)

Run: `python paper/reproduction/meg/run.py all`  
Seed: 20240904 (NumPy PCG64). MATLAB not used. SciPy `filtfilt` ≠ MATLAB `filtfilt` (D8).  
Library: `from redisca import ReDisCA` on `paper_faithful` only.  
AIRI: `common.source_faithful.fit_condition_averages` / `airi_bandpass_trials` (no `redisca` fit).

## Commands

```bash
python paper/reproduction/meg/run.py all
python paper/reproduction/meg/run.py paper
python paper/reproduction/meg/run.py airi
python -m pytest paper/reproduction/meg/tests -q
```

Documented Monte Carlo (this run; not `--quick`):

| Quantity | Value | Source |
| --- | --- | --- |
| paper component *B* | **500** | unspecified in paper; *p*=count/*B* (0 possible) |
| paper time *Nmc* | **200** | unspecified in paper; FWER max-stat over time |
| AIRI SPoC *B* | **1000** | MATLAB default; **not reduced** |
| AIRI time *Nmc* | **100** | MATLAB `Nmc` |
| pair-order diagnostic | 5 shuffles × *B*=200 | diagnostic only |
| data | 204 planars × 1501 × 880; 80×6 used | `.dat` not loaded |

Rank of \(\bar R\) / `Cxx`: **68** on both paths (AIRI OSF topo file is 204×67; scipy `eigh` vs MATLAB `eig`).

## Dual-path statuses

| Item | paper_faithful | airi_executable |
| --- | --- | --- |
| fig12 theoretical RDMs | **emitted** binary 0/1 and AIRI 0.1/1 | **emitted** AIRI numeric |
| fig13 face | **partial** | **numeric run complete** (not paper methods) |
| fig14 tool | **partial** | **numeric run complete** |
| fig15 meaning | **partial** | **numeric run complete** |
| fig16 non-binary RDM | AIRI 0.1/0.5/1 emitted | AIRI default `facevstool` |
| fig17 non-binary components | **partial** | closest unmodified AIRI run; still not paper |
| airi-executable-meg-facevstool | n/a | **ran** *B*=1000, *Nmc*=100 |

“Partial” = the labeled estimator ran, but paper-described condition-label permutation with a `max|λ|` null does **not** yield three *p*<0.05 components (the published figure count). AIRI random-phase *does* yield three–four *p*<0.05 components. That is D5, not a silent fix.

## Eigenvalues and component *p*-values

### paper_faithful — `ReDisCA(demean_time=False)`, unique unordered pairs, unscaled Gram, full −500…+1000 ms, no bandpass

| RDM (fill) | rank | λ₁, λ₂, λ₃ | *p* max\|λ\| *B*=500 | *p* matched (exploratory) | emp. RDM Pearson c1 | FWER onset c1 (ms) | contrast peak c1 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| face binary | 68 | 0.877, 0.786, 0.691 | **0, 1, 1** | 0, 0.118, 0.32 | 0.9995 | 107 | 308 |
| tool binary | 68 | 0.857, 0.789, 0.723 | **0.478, 1, 1** | 0.478, 0.070, 0.000 | 0.9999 | 165 | 384 |
| meaning binary | 68 | 0.870, 0.775, 0.717 | **0.204, 1, 1** | 0.204, 0.136, 0.056 | 0.9996 | 128 | 315 |
| facevstool 0.1/0.5/1 | 68 | 0.836, 0.753, 0.684 | **0.302, 1, 1** | 0.302, 0.116, 0.130 | 0.966 | 113 | 564 |

Eq. 7 **sum** scale (D4): multiply λ by *n_pairs*=15. Filter rays unchanged. Reported λ uses the library **mean**.

`max|λ|` is family-wise over the full rank-68 spectrum. With slowly decaying λ, components 2–3 almost always have *p*=1 under that null. The paper’s “first three significant components” is **not** reproduced by this reading of §2.3. Matched-component *p* is still ≥0.05 for face c2–c3.

D7 extra: AIRI 0.1/1 fill vs binary 0/1 is **identical** after sample-SD z-scoring for the two-level RDMs (face/tool/meaning). Filter Pearson = 1. The 0.1 vs 0 fill does not move the GEP when the high/low pair partition is unchanged.

`demean_time=True` extra (D2): first-three sign-aligned filter Pearson 0.80 (face c1), 0.99 (tool c1), 0.96 (meaning c1), 0.92 (facevstool c1). Demeaning moves λ₁ more than the later filters.

### airi_executable — directed *i≠j* (30), MATLAB cov, 99–999 ms, butter(3) 0.25–20 Hz scipy filtfilt

| RDM | rank | λ₁…λ₄ | SPoC random-phase *p* *B*=1000 | emp. RDM Pearson c1 | AIRI-asterisk onset c1 (ms) | contrast peak c1 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| face | 68 | 0.880, 0.859, 0.833, 0.806 | **0.006, 0.008, 0.014, 0.032** | 0.998 | 253 | 319 |
| tool | 68 | 0.873, 0.861, 0.834, 0.823 | **0.001, 0.003, 0.008, 0.010** | 0.999 | 100 | 424 |
| meaning | 68 | 0.874, 0.855, 0.832, 0.812 | **0.005, 0.012, 0.018, 0.026** | 0.998 | 296 | 499 |
| facevstool (default) | 68 | 0.848, 0.831, 0.810, 0.782 | **0.016, 0.019, 0.028, 0.054** | 0.965 | 268 | 321 |

First three components *p*<0.05 on all four AIRI RDMs; facevstool c4 *p*=0.054. This **count** matches the paper’s “three components” better than the paper-described permutation — but it is the **AIRI/SPoC test (D5)**, not §2.3.

D1 scale: directed vs unique sample-SD ratio \(\sqrt{28/29}\approx 0.983\). Small for *C*=6.

## Actual peaks vs paper qualitative onsets

Paper numbers are **not** force-matched.

| Figure | Paper prose | paper_faithful (this run) | airi_executable (this run) |
| --- | --- | --- | --- |
| 13a face c1 | from **65** ms; peak **160** ms; again from **311** | FWER onset **107** ms; contrast peak **308** ms; class1 peak 629 ms | random-phase *p*=0.006; contrast peak **319** ms; class1 peak 518 ms |
| 13b face c2 | **218** ms | FWER onset **273** ms; contrast peak 805 ms | *p*=0.008; contrast peak 184 ms; class1 peak **191** ms |
| 13c face c3 | from **273** ms | FWER onset **89** ms; contrast peak 174 ms; class1 peak **197** ms | *p*=0.014; contrast peak 289 ms |
| 14a tool c1 | **210** ms | FWER onset **165** ms; contrast peak 384 ms; class1 peak **391** ms | *p*=0.001; class1 peak **450** ms |
| 15 meaning | from **160** ms | FWER onset **128** ms; contrast peak 315 ms; class1 peak **119** ms | *p*=0.005; contrast peak 499 ms |
| 17a non-binary | tools vs faces from **202** ms | FWER onset **113** ms; contrast peak 564 ms | *p*=0.016; contrast peak 321 ms; class2 (tools) peak **161** ms |
| 17c non-binary | face peak ~**160** ms | c3 class2 peak **140** ms; contrast peak 290 ms | c1 class2 peak **161** ms |

On the **99–999 ms overlap**, the mean-of-six-traces peak for AIRI face c1 is **160 ms** (`comparison/paper_vs_airi.json`). That is the closest numeric neighbour of the published 160 ms face peak, and it lives on the AIRI-window path, not on the paper-faithful full-epoch contrast.

Do **not** compare paper FWER asterisks with AIRI half-split asterisks (D5b).

## Path comparison (sign-aligned; not a mixed figure)

From `comparison/paper_vs_airi.json`. Filters are **not** interchangeable.

| RDM | filter Pearson c1–c3 | pattern Pearson c1–c3 | subspace min cos (filters) | time-course mean Pearson on 99–999 ms |
| --- | --- | --- | --- | --- |
| face | 0.42, 0.24, 0.18 | 0.50, 0.23, 0.69 | 0.42 | 0.48, 0.52, 0.55 |
| tool | 0.68, 0.75, 0.69 | 0.43, 0.07, 0.53 | 0.009 | 0.59, 0.55, 0.25 |
| meaning | 0.79, 0.79, 0.44 | 0.34, 0.72, 0.44 | 0.49 | 0.37, 0.48, 0.12 |
| facevstool | 0.61, 0.37, 0.39 | 0.38, 0.29, 0.96 | 0.52 | 0.44, 0.48, 0.85 |

Leading λ is similar (~0.84–0.88) across paths; later λ decay faster on the paper Gram / full-epoch path. D2 (demean) + D6 (window) + D8 (bandpass) dominate filter disagreement. D1 λ scale is ~2%. D4 is a labeled conversion only.

## Pair-order diagnostic (AIRI `facevstool`, not a replacement test)

Default directed-loop order, diagnostic *B*=200: *p* = (0.010, 0.015, 0.020, 0.045).  
Five pair-sequence shuffles: *p*₁ in {0, 0, 0, 0.005, 0.005}. Random-phase *p* **moves with the accidental FFT of the length-30 z sequence** (D5 / stock SPoC). Do not treat default-order *p* as unique.

## Discrepancies exercised

| ID | This run |
| --- | --- |
| D1 | Unique 15 vs directed 30. λ compared only within a path. Scale factor ~0.983. |
| D2 | Printed Gram vs MATLAB cov. `demean_time=True` extra recorded. |
| D3 | Sample SD on both; not a bug. |
| D4 | Mean aggregation on both executable estimators. Sum conversion = ×15 (paper) / ×30 (AIRI). |
| D5 | Paper permutation *B*=500 vs SPoC random-phase *B*=1000. **Component significance count disagrees.** |
| D5b | Paper FWER *Nmc*=200 vs AIRI half-split *Nmc*=100 on channel-time std over 880 trials. Asterisks not compared. MATLAB `rpm` indexing hazard documented in JSON. |
| D6 | 1501 samples −500…+1000 vs 901 samples 99–999 ms. |
| D7 | Fig. 12 binary (+ identical 0.1/1 z-score) then Fig. 16 `facevstool`. AIRI default is Fig. 16-like. |
| D8 | Paper: no bandpass. AIRI: scipy `filtfilt` butter(3) 0.25–20 Hz; not MATLAB parity. |
| D9 | Haufe; invert-*W* undefined at rank 68. |
| D16 | File is 1501 samples; paper_faithful uses 1501. |

## SciPy vs MATLAB `filtfilt`

`scipy.signal.filtfilt`, `padtype='odd'`, default `padlen=3*(max(len(a),len(b))-1)`. MATLAB Signal Processing Toolbox uses the Gustafsson method / MATLAB padding. **Not bit-exact.** No MATLAB in this environment.

## Blocked / not claimed

- FieldTrip / `prepare4topoNMG` helmet topographies (not in AIRI repo). Patterns are 12×17 planar heatmaps + planar-pair RMS.
- Bit-exact MATLAB `eig` / `rand` / `filtfilt`.
- Fig. 18 MUSIC (source-localization track).
- Paper onset times as numeric targets.
- Three *p*<0.05 paper_faithful components under `max|λ|` permutation.

## Environment

- Library parent commit: `5a5c865`.
- AIRI pin: `15bc19c`. Stock SPoC pin: `18e4754`.
- Data: OSF `8rk67` `MEG_AD_run1.mat` + SPM labels. `.dat` not loaded.
- Compact metrics: `paper/results/meg/**/*.json`.
- Figures PNG / arrays NPZ: generated, gitignored.
