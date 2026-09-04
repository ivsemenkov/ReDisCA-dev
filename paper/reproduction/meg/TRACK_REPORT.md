# MEG track report (Ossadtchi et al. 2024, Figs 12–17)

Status of numeric cells: **to be filled from `paper/results/meg/` after `run.py all`**.
Methods below are the executed contract, not a claim that the figures matched.

## Commands

```bash
python paper/reproduction/meg/run.py all
# equivalent split:
python paper/reproduction/meg/run.py paper
python paper/reproduction/meg/run.py airi
```

Documented Monte Carlo (this run):

| Quantity | Value | Source |
| --- | --- | --- |
| paper component *B* | 500 | unspecified in paper; documented here; *p*=count/*B* |
| paper time *Nmc* | 200 | unspecified in paper; FWER max-stat over time |
| AIRI SPoC *B* | 1000 | MATLAB `n_bootstrapping_iterations` |
| AIRI time *Nmc* | 100 | MATLAB `Nmc` |
| pair-order diagnostic | 5 shuffles × *B*=200 | diagnostic only |
| RNG | NumPy PCG64, seed 20240904 | `common.rng.numpy_generator` |

If AIRI *B* is reduced, that is recorded as `reduced_B` in
`airi_executable/*.json` and must not be described as the MATLAB default.

## Dual-path statuses

| Item | paper_faithful | airi_executable |
| --- | --- | --- |
| fig12 theoretical RDMs | binary 0/1 emitted; AIRI 0.1/1 emitted | AIRI numeric emitted |
| fig13 face | pending run | pending run |
| fig14 tool | pending run | pending run |
| fig15 meaning | pending run | pending run |
| fig16 non-binary RDM | AIRI 0.1/0.5/1 (only numeric source) | AIRI default `facevstool` |
| fig17 non-binary components | pending run | closest unmodified AIRI run; still not paper |
| airi-executable-meg-facevstool | n/a | pending run |

## Eigenvalues and *p*-values

Filled after the run from `summary.json` / per-figure JSON.

### paper_faithful (`ReDisCA(demean_time=False)`, unique Gram, 1501 samples)

| RDM (fill) | rank | λ (first 3) | *p* max\|λ\| (first 3) | empirical RDM Pearson comp1 | contrast peak (ms) |
| --- | --- | --- | --- | --- | --- |
| face binary | | | | | |
| tool binary | | | | | |
| meaning binary | | | | | |
| facevstool 0.1/0.5/1 | | | | | |

`demean_time=True` extra and AIRI 0.1/1 fill extra: see
`paper_faithful/demean_time_extra.json` and `rdm_fill_extra.json`.

λ under paper Eq. 7 **sum** would be *n_pairs* × library mean λ (D4). Filter
rays are invariant. Reported λ is the library/SPoC **mean** convention.

### airi_executable (directed MATLAB cov, 99–999 ms, 0.25–20 Hz)

| RDM | rank | λ (first 4) | SPoC random-phase *p* (first 4) | empirical RDM Pearson comp1 | contrast peak (ms) |
| --- | --- | --- | --- | --- | --- |
| face | | | | | |
| tool | | | | | |
| meaning | | | | | |
| facevstool (default) | | | | | |

## Actual peaks vs paper qualitative onsets

Paper numbers are qualitative targets, **not** force-matched.

| Figure | Paper prose | This run (paper_faithful) | This run (airi_executable) |
| --- | --- | --- | --- |
| 13a face c1 | 65 / 160 / 311 ms | | |
| 13b face c2 | 218 ms | | |
| 13c face c3 | 273 ms | | |
| 14a tool c1 | 210 ms | | |
| 15 meaning | from 160 ms | | |
| 17a non-binary | 202 ms | | |
| 17c non-binary | ~160 ms face peak | | |

## Path comparison (sign-aligned)

From `comparison/paper_vs_airi.json`. Do not read this as a mixed reproduction
figure. D1/D2/D4/D6/D8 all move the two fits.

| RDM | filter Pearson (c1) | pattern Pearson (c1) | subspace min cos | time-course Pearson on 99–999 ms |
| --- | --- | --- | --- | --- |
| face | | | | |
| tool | | | | |
| meaning | | | | |
| facevstool | | | | |

## Discrepancies exercised

| ID | How this track treats it |
| --- | --- |
| D1 | Unique pairs on paper path; directed 30 on AIRI path. λ compared only within a path. |
| D2 | Printed Gram `demean_time=False` vs MATLAB cov. `demean_time=True` extra on paper path. |
| D3 | Sample SD (`ddof=1`) on both; not a bug. |
| D4 | Both executable estimators mean-aggregate. Eq. 7 sum scale is reported as a labeled conversion. |
| D5 | Paper: condition-label permutation. AIRI: random-phase *B*=1000. |
| D5b | Paper: label-shuffle averages, FWER max-*T*. AIRI: *Nmc*=100 half-split on std data. Asterisks not compared. |
| D6 | Paper: full 1501 samples −500…+1000 ms. AIRI: 99–999 ms. |
| D7 | Paper Fig. 12 binary + Fig. 16 numeric. AIRI default `facevstool`. |
| D8 | Paper: no bandpass. AIRI: scipy `filtfilt` reconstructing butter(3) 0.25–20 Hz; not MATLAB parity. |
| D9 | Haufe patterns; invert-*W* undefined at rank 67. |
| D16 | File is 1501 samples; paper_faithful uses 1501, does not silently drop the last sample. |

## Pair-order diagnostic

AIRI random-phase *p*-values as a function of directed pair sequence:
see `airi_executable/pair_order_diagnostic.json`. **Not** a replacement test.

## SciPy vs MATLAB `filtfilt`

Documented in `prepare.prepare_provenance` / airi JSON `bandpass.matlab_parity=false`.
Gustafsson vs odd-padding. No MATLAB in this environment.

## Blocked / not claimed

- FieldTrip / `prepare4topoNMG` helmet topographies (not in AIRI repo).
- Bit-exact MATLAB `eig` / `rand` / `filtfilt`.
- Fig. 18 MUSIC (source-localization track).
- Forcing paper onset times.

## Environment

Python `redisca` library commit on this branch parent: `5a5c865`.
AIRI pin: `15bc19c`. Stock SPoC pin: `18e4754`.
Data: OSF `8rk67` `MEG_AD_run1.mat` + SPM labels. `.dat` not loaded.
