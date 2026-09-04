# Ossadtchi et al. 2024 — paper-branch reproduction

Ossadtchi, A., Semenkov, I., Zhuravleva, A., Kozunov, V., Serikov, O.,
& Voloshina, E. (2024). Representational dissimilarity component analysis
(ReDisCA). *NeuroImage*, 301, 120868.
https://doi.org/10.1016/j.neuroimage.2024.120868

This directory is the scientific reproduction program. It lives on the
permanent `paper` branch. The lightweight public library remains on `main`.
Do not merge `paper` into `main`. Do not change library semantics on this
branch merely to improve figure match.

Integration review snapshot: merge commit `0a2d9ed` (N170, MEG, simulations,
and source-localization tracks). Statuses below are from committed JSON and
code, not from TRACK_REPORT prose.

**No MATLAB parity is claimed.** Historical AIRI/SPoC numerics are a
source-faithful Python reconstruction in
`paper/reproduction/common/source_faithful.py`. SciPy `eigh` / `filtfilt` /
`Generator` are not MATLAB `eig` / `filtfilt` / `rand`.

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
`paper/results_summary.json`.

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

MEG **must** be run as separate `paper_faithful` and `airi_executable`
paths (`run.py paper` / `run.py airi` / `run.py all`). Never mix those
settings in one figure. Comparison metrics live in
`paper/results/meg/comparison/paper_vs_airi.json` and are not a mixed plot.

Canonical deterministic fits: `from redisca import ReDisCA` with
`demean_time=False` (printed Gram). AIRI-executable MEG fits:
`common.source_faithful.fit_condition_averages` (no `redisca` fit).

## Expected runtime / compute

From TRACK_REPORTs of the recorded runs (not re-timed here):

| Track | Recorded compute | Notes |
| --- | --- | --- |
| Unit tests | 123 passed at merge | `common` + library + per-track tests |
| N170 | **~13 s** | exact 24-permutation + B=1000 MC; 28 ch × 256 samples |
| MEG | wall time **not printed** | full MC, not `--quick`; paper B=500 / Nmc=200 and AIRI B=1000 / Nmc=100 on 204 planars × 1501/901 samples × 480 used trials. Plan minutes to tens of minutes. MEG file is ~1.24 GB. |
| Simulations | **~108 min, 4 cores** | 100 MC; RSA was **not** reduced (100 MC × 5002 vertices). `--quick` is smoke only. |
| Source localization | wall time **not printed** | local MEG fit + condition-label B=200 + Eq. 14 scan on 5002 vertices. Plan minutes. |

RNG: NumPy PCG64. N170 / simulations master seed `20240904`. MEG CLI seed
`20240904` splits streams (`seed+1` permutation, `+2` FWER, `+3` AIRI,
`+10` pair-order diagnostic). Source loc permutation seed `20240915`.
Paper text prints no seeds.

## Result locations

| Track | Code | Compact results |
| --- | --- | --- |
| Manifest | `paper/reproduction_manifest.md` | `paper/reproduction_manifest.json` |
| N170 | `paper/reproduction/n170/` | `paper/results/n170/` |
| MEG | `paper/reproduction/meg/` | `paper/results/meg/{paper_faithful,airi_executable,comparison}/` |
| Simulations | `paper/reproduction/simulations/` | `paper/results/simulations/` |
| Source localization | `paper/reproduction/source_localization/` | `paper/results/source_localization/` |
| **This review** | this README | `paper/results_summary.md`, `paper/results_summary.json` |

PNG / NPZ under `paper/results/` are gitignored. Compact JSON is committed.

## Reproduction status

Allowed labels: `reproduced numerically` | `reproduced qualitatively` |
`approximate` | `blocked by missing source asset` | `paper/code discrepancy`
| `stochastic mismatch` | `not yet reproduced`.

**No paper figure is classified `reproduced numerically`.** The only
strict numeric paper targets that were checked (N170 0.82 / p-maps,
simulation 85% hit and <2 cm, MEG three-component significance) are not
recovered on the paper-described estimator. Fig. 11 window RDM correlation
does meet `>0.99`; the rest of that figure does not.

| Figure / result | Status | One-line evidence |
| --- | --- | --- |
| Table 1 SPoC correspondence | reproduced qualitatively | Methods identity; library uses unique pairs + Haufe, not invert-`W` |
| Fig. 1 source-space RSA diagrams | reproduced qualitatively | Visual/methods; RSA AV/S.T. baselines implemented in simulations |
| Fig. 2 ReDisCA diagram | reproduced qualitatively | Visual/methods; GEP + Haufe patterns on rank-deficient data (D9) |
| Fig. 3 simulated multi-source RDMs | approximate | Visual exemplar; unnamed mesh / Υ_d (D13) |
| Fig. 4 single-source ROC / traces | approximate | ReDisCA ranks above RSA; **~85% hit @ ~0 FA not recovered** (TPR@FPR=0 is 0 at SNR 0.1). Exact numbers blocked by missing forward model |
| Fig. 5 four-source Monte Carlo | approximate | ReDisCA better than MNE/BF S.T.; **“largest mass <1 cm” not recovered** (~23–26% <1 cm). C=5 and C=6 both recorded (D14) |
| Fig. 6 localization error vs C | approximate | ReDisCA best at every C; **mean median 3.36 cm at C=6, not <2 cm** |
| Fig. 7 N170 meaning p-map | paper/code discrepancy | Comp-1 p at 400 ms is 8/24 ≈ 0.333 (C=4 floor). Paper uncorrected p<0.05 is unreachable under condition-label permutation of a 2–2 RDM |
| Fig. 8 N170 meaning patterns | reproduced qualitatively | Adjacent 375/400/425 ms windows; occipital (PO4/PO8/Oz); empirical RDMs match the meaning partition |
| Fig. 9 N170 face/car theoretical RDMs | reproduced qualitatively | 0/1 structure encoded from §4.2.1; figures are images, fill not printed |
| Fig. 10 N170 face-specific | paper/code discrepancy | Window RDM corr **0.99988 vs paper 0.82** (not tuned). Face peak 171.9 ms. **0** components with exact-24 p<0.05 (p=0.75) |
| Fig. 11 N170 car-specific | paper/code discrepancy | Window corr **0.99992 meets >0.99**. **0** components with exact-24 p<0.01 (floor 0.25). Two-component claim not recovered |
| Fig. 12 MEG theoretical RDMs | reproduced qualitatively | Binary 0/1 and AIRI 0.1/1 emitted; figures are images (D7) |
| Fig. 13 MEG face-specific | paper/code discrepancy | paper_faithful `max\|λ\|` p = (0, 1, 1): not three p<0.05 components. Contrast peak 308 ms vs paper 160 ms. AIRI-window mean-trace peak **160 ms** is the AIRI path, not paper methods |
| Fig. 14 MEG tool-specific | paper/code discrepancy | paper_faithful p = (0.478, 1, 1). FWER onset 165 ms vs paper 210 ms |
| Fig. 15 MEG meaning | paper/code discrepancy | paper_faithful p = (0.204, 1, 1). FWER onset 128 ms vs paper 160 ms |
| Fig. 16 MEG non-binary RDM | reproduced qualitatively | AIRI `facevstool` 0.1/0.5/1 emitted; not Fig. 12a (D7) |
| Fig. 17 MEG non-binary components | paper/code discrepancy | paper_faithful p = (0.302, 1, 1). AIRI random-phase first three p<0.05 is D5, not §2.3 |
| `airi-executable-meg-facevstool` | approximate | Literal AIRI defaults ran (B=1000); SciPy ≠ MATLAB; random-phase p moves with pair order |
| Fig. 18 MEG MUSIC | approximate | Eq. 14 ran on public AD Gain; argmax **left cuneus / V2**, not paper right FG / insula / left IPS. Individual T1 and `show_on_cortex` missing |
| `airi-source-loc-precomp` | reproduced numerically | `abs(W @ A1[:,3])` peak vertex 394 / lingual L. **Not Fig. 18.** Screenshot blocked (`show_on_cortex`) |
| AIRI `music` `P=eye(1)` | paper/code discrepancy | Non-executable dimension error in committed MATLAB |
| AIRI `music` `P=eye(Nsns)` | approximate | Obvious fix; still `A1(:,4)` only; not Fig. 18 |

Nothing is `not yet reproduced` in this snapshot: every in-scope figure was
run or classified. Nothing is `stochastic mismatch`: published numeric
targets that failed are far from Monte Carlo noise (e.g. 0.82 vs 0.99988;
85% hit vs TPR@0 = 0).

## Paper vs AIRI implementation discrepancies

Verified against the NeuroImage text, AIRI `15bc19c`, stock SPoC `18e4754`,
and library `5a5c865`. Full write-up:
`paper/reference/source_notes/discrepancies.md`. Starting hypotheses 1, 2,
4, 5, 6, 7: **confirmed**. Hypothesis 3 (target SD): **same convention**,
not a bug.

| ID | Topic | Paper | AIRI / SPoC / library | This review |
| --- | --- | --- | --- | --- |
| D1 | Pairs | unique triangle | AIRI `i≠j` directed (30 vs 15). Library unique | MEG dual path respects this. For symmetric D, filters stay on the same rays; λ scales with sample SD |
| D2 | Pair matrix | unscaled Gram | MATLAB `cov` (demean + `/(T-1)`). Library: demean optional, no `T-1` | `1/(T-1)` cancels in the GEP; **demeaning does not**. N170 and MEG store `demean_time=True` as a labeled extra. Simulation `demean_time=True` is near chance (2 Hz ERPs ≈ DC) |
| D3 | Target SD | unspecified | MATLAB `std` = library `ddof=1` | **Not a library bug** |
| D4 | Aggregation | Eq. 7 **sum** (prose: average) | SPoC/library **mean** | Filter rays invariant; quote λ only within one convention. Reported λ is mean aggregation |
| D5 | Component p | condition-label permutation | SPoC random-phase, `max\|λ\|`, `p=count/B` (0 possible). Library: none | **This is why MEG “three significant components” appears on the AIRI path and not on paper_faithful.** N170 primary test is label permutation; random-phase is exploratory only |
| D5b | MEG time p | subcategory-label shuffle, FWER max-T | `Nmc=100` half-split, pointwise, std-normalized data | Asterisks are **not** compared across paths |
| D6 | MEG window | entire 1500 ms (−500…+1000) | `trange=600:1500` → **99–999 ms** | Dual path. File is **1501** samples (D16) |
| D7 | Default RDM | Fig. 12 then Fig. 16 | `ThRDMArr(2)='facevstool'` (0.1/0.5/1) | AIRI default is Fig. 16-like, **not** Fig. 12a. Two-level 0 vs 0.1 fill is identical after z-scoring |
| D8 | MEG filter | none stated | butter(3) 0.25–20 Hz filtfilt | Paper-faithful: no bandpass. AIRI path: SciPy `filtfilt`, not MATLAB |
| D9 | Patterns | `A=W^{-1}` | Haufe; MEG rank 68 < 204 | Invert-`W` undefined. All executable paths use Haufe |
| D11 | N170 ICA | “three ocular+cardiac” | ERP CORE subject 1: comps **2, 7** | Official averages used; a third component is not invented |
| D12 | N170 code | paper + ERP CORE | no AIRI N170 script | Confirmed |
| D13 | Simulations | unnamed forward model | no AIRI script | Public AD Gain is a **hypothesis**. Never fsaverage |
| D14 | Fig. 5 C | caption C=6 vs body C=5 | — | Both recorded; not silently “fixed” |
| D15 | Fig. 18 | MUSIC of Fig. 17 subspace | default sLORETA of component 4 + index vector | On this kernel `megplanarbst` **is** 204 GRAD. Hazard remains if MAG-first |
| D16 | Sample count | 204 × 1500 | file 204 × **1501** | paper_faithful uses 1501 |
| D17 | Saved topo | — | script `return`s before `save`; OSF `filt15` | Precomp is author-saved `A1`, not a vanilla script output |
| D18 | `toolvsface` | — | named but unimplemented (zero RDM) | Not used |

## Integration QC (did not rewrite track science)

Looked for, and did **not** find: `redisca` imported inside
`source_faithful.py`; student N170/synthetic code used as an oracle;
silent retuning to 0.82 / 85% hit / <2 cm; paper/AIRI settings mixed in
one figure; fsaverage substituted for AD Gain.

Did find, and classified rather than “fixed”:

- N170 C=4 condition-label permutation has a discrete p floor (8/24 meaning,
  6/24 face/car). Paper Fig. 7 `p<0.05` and Fig. 11 `p<0.01` are
  incompatible with that null.
- Library GEP on 28 channels × ~26 samples matches a 6-entry two-level RDM
  almost perfectly (corr ≈ 1). Paper face **0.82** is not recovered from
  official subject-1 averages. Parameters were not tuned downward.
- MEG paper_faithful vs AIRI-executable filters are **not** interchangeable
  (face c1 Pearson 0.42; tool subspace min cosine 0.009). D2+D6+D8 dominate.
- MEG published 160 ms face peak is the closest neighbour of the
  **AIRI-window** mean-of-six-traces peak, not of the paper-faithful
  full-epoch contrast.
- Simulations: ReDisCA > RSA ranking holds on the public AD Gain;
  published operating points do not. `tuned_to_85pct_hit_rate: 0`.
- Fig. 18 argmax is left cuneus, not the paper’s qualitative peaks.
- Canonical library: **no bug** identified. D1–D9 are documented design
  differences (unique pairs, optional demean, mean aggregation, no
  inference, Haufe). Default `demean_time=True` is **not** the printed
  Gram; paper-faithful fits set `False`.

See `paper/WORKER_CONTRACT.md` for ownership. See
`paper/results_summary.md` for the per-item matrix.
