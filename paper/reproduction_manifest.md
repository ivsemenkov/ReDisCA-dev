# Reproduction manifest (Ossadtchi et al., NeuroImage 2024)

This file is the human coordination contract for paper-branch tracks.
The machine companion is `paper/reproduction_manifest.json`.
Authority notes live in `paper/reference/source_notes/` and
`paper/reference/provenance/`. Pins: `paper/reference/dependency_pins/pins.json`.

Tracks must wait for this file before claiming a figure is in or out of
scope. Do not implement from memory of “which figures matter”: every
item below was inventoried from the published NeuroImage text.

**Never collapse** (A) paper text, (B) AIRI MATLAB, (C) stock SPoC,
(D) Python `redisca`, (E) what actually reproduces a published panel.
When they disagree, run labeled variants.

Canonical deterministic fits: `from redisca import ReDisCA`.
Historical AIRI/SPoC reconstruction: `paper/reproduction/common/source_faithful.py`
(do not import `redisca` there). MATLAB is not required.

## Pins

| Source | Pin |
| --- | --- |
| Paper | Ossadtchi et al., *NeuroImage* 301 (2024) 120868, doi 10.1016/j.neuroimage.2024.120868 |
| Library `main` | `5a5c8658452172e4011445c9a394c1cbbd3c5f7e` |
| AIRI MATLAB | https://github.com/AIRI-Institute/ReDisCA `15bc19cdc76989da202714b257f6de4d26a42c51` |
| Stock SPoC | https://github.com/svendaehne/matlab_SPoC `18e4754aec1411160fd5b7ef0db852f1e0a87d90` |
| MEG / source models | https://osf.io/8rk67/ |
| N170 | https://osf.io/pfde9/ (ERP CORE node; parent `thsqg`) |
| ERP CORE scripts | https://github.com/lucklab/ERP_CORE `c18b43d70d791ca914d90410afe4ff06d6f7f429` |

The paper’s data-availability URL `osf.io/pfde9` is **N170 only**.
MEG lives on `osf.io/8rk67`.

## Classes (repeatable tags)

`deterministic`, `stochastic`, `preprocessing-dependent`,
`source-model-dependent`, `visual/qualitative`, `numeric`,
`currently blocked`.

Status guesses in the JSON are not results. After a track runs, the
integration worker replaces them with evidence-backed labels from
`paper/README.md`.

## Paper-internal conflicts tracks must not “fix quietly”

1. Fig. 5 caption vs body: \(C=6\) vs \(C=5\); panel identities do not
   match (see D14).
2. Pair set: Table 1 / Eq. 6 unique triangle vs Eq. 5 `i,j=1…C`.
3. Eq. 7 is a **sum** while the sentence calls it an average.
4. Printed 204×1500 vs OSF 204 planars × **1501** samples.
5. N170 “three ICA components, ocular+cardiac” vs ERP CORE subject `"1"`
   removing components **2 and 7** only.

## Dual-path rule (MEG)

Every MEG figure gets **two** result folders/tags:

- `paper_faithful`: unique pairs, unscaled Gram (`demean_time=False`)
  and/or demeaned Gram as a labeled extra, full −500…+1000 ms, no
  AIRI bandpass (or a labeled Kozunov-only path), condition-label
  permutation inference with documented \(B\).
- `airi_executable`: directed pairs, MATLAB cov, `trange` 99–999 ms,
  butter(3) 0.25–20 Hz filtfilt, SPoC random-phase \(B=1000\), default
  RDM only when the figure is Fig. 16/17.

Never mix them in one plot and call it reproduction.

---

## Inventory

### Methods diagrams (not numeric targets)

| ID | Figure | Track | Classes | Expected |
| --- | --- | --- | --- | --- |
| `table1-spoc-correspondence` | Table 1 | all (read) | deterministic, visual | ReDisCA ≡ SPoC covariance max on pair-difference matrices |
| `fig01-source-space-rsa-diagrams` | Fig. 1a,b | simulations (baseline defs) | visual, source-model-dependent | Four RSA versions later used in Figs 4–6 |
| `fig02-redisca-diagram` | Fig. 2 | all (read) | visual, deterministic | GEP diagram; invert-\(W\) patterns if full rank |

### Simulations — **currently blocked** for exact numbers

Owner: `paper/reproduction/simulations/`. No AIRI code. Forward model
unnamed. Public candidate only: AD `tess_cortex_pial_low` (5002 vtx)
+ overlapping-spheres `Gain`. If used, label `approximate` /
`blocked by missing source asset`. Do **not** silently use fsaverage.

Shared generative facts (paper §2.4): \(T=200\) ms; 6th-order
Butterworth 2 Hz on source time series; \(\sigma_\delta=0.15\|g\|\);
1000 1/\(f\) noise sources; \(r_{\max}=0.01\) m; **100 MC**;
\(D=D_0+\Upsilon_d\) (\(\Upsilon_d\) unspecified); \(I_c\) and \(f_s\)
unspecified.

| ID | Figure | What to produce | Expected (qualitative / numeric) |
| --- | --- | --- | --- |
| `fig03-sim-four-source-rdms` | Fig. 3 | exemplar 4 RDMs, \(C=6\), \(P=4\) | visual; not a fixed matrix to copy |
| `fig04-single-source-roc` | Fig. 4a,c | ROC ReDisCA vs MNE/BF × AV/S.T., \(C=5\) | ReDisCA dominates; ~85% hit @ ~0 FA at SNR 0.1. High SNR panel: preprint overlay **0.2** |
| `fig04-single-source-traces` | Fig. 4b,d | noisy/clean ERPs + RDM inset | visual |
| `fig05-four-source-mc` | Fig. 5 | error hist + corr(\(a,g\)), corr(\(w,g\)), RDM corr | ReDisCA most mass <1 cm; SNR overlays 0.4 / 0.2; **record C=5 vs C=6 choice** |
| `fig06-error-vs-C` | Fig. 6 | median error vs \(C=3,4,5,6\) | ReDisCA best; mean median **< 2 cm at \(C=6\)** |

### N170 EEG — owner `paper/reproduction/n170/`

No AIRI MATLAB. Use ERP CORE official pipeline / precomputed
`1_N170_erp_ar.erp` from pfde9 “N170 All Data and Scripts/1/”.
Subject folder `"1"`. ICA list: components **2, 7** (xlsx SHA in
pins). Conditions: face, car, scrambled face, scrambled car
(correct-response bins). Epoch [−200, 800] ms, 256 Hz.

Sliding **step is not in the paper** — pick one, document it
(suggestion: 10 or 25 ms; do not pretend it is specified).

| ID | Figure | Window / RDM | Metric / expected |
| --- | --- | --- | --- |
| `fig07-n170-meaning-pmap` | Fig. 7a–d | sliding \(T=150\) ms; meaning RDM | first component uncorrected \(p<0.05\) around **t=400 ms**; occipital |
| `fig08-n170-meaning-patterns` | Fig. 8 | three adjacent windows ~400 ms | meaningless traces split from meaningful; face longer than car |
| `fig09-n170-face-car-rdms` | Fig. 9a,b | 4×4 theoretical RDMs | encode 0/1 structure; optional 0.1-within variant |
| `fig10-n170-face` | Fig. 10 | \(T=100\) ms centered at **200 ms**; face RDM | **RDM corr 0.82**; one significant component; right-FG-like; face burst ~170 ms |
| `fig11-n170-car` | Fig. 11 | at **t=170 ms** (duration not restated); car RDM | **two** comps \(p<0.01\); RDM corr **> 0.99** |

Library path: `ReDisCA(demean_time=False)` (printed Gram) and
`demean_time=True` as a labeled extra. Inference: paper-style
condition-label permutation with documented \(B\) (e.g. 1000). Do
not call SPoC random-phase the paper N170 test.

### MEG sensor space — owner `paper/reproduction/meg/`

Data: `MEG_AD_run1.mat` + SPM labels. Verified 80 epochs × 6
subcategories. 204 planars. True time −500…+1000 ms at 1 kHz
(1501 samples).

AIRI default RDM is **`facevstool`** (Fig. 16 geometry), **not**
Fig. 12a.

| ID | Figure | RDM | Paper window | AIRI window |
| --- | --- | --- | --- | --- |
| `fig12-meg-theoretical-rdms` | Fig. 12a–c | face / tool / meaning | n/a (targets) | AIRI names `face`,`tool`,`meaning` with 0.1/1 fill |
| `fig13-meg-face` | Fig. 13a–c | face | full epoch | 99–999 ms + 0.25–20 Hz |
| `fig14-meg-tool` | Fig. 14a–c | tool | full epoch | same |
| `fig15-meg-meaning` | Fig. 15a–c | meaning | full epoch | same |
| `fig16-meg-nonbinary-rdm` | Fig. 16a,b | non-binary geometry | n/a | **AIRI default** 0.1/0.5/1 |
| `fig17-meg-nonbinary-components` | Fig. 17 | same | full epoch | closest unmodified AIRI run, still not identical |
| `airi-executable-meg-facevstool` | not a paper figure | facevstool | n/a | the literal AIRI defaults |

Paper time-series asterisks: permute **subcategory labels**, FWER
**max over time**. AIRI asterisks: `Nmc=100` half-split on
channel-std data. Do not compare asterisks across those tests.

Reported paper onsets (qualitative targets): see JSON `fig13`–`fig17`
and `paper/reference/source_notes/paper_methods.md` §6.

### Source localization — owner `paper/reproduction/source_localization/`

| ID | Figure | Algorithm | Notes |
| --- | --- | --- | --- |
| `fig18-meg-music` | Fig. 18 | paper Eq. 14 MUSIC of **Fig. 17 subspace**, free orientation | Public `Gain`+`tess` suffice for a scan on AD 5002; individual T1 not released |
| `airi-source-loc-precomp` | not Fig. 18 | constrained sLORETA × **A1(:,4)** | MAG/GRAD index hazard; `show_on_cortex` missing; topo file is author-saved `filt15` |

Sim cosine-similarity localization (Eq. 13) belongs to the
simulations track (Figs 4–6), not this track.

---

## Material paper vs AIRI discrepancies (verified)

Full write-up: `paper/reference/source_notes/discrepancies.md`.

| ID | Topic | Paper | AIRI / SPoC / library |
| --- | --- | --- | --- |
| D1 | Pairs | unique triangle | AIRI \(i\neq j\) directed (30 vs 15). Library unique. |
| D2 | Pair matrix | unscaled Gram | MATLAB `cov` (demean + `/(T-1)`). Library: demean optional, no `T-1`. |
| D3 | Target SD | unspecified | MATLAB sample SD = library `ddof=1`. **Not a bug.** |
| D4 | Aggregation | Eq. 7 **sum** | SPoC/library **mean**. Filters invariant; \(\lambda\) not. |
| D5 | Component \(p\) | condition-label permutation | SPoC random-phase, \(\max\|\lambda\|\), \(p=count/B\) (0 possible). AIRI \(B=1000\). Library: none. |
| D5b | MEG time \(p\) | label-shuffle averages, FWER max-T | `Nmc=100` half-split, pointwise, std-normalized data |
| D6 | MEG window | entire 1500 ms | `600:1500` → **99–999 ms** |
| D7 | Default RDM | Fig. 12 then Fig. 16 | `ThRDMArr(2)='facevstool'` (Fig. 16-like, 0.1/0.5/1) |
| D8 | MEG filter | none stated | butter(3) 0.25–20 Hz filtfilt |
| D9 | Patterns | \(A=W^{-1}\) | Haufe; MEG rank 67. Use Haufe. |
| D11 | N170 ICA | “three ocular+cardiac” | ERP CORE subject 1: comps **2, 7** |
| D15 | Fig. 18 | MUSIC of subspace | default sLORETA of component 4 + bad planar index |
| D17 | Saved topo | — | `return` before `save`; OSF `filt15` artifact |

Starting hypotheses 1,2,4,5,6,7: **confirmed**. Hypothesis 3: **same
convention**, not a discrepancy.

## Currently blocked

| Item | Why |
| --- | --- |
| Figs 3–6 exact numbers | No simulation script; unnamed forward model; unnamed \(I_c\), \(f_s\), \(\Upsilon_d\); Fig. 5 \(C\) conflict |
| AIRI cortical screenshots | `show_on_cortex` / `prepare4topoNMG` / FieldTrip not in AIRI repo |
| Rebuilding Gain from MRI | Individual T1 not on OSF |
| Bit-exact MATLAB SPoC | No MATLAB in this environment (Python source-faithful is the substitute; `eig`/`filtfilt`/`rand` differ) |
| N170 sliding step | not printed (choose and document; do not invent a paper value) |
| Literal AIRI `save topo_*` | committed script `return`s first |

N170 and MEG **sensor-space** paper figures are **not** blocked on
missing data: ERP CORE subject 1 and OSF MEG files are public.

## Track checklist

**N170:** Figs 7–11. Dual library `demean_time`. Document window step.
Use ERP CORE ICA list, not “3 components”.

**MEG:** Figs 12–17 + AIRI-executable default. Dual path mandatory.
Do not use `trange=600:1500` on a `paper_faithful` tag.

**Simulations:** Figs 3–6. If AD Gain is used, say so. Status
`blocked` or `approximate`.

**Source loc:** Fig. 18 MUSIC from locally fitted Fig. 17 patterns.
AIRI precomp as a separately labeled negative-control / executable
path, not as Fig. 18.

## JSON schema

Top-level keys: `paper`, `sources`, `items`, `discrepancies`,
`n170_notes`, `meg_notes`, `simulation_notes`, `localization_notes`.
Each `items[]` object has: `id`, `figure`, `panel`, `section`,
`title`, `dataset`, `rdm`, `time_window`, `preprocessing`,
`algorithm_variant`, `metric`, `statistical_test`, `n_monte_carlo`,
`expected`, `sources_needed`, `classes`, `evidence`, `status_guess`,
`paper_vs_airi_notes`, `blocked_reason`.
