# Reproduction matrix (integration review)

Ossadtchi et al., *NeuroImage* 301 (2024) 120868.
Evidence snapshot: `paper` HEAD `0a2d9ed`. Reviewer: integration worker.
Statuses are from committed JSON/code, not TRACK_REPORT claims.

Status vocabulary: `reproduced numerically` | `reproduced qualitatively` |
`approximate` | `blocked by missing source asset` | `paper/code discrepancy`
| `stochastic mismatch` | `not yet reproduced`.

Paths: `paper_faithful` (printed Gram, unique pairs, paper windows/tests) |
`airi_executable` (source_faithful MATLAB cov, directed pairs, AIRI
window/filter/tests) | `library` (`from redisca import ReDisCA`) |
`methods_diagram`.

Canonical-library issues: **none** (design differences D1–D9 only).
`source_faithful.py` does **not** import `redisca`.

**Headline:** no paper figure is `reproduced numerically`. Several
theoretical-RDM / methods panels are `reproduced qualitatively`.
Simulations and Fig. 18 are `approximate` because the named forward model /
individual T1 / FieldTrip screenshots are missing. N170 inference and MEG
component counts are `paper/code discrepancy`.

---

## Methods

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `table1-spoc-correspondence` | reproduced qualitatively | library | n/a | — | library has no p-values | ReDisCA ≡ SPoC covariance max on pair-difference matrices. Library unique pairs + Haufe, not invert-`W` (D9) |
| `fig01-source-space-rsa-diagrams` | reproduced qualitatively | methods_diagram | n/a | — | — | Four RSA versions implemented for Figs 4–6. Pairing rule for S.T. assumed (index `l=1…I_c`) |
| `fig02-redisca-diagram` | reproduced qualitatively | methods_diagram | n/a | — | — | GEP diagram. Rank 68 MEG: invert-`W` undefined |

## Simulations (Figs 3–6)

Canonical path: `library` / paper Gram (`ReDisCA(demean_time=False)`), unique
pairs, public AD overlapping-spheres Gain (5002 vtx) — **not** a paper-named
mesh (D13). 100 MC, seed `20240904`. `tuned_to_85pct_hit_rate: 0`.
Runtime ~108 min, 4 cores. Exact published numbers: **blocked by missing
source asset** (unnamed mesh, `I_c`, `f_s`, `Υ_d`, 1/f recipe).

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `fig03-sim-four-source-rdms` | approximate | library | visual; no fixed matrix | missing named mesh / Υ_d | none | Exemplar C=6, P=4 RDMs. JSON: `paper/results/simulations/fig03_rdms.json` |
| `fig04-single-source-roc` | approximate | library | Paper: ReDisCA dominates; ~85% hit @ ~0 FA at SNR 0.1. **This run SNR 0.1:** ReDisCA AUC **0.874**, TPR@FPR=0 **0.000**, TPR@FPR≤0.01 **0.381**, median error 0.86 cm. MNE S.T. AUC 0.693. `demean_time=True` extra AUC **0.500**. Secondary seed 20 MC AUC 0.845 | unnamed forward model, I_c=40 assumed, fs=1000 Hz assumed | none (localization ROC) | Ranking ReDisCA > RSA holds. **85% operating point not recovered.** High-SNR panel uses preprint overlay 0.2 (AUC 0.880, TPR@0 = 0.002). JSON: `fig04_roc.json` |
| `fig04-single-source-traces` | approximate | library | visual | same as Fig. 4 ROC | — | JSON: `fig04_traces.json` |
| `fig05-four-source-mc` | approximate | library | Paper: largest mass <1 cm. **This run:** ReDisCA mean median 3.56–4.14 cm; frac<1 cm **0.23–0.26**. corr(RDM) ~0.95; corr(a,g)~0.29; corr(w,g)~0. C=5 and C=6 both recorded (D14). SNR 0.4 preprint / 0.2 body | same + Fig. 5 C conflict | none | Qualitative: ReDisCA better than MNE/BF S.T.; patterns align better than weights. **“Most mass <1 cm” not recovered.** JSON: `fig05_four_source.json` |
| `fig06-error-vs-C` | approximate | library | Paper: ReDisCA best; mean median **<2 cm at C=6**. **This run SNR assumed 0.2:** C=3…6 mean median **5.42, 4.32, 4.13, 3.36 cm**; frac<2 cm at C=6 = 0.48 | same + Fig. 6 SNR unspecified | none | ReDisCA best at every C and improves with C. **<2 cm claim not met.** JSON: `fig06_error_vs_C.json` |

## N170 EEG (Figs 7–11)

Path: `library` paper Gram, 28 scalp channels, ERP CORE subject `"1"`
averages (`1_N170_erp_ar.erp` SHA-256 `53e74e93…9bbc72`). Conditions
Faces / Cars / Scrambled Faces / Scrambled Cars. Sliding step **25 ms**
(not printed). Car duration **100 ms** centered at 170 ms (not restated).
No AIRI N170 code (D12). Student examples not used. ~13 s.

Primary inference: permute condition order of D; exact 24 permutations.
Meaning RDM floor 8/24; face/car floor 6/24. SPoC random-phase is
exploratory only (D5).

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `fig07-n170-meaning-pmap` | paper/code discrepancy | library (`demean_time=False`) | Paper: uncorrected p<0.05 around t=400 ms, occipital. **This run:** p(400 ms)=8/24≈0.333; no p<0.05 window. RDM corr @400 ms **0.99981**. max \|a\| **PO8**; occipital energy **0.61** | unpublished B / p formula; sliding step chosen | condition-label exact-24 `P(λ*≥λ)`; C=4 floor | Topography at 400 ms is occipital (qualitative). The **p-map is not reproduced** and cannot be under this null. JSON: `fig07_meaning_pmap.json` |
| `fig08-n170-meaning-patterns` | reproduced qualitatively | library | Paper: three adjacent windows ~400 ms; meaningless vs meaningful split. **This run:** 375/400/425 ms; window corr 0.99981 / 0.99981 / 0.99991; max \|a\| PO4 / PO8 / Oz | step not printed | same floor p=0.333 | Empirical RDMs show the 2–2 meaning partition. JSON: `fig08_meaning_windows.json` |
| `fig09-n170-face-car-rdms` | reproduced qualitatively | n/a (targets) | 0/1 meaning / face / car encoded from §4.2.1. 0.1-within extra is z-score equivalent | numeric fill not printed (figures are images) | — | JSON: `fig09_rdms.json`, `fingerprints.json` |
| `fig10-n170-face` | paper/code discrepancy | library | Paper: RDM corr **0.82**; one significant component; face burst ~170 ms; right-FG-like. **This run:** window corr **0.99988** (Δ +0.18); full-epoch corr 0.945; exact-24 p=**0.75** (0 components p<0.05); Faces peak **171.9 ms**; max \|a\| **P7**; occipital energy 0.53 | unpublished permutation B; channel list unnamed; P9/P10 absent | exact-24 `≥`; MC B=1000 p=0.741 agrees. Random-phase exploratory p=0.000 would *call* it significant — **not** the paper N170 test | 0.82 **not** recovered; not tuned. Timing is close. JSON: `fig10_face.json` |
| `fig11-n170-car` | paper/code discrepancy | library | Paper: two comps p<0.01; corr **>0.99**. **This run:** window corr **0.99992 / 0.99968** (meets >0.99); exact-24 p=**0.25 / 0.25** (0 comps p<0.01); Cars peak 136.7 ms vs ~150 ms; max \|a\| O1 then Pz | car duration not restated (100 ms used) | floor 6/24; strict-`>` p=0 for comps 1–2 (uniquely best detector) | Correlation matches; **two-component p<0.01 does not**. JSON: `fig11_car.json` |

`demean_time=True` extra (not mixed into primary plots): face window corr
0.99858 (PO4); car 0.99909 / 0.99650 (P8 / PO8); meaning @400 ms corr
0.99105 with max \|a\| at Pz (less occipital).

## MEG sensor space (Figs 12–17)

Dual path mandatory. Rank 68 on both (OSF topo is 204×67; scipy vs MATLAB
eig). Seed CLI `20240904`. Comparison JSON is metrics-only — not a mixed
figure.

### paper_faithful

`ReDisCA(demean_time=False)`, unique 15 pairs, unscaled Gram, full
−500…+1000 ms (1501 samples), no bandpass. Component test: condition-label
permutation, primary null `max\|λ\|`, B=500. Time test: subcategory-label
FWER, Nmc=200.

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `fig12-meg-theoretical-rdms` | reproduced qualitatively | paper_faithful + airi_executable | binary 0/1 and AIRI 0.1/1 emitted | fill 0 vs 0.1 unreadable from figures | n/a | Two-level 0 vs 0.1 is identical after z-score. JSON: `paper_faithful/fig12_theoretical_rdms.json` |
| `fig13-meg-face` | paper/code discrepancy | paper_faithful | Paper: 3 significant comps; from 65 ms; peak 160 ms; again from 311. **This run:** λ=(0.877, 0.786, 0.691); p_maxabs=**(0, 1, 1)**; emp. RDM Pearson 0.9995; FWER onset **107 ms**; contrast peak **308 ms** | FieldTrip helmet topos; unpublished B | `max\|λ\|` FWER over rank 68 → comps 2–3 almost always p=1. Matched-comp p still ≥0.05 for c2–c3 | **Not three p<0.05 components.** Closest 160 ms neighbour is AIRI-window mean-trace peak (see comparison). JSON: `paper_faithful/fig13_face.json` |
| `fig14-meg-tool` | paper/code discrepancy | paper_faithful | Paper: 210 ms. **This run:** λ=(0.857, 0.789, 0.723); p=**(0.478, 1, 1)**; Pearson 0.9999; FWER onset **165 ms**; contrast peak 384 ms | same | no component p<0.05 under `max\|λ\|` | JSON: `paper_faithful/fig14_tool.json` |
| `fig15-meg-meaning` | paper/code discrepancy | paper_faithful | Paper: from 160 ms. **This run:** λ=(0.870, 0.775, 0.717); p=**(0.204, 1, 1)**; Pearson 0.9996; FWER onset **128 ms**; contrast peak 315 ms | same | no component p<0.05 under `max\|λ\|` | JSON: `paper_faithful/fig15_meaning.json` |
| `fig16-meg-nonbinary-rdm` | reproduced qualitatively | both | AIRI `facevstool` 0.1/0.5/1 emitted | exact printed fill unknown | n/a | This **is** AIRI default `ThRDMArr(2)`, not Fig. 12a (D7) |
| `fig17-meg-nonbinary-components` | paper/code discrepancy | paper_faithful | Paper: tools vs faces from 202 ms; face peak ~160 ms. **This run:** λ=(0.836, 0.753, 0.684); p=**(0.302, 1, 1)**; Pearson 0.966; FWER onset **113 ms**; contrast peak 564 ms | same | no p<0.05 under `max\|λ\|` | JSON: `paper_faithful/fig17_nonbinary_components.json` |

Eq. 7 sum scale (D4): multiply λ by n_pairs=15. Filter rays unchanged.

### airi_executable (not paper methods)

Directed 30 pairs, MATLAB cov, 99–999 ms, butter(3) 0.25–20 Hz SciPy
filtfilt, SPoC random-phase B=1000, half-split Nmc=100.

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Fig. 13 AIRI face | approximate | airi_executable | λ=(0.880, 0.859, 0.833, 0.806); random-phase p=**(0.006, 0.008, 0.014, 0.032)** — first **four** p<0.05, not three; Pearson 0.998; contrast peak 319 ms; overlap mean-trace peak **160 ms** | SciPy ≠ MATLAB filtfilt/eig | D5 random-phase, **not** §2.3 | Component *count* looks more like the paper figures than paper_faithful does — because it is the AIRI test |
| Fig. 14 AIRI tool | approximate | airi_executable | p=(0.001, 0.003, 0.008, 0.010); Pearson 0.999 | same | D5 | |
| Fig. 15 AIRI meaning | approximate | airi_executable | p=(0.005, 0.012, 0.018, 0.026); Pearson 0.998 | same | D5 | |
| Fig. 17 AIRI facevstool | approximate | airi_executable | p=(0.016, 0.019, 0.028, 0.054); Pearson 0.965; class2 (tools) peak 161 ms | same | first three p<0.05; c4 = 0.054 | Closest unmodified AIRI run; still not paper |
| `airi-executable-meg-facevstool` | approximate | airi_executable | same as Fig. 17 AIRI | not MATLAB parity | pair-order diagnostic: default p≈(0.010, 0.015, 0.020, 0.045); five shuffles move p₁ in {0, 0.005} | Random-phase p is not unique in the directed length-30 z sequence |

### Path comparison (sign-aligned; not a mixed figure)

From `comparison/paper_vs_airi.json`. Filter Pearson c1–c3: face 0.42 / 0.24
/ 0.18; tool 0.68 / 0.75 / 0.69; meaning 0.79 / 0.79 / 0.44; facevstool
0.61 / 0.37 / 0.39. Tool filter subspace min cosine **0.009**. Leading λ is
similar (~0.84–0.88); later λ decay faster on the paper Gram / full epoch.
D2+D6+D8 dominate disagreement. Do not compare FWER vs half-split asterisks.

## Source localization (Fig. 18)

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `fig18-meg-music` | approximate | paper_faithful (local fit) | Paper: MUSIC of Fig. 17 subspace; qualitative right FG / right insula / left IPS / anterior central gyrus. **This run:** Eq. 14 peak vertex **117** (MATLAB 118), subcorr **0.821**, Mindboggle **cuneus L** / V2 L. Scout maxima: fusiform R 0.754, insula R 0.722, left IPS 0.592 — on the map, **not** the argmax. Local fit p_maxabs=(0.325, 1, 1); three leading comps used anyway | individual T1; `show_on_cortex` / FieldTrip A/P/S/L/R views; unnamed paper B | condition-label `max\|λ\|` B=200, seed 20240915; **no** component p<0.05 | Public AD Gain, not fsaverage. Haufe (D9). **Not screenshot parity.** JSON: `fig18_meg_music.json` |
| `airi-source-loc-precomp` | reproduced numerically | airi_executable | AIRI formula `abs(ImagingKernel[:, megplanarbst] @ A1[:,3])` peak vertex **394**, lingual L, max \|Wa\| 8.50×10⁻¹¹ (saved A1 columns have ~10⁻¹² norms) | `show_on_cortex` for the visual; D17 author-saved `filt15` (script `return`s before `save`) | n/a (sLORETA of one topography) | **Not Fig. 18.** Numeric map matches the committed AIRI default. JSON: `airi_source_loc_precomp.json` |
| `airi-music-literal-bug` | paper/code discrepancy | airi_executable | `P = eye(size(Nsns,1))` → `eye(1)` → dimension error | non-executable committed MATLAB | — | JSON: `airi_music_literal_bug.json` |
| `airi-music-eye-Nsns-fix` | approximate | airi_executable | peak vertex 294, subcorr 0.576, lingual L | RAP never deflates (`nRAP=1`); still A1(:,4) | — | Equals Eq. 14 K=1 on component 4. Not Fig. 18 |
| author-saved `A1(:,1:3)` MUSIC | approximate | airi_executable | peak vertex 2595, lateraloccipital R, subcorr 0.823 | D17 | — | Same peak as local AIRI-executable K=3 MUSIC (0.838). Consistent with D17 being an AIRI-like fit, **not** paper-faithful Fig. 18 |
| D15 index audit | n/a (hazard documented) | — | On this kernel ChannelTypes = GRAD, GRAD, MAG so AIRI `megplanarbst` **is** 204 planars | MAG-first files would mix types | — | JSON: `index_audit.json`. GRAD1+MAG negative control peaks at a different vertex |

## Canonical library

| Topic | Verdict |
| --- | --- |
| Unique pairs vs AIRI directed | design (D1), not a bug |
| `demean_time` default `True` vs printed Gram | design (D2). Paper-faithful runs set `False` |
| Sample SD `ddof=1` | same as MATLAB (D3), **not a bug** |
| Mean vs sum aggregation | design (D4); filters invariant |
| No built-in inference | design (D5). Tracks own permutation / random-phase layers |
| Haufe vs invert-`W` | design (D9); invert-`W` undefined at MEG rank 68 |
| Student code as oracle | **not found** |
| `source_faithful` imports `redisca` | **no** |
| Hidden parameter tuning | **not found** (`tuned_to_85pct_hit_rate: 0`; N170 0.82 not chased) |

## Status changes vs track self-reports

See also the integration return note. Tracks often wrote “ran” / “partial”
rather than the seven official labels. This review **does not** upgrade
those to `reproduced numerically`.

| Item | Track self-report | Integration status | Change |
| --- | --- | --- | --- |
| Fig. 7 | ran; p<0.05 not obtained | paper/code discrepancy | classified; not upgraded |
| Fig. 8 | ran | reproduced qualitatively | **upgrade** of “ran” to qualitative (patterns/windows only) |
| Fig. 9 | encoded | reproduced qualitatively | **upgrade** of encoding to qualitative (images, not numeric fill) |
| Fig. 10 | ran; report 0.99988 vs 0.82 | paper/code discrepancy | classified; **not** qualitative despite ~172 ms peak |
| Fig. 11 | ran; corr agrees >0.99; p<0.01 not obtained | paper/code discrepancy | **downgrade** vs reading the corr match as figure reproduction |
| Fig. 12 | emitted | reproduced qualitatively | **upgrade** of “emitted” |
| Figs 13–15, 17 paper_faithful | partial | paper/code discrepancy | **downgrade** of “partial” (methods ran; paper claims not recovered) |
| Figs 13–15, 17 AIRI | numeric run complete | approximate | **downgrade** vs treating AIRI significance count as paper reproduction |
| Fig. 16 | emitted | reproduced qualitatively | **upgrade** of “emitted” |
| `airi-executable-meg-facevstool` | ran B=1000 | approximate | not MATLAB parity |
| Figs 3–6 | approximate / blocked | approximate | **kept**; exact numbers remain blocked; 85% / <1 cm / <2 cm **not** upgraded to qualitative reproduction of those claims |
| Fig. 18 | approximate | approximate | **kept**; peak region mismatch recorded |
| AIRI precomp | approximate; numeric map reproduced | reproduced numerically (map only) | **upgrade of the map formula only**; still not Fig. 18; visual blocked |
| AIRI `P=eye(1)` | blocked | paper/code discrepancy | classified as MATLAB bug |
