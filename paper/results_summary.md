# Reproduction matrix (overnight integration review)

Ossadtchi et al., *NeuroImage* 301 (2024) 120868.
Evidence snapshot: `paper` branch commit `SNAPSHOT_SHA` (Tracks A–F at
`38466de`). Reviewer: overnight integration. Statuses are from committed
JSON/code, not TRACK_REPORT claims.

Status vocabulary: `reproduced numerically` | `reproduced qualitatively` |
`approximate` | `blocked by missing source asset` | `paper/code discrepancy`
| `stochastic mismatch` | `not yet reproduced`.

Paths:

- `library` — `from redisca import ReDisCA`, unique pairs, printed Gram
  (`demean_time=False`). Separate from historical freeze. Do not change
  semantics.
- `historical` — `common.source_faithful` frozen candidate
  (`airi_directed` + `matlab_cov` + `spoc_random_phase`, B=1000).
  **PRIMARY historical reproduction.**
- `paper_faithful` — MEG unique+Gram, full epoch, no bandpass, §2.3
  condition-label `max|λ|` (printed-method MEG path).
- `airi_executable` — directed + matlab_cov + 99–999 ms + 0.25–20 Hz +
  random-phase. **Not paper methods.**
- `methods_diagram`

**PRIMARY historical inference:** stock SPoC random-phase
(`max|λ|`, `p=count/B`, B=1000).
**SECONDARY printed inference:** paper §2.3 condition-label permutation.
Do not call §2.3 the historical oracle. Previous snapshot `4813b38` did.

Canonical-library issues: **none** (design differences D1–D9 only).
`source_faithful.py` does **not** import `redisca`. Unique+Gram library λ
matches source_faithful unique+gram to ~1e-15.
`matlab_parity_claimed: false`.

**Headline:** no paper figure is `reproduced numerically`. Fig. 10 is
still `paper/code discrepancy` (corr 0.82 unmatched) even though PRIMARY
p1=0 and λ is close. Fig. 11 is `approximate` (λ/p2 envelope/corr), not
numerical. Fig. 7 PRIMARY p<0.05 at 400 ms only (`approximate`). MEG
Figs 13–15, 17 remain `paper/code discrepancy` for **three** components:
paper-epoch freeze got **two**. Simulations and Fig. 18 stay `approximate`.

---

## Methods

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `table1-spoc-correspondence` | reproduced qualitatively | library | n/a | — | library has no p-values | ReDisCA ≡ SPoC covariance max on pair-difference matrices. Library unique pairs + Haufe, not invert-`W` (D9) |
| `fig01-source-space-rsa-diagrams` | reproduced qualitatively | methods_diagram | n/a | — | — | Four RSA versions implemented for Figs 4–6 |
| `fig02-redisca-diagram` | reproduced qualitatively | methods_diagram | n/a | — | — | GEP diagram. Rank 68 MEG: invert-`W` undefined |

## Simulations (Figs 3–6)

Unchanged vs prior integration. Canonical path: `library` / paper Gram,
unique pairs, public AD overlapping-spheres Gain (5002 vtx) — **not** a
paper-named mesh (D13). 100 MC, seed `20240904`. `tuned_to_85pct_hit_rate: 0`.
Runtime ~108 min, 4 cores. Exact published numbers: **blocked by missing
source asset**.

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `fig03-sim-four-source-rdms` | approximate | library | visual; no fixed matrix | missing named mesh / Υ_d | none | JSON: `paper/results/simulations/fig03_rdms.json` |
| `fig04-single-source-roc` | approximate | library | Paper: ~85% hit @ ~0 FA at SNR 0.1. **This run SNR 0.1:** ReDisCA AUC **0.874**, TPR@FPR=0 **0.000**, TPR@FPR≤0.01 **0.381**, median error 0.86 cm. Ranking ReDisCA > RSA holds. **85% operating point not recovered.** | unnamed forward model | none | JSON: `fig04_roc.json` |
| `fig04-single-source-traces` | approximate | library | visual | same as Fig. 4 ROC | — | JSON: `fig04_traces.json` |
| `fig05-four-source-mc` | approximate | library | Paper: largest mass <1 cm. **This run:** ReDisCA mean median 3.56–4.14 cm; frac<1 cm **0.23–0.26**. C=5 and C=6 both recorded (D14). | same + Fig. 5 C conflict | none | **“Most mass <1 cm” not recovered.** JSON: `fig05_four_source.json` |
| `fig06-error-vs-C` | approximate | library | Paper: mean median **<2 cm at C=6**. **This run SNR assumed 0.2:** C=3…6 mean median **5.42, 4.32, 4.13, 3.36 cm**; frac<2 cm at C=6 = 0.48 | same + Fig. 6 SNR unspecified | none | ReDisCA best at every C. **<2 cm claim not met.** JSON: `fig06_error_vs_C.json` |

## N170 EEG (Figs 7–11)

**PRIMARY historical path:** frozen candidate `airi_directed` +
`matlab_cov` + `spoc_random_phase` B=1000 on official
`1_N170_erp_ar.erp` (SHA-256 `53e74e93…9bbc72`), 28 scalp channels.
Face: T=100 ms @ 200 ms. Car: T=100 ms @ 170 ms. Meaning sliding: T=150 ms,
step 25 ms (not printed; not searched).

**SECONDARY:** exact-24 condition-label permutation (floors: meaning 8/24,
face/car 6/24). Cannot produce printed p2≈0.009.

**Library path (separate):** unique+Gram `ReDisCA(demean_time=False)`.
λ agrees with `source_faithful` unique+gram to ~1e-15. Unique-pair
random-phase car p2 ~0.12 does **not** match printed 0.009.

Preprocessing (Track C): keep `1_N170_erp_ar.erp`; lpfilt not paper
analysis; ICA 2 and 7 only; no third IC; P9/P10 absent.

### Compact Track A table

Full 12 rows: `paper/results/n170/historical/track_a_table.json`.

Face (printed λ1≈0.87209, p1=0, corr≈0.82):

| variant | λ1 | PRIMARY p1 | corr wᵀRw |
| --- | ---: | ---: | ---: |
| unique Gram | 0.88006 | 0.000 | 0.99988 |
| unique cov | 0.83915 | 0.000 | 0.99858 |
| directed Gram | 0.92301 | 0.000 | 0.99988 |
| **directed cov (freeze)** | **0.88010** | **0.000** | **0.99858** |

Car @ 170 ms (printed λ1≈0.91639, λ2≈0.77036, p2≈0.009):

| variant | λ1, λ2 | PRIMARY p1, p2 | corr wᵀRw |
| --- | --- | --- | ---: |
| unique Gram | 0.88691, 0.79170 | 0.000, **0.141** | 0.99992 |
| unique cov | 0.87063, 0.75365 | 0.000, **0.115** | 0.99909 |
| directed Gram | 0.93020, 0.83035 | 0.000, 0.009 | 0.99992 |
| **directed cov (freeze)** | **0.91312, 0.79043** | 0.000, 0.003 | **0.99909** |

Track B freeze p2 (20 seeds): mean 0.0074, min 0.003, max 0.014, 2/20
exactly 0.009. Printed 0.009 is inside the envelope.

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `fig07-n170-meaning-pmap` | approximate | historical | Paper: uncorrected p<0.05 around t=400 ms, occipital. **PRIMARY:** p1(400 ms)=**0.018**; neighbors 0.076 / 0.050; one isolated window. λ1=0.63587; corr wᵀRw=0.991; max-abs **Pz**; occipital energy **0.218**. **SECONDARY** exact-24 still 8/24. Library unique+Gram exact-24 was also 8/24 with no p<0.05 | unpublished sliding step (25 ms documented) | PRIMARY `spoc_random_phase` B=1000; secondary exact-24 | Deterministic: pattern not occipital. Stochastic PRIMARY: p<0.05 at 400 ms only. Secondary: floor. Printed vs executable: §2.3 cannot reach p<0.05. JSON: `historical_apply/fig07_meaning_pmap.json` |
| `fig08-n170-meaning-patterns` | reproduced qualitatively | historical | Paper: three adjacent windows ~400 ms. **This freeze:** 375/400/425 ms; λ1=0.616/0.636/0.620; PRIMARY p1=0.076/0.018/0.050; corr 0.984/0.991/0.997; max-abs Oz/Pz/Pz | step not printed | same as Fig. 7 | Meaning partition visible. 400/425 ms max-abs is Pz, not occipital. JSON: `historical_apply/fig08_meaning_windows.json` |
| `fig09-n170-face-car-rdms` | reproduced qualitatively | n/a (targets) | 0/1 meaning / face / car encoded from §4.2.1 | numeric fill not printed | — | JSON: `fig09_rdms.json` |
| `fig10-n170-face` | paper/code discrepancy | historical | Paper: λ1≈0.87209, p1=0, corr **0.82**, burst ~170 ms. **Freeze:** λ1=**0.88010** (Δ+0.008); PRIMARY p1=**0 in 20/20**; corr wᵀRw=**0.99858** (undemeaned 0.94819); Faces peak 167.97 ms. **SECONDARY** exact-24 p1=0.50. Library unique+Gram: λ1=0.88006, corr=0.99988, exact-24 p=0.75 | unpublished permutation B; channel list unnamed; P9/P10 absent; D11 ICA wording | PRIMARY random-phase (historical); secondary §2.3 | Deterministic: λ close, corr **not** 0.82. Stochastic PRIMARY: p1=0 matches printed p1 under the **historical** test. Secondary does not call it significant. Preproc: official `.erp` kept. **Not upgraded to numerical.** JSON: `historical/leading_candidate.json`, `rdm_correlation/` |
| `fig11-n170-car` | approximate | historical | Paper: λ1≈0.91639, λ2≈0.77036, p1=0, p2≈0.009, corr **>0.99**. **Freeze:** λ1=**0.91312**, λ2=**0.79043**; PRIMARY p1=0 in 20/20; p2 mean **0.0074** (min 0.003 max 0.014); **2/20 exactly 0.009**; corr **0.99909**. Unique-pair p2 **0.141 / 0.115 does NOT match**. **SECONDARY** 0.25/0.25 | car duration not restated (100 ms used) | PRIMARY random-phase; secondary floor 6/24 | Deterministic: λ1 close, λ2 +0.020, corr matches >0.99. Stochastic PRIMARY: p2 envelope contains 0.009. Secondary cannot equal 0.009. Printed vs executable: directed-pair SPoC null is the lever. Not numerical. JSON: `historical/track_b.json` |

RDM 0.82 investigation (Track D): no endorsed Eq. 1/2 / `w^T R w` /
AIRI `corrcoef` reading of this ERP is 0.82. Eq. 2 sample-SD inner
product ~0.833 is ruled out by the car control (also ~0.833, not >0.99).
`canonical_library_bug: false`.

## MEG sensor space (Figs 12–17)

Three labeled paths. Rank **68** on local Cxx (author A1 is 204×67; not a
SciPy/MATLAB flip of these matrices — Track F).

### historical paper-epoch freeze (PRIMARY; no AIRI extras)

Directed + matlab_cov + random-phase B=1000, −500…+1000 ms, 1501 samples,
no bandpass. Secondary: exact 720 condition-label `max|λ|`.

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `fig12-meg-theoretical-rdms` | reproduced qualitatively | paper_faithful + airi_executable | binary 0/1 and AIRI 0.1/1 emitted | fill 0 vs 0.1 unreadable from figures | n/a | Two-level 0 vs 0.1 identical after z-score |
| `fig13-meg-face` | paper/code discrepancy | historical paper-epoch | Paper: **3** significant comps; peak 160 ms. **Paper-epoch freeze:** λ=(0.862, 0.796, 0.694); PRIMARY p=**(0.006, 0.019, 0.114)** → **2** comps p<0.05; contrast peak **309 ms**. Secondary maxabs p1=0 then p2=0.867. **AIRI extras (not paper methods):** p=(0.006, 0.008, 0.014, 0.032) → first **3** (actually 4) p<0.05; overlap mean-trace peak 160 ms | FieldTrip helmet topos | PRIMARY random-phase B=1000; secondary exact-720 | Deterministic: peak 309 ms ≠ 160 ms. Stochastic PRIMARY: 2 not 3. AIRI extras recover the count via D6+D8, not paper methods. Rank 68. JSON: `meg/historical_candidate/summary.json` |
| `fig14-meg-tool` | paper/code discrepancy | historical paper-epoch | Paper: 210 ms; three comps implied. **Freeze:** λ=(0.818, 0.805, 0.738); p=**(0.006, 0.010, 0.057)** → **2**; contrast 392 ms. AIRI extras p=(0.001, 0.003, 0.008, 0.010) | same | PRIMARY 2 comps; secondary p1=0.533 | Do not call 3-comp numerical reproduction |
| `fig15-meg-meaning` | paper/code discrepancy | historical paper-epoch | Paper: from 160 ms. **Freeze:** λ=(0.845, 0.787, 0.731); p=**(0.001, 0.018, 0.061)** → **2**; contrast 315 ms. AIRI extras p=(0.005, 0.012, 0.018, 0.026) | same | PRIMARY 2 comps | Component 3 sits just above 0.05 |
| `fig16-meg-nonbinary-rdm` | reproduced qualitatively | both | AIRI `facevstool` 0.1/0.5/1 emitted | exact printed fill unknown | n/a | AIRI default `ThRDMArr(2)`, not Fig. 12a (D7) |
| `fig17-meg-nonbinary-components` | paper/code discrepancy | historical paper-epoch | Paper: three comps; tools vs faces from 202 ms. **Freeze:** λ=(0.808, 0.766, 0.697); p=**(0.010, 0.024, 0.104)** → **2**. AIRI extras p=(0.016, 0.019, 0.028, 0.054) → first **3** | same | PRIMARY 2; AIRI extras 3 (c4=0.054) | AIRI extras are D5/D6/D8, not §2.3 |

Library `paper_faithful` unique+Gram + label-perm `max|λ|` B=500 remains
on disk: face p=(0, 1, 1) → 1 comp; tool/meaning/facevstool 0 comps.
That is the printed-method MEG path, not the historical freeze.

### airi_executable (not paper methods)

Directed 30 pairs, MATLAB cov, 99–999 ms, butter(3) 0.25–20 Hz SciPy
filtfilt, SPoC random-phase B=1000. First three components p<0.05 on
face/tool/meaning/facevstool. Status `approximate` for the AIRI-executable
item. SciPy ≠ MATLAB.

### Rank 67 vs 68 (Track F)

Local Cxx/Rbar **68** on both owned paths. A1 **204×67**. λ₆₈/λ_max
4.36×10⁻⁶ (paper) / 2.48×10⁻⁶ (AIRI) vs cutoff 1e-6. MATLAB-eig-only
flip **not_borderline**. Do not force rank 67.
`paper/reproduction/meg/RANK_AUDIT.md`.

### Path comparison (library paper_faithful vs airi_executable)

Unchanged: face filter Pearson c1–c3 0.42 / 0.24 / 0.18; tool subspace
min cosine **0.009**. D2+D6+D8 dominate. Do not mix paths in one figure.

## Source localization (Fig. 18)

Unchanged vs prior integration.

| ID | Status | Path | Quantitative vs paper | Blocked | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `fig18-meg-music` | approximate | paper_faithful (local fit) | Paper: right FG / right insula / left IPS. **This run:** Eq. 14 peak vertex **117**, subcorr **0.821**, Mindboggle **cuneus L** / V2 L. Scout maxima exist on the map but are **not** the argmax | individual T1; `show_on_cortex` missing | condition-label `max|λ|` B=200; no component p<0.05 | Public AD Gain, not fsaverage. **Not screenshot parity.** JSON: `fig18_meg_music.json` |
| `airi-source-loc-precomp` | reproduced numerically | airi_executable | `abs(W @ A1[:,3])` peak vertex **394**, lingual L | `show_on_cortex`; D17 author-saved `filt15` | n/a | **Not Fig. 18.** |
| `airi-music-literal-bug` | paper/code discrepancy | airi_executable | `P=eye(1)` dimension error | non-executable committed MATLAB | — | |
| `airi-music-eye-Nsns-fix` | approximate | airi_executable | peak vertex 294, subcorr 0.576, lingual L | RAP never deflates; still A1(:,4) | — | Not Fig. 18 |

## Canonical library

| Topic | Verdict |
| --- | --- |
| Unique pairs vs AIRI directed | design (D1), not a bug. Stochastic random-phase **is** pair-order-sensitive |
| `demean_time` default `True` vs printed Gram | design (D2). Paper-faithful / library Gram runs set `False` |
| Sample SD `ddof=1` | same as MATLAB (D3), **not a bug** |
| Mean vs sum aggregation | design (D4); filters invariant |
| No built-in inference | design (D5). Historical path owns random-phase; §2.3 is secondary |
| Haufe vs invert-`W` | design (D9); invert-`W` undefined at MEG rank 68 |
| unique+Gram library vs source_faithful unique+gram | λ match ~1e-15 (face 1.55e-15; car 7.8e-16 / 8.9e-16). Expected |
| Near-perfect 6-entry two-level RDM corr | expected GEP optimum, **not a bug** |
| Student code as oracle | **not found** |
| `source_faithful` imports `redisca` | **no** (AST) |
| Hidden parameter tuning | **not found** |
| Canonical-library bug | **false** |

## Status changes vs prior integration (`4813b38`)

That snapshot treated §2.3 condition-label permutation as the historical
oracle. Overnight review reverses the hierarchy. Simulations / Fig. 18
labels are **kept**.

| Item | Prior integration (`4813b38`) | This review | Change |
| --- | --- | --- | --- |
| Fig. 7 | paper/code discrepancy (exact-24 p=0.333, unreachable p<0.05) | **approximate** | PRIMARY random-phase **does** give p=0.018 at 400 ms only; pattern/continuity/secondary still disagree |
| Fig. 8 | reproduced qualitatively | reproduced qualitatively | **kept**; freeze max-abs is Oz/Pz/Pz not all occipital |
| Fig. 10 | paper/code discrepancy (0.99988 vs 0.82; exact-24 p=0.75) | paper/code discrepancy | **kept** as overall label. PRIMARY p1=0 now matches printed p1 under the **historical** test. corr 0.82 still unmatched. **Not upgraded to numerical** |
| Fig. 11 | paper/code discrepancy (corr>0.99 but exact-24 p=0.25) | **approximate** | PRIMARY p1=0 and p2 envelope contains 0.009; unique-pair p2 ~0.12 does not; λ2 still +0.020 |
| Figs 13–15, 17 | paper/code discrepancy (paper_faithful 0–1 comps) | paper/code discrepancy | **kept**. Historical paper-epoch freeze: **2** comps, not 3. AIRI extras: 3, not paper methods. **Not upgraded to numerical** |
| Figs 3–6, 18 | approximate | approximate | **kept** |
| Canonical library bug | false | false | **kept**; new ~1e-15 unique+Gram identity |

## Remaining unresolved

1. Fig. 10 observed–theoretical RDM corr **0.82**.
2. MEG **2 vs 3** significant components on the paper epoch without AIRI extras.
3. D11 ICA wording (“three” vs subject-1 components 2 and 7).
4. Fig. 18 anatomy (left cuneus vs paper right FG / insula / left IPS).
