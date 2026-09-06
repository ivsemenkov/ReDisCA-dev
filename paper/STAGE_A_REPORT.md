# Stage A report: ReDisCA paper reproduction

**Status: Stage A incomplete / no final verdict yet.**

Ossadtchi et al., *NeuroImage* 301 (2024) 120868, using **exactly one**
ReDisCA configuration: the AIRI → stock-SPoC settings on current `main`.

This is a scientific reproduction report, not a method-development or
ablation note. The ReDisCA constructor was not changed anywhere.

Do **not** read this file as a claim that no pipeline reproduces the
paper, or that one candidate has been selected as the winner. The
pre-registered multi-seed simulation matrix and the review-required
Fig. 17→18 / AIRI-temporal branches are still being executed.

Machine-readable companions:

- `paper/reproduction_manifest.json` — original candidate matrix frozen
  **before** full-result selection. Not rewritten after review.
- `paper/reproduction_manifest_addendum.json` — review-added forensic
  branches, each marked `added_after_stage_a_review: true`
- `paper/results/coverage.json` — what was actually executed
- `paper/results/stage_a_summary.json` — numeric extracts vs paper anchors
  (`stage_a_status.final_verdict_allowed` is false until the required
  matrix exists)
- per-run JSON under `paper/results/`

`--quick` leftovers must not be read as reproduction results.

---

## 0. How to read claims in this report

Every reconstruction choice is labeled:

| Tag | Meaning |
| --- | --- |
| **A** | Directly specified by the paper / figure / caption |
| **B** | Literal historical executable behavior (AIRI MATLAB @ `15bc19cd`) |
| **C** | Source-supported ambiguity (paper/preprint/citation leaves more than one reading) |
| **D** | Post-review forensic interpretation added because a literal reading is internally pathological. **Not** paper-faithful. **Not** pre-registered. |

These are external analysis / simulation-generation branches. They are
not ReDisCA method ablations.

---

## 1. Working branch and commit

| Item | Value |
| --- | --- |
| Working branch | `cursor/stage-a-reproduction-093e` |
| Report commit | recorded at commit time in git; see `HEAD` on that branch |
| Pinned library `main` | `f657b954da7d48d05b50f6f4dc967595a155f7ae` |
| `src/redisca` vs pinned main | unchanged |
| PR against `main` | draft https://github.com/ivsemenkov/ReDisCA-dev/pull/8 (do not merge) |

Old `paper` branch `e54dd260d11cd1b5b71b73af30519d5f3f3b8aef` was source
material only. Experiment code imports `redisca.ReDisCA` through
`paper.reproduction.common.method.make_redisca`. It does not depend on
`source_faithful.py`, `git show`, another branch, MATLAB, or fixtures
from another implementation.

## 2. Files added or changed

All new work is under `paper/` plus small pytest wiring
(`pyproject.toml`, `tests/conftest.py`, `.gitignore`).
**Nothing in `src/redisca`.**

Principal code:

- `paper/reproduction/common/method.py` — single factory / AIRI-SPoC kwargs
- `paper/reproduction/common/inference_secondary.py` — paper-described permutation and **both** AIRI temporal indexings
- `paper/reproduction/validation/oracle.py` — test-only independent AIRI/SPoC formula check
- `paper/reproduction/{n170,meg,simulations,source_localization}/`
- `paper/reproduction/__main__.py` — `test` / `download` / `stage-a`
- `paper/reproduction_manifest_addendum.json` — review-added candidates

Lightweight result JSON is in `paper/results/`. Large arrays stay in
gitignored `.reproduction_data/`.

## 3. Tests

```text
python -m paper.reproduction test
141 passed
```

Historical
validation is the independent Python oracle in
`paper/reproduction/validation/oracle.py` compared with
`ReDisCA(**AIRI_SPOC_KWARGS)`. MATLAB is unavailable. NumPy RNG is not
MATLAB `rand` bitwise parity.

Added after the Stage A review (not a method change):

- literal vs corrected AIRI half-split indexing
- Fig. 17 lowest-*p* three-component selection (not *p*<0.05)
- simulation-generation branches (I_c=100, norm-15% δ, global γ,
  fixed noise loci, C=5-from-scratch)
- original-manifest vs addendum provenance

## 4. What was run versus only prepared

This inventory is the reason there is **no final verdict**.

| Track | Status |
| --- | --- |
| Historical AIRI/SPoC validation | **Run** (unit/oracle) |
| N170 both ERP states × 5 seeds, B=1000 | **Run** |
| MEG component inference: AIRI / PAPER-1501 / PAPER-1500 × 5 seeds, B=1000 | **Run** |
| MEG temporal: AIRI-LITERAL-INDEXING and AIRI-CORRECTED-POOLED, Nmc=100, 5 seeds | **Run** (companion `temporal_airi_seed*.json`; does not replace `seed*.json` *p*-values) |
| Fig. 18 MUSIC on *p*<0.05 components (2-D when n_sig=2) | Run for seed 20240904 only; **not** the paper Fig. 17 rule |
| Fig. 18 MUSIC on the three lowest-*p* components (3-D) | **Run** (5 seeds × MEG-PAPER-1501 and MEG-AIRI). Selection and patterns are deterministic across seeds. |
| SIM-P1 Fig. 4 SNR=0.2 seeds 20240904–20240907; SNR=0.1 seeds 20240904–20240907 | **Run** (near-chance / ~7 cm) |
| SIM-P1 Figs 5–6 both SNRs seeds 20240904–20240906; SNR=0.4 seed 20240907 | **Run** (~7 cm, no C-trend) |
| SIM-P1 remaining (Fig. 4 both SNRs seed 08; Fig. 5/6 0.2/07 and both SNRs seed 08) | **Running without RSA** |
| SIM-P2 Fig. 4 both SNRs × 5 seeds | **Run** (near-chance). Fig. 5/6 not in the required 160-job matrix |
| SIM-P3 Fig. 4 both SNRs × 5 seeds | **Run** (paper-like localization) |
| SIM-P3 Figs 5–6 both SNRs × 5 seeds | **Run** (C-trend on all seeds; C=6 still >2 cm) |
| SIM-P4 / SIM-P6 Fig. 4 both SNRs × 5 seeds | **Run** (near-chance) |
| SIM-P6 Figs 5–6 SNR=0.4 seeds 20240904–20240907; SNR=0.2 seeds 20240904–20240907 | **Run** (~7.1–7.6 cm, no C-trend). Remaining P6 / all P4 Fig. 5/6 unfinished |
| SIM-P5 Fig. 4 both SNRs seeds 20240904–20240907; Figs 5–6 seeds 20240904–20240906 | **Run** (~7 cm). Remaining seeds running |
| SIM-P7 Fig. 4 both SNRs × 5 seeds | **Run** (near-chance). Fig. 5/6 not started |
| SIM-R1 Fig. 4 seeds 20240904–20240906 both SNRs | **Run** (near-chance / ~7–8 cm). Remaining R1 and all SIM-P8 unfinished |

Existing `seed*.json` MEG files store a `temporal_airi` block that is the
**corrected pooled** indexing and must not be labeled “literal AIRI
executable.” Both indexings are now in `temporal_airi_seed*.json`.

## 5. Fixed AIRI-SPoC configuration (everywhere)

```python
AIRI_SPOC_KWARGS = dict(
    n_components=None,
    demean_time=True,
    divide_by_t_minus_1=True,
    directed_pairs=True,
    aggregation="mean",
    solver="whitening",
    rank=None,
    rank_tol=1e-6,
)
```

Every fit is equivalent to `ReDisCA(**AIRI_SPOC_KWARGS).fit(X, rdm)`
via `make_redisca()` / `fit_redisca()`. No unique-pair, printed-Gram,
no-`1/(T-1)`, sum aggregation, generalized solver, library-default, or
“paper core” ReDisCA runs.

Primary component inference is `redisca.random_phase_test()`:
stock-SPoC random-phase, B=1000, null = max(|λ|), p = count/B, no +1,
p=0 allowed, observed estimator not refit.

## 6. Candidate matrix and seeds

**Seeds (frozen before results):** `20240904`, `20240905`, `20240906`,
`20240907`, `20240908`. Do not cherry-pick seeds.

Exact original values: `paper/reproduction_manifest.json`.
Review-added branches: `paper/reproduction_manifest_addendum.json`.

### Original freeze (not rewritten)

**Simulations**

- `SIM-P1` (primary, **C** on unspecified I_c / mesh / Υ_d / fs):
  fs=1000 Hz, I_c=40, zero-phase 6th-order 2 Hz Butterworth, AD
  204-planar constrained forward, Υ_d = symmetric Gaussian on unique
  pairs with σ=0.05·std(D0), clip ≥0. Literal δ covariance (**A**
  printed formula). Per-trial γ. Noise loci redrawn every epoch.
  Fig. 4 SNR {0.2, 0.1}. Fig. 5/6 from C=6 generation.
- `SIM-P2` (**C**): I_c=80. After review, Fig. 5/6 coverage is also run.
- `SIM-P3` (**C**): causal Butterworth. Original freeze: Fig. 4
  SNR=0.1 only. **2026-09-06 addendum:** also Fig. 4 SNR=0.2 and
  Figs 5–6, because the first finished seed is the first
  paper-like simulation localization. Not pre-registered as fig5.

**N170 / MEG / source (original)** — unchanged IDs. See the original
manifest.

### Review-added forensic / implementation branches

These were **not** in the original freeze. Each is tagged
`added_after_stage_a_review: true`.

| ID | Tag | What | Why added |
| --- | --- | --- | --- |
| `AIRI-LITERAL-INDEXING` | **B** | MATLAB `data(:,:,rpm)` half-split | Committed AIRI indexes `rpm`, not `idxAll(rpm)` |
| `AIRI-CORRECTED-POOLED` | **C** | `data(:,:,idxAll(rpm))` | Intended pooled-class split; previous Python only did this |
| `FIG18-MUSIC-LOWESTP` | **A** | 3-D MUSIC on the three lowest-*p* Fig. 17 components | Paper: “The three components with the lowest p-values are shown in Fig. 17.” Previous scan used *p*<0.05 (2-D) |
| `SIM-P4` | **C** | I_c=100 | Fig. 4 panel titles show “100 trials”; caption also says 100 MC. Not pre-registered |
| `SIM-P5` | **D** | rescale ‖δ‖ = 0.15‖g‖ | Literal Σ=σ²I makes typical ‖δ‖ ≳ 2‖g‖ in 204-D. Not called paper-faithful |
| `SIM-P6` | **C** | one global γ per realization | Paper: ratio of RMS of the noiseless and noise **matrices** |
| `SIM-P7` | **C** | 1000 noise loci seeded once | “Randomly seeded” + new time series per trial |
| `SIM-P8` | **C** | generate C=5 from scratch for Fig. 5 | Body C=5 vs caption C=6 |
| `SIM-R1` | **D** | I_c=100 + norm-15% + global γ + fixed loci | Coherent composite, **not** tuned to 85% |

ReDisCA kwargs are identical on every branch.

### Source re-inspection notes (2026-09-05)

Published PDF and bioRxiv preprint were re-read before adding the
forensic branches. Ossadtchi et al. 2018 public PSIICOS
(`sim_psiicos.m`) is **not** this RDM recipe; `n_tr=100` there is not
evidence for ReDisCA I_c.

- Figs 13–15: “we considered only the first three statistically
  significant ReDisCA components.” That is a presentation choice. It
  does **not** say that exactly three components were significant.
  `n_sig >= 3` is compatible with that sentence. A fourth/fifth
  *p*<0.05 component is **not** a mismatch to it. The first three
  components, their *p*-values, patterns, and timing are still compared
  to the paper.
- Fig. 17: three **lowest *p*** components; Fig. 18 uses that subspace.
- Fig. 5 caption C=6 vs body C=5 remains a paper-internal conflict.
- Printed δ is Σ = σ²I, σ = 0.15‖g‖. The preprint says “variance equal
  to 15% of gm0’s norm.” Both readings are retained and labeled.
- SNR sentence about two **matrices** supports a singular γ. Per-trial
  γ stays as the original SIM-P1 reconstruction.

## 7–8. Per-experiment results and published anchors

### N170 EEG (Figs 7–11)

**Complete** (5 seeds, B=1000). This track is finished; it is not a
Stage A winner by itself.

**`N170-UNFILT` is the only N170 candidate in the same neighborhood as
the paper.** Eigenvalues and RDM correlations are deterministic across
the five seeds; random-phase *p* varies.

| Analysis | Paper | `N170-UNFILT` |
| --- | --- | --- |
| Face @ 200 ms λ | 0.87209 | **0.88010** (Δ +0.008) |
| Face *p* | 0 | **0** (5/5) |
| Face RDM corr | 0.81556 / caption 0.82 | **0.9482** (not 0.82) |
| Face pattern | occipital / rFG | max-abs **PO4** |
| Car @ 170 λ1, λ2 | 0.91639, 0.77036 | **0.91312, 0.79043** |
| Car @ 170 *p*1 | 0 | **0** (5/5) |
| Car @ 170 *p*2 | 0.009 | 0.011, 0.006, 0.012, 0.010, 0.006 (median **0.010**) |
| Car @ 170 RDM corr | 0.99074, 0.93002 | **0.9798, 0.9507** |
| Car @ 200 (panel *t*=0.2 s) | — | λ1=0.91489 *p*1=0; λ2=0.831; *p*2≈0.002–0.006 |
| Meaning ~400 ms, 25 ms step | three adjacent *p*<0.05 | **400 and 425 ms** only; **375 ms** *p*≈0.073–0.082 |

1-sample stepping finds a contiguous significant cluster ≈387–426 ms,
plus an isolated early window at 7.8 ms.

**`N170-LP20` does not match this N170 track.** Face λ=0.957, RDM corr
**−0.16**; meaning *p*≈0.17 at 375/400/425 ms; no significant 400 ms
segment.

**Secondary paper-described condition-label permutation** (exact C!=24
for N170; C!=720 for MEG) does **not** produce paper-like *p*=0
(N170 face *p*_maxabs=0.5; MEG first-component *p*_maxabs≈0.2–0.4).
Random-phase is what produces small *p*. This is a labeled source
discrepancy (paper text vs AIRI/stock-SPoC executable), not a method
ablation.

### MEG (Figs 12–17)

**Component inference is complete** (5 seeds, B=1000). **Both AIRI
temporal indexings are complete** (Nmc=100, five seeds). Fig. 18
lowest-*p* is complete. Simulations are not. Do not treat Stage A as
closed.

**`n_sig` vs “three components” (corrected reading).**

Paper Figs 13–15: “we considered only the first three statistically
significant ReDisCA components.” Compatible statement: the paper
*shows* the first three significant components. Incompatible
over-reading (now withdrawn): the paper claims that *exactly* three
were significant.

| Candidate | Face n_sig (*p*<0.05) | Compatible with “first three significant”? |
| --- | --- | --- |
| `MEG-AIRI` | **4–5** | **Yes** (`n_sig >= 3`). Still compare the first three λ / *p* / patterns / timing |
| `MEG-PAPER-1501` | **2** (c3 *p*≈0.12) | **No** for that sentence (fewer than three significant) |
| `MEG-PAPER-1500` | **2** (same as 1501 to ~10⁻⁴ in λ) | **No** for that sentence |

AIRI is **not** penalized merely for a fourth/fifth significant
component. PAPER-1501 under-discovering face (2 vs ≥3) remains a
mismatch to “first three statistically significant.”

**`MEG-AIRI` (5/5 seeds), first three face components.** First-component
random-phase *p* median ≈0.006 (range 0.005–0.008).

AIRI temporal, **both** indexings (Nmc=100, five seeds): face component 1
first *p*− interval is **≈250–355 ms**. Neither literal MATLAB indexing
nor corrected pooled indexing recovers **65 ms** or **160 ms** as a
first onset. The two indexings are close to each other on this contrast
(the max/min-over-time test is the same). Paper-FWER face component 1
starts ≈244 ms.

**`MEG-PAPER-1501` (5/5).** Face has 2 significant components.
Paper-FWER face component 1 starts ≈113–115 ms: that interval can cover
160 ms and 311 ms inside a long later block, but **still misses 65 ms**.
Two-level AIRI vs binary RDMs are identical after target
standardization.

**`MEG-PAPER-1500` (5/5).** Matches 1501 to ~10⁻⁴ in λ.

| MEG timing anchor (paper) | MEG-AIRI literal / corrected | MEG-PAPER-1501 literal / corrected / FWER |
| --- | --- | --- |
| Face c1 ~65 ms | no / no | no / no (AIRI *p*− starts ~105–110 ms; FWER ~113 ms) |
| Face c1 peak ~160 ms | no (AIRI *p*− ~250 ms; FWER ~244 ms) | later block after ~105–113 ms |
| Face c1 second ~311 ms | yes, inside later block | yes, inside later block |
| Face c2 ~218 ms | often yes | not as a first onset |
| Tool c1 ~210 ms | often in AIRI *p*− / FWER | FWER starts ~163–202 ms |
| Meaning c1 ~160 ms | no | FWER ~129 ms then long later block |
| Face vs tool ~202 ms | mixed / seed-dependent | FWER ~186–202 ms (long block) |

Literal AIRI indexing (**B**) and corrected pooled (**C**) are both
finished. Neither changes `max(aa,[],2)` / `min(aa,[],2)`. Switching
the index lookup does not create the 65 ms face onset.

### Simulations (Figs 3–6)

**Incomplete (103 / 160 required n_mc=100 jobs).** The pre-registered
five-seed design and the review-expanded generation matrix are still
running. Numbers below are interim. They are **not** a failure verdict.
`stage_a_status.final_verdict_allowed` remains false.

**Fig. 4, `SIM-P1`, seed 20240904, n_mc=100** (not `--quick`):

| SNR | ReDisCA AUC | TPR @ FPR=0.01 | Median loc. error |
| --- | --- | --- | --- |
| 0.2 | 0.527 | 0.020 | **7.37 cm** |
| 0.1 | 0.522 | 0.004 | **7.91 cm** |

RDM recovery on the fitted component is excellent (corr ≈0.999). RSA
baselines are also weak at low FPR. This does not yet match the paper’s
ReDisCA-dominates-near-zero-FPR / tight-localization picture **under
this one reconstruction**. Other generation branches (I_c=100,
norm-15% δ, global γ, fixed loci, and the SIM-R1 composite) are
required before that sentence can be generalized.

A noiseless diagnostic (not a retune): the fitted pattern matches the
**perturbed** topography exactly, but the paper-literal
`σ = 0.15‖g‖` isotropic Gaussian in 204-D makes `‖δ‖` typically
`≳ 2‖g‖`. Even without brain noise, cosine scan against `g_true` is
poor. That is why SIM-P5 exists as a labeled **D** candidate. The
literal covariance branch is kept.

**Figs 5–6, `SIM-P1`, SNR=0.4, seed 20240904, n_mc=100**
(C=6 generation; evaluate C∈{3,4,5,6}):

| C | Mean median loc. error | Frac. error <1 cm | Mean pattern corr | Mean weight corr | Mean RDM corr |
| --- | --- | --- | --- | --- | --- |
| 3 | 7.44 cm | 0.005 | 0.174 | 0.001 | 0.455 |
| 4 | 7.44 cm | 0.003 | 0.176 | 0.001 | 0.669 |
| 5 | 7.48 cm | 0.003 | 0.170 | 0.001 | 0.707 |
| 6 | 7.55 cm | 0.003 | 0.172 | 0.001 | 0.711 |

Paper Fig. 6: ReDisCA mean median error **< 2 cm at C=6**.
Paper Fig. 5: patterns much better aligned with true topographies than
weights. Under **this** SIM-P1 reconstruction, weights are ~0, patterns
~0.17, localization stays ~7.5 cm, and error does not fall as C
increases. That is an interim observation, not a closed Stage A result.

**SIM-P5 (D, norm-15% δ only), seed 20240904, n_mc=100:**

| Job | AUC / TPR@0.01 | Median loc. | Pattern corr | Notes |
| --- | --- | --- | --- | --- |
| Fig. 4 SNR 0.2 | 0.519 / 0.008 | 7.08 cm | — | no RSA |
| Fig. 4 SNR 0.1 | 0.519 / 0.007 | 7.90 cm | — | no RSA |
| Fig. 5/6 SNR 0.4, C=3…6 | — | 7.37–7.57 cm | ≈0.17 | frac&lt;1 cm ≈0; RDM corr 0.23–0.33 |

Rescaling ‖δ‖ to 0.15‖g‖ **alone** does not produce the published ROC
or &lt;2 cm Fig. 6 picture.

**SIM-P2 (C, I_c=80), Fig. 4, all five seeds × both SNRs, n_mc=100
(this branch now complete for Fig. 4).**
AUC 0.478–0.530; median localization 7.08–8.43 cm. Doubling I_c
does not recover the paper ROC on any registered seed. Fig. 5/6
for SIM-P2 is not in the required verdict matrix.

Second-seed Fig. 4 SNR=0.2 (n_mc=100) is in the same neighborhood:
SIM-P1 AUC 0.487, median 6.72 cm; SIM-P5 AUC 0.495, median 7.00 cm;
SIM-P2 seed 20240905 AUC 0.478, median 7.54 cm. The near-chance /
~7 cm picture is not a one-seed fluke under these reconstructions.

**SIM-P6 (C, one global γ), Fig. 4, all five seeds × both SNRs,
n_mc=100 (this branch now complete for Fig. 4):**

| SNR | AUC range (5 seeds) | TPR@0.01 range | Median loc. range |
| --- | --- | --- | --- |
| 0.2 | 0.482–0.523 | 0.006–0.020 | 6.75–7.65 cm |
| 0.1 | 0.446–0.523 | 0.004–0.028 | 6.60–7.85 cm |

Switching from per-trial γ to one matrix-wide γ does **not** recover
the paper ROC on any registered seed.

**SIM-P6 Figs 5–6** (eight finished jobs, n_mc=100):

| SNR | Seed | C=3 | C=4 | C=5 | C=6 | Pattern @ C=6 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 20240904 | 7.48 | 7.45 | 7.48 | **7.59 cm** | 0.172 |
| 0.4 | 20240905 | 7.48 | 7.18 | 7.37 | **7.34 cm** | 0.187 |
| 0.4 | 20240906 | 7.36 | 7.14 | 7.12 | **7.20 cm** | 0.178 |
| 0.4 | 20240907 | 7.28 | 7.25 | 7.43 | **7.26 cm** | 0.182 |
| 0.2 | 20240904 | 7.24 | 7.29 | 7.36 | **7.31 cm** | 0.169 |
| 0.2 | 20240905 | 7.22 | 7.26 | 7.23 | **7.27 cm** | 0.186 |
| 0.2 | 20240906 | 7.19 | 7.08 | 7.04 | **7.44 cm** | 0.179 |
| 0.2 | 20240907 | 7.51 | 7.38 | 7.34 | **7.14 cm** | 0.172 |

No C-trend. Pattern ~0.17–0.19, same neighborhood as zero-phase
SIM-P1. Global γ alone does **not** produce paper Fig. 6. Remaining
SIM-P6 Fig. 5/6 seeds are running. This is an interim observation,
not a Stage A verdict.

**SIM-P4 (C, I_c=100, post-review), Fig. 4, all five seeds × both
SNRs, n_mc=100 (this branch now complete for Fig. 4).**
AUC 0.471–0.553; median loc. 6.38–8.31 cm. Reading the Fig. 4
title “100 trials” as within-condition I_c does **not** recover
the paper ROC.

**SIM-P3 (C, causal Butterworth; original freeze).** Fig. 4, SNR=0.1,
n_mc=100, **all five registered seeds finished.**

| Seed | AUC | TPR@0.01 | Median loc. |
| --- | --- | --- | --- |
| 20240904 | **0.836** | **0.296** | **0.88 cm** |
| 20240905 | **0.856** | **0.326** | **1.04 cm** |
| 20240906 | **0.847** | **0.322** | **1.11 cm** |
| 20240907 | **0.862** | **0.329** | **1.13 cm** |
| 20240908 | **0.859** | **0.314** | **0.95 cm** |
| SIM-P1 zero-phase, seed 20240904, same SNR | 0.523 | 0.022 | 6.44 cm |

Range at the published SNR=0.1: AUC 0.836–0.862, TPR@0.01
0.296–0.329, median loc. 0.88–1.13 cm. This is the first
generation setting under the frozen AIRI-SPoC factory that is
not pathologically far from the paper Fig. 4 picture, and it is
stable across the pre-registered seeds.

Fig. 4 SNR=0.2 is now also complete on all five seeds:
AUC 0.874–0.887, TPR@0.01 0.348–0.382, median loc. 0.61–0.80 cm.

Figs 5–6 under the same causal generation: **all five registered
seeds finished** at both SNRs (20240904–20240908). This SIM-P3
Fig. 5/6 branch is complete. It is **not** a reproducing pipeline.

| SNR | Seed | C=3 | C=4 | C=5 | C=6 |
| --- | --- | --- | --- | --- | --- |
| 0.4 | 20240904 | 5.49 | 4.59 | 3.88 | **3.46 cm** |
| 0.4 | 20240905 | 5.58 | 4.57 | 3.91 | **3.37 cm** |
| 0.4 | 20240906 | 5.56 | 4.62 | 3.65 | **3.52 cm** |
| 0.4 | 20240907 | 5.61 | 4.38 | 3.84 | **3.53 cm** |
| 0.4 | 20240908 | 5.34 | 4.81 | 4.12 | **4.00 cm** |
| 0.2 | 20240904 | 5.62 | 4.81 | 4.27 | **4.09 cm** |
| 0.2 | 20240905 | 5.81 | 4.65 | 4.30 | **3.94 cm** |
| 0.2 | 20240906 | 5.58 | 5.08 | 4.56 | **4.06 cm** |
| 0.2 | 20240907 | 5.96 | 5.03 | 4.40 | **3.93 cm** |
| 0.2 | 20240908 | 5.46 | 4.93 | 4.40 | **4.24 cm** |

At SNR=0.4, C=6, frac&lt;1 cm is 0.24–0.29 and mean pattern corr
is 0.28–0.29 (weights ~0). RDM corr is 0.95–1.00. Error **falls as C
increases** on all five seeds. Seed 20240908 is the weakest of the
five (C=6 = 4.00 / 4.24 cm).

Compared with SIM-P1 (~7.5 cm, no C trend, pattern ~0.17) this is
better. Paper Fig. 6 asks for **< 2 cm at C=6**; five seeds are
**3.37–4.00 cm** (SNR=0.4) and **3.93–4.24 cm** (SNR=0.2). Do not
treat this as a reproducing pipeline.

Mechanism (same vertex / same literal δ on a matched one-MC
replay): zero-phase `filtfilt` makes source RMS ~10³× larger and
the fitted pattern **does not** match the used topography
(pattern–topo corr ≈0.07). Causal `sosfilt` yields a pattern that
**does** match the used topography (corr ≈0.98). The leadfield
scan of that recovered topography then peaks near the true vertex
even though ‖δ‖≈2‖g‖, because MEG columns are correlated. The
paper states a 6th-order 2 Hz Butterworth and does **not** say
zero-phase; MATLAB `filter` is causal. SIM-P3 was already in the
original freeze.

On 2026-09-06 the addendum therefore expands SIM-P3 to Fig. 4
SNR=0.2 and Figs 5–6. That expansion is **not** pre-registered.
ReDisCA kwargs are unchanged.

**SIM-P7 (C, fixed 1000 noise loci)** Fig. 4 is now complete
(all five seeds × both SNRs): AUC 0.492–0.544, TPR@0.01
0.005–0.021, median 6.84–8.15 cm. Seeding loci once does not
recover the paper ROC. Fig. 5/6 for SIM-P7 is still required.

**SIM-R1 (D, composite)** Fig. 4, six finished jobs, n_mc=100.
Seeds 20240904–20240906 both SNRs: AUC 0.493–0.547, TPR@0.01
0.012–0.023, median 6.80–7.94 cm. Newest SNR=0.1 seed 20240906:
AUC 0.505 / 7.69 cm. I_c=100 + norm-15% δ + global γ + fixed loci
does **not** recover the paper ROC on these seeds. Remaining R1
jobs are running.

**SIM-P5 (D)** Fig. 4 seed 20240907 now has both SNRs: SNR=0.2 AUC
0.504 / 7.43 cm; SNR=0.1 AUC 0.462 / 7.62 cm. Figs 5–6 seeds
20240904–20240906 at both SNRs still have C=6 at 7.14–7.66 cm, no
C-trend.

**SIM-P1** Fig. 5/6 SNR=0.4 seed 20240907: C=6 **7.28 cm**, pattern
0.18, no C-trend — same neighborhood as seeds 20240904–20240906.
Fig. 4 SNR=0.1 seed 20240907 remains AUC 0.446 / 7.39 cm.

**Still unfinished (no final verdict):** remaining SIM-P1 / SIM-P5
seeds, SIM-P4 / SIM-P6 / SIM-P7 / SIM-P8 Fig. 5/6, and remaining
SIM-R1.

Default SIM-P1 generation (literal δ, per-trial γ, per-epoch loci,
I_c=40, generate C=6) is hash-compatible with the original freeze.
Review knobs are opt-in.

### Source localization (Fig. 18)

**The completed `fig18_seed20240904.json` is not the paper path.**

That file used *p*<0.05 components of MEG-PAPER-1501 facevstool
(components 0 and 1; 2-D subspace) and peaked at **left V2 / superior
occipital / lateral occipital** (subcorr 0.79), not the paper’s right
fusiform, right insula, left intraparietal, and anterior-central set.

Paper Fig. 17: “The three components with the lowest *p*-values are
shown.” For MEG-PAPER-1501 facevstool those *p*-values (already on
disk, B=1000, all five seeds) are always increasing in component
index, so the lowest three are **[0, 1, 2]** even though only two
cross 0.05. Patterns are **deterministic** across the five registered
seeds (`patterns_hash` identical). MEG-AIRI facevstool likewise always
selects **[0, 1, 2]**; four components are *p*<0.05, so the paper rule
still takes the three lowest, not all significant ones.

**`FIG18-MUSIC-LOWESTP` is now complete** (5 seeds × MEG-PAPER-1501 and
MEG-AIRI). Selection is `[0, 1, 2]` for every registered seed.
`patterns_hash` is identical across seeds, so MUSIC was computed once
per MEG candidate and reused.

| MEG candidate | Selected (lowest *p*) | Selected *p* (seed 20240904) | n_sig *p*<0.05 | Peak | Paper regions? |
| --- | --- | --- | --- | --- | --- |
| MEG-PAPER-1501 | 0, 1, 2 | 0.007, 0.021, 0.093 | 2 | vertex 89, left V2 / G_occipital_sup / lateral occipital, subcorr 0.787 | **No** (not rFG / insula / IPS / anterior-central) |
| MEG-AIRI | 0, 1, 2 | 0.006, 0.010, 0.017 | 4 (paper rule still takes 3, not 4) | vertex 2595, right occipital pole / lateral occipital, subcorr 0.838 | **No** |

Adding the third (non-significant) PAPER-1501 component does **not**
move the peak off the 2-D *p*<0.05 scan: same vertex 89. The
paper-described 3-D input therefore still does not recover the published
Fig. 18 anatomy on this OSF AD overlapping-spheres Gain.

AIRI sLORETA of author-saved `A1(:,4)` and of the local MEG-AIRI
facevstool pattern peak in lingual / occipital cortex.
Literal AIRI `P = eye(size(Nsns,1))` is `eye(1)` and non-executable;
the labeled `P=eye(Nsns)` branch peaks at the right occipital pole.

## 9. Which external pipeline best reproduces the complete paper?

**No answer yet. Stage A is incomplete. No winner. No failure
declaration.**

What can be said without pretending the matrix is finished:

- **N170 (complete):** `N170-UNFILT` is in the paper’s neighborhood;
  `N170-LP20` is not. Face RDM corr is 0.95 vs printed 0.82.
- **MEG component count (complete inference, corrected reading):**
  AIRI `n_sig >= 3` is compatible with “first three statistically
  significant.” PAPER-1501 face `n_sig=2` is not.
- **MEG early timing (both AIRI indexings finished):** neither literal
  nor corrected recovers 65 ms. PAPER-1501 AIRI *p*− starts ~105 ms;
  AIRI window starts ~250 ms.
- **Fig. 18 (now run on the paper 3-D input):** lowest-*p* three
  components, deterministic across seeds. Peaks remain occipital (left
  V2 for PAPER-1501; right occipital pole for AIRI), not the published
  rFG / insula / IPS set. This is one finished track, not a global
  verdict.
- **Simulations (103/160, not a verdict):** SIM-P3 causal Butterworth
  Fig. 4 is paper-like on all five seeds (median 0.88–1.13 cm at
  published SNR=0.1). SIM-P3 Fig. 5/6 is complete: C-trend present,
  but C=6 is 3.37–4.00 cm (paper &lt;2 cm). Eight SIM-P6 Fig. 5/6
  jobs and SIM-P1 Fig. 5/6 SNR=0.4 seed 20240907 stay ~7.1–7.6 cm
  with no C-trend. SIM-R1 Fig. 4 (six jobs) stays near-chance /
  ~7–8 cm. Finish the required matrix before any pipeline claim.
  Do not change ReDisCA.

## 10. Remaining irreducible discrepancies or missing source information

Interim list only. Several items may move after the missing runs.

- Face RDM corr 0.82 is **not** recovered from official
  `1_N170_erp_ar.erp` with Eq. 2 Pearson (got 0.95). Same mismatch as
  earlier forensics; not treated as a library bug.
- Meaning: two adjacent 25 ms windows, not three including 375 ms.
- Paper text “three ICA components” vs official ERP CORE subject-1
  list **2 and 7** only.
- MEG 65 ms face onset is not produced by the finished inferential
  pieces under the fixed fit.
- Simulation mesh, numeric I_c, fs, Υ_d law, 1/f construction,
  MNE/LCMV details, and Fig. 6 SNR are unspecified. `SIM-P1` is a
  frozen MEG-like reconstruction, not an exact paper recipe.
- Paper-literal `σ_δ = 0.15‖g‖` with `Σ = σ²I` is destructive in
  204-D. The **D** alternative is labeled and was **not** used to
  overwrite the literal branch.
- Fig. 5 C=5 vs C=6 is a paper-internal conflict; both generation
  stories are now instantiated.
- Individual MRI for Fig. 18 is not public; the OSF AD
  overlapping-spheres Gain was used.

## 11. Suspected core-library issue

**None found.** `src/redisca` matches pinned `main`. Failures sit in
external analysis (preprocessing, windows, RDM fill, inference wording,
simulation reconstruction) or in the historical AIRI-SPoC configuration
itself. `main` was not patched.

## 12. Commands

```bash
python -m paper.reproduction test
python -m paper.reproduction download all
python -m paper.reproduction.summarize
python -m paper.reproduction.meg.run_temporal
python -m paper.reproduction.source_localization.run --skip-airi-aux
python -m paper.reproduction.simulations.run --candidate SIM-P1
python -m paper.reproduction.simulations.run --candidate SIM-P2
python -m paper.reproduction.simulations.run --candidate SIM-P3
python -m paper.reproduction.simulations.run --candidate SIM-P4
# … SIM-P5 SIM-P6 SIM-P7 SIM-P8 SIM-R1
# missing required jobs, Fig. 4 first, file-locked:
python -m paper.reproduction.simulations.queue --prefer fig4
```

`--quick` is **non-reproduction**. Do not report reduced MC/B output as
a paper result. The runner skips complete `n_mc>=100` files so finished
SIM-P1 Fig. 4 / Fig. 5 SNR=0.4 seed 20240904 are not overwritten.

---

Stage A continues on this branch until the required matrix exists.
No method ablation has been started. No reproducing-pipeline verdict
is offered in this revision.
