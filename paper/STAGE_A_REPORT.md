# Stage A report: ReDisCA paper reproduction

Ossadtchi et al., *NeuroImage* 301 (2024) 120868, using **exactly one**
ReDisCA configuration: the AIRI → stock-SPoC settings on current `main`.

This is a scientific reproduction report, not a method-development or
ablation note. The ReDisCA constructor was not changed anywhere.

Machine-readable companions:

- `paper/reproduction_manifest.json` — candidate matrix frozen **before** full-result selection
- `paper/results/coverage.json` — what was actually executed
- `paper/results/stage_a_summary.json` — numeric extracts vs paper anchors
- per-run JSON under `paper/results/`

`--quick` leftovers must not be read as reproduction results.

---

## 1. Working branch and commit

| Item | Value |
| --- | --- |
| Working branch | `cursor/stage-a-reproduction-093e` |
| Report commit | recorded at commit time in git; see `HEAD` on that branch |
| Pinned library `main` | `f657b954da7d48d05b50f6f4dc967595a155f7ae` |
| `src/redisca` vs pinned main | unchanged |
| PR against `main` | not opened (Stage A instruction) |

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
- `paper/reproduction/common/inference_secondary.py` — paper-described permutation and MEG temporal tests (reproduction-only)
- `paper/reproduction/validation/oracle.py` — test-only independent AIRI/SPoC formula check
- `paper/reproduction/{n170,meg,simulations,source_localization}/`
- `paper/reproduction/__main__.py` — `test` / `download` / `stage-a`

Lightweight result JSON is in `paper/results/`. Large arrays stay in
gitignored `.reproduction_data/`.

## 3. Tests

```text
python -m paper.reproduction test
117 passed
```

MATLAB is unavailable. Historical validation is the independent Python
oracle in `paper/reproduction/validation/oracle.py` compared with
`ReDisCA(**AIRI_SPOC_KWARGS)`. NumPy RNG is not MATLAB `rand` bitwise
parity.

## 4. What was run versus only prepared

| Track | Status |
| --- | --- |
| Historical AIRI/SPoC validation | **Run** (unit/oracle) |
| N170 both ERP states × 5 seeds, B=1000 | **Run** |
| MEG-AIRI, MEG-PAPER-1501, MEG-PAPER-1500 × 5 seeds, B=1000, Nmc=100 | **Run** |
| Fig. 18 MUSIC + AIRI sLORETA + AIRI-MUSIC | **Run** (seed 20240904) |
| SIM-P1 Fig. 4, SNR 0.2 and 0.1, seed 20240904, n_mc=100 | **Run** |
| SIM-P1 Figs 5–6, SNR 0.4, seed 20240904, n_mc=100 | **Run** |
| SIM-P1 Figs 5–6, SNR 0.2 | Prepared; not written |
| SIM-P1 seeds 20240905–20240908; SIM-P2; SIM-P3 | Prepared; not started |

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

## 6. Pre-registered external candidate matrix and seeds

**Seeds (frozen before results):** `20240904`, `20240905`, `20240906`,
`20240907`, `20240908`.

Exact values and source rationales: `paper/reproduction_manifest.json`.

**Simulations**

- `SIM-P1` (primary): fs=1000 Hz, I_c=40, zero-phase 6th-order 2 Hz
  Butterworth, AD 204-planar constrained forward, Υ_d = symmetric
  Gaussian on unique pairs with σ=0.05·std(D0), clip ≥0.
  Fig. 4 SNR {0.2, 0.1}. Fig. 5 C∈{5,6}×SNR {0.4, 0.2}.
  Fig. 6 C∈{3,4,5,6}×SNR {0.2, 0.4}.
- `SIM-P2`: I_c=80 (not fully crossed).
- `SIM-P3`: causal Butterworth (not fully crossed).

**N170 (subject `"1"`)**

- `N170-UNFILT`: official ERP CORE `1_N170_erp_ar.erp`
- `N170-LP20`: documented 20 Hz low-pass `1_N170_erp_ar_lpfilt.erp`
- Face: 100 ms centered at 200 ms
- Car: **both** 170 ms (caption) and 200 ms (panel)
- Meaning: 150 ms windows; steps 25 ms and 3.90625 ms
- ICA: official subject-1 list is components **2 and 7 only**; no third
  component was invented

**MEG (subject AD, run 1, 204 planars, six subcategories)**

- `MEG-AIRI`: butter(3) 0.25–20 Hz `filtfilt`, MATLAB `trange` 600:1500
  → 99–999 ms, AIRI 0.1/0.5/1 RDMs, half-split Nmc=100
- `MEG-PAPER-1501`: no AIRI filter, full public −500…+1000 ms (1501 samples)
- `MEG-PAPER-1500`: no filter, first 1500 samples
- Filter and window were not mixed across candidates

**Source localization**

- `FIG18-MUSIC`: paper first-principal-angle scan of MEG-PAPER-1501
  facevstool significant patterns
- `AIRI-PRECOMP-SLORETA`: author-saved `A1(:,4)` and local MEG-AIRI pattern
- `AIRI-MUSIC-EYE-NSNS`: AIRI music with `P=eye(Nsns)` (literal `eye(1)` is
  non-executable)

These are external analysis branches, not method ablations.

## 7–8. Per-experiment results and published anchors

### N170 EEG (Figs 7–11)

**`N170-UNFILT` is the only N170 candidate that is in the same
neighborhood as the paper.** Eigenvalues and RDM correlations are
deterministic across the five seeds; random-phase *p* varies.

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

**`N170-LP20` does not reproduce.** Face λ=0.957, RDM corr **−0.16**;
meaning *p*≈0.17 at 375/400/425 ms; no significant 400 ms segment.

**Secondary paper-described condition-label permutation** (exact C!=24
for N170; C!=720 for MEG) does **not** produce paper-like *p*=0
(N170 face *p*_maxabs=0.5; MEG first-component *p*_maxabs≈0.2–0.4).
Random-phase is what produces small *p*. This is a labeled source
discrepancy (paper text vs AIRI/stock-SPoC executable), not a method
ablation.

### MEG (Figs 12–17)

**`MEG-AIRI` (5/5 seeds).** Face has **4–5** significant components
(*p*<0.05), not 3. First-component random-phase *p* median ≈0.006
(range 0.005–0.008). Neither AIRI half-split nor paper FWER recovers
the paper face onsets at **65 ms** or **160 ms**. Paper-FWER face
component 1 starts ≈244 ms. Later intervals (≈311 ms, ≈218 ms, some
tool/meaning late blocks) appear as pieces of longer significant
segments.

**`MEG-PAPER-1501` (5/5).** Face has **2** significant components
(component 3 *p*≈0.12). Paper-FWER face component 1 starts ≈113–115 ms:
that interval can cover 160 ms and 311 ms inside a long later block,
but **still misses 65 ms**. Two-level AIRI vs binary RDMs are identical
after target standardization (only two distinct off-diagonal values).

**`MEG-PAPER-1500` (5/5).** Matches 1501 to ~10⁻⁴ in λ. The extra public
sample is not the discrepancy.

| MEG timing anchor (paper) | MEG-AIRI | MEG-PAPER-1501 |
| --- | --- | --- |
| Face c1 ~65 ms | no | no |
| Face c1 peak ~160 ms | no (FWER starts ~244 ms) | inside later FWER block after ~113 ms |
| Face c1 second ~311 ms | yes, inside later block | yes, inside later block |
| Face c2 ~218 ms | often yes | not as a first onset |
| Tool c1 ~210 ms | often in AIRI *p*− / FWER | FWER starts ~163–202 ms |
| Meaning c1 ~160 ms | no | FWER ~129 ms then long later block |
| Face vs tool ~202 ms | mixed / seed-dependent | FWER ~186–202 ms (long block) |

### Simulations (Figs 3–6)

**Fig. 4, `SIM-P1`, seed 20240904, n_mc=100** (not `--quick`):

| SNR | ReDisCA AUC | TPR @ FPR=0.01 | Median loc. error |
| --- | --- | --- | --- |
| 0.2 | 0.527 | 0.020 | **7.37 cm** |
| 0.1 | 0.522 | 0.004 | **7.91 cm** |

RDM recovery on the fitted component is excellent (corr ≈0.999). RSA
baselines are also weak at low FPR. This does not match the paper’s
ReDisCA-dominates-near-zero-FPR / tight-localization picture.

A noiseless diagnostic (not a retune): the fitted pattern matches the
**perturbed** topography exactly, but the paper-literal
`σ = 0.15‖g‖` isotropic Gaussian in 204-D makes `‖δ‖` typically
`≳ 2‖g‖`. Even without brain noise, cosine scan against `g_true` is
poor. That is an implication of the published formula, not a parameter
search.

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
weights; RDM correlations high. Here weights are ~0, patterns ~0.17,
localization stays ~7.5 cm, and error does **not** fall as C increases.
RDM correlation does rise with C, but stays well below the
near-ceiling values implied by the figure.

Fig. 5 SNR=0.2 and the other four master seeds were not completed.

### Source localization (Fig. 18)

`FIG18-MUSIC` on MEG-PAPER-1501 facevstool significant components
(2 components, *p*<0.05) peaks at **left V2 / superior occipital /
lateral occipital** (subcorr 0.79), not the paper’s right fusiform,
right insula, left intraparietal, and anterior-central set.

AIRI sLORETA of author-saved `A1(:,4)` and of the local MEG-AIRI
facevstool pattern peak in lingual / occipital cortex.
Literal AIRI `P = eye(size(Nsns,1))` is `eye(1)` and non-executable;
the labeled `P=eye(Nsns)` branch peaks at the right occipital pole.

## 9. Which external pipeline best reproduces the complete paper?

**None**, without changing ReDisCA.

Closest fragments, reported without selecting a winner after the fact:

- **N170:** `N170-UNFILT` only. `N170-LP20` fails.
- **MEG component count:** AIRI over-discovers (4–5); paper-window
  under-discovers face (2 vs 3). Neither matches “three significant
  components” as shown.
- **MEG early timing:** `MEG-PAPER-1501` is closer (≈113 ms vs AIRI
  ≈244 ms) but still misses the 65 ms onset.
- **Simulations / Fig. 18:** no completed candidate reproduces the
  published ROC, <1–2 cm localization, or claimed cortical regions.

Different tracks prefer different external pipelines. Stage A does not
declare a single reproducing analysis stack.

## 10. Remaining irreducible discrepancies or missing source information

- Face RDM corr 0.82 is **not** recovered from official
  `1_N170_erp_ar.erp` with Eq. 2 Pearson (got 0.95). Same mismatch as
  earlier forensics; not treated as a library bug.
- Meaning: two adjacent 25 ms windows, not three including 375 ms.
- Paper text “three ICA components” vs official ERP CORE subject-1
  list **2 and 7** only.
- MEG 65 ms face onset is not produced by either inferential pipeline
  under the fixed fit.
- Simulation mesh, I_c, fs, Υ_d law, 1/f construction, MNE/LCMV
  details, and Fig. 6 SNR are unspecified. `SIM-P1` is a frozen
  MEG-like reconstruction, not an exact paper recipe.
- Paper-literal `σ_δ = 0.15‖g‖` with `Σ = σ²I` is destructive in
  204-D. That reading was **not** replaced after seeing chance ROC.
- Fig. 5 C=5 vs C=6 is a paper-internal conflict; both C values were
  evaluated from C=6 generation. Fig. 6 SNR=0.2 was not completed.
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
python -m paper.reproduction stage-a
# equivalent per-track:
python -m paper.reproduction.n170.run
python -m paper.reproduction.meg.run
python -m paper.reproduction.simulations.run --candidate SIM-P1
python -m paper.reproduction.source_localization.run
python -m paper.reproduction.summarize
```

`--quick` is **non-reproduction**. Do not report reduced MC/B output as
a paper result.

Remaining full runs that were prepared but not finished:

```bash
python -m paper.reproduction.simulations.run --candidate SIM-P1 --seeds 20240904
# then 20240905 20240906 20240907 20240908
python -m paper.reproduction.simulations.run --candidate SIM-P2
python -m paper.reproduction.simulations.run --candidate SIM-P3
```

Fig. 4 files for seed 20240904 already exist; the runner currently
recomputes them. Do not overwrite a finished `n_mc=100` file with
`--quick` output.

---

Stage A stops here. No method ablation was started.
