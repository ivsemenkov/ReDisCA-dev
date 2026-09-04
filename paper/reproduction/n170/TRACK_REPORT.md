# N170 track report (Ossadtchi et al. 2024, Figs 7–11)

Track owner: N170 worker. Branch: `cursor/paper-n170-f368`.
Working directory: `/tmp/redisca-worktrees/n170`.

This report records source evidence, decisions, commands, environment,
numbers, discrepancies, and blocked items. Compact machine copy:
`paper/results/n170/summary.json`. Production run 2026-09-04.

## Source evidence

| Claim | Source |
| --- | --- |
| Dataset ERP CORE N170, OSF pfde9, first participant index `"1"` | NeuroImage §4.2 / 4.2.1; manifest `n170_notes` |
| Four conditions: face, car, scrambled face, scrambled car | paper; ERP CORE `BDF_N170.txt` bins 1–4, correct only |
| Epoch [−200, 800] ms, 256 Hz | paper; ERP CORE Script 5/7 |
| ICA: paper “three ocular+cardiac”; subject `"1"` removes **2 and 7** | D11; `ICA_Components_N170.xlsx` SHA-256 `23373a2b…cf85`; Script 4 |
| Meaning: sliding T=150 ms, uncorrected p<0.05 around t=400 ms, occipital | Fig. 7, §4.2.1 |
| Sliding **step not printed** | paper_methods.md; manifest `sliding_step_ms: null` |
| Three adjacent windows ~400 ms | Fig. 8 |
| Face / car theoretical RDMs | Fig. 9a,b (images; numerics not printed) |
| Face: T=100 ms centered at 200 ms; RDM corr **0.82**; one significant component | Fig. 10 |
| Car: applied at t=170 ms; two comps p<0.01; corr **>0.99** | Fig. 11; duration not restated |
| Pair matrix: unscaled Gram, no demean | Eq. 4; D2 |
| Inference: permute condition labels | §2.3; D5 (AIRI/SPoC random-phase is a different test) |
| No AIRI N170 code | D12 |
| Preferred ERP file | `1_N170_erp_ar.erp` (manifest) |
| Library pin | `from redisca import ReDisCA` @ `5a5c865` |

Official ERP CORE pipeline (Luck lab commit `c18b43d`): event shift +26 ms,
downsample 1024→256 Hz, average reference of 33 EEG-typed channels, bipolar
HEOG/VEOG, 0.1 Hz Butterworth high-pass, ICA on chans 1:31, remove listed
components, epoch [−200, +800] ms, interpolate + AR, average good trials.

## Decisions (not silent paper values)

1. **Use official precomputed averages**, not a re-run of ICA, and not deleted
   student N170 code. ICA components 2 and 7 are already removed in the `.erp`.
   A third component is **not** invented (D11).
2. **Channel set = 28 scalp EEG** in ERPLAB order:
   `FP1 F3 F7 FC3 C3 C5 P3 P7 PO7 PO3 O1 Oz Pz CPz FP2 Fz F4 F8 FC4 FCz Cz C4 C6 P4 P8 PO8 PO4 O2`.
   Dropped: `HEOG_left`, `HEOG_right`, `VEOG_lower`, `(corr) HEOG`, `(corr) VEOG`,
   `(uncorr) HEOG`, `(uncorr) VEOG`. P9 and P10 are not in the ERP (ERP CORE
   Script 1 skipped them). The paper does not name N or the channel list.
3. **Primary ERP** = `1_N170_erp_ar.erp` (unfiltered averages). The 20 Hz
   low-pass file is not the default (`--lpfilt` exists). Short windows on the
   lpfilt file produce degenerate leading eigenvalues in a spot check; do not
   treat that as the paper path.
4. **`ReDisCA(demean_time=False)`** is the paper printed Gram. **`True`** is a
   labeled extra. Paths are stored separately and not mixed in one figure.
5. **Sliding step = 25 ms**, documented as *not* a paper value. Window time
   coordinate = **center**. Inclusive samples with `|t − center| ≤ T/2`.
6. **Car duration = 100 ms** centered at 170 ms, because 100 ms is the only
   other real-data T besides 150 ms. Not claimed as printed.
7. **RDMs** encoded as 0/1 from §4.2.1 prose. Optional within=0.1 extra: after
   library z-scoring of unique pairs these two-level RDMs are identical, so
   filters/λ match.
8. **Inference (paper N170 test):** permute the condition order of D.
   C=4 ⇒ **24** permutations. Meaning RDM has **3** unique matrices
   (multiplicity 8); face/car detectors have **4** unique matrices
   (multiplicity 6). Authoritative p-values = exact 24-permutation
   `P(λ*_k ≥ λ_k)`. Monte Carlo **B=1000** is also stored (a resample of the
   same 24). Strict `>` is stored as an extra reading of “exceeds”.
9. **SPoC random-phase** and pair-vector shuffle are labeled exploratory /
   alternative readings. They are **not** called the paper N170 test.
10. **Do not tune** rank, channels, lpfilt, or windows to hit 0.82 / 0.99.
    Report actual Pearson unique-triangle correlations of D vs `‖u_i−u_j‖²`.
11. Patterns are Haufe (library). Sign-flipped so occipital ROI mean ≥ 0.
12. Filters from the analysis window are also applied to the **full epoch**
    for traces (as in the paper figures). Window-restricted and full-epoch
    RDM correlations are both stored.

## Commands

```bash
cd /tmp/redisca-worktrees/n170   # or the repository root
python3 -m pytest paper/reproduction/n170/test_n170.py -q
python3 paper/reproduction/n170/run.py --B 1000 --step-ms 25 --seed 20240904
```

Production run (this report): 2026-09-04T18:06:57Z, ~13 s; tests 6 passed.

## Environment

From `paper/results/n170/environment.json` after the production run:

| Item | Value |
| --- | --- |
| Python | 3.12.3 |
| numpy | 2.4.4 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| redisca | 0.1.0 (`ReDisCA`, default `demean_time=True`) |
| MATLAB | not used |
| RNG | numpy PCG64, seed `20240904` |
| ERP | `1_N170_erp_ar.erp` SHA-256 `53e74e931e6f0adaf1e5be4d606d028fcc3e04ee8b066569c2ed2d033d9bbc72` |
| Times | −199.21875 … 796.875 ms, n=256, fs=256 Hz |
| Accepted trials | 52 / 38 / 49 / 52 (faces / cars / scr. faces / scr. cars), total 191 |

## Quantitative numbers vs paper

Primary path: 28 scalp channels, `ReDisCA(demean_time=False)`, binary 0/1 RDMs,
T as in the table, step 25 ms for the meaning scan. Pearson unique-triangle
RDM correlations are **not** tuned toward 0.82 / 0.99.

| Target | Paper | This run (`demean_time=False`) |
| --- | --- | --- |
| Fig 10 face RDM corr (window) | 0.82 | **0.99988** |
| Fig 10 face RDM corr (full epoch, same w) | — | 0.94466 |
| Fig 10 n significant (p<0.05, exact-24 `≥`) | 1 | **0** (comp 1 p=0.75; floor 0.25) |
| Fig 10 face burst | ~170 ms | **171.875 ms** (Faces, \|peak\| in 80–250 ms) |
| Fig 10 topography | right-FG-like | max \|a\| at **P7**; occipital energy 0.53 |
| Fig 11 car RDM corr (window, comp 1) | >0.99 | **0.99992** (meets >0.99) |
| Fig 11 car RDM corr (window, comp 2) | — | **0.99968** |
| Fig 11 n comps p<0.01 (exact-24 `≥`) | 2 | **0** (comp 1–2 p=0.25 = discrete floor) |
| Fig 11 car deflection | ~150 ms | **136.72 ms** (Cars, comp 1, 80–250 ms) |
| Fig 11 topography | lower occipital then dorsal | comp 1 max \|a\| **O1**; comp 2 **Pz** |
| Fig 7 p<0.05 around 400 ms | uncorrected, comp 1 | **no**: p=`8/24≈0.333` at 400 ms; no p<0.05 segment |
| Fig 7 topography @ 400 ms | highly occipital | max \|a\| **PO8**; occipital energy **0.61**; O2>O1 |
| Fig 7 meaning RDM corr @ 400 ms | visual resemblance | **0.99981** |
| Fig 8 windows | three adjacent ~400 ms | 375 / 400 / 425 ms; corr 0.9998 / 0.9998 / 0.9999; PO4 / PO8 / Oz |

`demean_time=True` extra (not mixed into primary plots): face window corr 0.99858
(PO4); car 0.99909 / 0.99650 (P8 / PO8); meaning @ 400 ms corr 0.99105 with
max \|a\| at Pz (less occipital than the paper Gram).

Monte Carlo B=1000 condition-label p-values agree with the exact-24 table
(face comp 1 p=0.741 vs 0.75; car 0.244 vs 0.25; meaning 0.331 vs 0.333).

Exploratory (explicitly **not** the paper N170 test):

| Null | Face c1 p | Car c1, c2 p | Meaning @400 c1 p |
| --- | --- | --- | --- |
| Pair-vector shuffle B=1000 `≥` | 0.151 | 0.044, 0.044 | 0.050 |
| SPoC random-phase B=1000 `≥` | **0.000** | **0.000, 0.000** | 0.547 |

Random-phase would call face/car components “significant” and would **not**
recover Fig 7’s p<0.05 at 400 ms. Stored only as a D5 diagnostic.

## Discrepancies encountered

- **D11.** Paper ICA gloss vs ERP CORE subject 1 (comps 2, 7). Documented; official averages used.
- **D12.** No AIRI N170 code. Paper + ERP CORE only.
- **D2.** `demean_time` False vs True both run, labeled separately.
- **D5.** Paper condition-label permutation vs SPoC random-phase. Primary test is label permutation. Random-phase is exploratory only.
- **C=4 permutation floor.** Meaning RDM invariant under 8/24 relabelings ⇒
  `P(λ* ≥ λ_obs)` cannot be < 8/24 ≈ 0.333 when this partition is uniquely
  best. Face/car detectors: floor 6/24 = 0.25. Paper Fig 7 `p<0.05` and
  Fig 11 `p<0.01` are **not compatible** with this discrete null unless they
  used a different scheme (pair shuffle, random-phase, different B/p formula,
  or more conditions). This is reported, not “fixed”.
- **Face RDM corr.** Library GEP on 28 channels × ~26 samples can match a
  6-entry two-level RDM almost perfectly (corr near 1), vs paper 0.82. Actual
  correlation is reported; parameters are not tuned downward.
- **Sliding step, channel count, car duration, B** are not printed. Choices
  above are documented.
- **P9/P10** absent from the official ERP; not silently interpolated.

## Blocked items

- Exact match to unpublished numeric RDM entries (figures only).
- Exact match to unpublished permutation B / p formula.
- Sliding-window step (choose and document; 25 ms used).
- Car-window duration (100 ms used; not restated in the paper).
- Bit-exact MATLAB (none for N170; D12).
- Recomputing ICA (Script 3 is commented to preserve original weights).

N170 sensor-space figures are **not** blocked on missing data: subject `"1"`
ERPs are public.

## Files changed (owned paths)

```
paper/reproduction/n170/__init__.py
paper/reproduction/n170/rdms.py
paper/reproduction/n170/prepare.py
paper/reproduction/n170/inference.py
paper/reproduction/n170/plotting.py
paper/reproduction/n170/run.py
paper/reproduction/n170/README.md
paper/reproduction/n170/test_n170.py
paper/reproduction/n170/TRACK_REPORT.md
paper/results/n170/summary.json
paper/results/n170/fingerprints.json
paper/results/n170/channel_selection.json
paper/results/n170/environment.json
paper/results/n170/fig07_meaning_pmap.json
paper/results/n170/fig08_meaning_windows.json
paper/results/n170/fig09_rdms.json
paper/results/n170/fig10_face.json
paper/results/n170/fig11_car.json
```

PNG under `paper/results/n170/` is gitignored.

## Figure status

| ID | Status |
| --- | --- |
| `fig07-n170-meaning-pmap` | **ran**. T=150 ms, step 25 ms, 33 windows (−100…700 ms). Comp-1 p(t) under condition-label permutation sits on the 8/24 floor for 27/33 windows (including 400 ms). Occipital pattern at 400 ms (PO8). Uncorrected p<0.05 around 400 ms **not reproduced** under the paper-described null. |
| `fig08-n170-meaning-patterns` | **ran**. Three adjacent windows 375/400/425 ms; traces + empirical RDMs; occipital topographies. |
| `fig09-n170-face-car-rdms` | **encoded** 0/1 meaning/face/car plus labeled 0.1-within extra (z-score equivalent). |
| `fig10-n170-face` | **ran**. Window corr **0.99988** vs paper **0.82**. Face peak **171.9 ms**. One significant component **not** obtained under exact-24 `≥` (p=0.75). |
| `fig11-n170-car` | **ran**. Window corr **0.99992** vs paper **>0.99** (match on correlation). Two comps p<0.01 **not** obtained under exact-24 `≥` (p=0.25 = floor); both comps have p_strict=0 (uniquely best detector). |

PNG copies are gitignored under `paper/results/n170/*.png`.

## Headline vs 0.82 / >0.99

- Face (Fig 10, paper Gram, window empirical RDM): **0.99988** (paper 0.82). Difference +0.18. Not tuned.
- Car (Fig 11, paper Gram, window empirical RDM, component 1): **0.99992** (paper >0.99). **Agrees** with the >0.99 claim.
- Likely reason the face number is near 1: 28 spatial degrees of freedom vs a two-level 6-pair target in a 26-sample window. The GEP can match D almost exactly. Paper 0.82 is not recovered from official subject-1 averages with the documented choices above. Full-epoch traces with the same filter give 0.945, still above 0.82.
