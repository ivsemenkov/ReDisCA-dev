# N170 track report (Ossadtchi et al. 2024, Figs 7–11)

Track owner: N170 worker. Branch: `cursor/paper-n170-f368`.
Working directory: `/tmp/redisca-worktrees/n170`.

This report records source evidence, decisions, commands, environment,
numbers, discrepancies, and blocked items. Headline numbers are filled after
`python paper/reproduction/n170/run.py`. See `paper/results/n170/summary.json`
for the compact machine copy.

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
cd /tmp/redisca-worktrees/n170
python -m pytest paper/reproduction/n170/test_n170.py -q
python paper/reproduction/n170/run.py --B 1000 --step-ms 25 --seed 20240904
```

Rerun from repo root:

```bash
python paper/reproduction/n170/run.py
```

## Environment

Filled after the run from `paper/results/n170/environment.json`.

- Python, numpy, scipy, scikit-learn, matplotlib, redisca
- MATLAB: not used (and not required)
- ERP SHA-256: filled after the run

## Quantitative numbers vs paper

Filled after the run. Placeholders:

| Target | Paper | This run (`demean_time=False`) |
| --- | --- | --- |
| Fig 10 face RDM corr | 0.82 | *run* |
| Fig 10 n significant | 1 | *run* |
| Fig 10 face burst | ~170 ms | *run* |
| Fig 11 car RDM corr | >0.99 | *run* |
| Fig 11 n comps p<0.01 | 2 | *run* |
| Fig 7 p<0.05 around 400 ms | yes, uncorrected, comp 1 | *run; expect discrete floor ~0.333* |
| Fig 7 topography | occipital | *run* |

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
paper/results/n170/*.json
```

PNG under `paper/results/n170/` is gitignored.

## Figure status

| ID | Status |
| --- | --- |
| `fig07-n170-meaning-pmap` | implemented; see JSON after run |
| `fig08-n170-meaning-patterns` | implemented; 375/400/425 ms |
| `fig09-n170-face-car-rdms` | encoded 0/1 + 0.1-within extra |
| `fig10-n170-face` | implemented; actual corr vs 0.82 |
| `fig11-n170-car` | implemented; actual corr vs >0.99 |
