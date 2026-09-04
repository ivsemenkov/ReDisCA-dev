# MEG track (Kozunov et al. 2018 subject AD, Ossadtchi et al. 2024 Figs 12–17)

Owner: MEG worker. Do not edit `paper/reproduction/common/` or `src/redisca`.

This track runs **two labeled deterministic paths**. They are never mixed in one
figure or one untagged metric file.

| Path | Pairs | Pair matrix | Window | Filter | RDM | Component *p* | Time asterisks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `paper_faithful` | unique unordered *i<j* (15) | `ReDisCA(demean_time=False)` printed Gram | full −500…+1000 ms, **1501** samples (D16) | none (paper states none) | Fig. 12a–c **binary 0/1**; Fig. 16 **AIRI 0.1/0.5/1** | condition-label permutation, documented *B* | permute subcategory labels of epochs; FWER max-stat over time; documented *Nmc* |
| `airi_executable` | directed *i≠j* (30) | `source_faithful` MATLAB `cov` | MATLAB `600:1500` → **99–999 ms** (901 samples) | `butter(3)` 0.25–20 Hz `filtfilt` per trial | AIRI numeric via `airi_rdm()`; default `facevstool` | SPoC random-phase *B*=1000 | `Nmc=100` half-split on channel-std data (**not** the paper test) |

Canonical paper fit: `from redisca import ReDisCA`.
AIRI fit: `common.source_faithful.fit_condition_averages` / `airi_bandpass_trials` (does **not** call `redisca`).

`ReDisCA(demean_time=True)` is recorded as a **labeled extra**, not as the paper Gram.

## Data

```bash
python paper/reproduction/common/download_osf.py meg-sensor
```

Cached files (gitignored):

- `.reproduction_data/meg/MEG_AD_run1.mat` — 207 × 1501 × 880; channels 1–204 are planars
- `.reproduction_data/meg/ibfctfprespm8_AD_run1_raw_tsss_mc.mat` — SPM labels only

Do **not** load the companion 1.09 GB `.dat`. Trial indices:
`common.meg_io.airi_condition_indices` (verified 80 × 6).

True time: 1 kHz, −500…+1000 ms (1501 samples). Paper prints 204 × 1500 (D16).

## Commands

From the repository root (after `python3 -m pip install -e ".[paper]"`):

```bash
python paper/reproduction/meg/run.py all
python paper/reproduction/meg/run.py paper
python paper/reproduction/meg/run.py airi
```

Documented Monte Carlo defaults (paper *B* / time *Nmc* are unspecified in the
NeuroImage text; AIRI values are from the pinned MATLAB script):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--paper-B` | 500 | condition-label permutations; *p* = count/*B* (0 possible) |
| `--paper-nmc` | 200 | FWER max-stat over time |
| `--airi-B` | 1000 | SPoC random-phase (reduce only if infeasible; then label `reduced_B`) |
| `--airi-nmc` | 100 | AIRI half-split |
| `--pair-order-shuffles` / `--pair-order-B` | 5 / 200 | diagnostic only |
| `--quick` | off | smoke-test MC; **not** the reported run |
| `--rdms` | all | subset, e.g. `--rdms facevstool` |
| `--seed` | 20240904 | NumPy PCG64 streams |

```bash
python -m pytest paper/reproduction/meg/tests -q
```

## Outputs

Compact JSON (committed) under `paper/results/meg/`:

```text
paper/results/meg/paper_faithful/     fig12, fig13–15, fig16–17, demean extra, fill extra
paper/results/meg/airi_executable/    AIRI RDMs, fig13–15, airi-executable-meg-facevstool,
                                      pair-order diagnostic
paper/results/meg/comparison/         paper_vs_airi.json (metrics only; no mixed figure)
```

PNG figures are generated beside the JSON (`figures/`) but are gitignored by
the paper-branch policy. Arrays (`.npz`) are gitignored.

After `run.py all`, compact numbers and statuses are in
`TRACK_REPORT.md` and `paper/results/meg/**/*.json`.

## RDMs

AIRI names (`source_faithful.airi_rdm`): `face`, `tool`, `meaning`, `facevstool`.

This track **emits** AIRI numeric 0.1/1 (and 0.1/0.5/1) matrices **and** 0/1
binary variants for face/tool/meaning. `facevstool` has no 0/1 analogue.

Paper-faithful **fits** use binary Fig. 12a–c and AIRI `facevstool` for Fig. 16/17.
AIRI 0.1/1 fill is a labeled extra (`rdm_fill_extra.json`).

AIRI default `ThRDMArr(2)='facevstool'` is Fig. 16-like, **not** Fig. 12a (D7).

## Inference (do not mix)

- Paper components: permute **condition labels** of the theoretical RDM; pair
  matrices fixed. Primary null = `max|λ|`. Matched-component *p* is exploratory.
- Paper time series: permute **subcategory labels** of the 480 epochs (80 each),
  surrogate averages, **fixed** filters, FWER `max_t |contrast|`.
  Contrast convention is paper prose (faces vs *others*, etc.).
- AIRI components: stock SPoC random-phase of *z* (not label permutation).
- AIRI time series: channel-time sample SD over **880** trials, then *Nmc*=100
  half-split of pooled class trials. Class split follows MATLAB (`face` is
  faces vs nons, not vs others). **Not the paper test.**
- Pair-order diagnostic: shuffle the directed pair sequence and recompute
  random-phase *p*. Labelled diagnostic; not a replacement test.

## SciPy `filtfilt` vs MATLAB

AIRI MATLAB uses Signal Processing Toolbox `filtfilt` (Gustafsson method /
MATLAB padding). This reconstruction uses `scipy.signal.filtfilt` (`padtype='odd'`,
default `padlen`). Not bit-exact (D8 numeric). No MATLAB in this environment.

## Patterns

MEG numerical rank is ≪ 204, so paper *A = W⁻¹* does not apply (D9). Both
executable paths use Haufe / SPoC patterns. Topographies are 12×17 planar
heatmaps plus planar-pair RMS — **not** FieldTrip `ft_topoplotER` helmets
(`prepare4topoNMG` is not in the AIRI repo).

## What not to do

- Do not put `trange=600:1500` on a `paper_faithful` tag.
- Do not call AIRI half-split asterisks the paper FWER test.
- Do not compare eigenvector signs without alignment.
- Do not treat AIRI default `facevstool` as Fig. 12a.
- Do not collapse D1/D2/D4 eigenvalue scales across paths.

Paper qualitative onsets (65/160/311 ms, …) are **not** numbers to force-match.
Report actual peaks in the JSON and `TRACK_REPORT.md`.
