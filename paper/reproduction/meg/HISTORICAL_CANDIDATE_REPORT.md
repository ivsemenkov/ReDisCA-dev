# MEG historical candidate (frozen N170 estimator)

Owner: historical-apply worker.
Branch: `cursor/paper-historical-apply-f368` (from `paper` @ `283ca1e`).
Working directory: `/tmp/redisca-worktrees/historical-apply`.

Applies the **frozen N170 Tracks A+B candidate** to MEG. Unmodified:

```text
pair_mode  = airi_directed
matrix_mode = matlab_cov
inference  = spoc_random_phase      # PRIMARY
B          = 1000
p          = count(max|lambda_surr| >= |lambda_obs|) / B
```

Do not mix paper-window averages with AIRI filters in one untagged file.
New modules do not import `redisca`. MATLAB was not used; not MATLAB parity.
Rank is recorded as estimated (expect 68; not forced to 67).

Compact machine copies under `paper/results/meg/historical_candidate/`.

N170 Fig. 7/8 apply (same freeze): `paper/reproduction/n170/HISTORICAL_APPLY_REPORT.md`.

## Two labeled applications

### A. Reuse `paper/results/meg/airi_executable/` (not rerun)

This is the frozen estimator **plus AIRI MEG extras**: 99–999 ms crop and
butter(3) 0.25–20 Hz `scipy.signal.filtfilt`. JSON was verified to match
directed pairs + `matlab_cov` + `spoc_random_phase` B=1000, `reduced_B=false`,
204 planars, 901 samples. **B=1000 was not recomputed.**

PRIMARY p heads (copied):

| RDM | rank | λ₁…λ₄ | p (random-phase B=1000) | first-3 p<0.05 | n p<0.05 in stored 8-head |
| --- | --- | --- | --- | --- | --- |
| face | 68 | 0.880, 0.859, 0.833, 0.806 | **0.006, 0.008, 0.014, 0.032** | **3** | 4 |
| tool | 68 | 0.873, 0.861, 0.834, 0.823 | **0.001, 0.003, 0.008, 0.010** | **3** | 5 |
| meaning | 68 | 0.874, 0.855, 0.832, 0.812 | **0.005, 0.012, 0.018, 0.026** | **3** | 4 |
| facevstool | 68 | 0.848, 0.831, 0.810, 0.782 | **0.016, 0.019, 0.028, 0.054** | **3** | 3 |

Published Figs 13–15, 17: **three** significant components. Path A matches
that **count** on the first three components (facevstool c4 p=0.054). This
count uses the AIRI extras + SPoC test (D5/D6/D8), not paper §2.3.

### B. Frozen estimator on the **paper MEG epoch** (required new run)

Same freeze. Condition averages via `meg.prepare` on the full file epoch:

- −500…+1000 ms, **1501** samples, 204 planars
- **no** AIRI bandpass
- RDMs: AIRI catalog `face`, `tool`, `meaning`, `facevstool` (`airi_rdm`)
- Seed: MEG CLI `20240904`; paper-epoch random-phase stream is **seed+20**
  (disjoint from `meg/run.py` +1/+2/+3/+10). One Generator consumed in RDM
  order face → tool → meaning → facevstool.
- Secondary: exact 6! = 720 condition-label permutations (no RNG).

**Status at first commit:** code and tests landed; B=1000 paper-epoch run
follows this commit (see updated numbers below after the run).

## Comparison vs paper_faithful (previous unique+Gram+label-permutation)

From existing `paper/results/meg/paper_faithful/summary.json` (not rerun):

| RDM | rank | λ₁…λ₃ | p max\|λ\| B=500 | first-3 p<0.05 | contrast peak c1 |
| --- | --- | --- | --- | --- | --- |
| face binary | 68 | 0.877, 0.786, 0.691 | **0, 1, 1** | 1 | 308 ms |
| tool binary | 68 | 0.857, 0.789, 0.723 | 0.478, 1, 1 | 0 | 384 ms |
| meaning binary | 68 | 0.870, 0.775, 0.717 | 0.204, 1, 1 | 0 | 315 ms |
| facevstool | 68 | 0.836, 0.753, 0.684 | 0.302, 1, 1 | 0 | 564 ms |

Paper-faithful unique+Gram + condition-label `max|λ|` does **not** recover
three significant components. That disagreement was already D5 in the MEG
track report.

## Commands

```bash
cd /tmp/redisca-worktrees/historical-apply
python3 -m pytest paper/reproduction/meg/historical_candidate/tests -q
python3 paper/reproduction/meg/historical_candidate/run.py reuse
python3 paper/reproduction/meg/historical_candidate/run.py paper-epoch
# default B=1000 from the freeze; do not reduce silently
```

`--quick` sets B=0 and is not the reported run.

## Environment

- Python 3.12.3, numpy 2.4.4, scipy 1.18.1
- MATLAB: **none**
- SciPy `filtfilt` on path A only (already stored); path B has no filter

## Path B results (filled after B=1000)

_Placeholder before the paper-epoch run. Replaced in the results commit._

## Does the N170 directed+cov freeze reproduce published MEG counts without AIRI extras?

**Pending path B.** Path A (freeze **with** AIRI window/filter) already
matches the published **three-component count**. Path B asks whether that
count survives on the paper epoch without those extras.

## Honest mismatches / not claimed

- Paper face peak ~160 ms is a qualitative fingerprint, not a tuning target.
  Path A contrast peaks are ~319/424/499/321 ms (see MEG TRACK_REPORT).
- Rank 68 vs author-saved 67 is recorded, not forced.
- MATLAB `eig` / `rand` / `filtfilt` parity is not claimed.
- Condition-label permutation remains secondary.

## Tests

AST: no `redisca` import. Smoke fit B=0. Freeze JSON accepted.
`airi_executable` reuse verifies directed+cov+random-phase B=1000 without
recomputing.
