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

Ran B=1000 (not reduced). Wall time ~13 s. Rank **68** on all four RDMs.

| RDM | rank | λ₁…λ₄ | p (B=1000) | first-3 p<0.05 | n p<0.05 (all rank) | contrast peak c1 | class1 peak c1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| face | 68 | 0.862, 0.796, 0.694, 0.683 | **0.006, 0.019, 0.114, 0.142** | **2** | 2 | 309 ms | 533 ms |
| tool | 68 | 0.818, 0.805, 0.738, 0.697 | **0.006, 0.010, 0.057, 0.107** | **2** | 2 | 392 ms | 393 ms |
| meaning | 68 | 0.845, 0.787, 0.731, 0.668 | **0.001, 0.018, 0.061, 0.171** | **2** | 2 | 315 ms | 120 ms |
| facevstool | 68 | 0.808, 0.766, 0.697, 0.654 | **0.010, 0.024, 0.104, 0.191** | **2** | 2 | 566 ms | 969 ms |

Component 3 sits just above 0.05 (tool 0.057, meaning 0.061, face 0.114,
facevstool 0.104). Secondary exact-720 label-perm `max|λ|` does **not**
give three p<0.05 components (face p₁=0 then p₂=0.867; tool p₁=0.533).

Face ~160 ms is not on c1. Face c3 class1 peak 197 ms is the closest
neighbour among the first three, and c3 is not significant (p=0.114).
facevstool c4 class2 peak 164 ms (p=0.191). Not force-matched.

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

## Path B results

See the table under application B. Files: `paper_epoch.json` and
`paper_epoch_{face,tool,meaning,facevstool}.json`.

## Does the N170 directed+cov freeze reproduce published MEG counts without AIRI extras?

**No.** On the paper epoch (full −500…+1000 ms, no bandpass) the freeze
yields **two** components with PRIMARY p<0.05 for every RDM, not three.
Path A (same freeze **plus** AIRI 99–999 ms crop and 0.25–20 Hz filtfilt)
does match the published three-component **count**. The extras, not the
N170 pair/matrix freeze alone, are what recover that count.

paper_faithful unique+Gram + label-permutation still yields 0–1 significant
components under `max|λ|`. Random-phase on directed+cov is the lever that
moves p₁ (and p₂) below 0.05; the AIRI window/filter is the remaining lever
for p₃.

## Honest mismatches / not claimed

- Paper face peak ~160 ms is a qualitative fingerprint, not a tuning target.
  Path B face c1 contrast peak is 309 ms (paper_faithful 308 ms). Path A
  contrast peaks are ~319/424/499/321 ms (MEG TRACK_REPORT).
- Rank 68 vs author-saved 67 is recorded, not forced.
- MATLAB `eig` / `rand` / `filtfilt` parity is not claimed.
- Condition-label permutation remains secondary.

## Tests

AST: no `redisca` import. Smoke fit B=0. Freeze JSON accepted.
`airi_executable` reuse verifies directed+cov+random-phase B=1000 without
recomputing.
