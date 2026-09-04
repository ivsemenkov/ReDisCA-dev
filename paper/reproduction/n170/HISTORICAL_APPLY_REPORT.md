# N170 historical apply (frozen Tracks A+B candidate → Figs 7–8)

Owner: historical-apply worker.
Branch: `cursor/paper-historical-apply-f368` (from `paper` @ `283ca1e`).
Working directory: `/tmp/redisca-worktrees/historical-apply`.

This applies the **frozen** N170 Tracks A+B leading candidate to Fig. 7
sliding meaning and Fig. 8 adjacent windows. It does **not** retune pair
order, RDM fill, window step, or B. MATLAB was not used. Results are
**not** MATLAB `eig` / `rand` parity. New modules do not import `redisca`.

Compact machine copies:

- `paper/results/n170/historical_apply/fig07_meaning_pmap.json`
- `paper/results/n170/historical_apply/fig08_meaning_windows.json`
- `paper/results/n170/historical_apply/summary.json`
- `paper/results/n170/historical_apply/environment.json`

Freeze file (read-only here): `paper/results/n170/historical/leading_candidate.json`.

## Frozen estimator (not retuned)

```text
pair_mode  = airi_directed          # AIRI i!=j nested loop (12 pairs, C=4)
matrix_mode = matlab_cov            # temporal demean + /(T-1)
inference  = spoc_random_phase      # PRIMARY
B          = 1000
p          = count(max|lambda_surr| >= |lambda_obs|) / B
executable = common.source_faithful.fit_condition_averages
```

Condition-label permutation (exact 24) is **secondary** only. Meaning RDM
has an 8/24 ≈ 0.333 floor when the partition is uniquely best. That floor
is expected and is not the historical primary.

## Data

- Official ERP CORE subject `"1"`: `1_N170_erp_ar.erp`
- SHA-256: `53e74e931e6f0adaf1e5be4d606d028fcc3e04ee8b066569c2ed2d033d9bbc72`
- 28 scalp channels (EOG/bipolar dropped; P9/P10 not in the ERP)
- RDM: meaning partition from `n170.rdms`, binary 0/1
- T = 150 ms (paper). Step = 25 ms (`DEFAULT_SLIDING_STEP_MS`). The paper
  does not print a step; it was **not** searched to obtain significance.
- Inclusive windows `|t − center| ≤ T/2` → 38 or 39 samples at 256 Hz.
- Sliding centers: −100 … +700 ms (33 windows).

## Seed policy

PCG64 master `20240904`. Fig. 7 window `i` uses
`PCG64(MASTER_SEED + 2000 + i)`, disjoint from Track A (offsets < 200)
and Track B (`MASTER+10000+i`). Fig. 8 375/400/425 ms reuse the same
stream as the matching sliding-grid index, so those three p-values match
the Fig. 7 map.

## Commands

```bash
cd /tmp/redisca-worktrees/historical-apply
python3 -m pytest paper/reproduction/n170/historical_apply/tests -q
python3 paper/reproduction/n170/historical_apply/run.py
# default: --freeze paper/results/n170/historical/leading_candidate.json --B 1000 --step-ms 25
```

`--quick` sets B=0 and is not the reported run.

## Environment

- Python 3.12.3, numpy 2.4.4, scipy 1.18.1, scikit-learn 1.9.0
- MATLAB: **none**
- Captured 2026-09-04T22:36:10Z

## Fig. 7 — does random-phase recover p₁<0.05 near 400 ms?

**Yes, at the 400 ms center, but not as a multi-window continuous interval.**

| Claim | This run |
| --- | --- |
| Paper | first-component uncorrected p<0.05 around t=400 ms; T=150 ms; occipital |
| Nearest-400 center | 400.0 ms |
| PRIMARY p₁ at 400 ms | **0.018** (18/1000) |
| λ₁ at 400 ms | 0.63587 |
| corr wᵀRw (unique triangle) | 0.99105 |
| pattern max-abs channel | **Pz** (occipital energy fraction 0.218) |
| Secondary exact-24 p₁ | 8/24 ≈ 0.333 (floor; expected) |
| p₁<0.05 segments | **one isolated window**: 400–400 ms |
| 375 / 425 ms neighbors | p₁ = 0.076 / **0.050** (0.050 is not strictly <0.05) |

p₁ along 300–500 ms (PRIMARY random-phase, B=1000):

| center ms | p₁ | λ₁ | corr wᵀRw | max-abs ch | exact-24 |
| --- | --- | --- | --- | --- | --- |
| 300 | 0.213 | 0.57951 | 0.989 | PO8 | 0.333 |
| 325 | 0.102 | 0.59814 | 0.998 | Pz | 0.333 |
| 350 | 0.074 | 0.62750 | 0.994 | Pz | 0.333 |
| 375 | 0.076 | 0.61617 | 0.984 | Oz | 0.333 |
| **400** | **0.018** | 0.63587 | 0.991 | Pz | 0.333 |
| 425 | 0.050 | 0.62016 | 0.997 | Pz | 0.333 |
| 450 | 0.089 | 0.59332 | 0.979 | FP1 | 0.333 |
| 475 | 0.330 | 0.50736 | 0.997 | FP1 | 1.000 |

No other sliding center has p₁<0.05. The local dip is at 400 ms.

Previous library N170 path (`ReDisCA` unique+Gram, exact-24 as primary)
had nearest-400 p = 8/24 ≈ 0.333 and **no** p<0.05 segment. Switching the
**test** to stock SPoC random-phase is what makes 400 ms reachable;
the secondary exact-24 numbers here still sit on the 8/24 floor.

## Fig. 8 — 375 / 400 / 425 ms (same freeze, same seeds as Fig. 7)

| center | λ₁ | PRIMARY p₁ | corr wᵀRw | max-abs ch | exact-24 |
| --- | --- | --- | --- | --- | --- |
| 375 | 0.61617 | 0.076 | 0.98397 | Oz | 0.333 |
| 400 | 0.63587 | 0.018 | 0.99105 | Pz | 0.333 |
| 425 | 0.62016 | 0.050 | 0.99680 | Pz | 0.333 |

## Honest mismatches

1. Paper Fig. 7 reads as a **continuous** uncorrected-p interval near 400 ms.
   This freeze produces a **single** 25 ms-grid hit (p=0.018). The next
   center is p=0.050, which is not strictly below 0.05. Step was not
   narrowed to merge them.
2. Paper describes an occipital pattern. At 400 ms the max-abs channel is
   **Pz**, occipital energy fraction ≈ 0.22. Not retuned.
3. Secondary exact-24 cannot go below 8/24. That is a property of the
   meaning RDM, not a failure of random-phase.
4. MATLAB unavailable. Python reconstruction of stock SPoC + AIRI pairs,
   not MATLAB parity.

## Tests

```bash
python3 -m pytest paper/reproduction/n170/historical_apply/tests -q
```

AST: no `redisca` import. Smoke fit B=0. Fig. 7 runner accepts
`leading_candidate.json` and rejects a retuned `pair_mode`.
