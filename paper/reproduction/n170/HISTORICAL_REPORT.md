# N170 historical estimator investigation (Tracks A + B)

Owner: overnight historical N170 worker.
Branch: `cursor/paper-n170-historical-f368` (from `paper` @ `ff49b9a`).
Working directory: `/tmp/redisca-worktrees/n170-historical`.

This is a **source-faithful Python reconstruction** of AIRI pair construction
plus stock SPoC inference. MATLAB is unavailable. Results are **not** MATLAB
`eig` / `rand` parity. Historical modules do not import `redisca`.

Compact machine copies:

- `paper/results/n170/historical/track_a_table.json`
- `paper/results/n170/historical/track_b.json`
- `paper/results/n170/historical/leading_candidate.json`
- per-variant JSON in the same directory

## Authority map (not collapsed)

| Layer | What it is here |
| --- | --- |
| A. Paper | Fig. 10/11 fingerprints; §2.3 condition-label permutation; Eq. 4 unscaled Gram; Table 1 unique pairs |
| B. AIRI MATLAB | Directed `i != j` pairs; MATLAB `cov` of pair differences; calls stock SPoC |
| C. Stock SPoC | Random-phase of `z`; null statistic `max\|λ\|`; `p = count/B` (p=0 allowed); `Cxxz` is a mean |
| D. Python library | Unique pairs + unscaled Gram (`ReDisCA(demean_time=False)`); not used inside historical modules |
| E. This reconstruction | `common.source_faithful.fit_condition_averages(..., inference="spoc_random_phase")` |

**PRIMARY** inference for every variant is stock SPoC random-phase (layer C).
Condition-label permutation (paper §2.3) is recorded as a **secondary**
diagnostic and is **not** called the historical oracle.

## Data (baseline; not tuned)

- Official ERP CORE subject `"1"`: `1_N170_erp_ar.erp` via
  `prepare.load_n170_subject1(lpfilt=False)`.
- SHA-256: `53e74e931e6f0adaf1e5be4d606d028fcc3e04ee8b066569c2ed2d033d9bbc72`
- 28 scalp channels already selected by that loader. EOG not added. P9/P10
  not invented (they are not in the ERP).
- Sampling: 256 Hz, 256 samples, times ≈ [−199.22, 796.88] ms.
- RDMs: binary 0/1 from `rdms.theoretical_rdm("face"|"car")`.
- Condition order: Faces, Cars, Scrambled Faces, Scrambled Cars.
- Windows: inclusive `|t − center| ≤ T/2`. T = 100 ms → 26 samples.

## Printed fingerprints (figure panels; not tuning targets)

Fig. 10 face (T=100 ms centered at 200 ms):

- λ1 ≈ 0.87209, p1 = 0, corr ≈ 0.82, burst ≈ 170 ms

Fig. 11 car (applied at t=170 ms, T=100 ms):

- λ1 ≈ 0.91639, p1 = 0; λ2 ≈ 0.77036, p2 ≈ 0.009; corr > 0.99

Existing unique+Gram **library** numbers (sanity only, not an oracle):

- Face λ1 = 0.88006, corr = 0.99988
- Car λ1, λ2 = 0.88691, 0.79170, corr = 0.99992

## Track A variants (source-motivated only)

Two pair orders only (random-phase is pair-order-sensitive):

1. `unique_unordered` — paper Table 1 / `pair_indices(..., "unique_unordered")`
   canonical upper triangle `(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)`
2. `airi_directed` — AIRI nested loop `i != j` (12 pairs for C=4)

Two pair matrices:

1. `unscaled_gram` — paper printed Gram `(X_i−X_j)(X_i−X_j)^T`
2. `matlab_cov` — MATLAB `cov` (temporal demean + `/(T−1)`)

Face: paper window only (100 ms @ 200 ms) × 2 × 2 = **4**.
Car: 170 ms and 200 ms × 2 × 2 = **8**.
Total **12**. No arbitrary pair permutations.

Master seed `20240904` (numpy PCG64). Variant RNG seed = master + offset
(see `historical/variants.py`). Matched-component extra uses seed+500.

PRIMARY call for every variant:

```python
from common.source_faithful import fit_condition_averages
result = fit_condition_averages(
    X_window, rdm,
    pair_mode=..., matrix_mode=...,
    n_bootstrapping_iterations=1000,
    rng=numpy PCG64 generator,
    inference="spoc_random_phase",
)
# result.p_values : p = count(max|lambda_surr| >= |lambda_obs|) / 1000
```

## Track A table (all 12)

All fits: n_channels=28, n_samples=26, whitening rank=25, B=1000.
`corr` is Pearson of unique `i<j` entries of the theoretical RDM vs
`d̂_ij = w^T R_ij w` (same `R_ij` as the fit). Un-demeaned
`‖u_i−u_j‖²` on `u = w^T X` is identical to `w^T R w` for Gram and
slightly lower for `matlab_cov` (because `R_ij` is temporally demeaned).

### Face (Fig. 10 window: 100 ms @ 200 ms). Printed λ1≈0.87209, p1=0, corr≈0.82

| variant | pair | matrix | λ1, λ2, λ3 | primary p1, p2, p3 | corr wᵀRw | corr ‖Δu‖² | rank | Faces peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `face_c200_d100_unique_unordered_unscaled_gram` | unique | Gram | 0.88006, 0.78438, 0.59962 | 0.000, 0.248, 0.725 | 0.99988 | 0.99988 | 25 | 171.875 ms |
| `face_c200_d100_unique_unordered_matlab_cov` | unique | cov | 0.83915, 0.68060, 0.63675 | 0.000, 0.417, 0.532 | 0.99858 | 0.94819 | 25 | 167.969 ms |
| `face_c200_d100_airi_directed_unscaled_gram` | directed | Gram | 0.92301, 0.82267, 0.62889 | 0.000, 0.014, 0.343 | 0.99988 | 0.99988 | 25 | 171.875 ms |
| `face_c200_d100_airi_directed_matlab_cov` | directed | cov | 0.88010, 0.71382, 0.66782 | 0.000, 0.058, 0.138 | 0.99858 | 0.94819 | 25 | 167.969 ms |

Face λ1 vs 0.87209: unique Gram **+0.0080** (closest Gram); directed cov
**+0.0080** (unique-cov scaled); unique cov **−0.033**; directed Gram **+0.051**.
All four have **primary p1 = 0**. No variant has window corr near 0.82.

### Car @ 170 ms (printed Fig. 11 application time). Printed λ1≈0.91639, λ2≈0.77036, p2≈0.009

| variant | pair | matrix | λ1, λ2, λ3 | primary p1, p2, p3 | corr wᵀRw | rank | Cars peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `car_c170_d100_unique_unordered_unscaled_gram` | unique | Gram | 0.88691, 0.79170, 0.70741 | 0.000, 0.141, 0.442 | 0.99992 | 25 | 136.719 ms |
| `car_c170_d100_unique_unordered_matlab_cov` | unique | cov | 0.87063, 0.75365, 0.67127 | 0.000, 0.115, 0.385 | 0.99909 | 25 | 140.625 ms |
| `car_c170_d100_airi_directed_unscaled_gram` | directed | Gram | 0.93020, 0.83035, 0.74194 | 0.000, **0.009**, 0.072 | 0.99992 | 25 | 136.719 ms |
| `car_c170_d100_airi_directed_matlab_cov` | directed | cov | **0.91312**, 0.79043, 0.70404 | 0.000, 0.003, 0.053 | 0.99909 | 25 | 140.625 ms |

### Car @ 200 ms (enumerated; not the printed application time)

| variant | pair | matrix | λ1, λ2, λ3 | primary p1, p2, p3 | corr wᵀRw | rank | Cars peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `car_c200_d100_unique_unordered_unscaled_gram` | unique | Gram | 0.88233, 0.81090, 0.77668 | 0.000, 0.062, 0.161 | 0.99994 | 25 | 136.719 ms |
| `car_c200_d100_unique_unordered_matlab_cov` | unique | cov | 0.87231, 0.79239, 0.69984 | 0.000, 0.058, 0.262 | 0.99965 | 25 | 148.438 ms |
| `car_c200_d100_airi_directed_unscaled_gram` | directed | Gram | 0.92539, 0.85048, 0.81459 | 0.000, 0.006, 0.013 | 0.99994 | 25 | 136.719 ms |
| `car_c200_d100_airi_directed_matlab_cov` | directed | cov | 0.91489, 0.83106, 0.73400 | 0.000, 0.004, 0.034 | 0.99965 | 25 | 148.438 ms |

Car λ1 closest to 0.91639 overall is directed+cov @ 200 ms (0.91489, Δ=−0.0015),
then directed+cov @ 170 ms (0.91312, Δ=−0.0033). Car λ2 closest to 0.77036 is
unique+cov @ 170 (0.75365, Δ=−0.017) then directed+cov @ 170 (0.79043, Δ=+0.020).
Car corr > 0.99 on every variant (`w^T R w`).

## What the table is saying (no scalar loss)

Directed vs unique, for a **symmetric** RDM, does not change `Cxx` or the
filter **rays**. MATLAB sample-SD z-scoring uses `N−1`, so λ scales by
`√(11/10) ≈ 1.0488` when going from 6 unique pairs to 12 directed pairs.
That is why unique Gram face λ1=0.88006 becomes directed Gram 0.92301, and
unique cov car λ1=0.87063 becomes directed cov 0.91312.

Random-phase **is** pair-order-sensitive: `z` length 6 vs 12 has a different
FFT amplitude spectrum, so the null changes even though the filters do not.
Unique-pair car p2 at 170 ms is 0.141 (Gram) / 0.115 (cov). Directed-pair
car p2 at 170 ms is 0.009 (Gram) / 0.003 (cov). That is the stochastic
lever. It is an accidental property of stuffing an RDM vector into SPoC’s
`z(e)`, not a paper-described test.

`matlab_cov` vs Gram **does** change the GEP (temporal demeaning). The
`1/(T−1)` scale itself cancels. Face/car Gram λ sit above cov λ for unique
pairs; after the directed z-scale, directed+cov face λ1 lands almost on the
unique-Gram face λ1 (0.88010 vs 0.88006) — a coincidence of two different
operations, not identity of the estimators.

Library sanity (unique+Gram, no `redisca` import; hardcoded fingerprints):

- Face λ1 delta = 1.6×10⁻¹⁵, corr delta = 0
- Car λ1, λ2 deltas = 7.8×10⁻¹⁶, 8.9×10⁻¹⁶

Secondary condition-label exact-24 p (signed `≥`) sits on the known discrete
floor 6/24=0.25 or 0.5/0.75. That test **cannot** produce printed p2≈0.009.
Matched-component random-phase p (labeled extra) is ~0 for leading
components and is **not** stock SPoC.

## Track B candidates (two jointly plausible settings)

Qualitative rule (lexicographic, **not** a weighted loss), applied after the
table:

1. PRIMARY face p1=0 and car p1=0 (all 12 already satisfy this on Track A).
2. Prefer printed Fig. 11 application time **170 ms** (200 ms stays in the table).
3. Then |car λ1 − 0.91639|, then |car p2 − 0.009|, then |car λ2 − 0.77036|,
   then |face λ1 − 0.87209|.
4. Face corr vs 0.82 is **not** used to discard anyone: every variant is ~1.

**Candidate 1 (leading freeze):** AIRI directed pairs + MATLAB cov, car
window 100 ms @ 170 ms, face window 100 ms @ 200 ms.

- Face λ1=0.88010 (Δ +0.008 vs 0.87209), p1=0, corr wᵀRw=0.99858
  (un-demeaned trace-sq=0.948; still not 0.82), Faces peak 167.97 ms
  (printed burst ~170 ms).
- Car λ1=0.91312 (Δ −0.003 vs 0.91639), λ2=0.79043 (Δ +0.020 vs 0.77036),
  Track A p1=0, p2=0.003, corr=0.99909 (>0.99), Cars peak 140.6 ms.

This is the least-bad **joint** source-supported setting: car λ1 is the
closest at the printed application time, face λ1 is as close as unique Gram,
and both PRIMARY p1 values are 0. Car λ2 is still ~0.02 high. Face corr
0.82 is not recovered.

**Candidate 2:** AIRI directed pairs + unscaled Gram, same windows.

- Face λ1=0.92301 (Δ +0.051, worse), p1=0, corr=0.99988, Faces peak 171.88 ms
  (slightly closer to 170 ms).
- Car λ1=0.93020 (Δ +0.014, worse), λ2=0.83035 (Δ +0.060, worse),
  Track A p1=0, **p2=0.009 exactly on the Track A seed**, corr=0.99992.

Kept because the Track A p2 hit is the printed stochastic fingerprint, and
because directed+Gram is the other AIRI-plausible pair matrix at 170 ms.
Deterministic λ are farther from the printed values than candidate 1.

## Track B: 20 independent seeds × B=1000

Seeds: `20240904 + 10000 + i` for `i = 0..19`, then `+ variant_offset`.
Deterministic λ do not change with seed. Only random-phase p-values do.

### Candidate 1 — directed + matlab_cov + car@170

| p | mean | median | min | max | q05 | q25 | q75 | q95 | fraction p=0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| face p1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **20/20** |
| car p1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **20/20** |
| car p2 | 0.0074 | 0.0075 | 0.003 | 0.014 | 0.0040 | 0.0050 | 0.0093 | 0.0112 | 0/20 |

Car component-2 full distribution (20 values):

`0.009, 0.009, 0.003, 0.005, 0.008, 0.004, 0.011, 0.006, 0.005, 0.010, 0.007, 0.008, 0.005, 0.005, 0.010, 0.006, 0.010, 0.008, 0.014, 0.005`

Exactly 0.009: **2/20**. Printed 0.009 lies inside min–max and inside the
5–95% interval.

### Candidate 2 — directed + unscaled Gram + car@170

| p | mean | median | min | max | q05 | q25 | q75 | q95 | fraction p=0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| face p1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **20/20** |
| car p1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **20/20** |
| car p2 | 0.0084 | 0.0080 | 0.005 | 0.011 | 0.0050 | 0.0068 | 0.0100 | 0.0110 | 0/20 |

Car component-2 full distribution:

`0.005, 0.011, 0.007, 0.006, 0.005, 0.008, 0.010, 0.011, 0.010, 0.006, 0.008, 0.008, 0.010, 0.009, 0.009, 0.006, 0.008, 0.011, 0.008, 0.011`

Exactly 0.009: **2/20**. Envelope even tighter around 0.009 (mean 0.0084).

### Does published p2≈0.009 lie naturally inside that variation?

**Yes, for both directed-pair candidates.** It is 9/1000 under `p=count/B`.
Across 20 seeds it appears twice for each candidate, and the whole envelope
is a few thousandths wide around ~0.007–0.008. It is **not** a one-seed
accident. It is also **not** recovered by unique-pair random-phase (Track A
p2≈0.12), so the printed stochastic fingerprint is pair-order-dependent.

Face p1=0 and car p1=0 are stable (20/20) for both candidates. Do not declare
figure parity from one seed; do record that the directed-pair SPoC null
naturally produces p2 values in the printed neighbourhood.

## Leading freeze

`paper/results/n170/historical/leading_candidate.json`

| Field | Frozen value |
| --- | --- |
| `pair_mode` | `airi_directed` |
| `matrix_mode` | `matlab_cov` |
| face window | 100 ms centered at 200 ms |
| car window | 100 ms centered at 170 ms |
| inference | `spoc_random_phase` (stock max\|λ\|, B=1000, p=0 allowed) |
| data file | `1_N170_erp_ar.erp` (28 scalp channels; no EOG; no P9/P10) |
| rdm fill | binary 0/1 |
| seed policy | PCG64, master `20240904`, per-variant offsets; Track B `master+10000+i` |

Why this one: jointly closest source-supported setting at the printed car
application time, with stable p1=0 and a p2 envelope that contains 0.009.
Second plausible candidate (`airi_directed` + `unscaled_gram` + car@170)
has a p2 mean even closer to 0.009 but worse λ1/λ2/face λ1.

### Remaining mismatches (not hidden)

1. **Face corr ≈ 0.82 is unmatched.** Every source-supported window fit gives
   `w^T R w` corr ≈ 0.999 (Gram) or 0.998 (cov). Un-demeaned `‖Δu‖²` for
   matlab_cov face is 0.948 — still not 0.82. The original library track
   already reported this. Historical variants do not fix it. Likely cause:
   25 spatial degrees of freedom vs a two-level 6-pair target in a 26-sample
   window; the GEP can match `D` almost exactly.
2. **Car λ2** is 0.790 vs printed 0.770 (Δ≈+0.020) on the freeze; candidate 2
   is worse (0.830).
3. **Face λ1** is 0.880 vs 0.872 (Δ≈+0.008). Same size as the unique-Gram
   library discrepancy. Not closed.
4. **Cars peak** ~141 ms vs a ~150 ms deflection in the figure; Faces peak
   168 ms vs ~170 ms burst (Gram face is 172 ms). Timing is in the right
   neighbourhood, not claimed as panel identity.
5. **MATLAB unavailable.** Numpy `eigh` / PCG64 ≠ MATLAB `eig` / `rand`.

This freeze is the least-bad source-supported candidate, not a claim that
Figs. 10–11 have been reproduced.

## Pair sequences and z (after `matlab_zscore`)

Unique unordered (C=4):

```
(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
z_face = [+0.91287]×3 + [−0.91287]×3
z_car  = [+0.91287, −0.91287, −0.91287, +0.91287, +0.91287, −0.91287]
```

AIRI directed:

```
(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0),(2,1),(2,3),(3,0),(3,1),(3,2)
z entries ±0.95743 (sample SD over 12, not 6)
```

Exact vectors are in each variant JSON (`z_after_matlab_zscore`, `pair_sequence`).

## Commands

```bash
cd /tmp/redisca-worktrees/n170-historical   # repository root
python3 -m pytest paper/reproduction/n170/historical/tests/test_historical.py -q
PYTHONPATH="src:paper/reproduction:paper/reproduction/n170" \
  python3 paper/reproduction/n170/historical/run.py --track all --B 1000 --n-track-b-seeds 20
```

Track A only / Track B only: `--track a` / `--track b`.
Do not require B=1000 in unit tests; smoke fits use B=0 or B=2.

## Environment

From `paper/results/n170/historical/environment.json` after Track B
(2026-09-04T22:18:53Z):

| Item | Value |
| --- | --- |
| Python | 3.12.3 |
| numpy | 2.4.4 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| redisca (installed; **not imported** by historical modules) | 0.1.0 |
| MATLAB | not used |
| RNG | numpy PCG64, master seed `20240904` |
| ERP | `1_N170_erp_ar.erp` SHA-256 `53e74e93…9bbc72` |
| Tests | 6 passed (`test_historical.py`) |

## What is still unmatched / blocked

- Printed face RDM corr ≈ 0.82 (all 12 variants ~1 or 0.95).
- Printed car λ2 ≈ 0.770 (best joint freeze 0.790).
- Printed face λ1 ≈ 0.872 (best 0.880).
- MATLAB numerical parity (blocked: no MATLAB).
- Paper §2.3 condition-label p-values cannot equal 0.009 at C=4 (discrete
  floor 0.25). If the figure used that test, the printed p2 is unexplained
  by this data/C; if it used stock SPoC random-phase on directed pairs, p2
  around 0.009 is natural.

No undocumented parameters were tuned. No scalar objective was optimized.
