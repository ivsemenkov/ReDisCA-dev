# Overnight historical-reproduction investigation

Permanent branch: `paper`. Do not modify `main`. Do not merge into `main`.
Do not rerun simulations or source localization.

## Inference hierarchy (correction)

**PRIMARY historical inference** (stock SPoC executable):

```text
random_phase_surrogate(z)
recompute surrogate eigenspectrum in the same whitened space
surrogate statistic = max(abs(lambda))
p = count / B
B = 1000
```

Implemented by `common.source_faithful.fit_condition_averages(...,
inference="spoc_random_phase", n_bootstrapping_iterations=1000)`.

**SECONDARY** printed-method diagnostic: paper §2.3 condition-label
permutation of the theoretical RDM. Record it. Do **not** call it the
historical reproduction oracle.

Canonical library fits (`from redisca import ReDisCA`) remain the
library path (unique pairs, printed Gram with `demean_time=False`).
Do not change library semantics. Historical estimator variants use
`source_faithful` and must not import `redisca` inside historical
modules. `source_faithful.py` does not import `redisca` (AST-checked).

Do **not** treat paper §2.3 condition-label permutation as the
historical reproduction oracle. The previous integration snapshot
(`4813b38`) did that; overnight Tracks A–F reverse it.

## Printed N170 fingerprints (from figures / text)

Fig. 10 face:

- `lambda1 ≈ 0.87209`
- `p1 = 0`
- observed-RDM corr ≈ 0.82
- face response burst ≈ 170 ms
- window: T=100 ms centered at 200 ms

Fig. 11 car:

- `lambda1 ≈ 0.91639`, `p1 = 0`
- `lambda2 ≈ 0.77036`, `p2 ≈ 0.009`
- observed-RDM corr > 0.99
- applied at t=170 ms; duration 100 ms is the only other real-data T

Fig. 7:

- first-component uncorrected p < 0.05 around 400 ms
- T=150 ms; step not printed (existing documented step = 25 ms)

Paper text does not print the lambda values; they are figure-panel
fingerprints supplied for this investigation. Do not tune parameters
to hit them.

## Canonical library unique+Gram numbers (separate path; do not re-tune)

These are `from redisca import ReDisCA` / `source_faithful` unique+Gram
fingerprints, **not** the historical freeze. Official
`1_N170_erp_ar.erp`, 28 scalp channels, unique pairs, unscaled Gram,
`demean_time=False`. `source_faithful` unique+Gram λ matches the library
to ~1e-15 (face λ1 Δ=1.55×10⁻¹⁵; car λ1/λ2 Δ=7.8×10⁻¹⁶ / 8.9×10⁻¹⁶).
That agreement is expected GEP identity, **not** a library bug.

| Item | Value |
| --- | --- |
| Face λ1 | 0.88006 |
| Face window RDM corr | 0.99988 |
| Face full-epoch RDM corr | 0.94466 |
| Face exact-24 p | 0.75 |
| Face exploratory random-phase matched p | 0.000 |
| Car λ1, λ2 | 0.88691, 0.79170 |
| Car window corr | 0.99992, 0.99968 |
| Car exact-24 p | 0.25, 0.25 |
| Car exploratory random-phase matched p | 0.000, 0.000 |
| Car random-phase max\|λ\| p (library GEP) | 0.000, 0.122 |
| Meaning @400 exact-24 p | 8/24 ≈ 0.333 |

## Overnight freeze (Tracks A–F; committed JSON only)

Leading historical candidate
(`paper/results/n170/historical/leading_candidate.json`):

```text
pair_mode   = airi_directed
matrix_mode = matlab_cov
inference   = spoc_random_phase
B           = 1000
data        = 1_N170_erp_ar.erp (28 scalp; keep this file)
face window = T=100 ms centered at 200 ms
car window  = T=100 ms centered at 170 ms
```

This is the least-bad source-supported joint setting, **not** printed-figure
parity and **not** MATLAB `eig`/`rand` parity.

Evidence (do not copy TRACK_REPORT self-labels):

| Track | Report | Compact JSON |
| --- | --- | --- |
| A+B N170 matrix | `paper/reproduction/n170/HISTORICAL_REPORT.md` | `paper/results/n170/historical/{track_a_table,track_b,leading_candidate}.json` |
| C preprocessing | `paper/reproduction/n170/PREPROCESSING_REPORT.md` | `paper/results/n170/preprocessing/` |
| D RDM corr 0.82 | `paper/reproduction/n170/RDM_CORR_REPORT.md` | `paper/results/n170/rdm_correlation/` |
| E Fig. 7/8 apply | `paper/reproduction/n170/HISTORICAL_APPLY_REPORT.md` | `paper/results/n170/historical_apply/` |
| E MEG freeze | `paper/reproduction/meg/HISTORICAL_CANDIDATE_REPORT.md` | `paper/results/meg/historical_candidate/` |
| F rank 67/68 | `paper/reproduction/meg/RANK_AUDIT.md` | `paper/results/meg/rank_audit/` |

Headline mismatches that remain after the freeze (not hidden):

1. Face window RDM corr is still ~0.999 (`w^T R w`) / 0.948 (undemeaned
   trace-sq), **not** printed 0.82. No paper/Eq-named correlation on this
   ERP is 0.82. Car control kills Eq. 2 sample-SD inner product ~0.83.
2. MEG paper-epoch freeze (no AIRI extras): **2** components p<0.05, not 3.
   The AIRI 99–999 ms + 0.25–20 Hz path gets 3; those extras are **not**
   paper methods.
3. D11: paper says three ICA components; subject `"1"` list is 2 and 7.
4. Fig. 18 anatomy (left cuneus vs paper right FG / insula / left IPS).
5. Unique-pair random-phase car p2 ~0.12 does **not** match printed 0.009;
   directed-pair p2 envelope does contain 0.009.

## Forbidden

Do not: modify `main`; rerun 108-minute simulations; rerun source
localization; arbitrary hyperparameter search; enumerate arbitrary
windows/channels/pair orders; invent a third ICA component; change the
target RDM without source evidence; declare success from one lucky seed;
hide disagreement; invent a scalar optimization objective.
