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
library path. Historical estimator variants use `source_faithful` and
must not import `redisca` inside new historical modules.

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

## Existing unique+Gram library numbers (do not re-tune)

Official `1_N170_erp_ar.erp`, 28 scalp channels, unique pairs, unscaled
Gram, `demean_time=False`:

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

The previous integration treated condition-label permutation as primary.
This investigation reverses that for **historical** reproduction.

## Forbidden

Do not: modify `main`; rerun 108-minute simulations; rerun source
localization; arbitrary hyperparameter search; enumerate arbitrary
windows/channels/pair orders; invent a third ICA component; change the
target RDM without source evidence; declare success from one lucky seed;
hide disagreement; invent a scalar optimization objective.
