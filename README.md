# ReDisCA

ReDisCA is a Python implementation of Representational Dissimilarity Component
Analysis for EEG/MEG-style evoked responses.

The library finds spatial components whose condition-to-condition
dissimilarity structure matches a user-defined target representational
dissimilarity matrix (RDM).

The public interface is the scikit-learn estimator `ReDisCA` (`fit`,
`transform`, `inverse_transform`) plus `random_phase_test`, which runs a
stock-SPoC random-phase significance test on an already fitted model.
Sliding-window analysis and MNE adapters are not included. Paper
reproduction is maintained separately from this library.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

For tests:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

## Data Shape

`fit` expects condition-average data and a target RDM:

```text
X.shape = (n_conditions, n_channels, n_times)
y.shape = (n_conditions, n_conditions)
```

`y` is the target representational dissimilarity matrix. It must be square,
symmetric, finite, and have a zero diagonal. At least three conditions are
required.

`transform` accepts any number of observations and any time length. Only the
channel axis must match the fitted estimator:

```text
X_new.shape = (n_observations, n_channels, n_times_new)
U.shape     = (n_observations, n_components_, n_times_new)
```

## Constructor

```python
ReDisCA(
    n_components=None,
    *,
    demean_time=True,
    divide_by_t_minus_1=False,
    directed_pairs=False,
    aggregation="mean",
    solver="generalized",
    rank=None,
    rank_tol=1e-6,
)
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_components` | `None` | Leading components used by `transform` / `inverse_transform`. `None` uses all `rank_` components. Does not change the fitted decomposition. |
| `demean_time` | `True` | Temporally center each condition-pair difference per channel before forming the pair matrix. |
| `divide_by_t_minus_1` | `False` | Divide each pair matrix by `T-1`. Independent of `demean_time`. |
| `directed_pairs` | `False` | `False`: unique unordered pairs `i < j`. `True`: every `i != j` in nested-loop order. |
| `aggregation` | `"mean"` | How to form the weighted centered matrix `R_bar_d`: `"mean"` or `"sum"`. `R_bar` is always a mean. |
| `solver` | `"generalized"` | `"generalized"` solves the GEP in the principal subspace of `R_bar`. `"whitening"` uses explicit whitening, then an ordinary eigenproblem. |
| `rank` | `None` | Principal-space rank of `R_bar`. `None` uses the effective numerical rank from `rank_tol`. |
| `rank_tol` | `1e-6` | Keep directions of `R_bar` with `eigval > rank_tol * max_eigval`. |

`ReDisCA()` with no arguments preserves the library defaults that were in
place before the reproduction-oriented switches were added: unique pairs,
`demean_time=True`, no `1/(T-1)` scale, mean aggregation, and the
generalized solver.

## Quick Start

```python
import numpy as np
from redisca import ReDisCA, random_phase_test

X = np.random.randn(4, 16, 200)
y = np.array(
    [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]
)

model = ReDisCA(n_components=3).fit(X, y)
U = model.transform(X)
X_hat = model.inverse_transform(U)

print(model.eigenvalues_)
print(model.filters_.shape, model.patterns_.shape)

result = random_phase_test(model, n_surrogates=200, random_state=0)
print(result.p_values)
```

A runnable copy of the fit / transform workflow is `examples/basic_usage.py`.

Fitted arrays use components as rows:

```text
filters_.shape  == (rank_, n_channels)
patterns_.shape == (rank_, n_channels)
```

`rank` is the principal-space size used to solve the eigenproblem.
`n_components` only slices `transform` / `inverse_transform`; the full
decomposition remains in `filters_`, `patterns_`, and `eigenvalues_`.

`fit` also stores `r_bar_`, `r_bar_d_`, `z_`, `centered_pair_stack_`,
`aggregation_`, `solver_`, and `rank_tol_` so inference can reuse the
fitted state rather than mutable constructor parameters.

## Inference

`random_phase_test` is not part of `fit()`. It takes a fitted `ReDisCA`
and does not refit the model when the number of surrogates changes.

```python
from redisca import random_phase_test

result = random_phase_test(model, n_surrogates=1000, random_state=0)
```

| Argument | Meaning |
| --- | --- |
| `estimator` | Fitted `ReDisCA`. |
| `n_surrogates` | Number of random-phase surrogates `B`. |
| `random_state` | `int`, `numpy.random.Generator`, or `None`. |

The return type is `RandomPhaseResult`:

| Field | Meaning |
| --- | --- |
| `p_values` | One value per fitted component. |
| `null_statistic` | `max(abs(lambda))` for each surrogate. |
| `n_surrogates` | `B`. |

Each surrogate randomizes the phase of the stored standardized target,
rebuilds only the weighted matrix, and recomputes the eigenspectrum in the
fitted `Cxx` subspace. Component p-values are
`count(null >= abs(lambda_observed)) / B` with no `+1` correction, so a
value of 0 is allowed.

This is the stock-SPoC random-phase test. Condition-label permutation
testing is not included.

## Current limitations

This release does not include:

- condition-label permutation tests
- sliding-window analysis
- MNE adapters or plotting
- report / export helpers
- paper-reproduction workflows
