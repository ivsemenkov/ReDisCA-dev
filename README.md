# ReDisCA

ReDisCA is a Python implementation of Representational Dissimilarity Component
Analysis for EEG/MEG-style evoked responses.

The library finds spatial components whose condition-to-condition
dissimilarity structure matches a user-defined target representational
dissimilarity matrix (RDM).

The public interface is the scikit-learn estimator `ReDisCA`. This package
currently provides the deterministic scientific core only: `fit`, `transform`,
and `inverse_transform`. Statistical inference, sliding-window analysis, and
MNE adapters are not included. Paper reproduction is maintained separately
from this lightweight library.

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

## Quick Start

```python
import numpy as np
from redisca import ReDisCA

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
```

A runnable copy of this workflow is `examples/basic_usage.py`.

Fitted arrays use components as rows:

```text
filters_.shape  == (rank_, n_channels)
patterns_.shape == (rank_, n_channels)
```

`rank` is the principal-space size used to solve the generalized eigenproblem.
`n_components` only slices `transform` / `inverse_transform`; the full
decomposition remains in `filters_`, `patterns_`, and `eigenvalues_`.

## Current limitations

This release does not include:

- statistical inference / permutation tests
- sliding-window analysis
- MNE adapters or plotting
- report / export helpers
- paper-reproduction workflows
