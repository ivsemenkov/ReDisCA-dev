"""Minimal ReDisCA example using the current public estimator API."""

import numpy as np

from redisca import ReDisCA

rng = np.random.default_rng(0)
n_conditions, n_channels, n_times = 4, 8, 50
X = rng.standard_normal((n_conditions, n_channels, n_times))
y = np.array(
    [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]
)

model = ReDisCA(n_components=2).fit(X, y)
U = model.transform(X)
X_hat = model.inverse_transform(U)

print("eigenvalues_:", model.eigenvalues_)
print("filters_.shape:", model.filters_.shape)
print("patterns_.shape:", model.patterns_.shape)
print("transform output shape:", U.shape)
print("inverse_transform output shape:", X_hat.shape)
