"""The single Stage A ReDisCA factory.

Every experiment fit must be equivalent to::

    ReDisCA(**AIRI_SPOC_KWARGS).fit(X, rdm)

Do not change these settings anywhere in Stage A. Future method ablations
change this module only.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike
from redisca import ReDisCA

AIRI_SPOC_KWARGS: dict[str, Any] = dict(
    n_components=None,
    demean_time=True,
    divide_by_t_minus_1=True,
    directed_pairs=True,
    aggregation="mean",
    solver="whitening",
    rank=None,
    rank_tol=1e-6,
)

LIBRARY_MAIN_SHA = "f657b954da7d48d05b50f6f4dc967595a155f7ae"
AIRI_SHA = "15bc19cdc76989da202714b257f6de4d26a42c51"
SPOC_SHA = "18e4754aec1411160fd5b7ef0db852f1e0a87d90"
ERP_CORE_SHA = "c18b43d70d791ca914d90410afe4ff06d6f7f429"


def make_redisca() -> ReDisCA:
    """Return an unfitted ReDisCA with the frozen AIRI→stock-SPoC configuration."""
    return ReDisCA(**dict(AIRI_SPOC_KWARGS))


def fit_redisca(X: ArrayLike, rdm: ArrayLike) -> ReDisCA:
    """Fit the frozen Stage A estimator. No constructor overrides."""
    return make_redisca().fit(X, rdm)
