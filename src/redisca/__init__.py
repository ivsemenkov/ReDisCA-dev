"""ReDisCA — Representational Dissimilarity Component Analysis.

This package provides a scikit-learn style estimator that finds spatial
components whose condition-to-condition dissimilarity structure matches a
target representational dissimilarity matrix (RDM).
"""

from ._redisca import ReDisCA

__version__ = "0.1.0"

__all__ = [
    "ReDisCA",
    "__version__",
]
