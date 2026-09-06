"""Pre-registered Stage A constants. Do not change after seeing results."""

from __future__ import annotations

# Five pre-registered master seeds. Frozen before any full-run selection.
MASTER_SEEDS: tuple[int, ...] = (
    20240904,
    20240905,
    20240906,
    20240907,
    20240908,
)

# Literal historical inference budgets. Do not inflate and call that reproduction.
RANDOM_PHASE_B = 1000
AIRI_NMC_TEMPORAL = 100

# Paper MEG temporal Nmc is unspecified. Assumed equal to the AIRI budget.
PAPER_TEMPORAL_NMC_ASSUMED = 100

# Paper component-label permutation B is unspecified. C=4 and C=6 label
# permutations are enumerated exactly when feasible.
N170_N_CONDITIONS = 4
MEG_N_CONDITIONS = 6
