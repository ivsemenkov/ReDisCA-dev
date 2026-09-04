# Python library (`src/redisca`, paper-branch baseline)

Library commit (paper branch parent / `main`):
`5a5c8658452172e4011445c9a394c1cbbd3c5f7e`

Public API: `from redisca import ReDisCA` (`src/redisca/_redisca.py`,
primitives in `_core.py`). Sklearn estimator. **Deterministic**. No
p-values, no sliding window, no MEG I/O, no MUSIC.

Paper-branch policy: do not change library semantics merely to improve
reproduction. Historical AIRI/SPoC reconstruction lives in
`paper/reproduction/common/source_faithful.py` and must not import
`redisca`.

## What the library implements

| Step | Function | Behaviour |
| --- | --- | --- |
| Pairs | `pair_indices` | unique unordered \(i<j\) only |
| Pair matrix | `pair_matrix` / `pair_matrices` | unscaled Gram \(\Delta\Delta^\top\); optional **temporal demean** (`demean_time`, default **True**). MATLAB `1/(T-1)` is **omitted by design** (global scale). |
| Vectorize \(D\) | `vectorize_rdm` | same \(i<j\) order |
| Standardize | `standardize_target` | sample SD (`ddof=1`), scale-free via amplitude pre-normalization |
| \(\bar R\) | `mean_pair_matrix` | `mean` over pairs |
| \(\bar R_d\) | `weighted_centered_mean` | `mean_k z_k (R_k - R_bar)` — **mean, not paper sum** |
| GEP | `solve_generalized_eigenproblem` | principal subspace of \(\bar R\), `rank_tol=1e-6` relative, `scipy.linalg.eigh`, signed-descending \(\lambda\), filters normalized to \(w^\top \bar R w = 1\) |
| Patterns | `compute_patterns` | Haufe \(A = \bar R W (W^\top \bar R W)^{-1}\) |

`demean_time=False` is the paper-printed uncentered Gram.
`demean_time=True` matches MATLAB `cov` **centering** but not the
`1/(T-1)` factor. The library comparison test
`test_library_comparison.py` records that unique-pair + matlab-cov
(faithful) eigenvalues match `demean_time=True` because the missing
`T-1` scale cancels in the GEP.

Validation (`_validation.py`): \(C\ge 3\), symmetric `y` with zero
diagonal, finite arrays. Constant RDM pair vector raises.

## What the library does not implement

- Directed AIRI pairs \(i\neq j\).
- SPoC random-phase or paper condition-label permutation.
- MEG/N170 I/O, Butterworth, sliding windows, MUSIC/sLORETA.
- Paper \(A=W^{-1}\) (it always uses Haufe, which is the right
  formula for rank-deficient MEG).

Eigenvector signs are free; compare after sign alignment.
`numpy`/`scipy` eigendecompositions are not MATLAB `eig` bit-exact.

## Source-faithful reconstruction (orchestrator-owned; do not edit)

`paper/reproduction/common/source_faithful.py` (not this audit’s
owned path) already encodes the variants the tracks must call:

- `pair_mode`: `airi_directed` vs `unique_unordered`
- `matrix_mode`: `matlab_cov` vs `unscaled_gram`
- `inference`: `spoc_random_phase` vs paper-style
  `condition_label_permutation_pvalues`
- `airi_rdm(name)` copies the six AIRI numeric matrices
- `AIRI_DEFAULT_RDM_NAME = "facevstool"`
- `AIRI_TRANGE_1BASED = (600, 1500)`
- `AIRI_FILTER = {order:3, low_hz:0.25, high_hz:20, fs:1000}`
- `AIRI_N_BOOTSTRAP = 1000`, `AIRI_N_MC_TIMECourse = 100`

This audit re-verified those constants against the pinned MATLAB
sources. Tracks should not treat that module as a substitute for
reading the paper: it is the AIRI/SPoC reconstruction, not the
published estimator.

## Recommended variant matrix for tracks

Canonical deterministic fit (library, paper pairs):

```python
ReDisCA(demean_time=False)   # printed Gram
ReDisCA(demean_time=True)    # MATLAB centering, still unique pairs
```

AIRI-executable reconstruction (source_faithful):

```python
fit_condition_averages(
    X, rdm,
    pair_mode="airi_directed",
    matrix_mode="matlab_cov",
    n_bootstrapping_iterations=1000,
    inference="spoc_random_phase",
)
```

Paper-described inference on unique pairs:

```python
condition_label_permutation_pvalues(..., max_abs_null=True)
```

Never silently mix these in one result folder. Label the path.
