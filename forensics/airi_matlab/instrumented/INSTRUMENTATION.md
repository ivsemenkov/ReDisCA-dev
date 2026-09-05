# Instrumentation relative to pinned AIRI `Redisca_tools_faces_3_random_norm_correct.m`

Pinned source (byte-identical copy in `../original/AIRI-ReDisCA/`):

- commit `15bc19cdc76989da202714b257f6de4d26a42c51`
- SHA-256 `44af60c421bbcc6321c5e65f73fedf8f2a9cd81dd776f094935585cdf7ab17f2`

The original file is never edited. All run-time changes live in clearly named copies under `instrumented/` and `scripts/`.

## Allowed modifications (literal sensor-space run)

These are the **only** changes in `Redisca_tools_faces_3_random_norm_correct_instrumented.m` versus the pinned original:

1. **Explicit paths** after `clear all`: forensic root, `data/`, SPoC path, output path.
2. **`addpath`** for pinned stock SPoC (`SPoC/` and `utils/`). The AIRI repo does not vendor SPoC.
3. **RNG seed** immediately before `spoc(...)`, default `rng(1,'twister')`.
4. **Save** `mx`, `D`, `Xspoc`, `z`, `W1`, `A1`, `lambda_values1`, `p_values1`, `Cxx1`, `Cxxz1`, `Cxxz1`, `Cxxe1`, `trange`, `idxTrial`, plus SPoC inputs and provenance, immediately after the `spoc` call.
5. **`return`** immediately after that save, **before** the Nmc time-series permutation loop, FieldTrip topoplots, `exportgraphics`, and the unreachable `save topo_*` statements after the original `return`.
6. **Environment / provenance logging** to JSON/text.
7. No change to: trial selection, `butter`/`filtfilt`, condition averages, RDM numeric values, directed `i!=j` pair loop, `trange`, `spoc` arguments, covariance construction, rank handling, solver, component ordering, or the SPoC null (random-phase surrogates).

`bExportGraphics` remains `false` as in the original. Plotting is skipped by returning, not by changing scientific flags.

## Instrumented SPoC copy

`spoc_save_surrogates.m` is a copy of pinned `matlab_SPoC` `SPoC/spoc.m` (`18e4754`) plus:

- save `lambda_samples` and `r_samples` after the bootstrap loop
- optional `rng` is **not** set inside SPoC; the caller seeds immediately before the call

The vendored `vendor/pinned_stock_SPoC/SPoC/spoc.m` is **not** overwritten.

## What was not changed

- Preprocessing (`butter(3,[0.25,20]/500)`, `filtfilt` on each trial).
- `bValid` / six condition index rules.
- `D` construction and `D = D+D'`.
- Pair loop `for i_cnd`, `for j_cnd`, skip `i_cnd==j_cnd`.
- `trange = 600:1500`.
- `spoc(Xspoc, z, 'n_bootstrapping_iterations', 1000)`.
- No substitution of unique `i<j` pairs in the literal script.
- No replacement of `cov` by an uncentered Gram.
- No replacement of random-phase surrogates by condition-label permutations.
