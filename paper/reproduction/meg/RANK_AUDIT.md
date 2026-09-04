# MEG rank audit: 67 vs 68

Track F. Owner: `paper/reproduction/meg/rank_audit/`. Does **not** rerun MEG Monte Carlo, SPoC B=1000, or source localization.

Question: local reconstructions report whitening rank **68**; author-saved AIRI `A1` is **204 × 67**. Is that a SciPy-vs-MATLAB eig borderline at `tol = λ_max · 1e-6`, or a different Cxx / an explicit PCA setting?

## Commands

```bash
PYTHONPATH=src:paper/reproduction python3 paper/reproduction/meg/rank_audit/run.py
PYTHONPATH=src:paper/reproduction python3 -m pytest paper/reproduction/meg/rank_audit -q
```

No bootstrap. Pair matrices only. Bandpass is the AIRI 0.25–20 Hz `scipy.signal.filtfilt` reconstruction (D8: not MATLAB parity).

## Source evidence (not tuned)

| Claim | Source |
| --- | --- |
| Rank cutoff `tol = ev(1)*1e-6`, `r = sum(ev > tol)` | stock SPoC `utils/whiten_data.m` @ `18e4754` |
| Optional PCA `pca_X_var_explained` default **1** | `SPoC/spoc.m` `set_defaults` |
| AIRI call does **not** pass PCA / rank | `spoc(Xspoc, z, 'n_bootstrapping_iterations',1000)` |
| AIRI bandpass 0.25–20 Hz, `trange=600:1500` | `Redisca_tools_faces_3_random_norm_correct.m` |
| `A1` is 204×67 Haufe patterns after whitening | `stock_spoc.md`; OSF `topo_face_vs_tool_correct_filt15.mat` |
| Committed script `return`s before `save topo_*` (D17) | same AIRI main script |
| Filename `filt15` is **not** `highCutOff` | committed `highCutOff=20`; `cfg.ylim=[15 20]` is a plot setting |

## Verdict (computed)

Paper-faithful unique-unordered unscaled Gram on the full 1501-sample unfiltered averages has rank **68** at `tol = λ_max·1e-6`. AIRI-executable directed MATLAB-cov on the 99–999 ms window after 0.25–20 Hz scipy filtfilt has rank **68**. Author-saved `A1` has **67** columns.

These numbers **disagree**. This audit does **not** truncate local whitening to 67 merely because the OSF file has 67 columns.

On both owned Cxx paths the 68th relative eigenvalue is `4.36047394e-06` (paper) and `2.48482986e-06` (AIRI), versus cutoff `1e-6`. The 69th is `1.08203078e-08` and `7.26533085e-09`. MATLAB-eig-only flip verdicts: paper **not_borderline**, AIRI **not_borderline**.

The gap from λ₆₈ down to the cutoff, and from the cutoff down to λ₆₉, is large compared with dense-Hermitian solver noise (~n·ε ≈ 4×10⁻¹⁴, padded to 10⁻¹²). A SciPy-vs-MATLAB eig difference **alone** is not a plausible explanation of 67 vs 68 on *these* Cxx matrices.

AIRI MATLAB does not pass `pca_X_var_explained` (`Xspoc, z, 'n_bootstrapping_iterations',1000`). SPoC default pca=1 keeps the numerical rank, so an explicit PCA setting is **not** present in the committed scripts. The `pca` interval that would yield exactly 67 on the AIRI Cxx is `pca in (0.9999979816751061, 0.99999932257837] yields find()=67 before min(., r=68)`.

Labeled extra: AIRI window + butter(3) **0.25–15 Hz** (filename `filt15` hypothesis) has rank **68**. That does **not** match 67, so the filename alone does not explain A1's column count on this reconstruction.

AIRI window **without** bandpass, MATLAB cov: rank **68**.

A1 is **AIRI-like in the leading columns** of a local facevstool Haufe fit (no bootstrap): c1|r|=0.985, c2|r|=0.954, c3|r|=0.976, c4|r|=0.931. The 67-D column space vs the first 67 of the rank-68 AIRI-executable patterns has min cosine `0.8675` (max principal angle `0.521` rad). Vs paper-faithful Haufe min cosine is `0.5675` (worse; D2/D6/D8). Matching leading topographies does **not** license truncating local Cxx to 67 columns.

Every labeled extra Cxx we formed is also rank **68**, with a cliff between λ₆₈ (few ×10⁻⁶ λ_max) and λ₆₉ (~10⁻⁸ λ_max) then a numerical floor (~10⁻¹⁴). Window, Gram-vs-cov, 0.25–20 Hz, 0.25–15 Hz, and no bandpass do not move the count. The rank looks like a property of the tSSS planar data, not of those analysis knobs.

Remaining untested path to 67: MATLAB Signal Processing Toolbox `filtfilt` plus MATLAB `eig` on **that** Cxx (D8 + D10 together), or some other unsaved MATLAB run (D17). That is a *different matrix*, not a solver-only flip of the SciPy Cxx computed here. It is not an explicit `pca_X_var_explained` setting in the committed AIRI scripts.

| Item | Value |
| --- | --- |
| paper_faithful rank | **68** |
| airi_executable rank | **68** |
| author-saved A1 columns | **67** |
| Force local rank to 67 because A1 has 67 columns? | **no** |
| MATLAB-eig-only flip (paper path) | not_borderline |
| MATLAB-eig-only flip (AIRI path) | not_borderline |
| Explicit AIRI PCA setting for 67? | none in AIRI scripts; SPoC default pca_X_var_explained=1 (numerical rank only) |

Labeled extras (still not a reason to force rank 67):

| extra Cxx | rank at 1e-6 |
| --- | --- |
| `airi_window_filt15_matlab_cov` | 68 |
| `airi_window_filtered_unscaled_gram` | 68 |
| `airi_window_unfiltered_matlab_cov` | 68 |
| `paper_window_matlab_cov` | 68 |

## Cxx / Rbar spectra

### `paper_faithful`

- Pairs: `unique_unordered` (15 pair matrices).
- Pair matrix: `unscaled_gram`.
- Window: -500.0…1000.0 ms, T=1501.
- Bandpass: none.
- λ_max = `9.046980e-20`, cutoff = λ_max·1e-6 = `9.046980e-26`.
- **Numerical rank at 1e-6 (numpy eigh): 68**.
- `whiten_from_covariance(..., pca_var_explained=1)` rows: 68.
- Solver ranks: numpy_eigh=68, scipy_eigh=68, scipy_eig_real=68 (agree).
- max |numpy eigh − scipy eigh| = `6.018531e-36`.
- Cxx asymmetry ‖C−Cᵀ‖_F / ‖C‖_F = `0`.

Indices 60–75 (1-based, descending λ):

| i (1-based) | eigenvalue | λᵢ / λ_max | λᵢ > λ_max·1e-6 | margin (ratio − 1e-6) |
| --- | --- | --- | --- | --- |
| 60 | 5.441477e-23 | 6.014689e-04 | yes | 6.004689e-04 |
| 61 | 5.043632e-23 | 5.574935e-04 | yes | 5.564935e-04 |
| 62 | 4.787437e-23 | 5.291752e-04 | yes | 5.281752e-04 |
| 63 | 4.224891e-23 | 4.669947e-04 | yes | 4.659947e-04 |
| 64 | 3.249619e-23 | 3.591938e-04 | yes | 3.581938e-04 |
| 65 | 2.750117e-23 | 3.039818e-04 | yes | 3.029818e-04 |
| 66 | 3.934808e-24 | 4.349306e-05 | yes | 4.249306e-05 |
| 67 | 7.113836e-25 | 7.863217e-06 | yes | 6.863217e-06 |
| 68 | 3.944912e-25 | 4.360474e-06 | yes | 3.360474e-06 |
| 69 | 9.789110e-28 | 1.082031e-08 | **no** | -9.891797e-07 |
| 70 | 1.651200e-35 | 1.825139e-16 | **no** | -1.000000e-06 |
| 71 | 1.451136e-35 | 1.604001e-16 | **no** | -1.000000e-06 |
| 72 | 1.303075e-35 | 1.440343e-16 | **no** | -1.000000e-06 |
| 73 | 1.234632e-35 | 1.364690e-16 | **no** | -1.000000e-06 |
| 74 | 1.108762e-35 | 1.225561e-16 | **no** | -1.000000e-06 |
| 75 | 9.826633e-36 | 1.086178e-16 | **no** | -1.000000e-06 |

Focus: λ₆₇/λ_max=`7.863217e-06`, λ₆₈/λ_max=`4.360474e-06`, λ₆₉/λ_max=`1.082031e-08`.
Margin of 68 above cutoff = `3.360474e-06`; margin of 69 below = `9.891797e-07`; gap 68−69 = `4.349654e-06`.
Relative drop of λ₆₈ needed to cross the cutoff: `0.770667`.
MATLAB-eig-flip verdict (gap argument, not parity): **not_borderline** (plausible solver-only 68→67: False).

SPoC `pca_X_var_explained` (AIRI does not pass it; default 1):

- default pca=1 selects **68** components (= numerical rank after `min(n, r)`).
- interval that would yield *exactly* 67 before `min(., r)`: pca in (0.9999963413119723, 0.9999986927796343] yields find()=67 before min(., r=68).
- cumulative variance at 66 / 67: `0.999996341312` / `0.99999869278`.

Extra:

- `alternate_pair_mode`: `airi_directed`
- `cxx_directed_vs_unique_max_abs`: `3.009265538105056e-36`
- `n_pairs_alternate_mode`: `30`
- `note_pairs`: `For a symmetric pair matrix (Gram or cov of ±Δ), directed duplication does not change Cxx. Rank is a property of Cxx, not of z or bootstrap.`

### `airi_executable`

- Pairs: `airi_directed` (30 pair matrices).
- Pair matrix: `matlab_cov`.
- Window: 99.0…999.0 ms, T=901.
- Bandpass: butter(3) 0.25–20.0 Hz. Note: scipy filtfilt; AIRI MATLAB Signal Processing Toolbox filtfilt is not bit-exact.
- λ_max = `4.366985e-23`, cutoff = λ_max·1e-6 = `4.366985e-29`.
- **Numerical rank at 1e-6 (numpy eigh): 68**.
- `whiten_from_covariance(..., pca_var_explained=1)` rows: 68.
- Solver ranks: numpy_eigh=68, scipy_eigh=68, scipy_eig_real=68 (agree).
- max |numpy eigh − scipy eigh| = `1.175494e-38`.
- Cxx asymmetry ‖C−Cᵀ‖_F / ‖C‖_F = `0`.

Indices 60–75 (1-based, descending λ):

| i (1-based) | eigenvalue | λᵢ / λ_max | λᵢ > λ_max·1e-6 | margin (ratio − 1e-6) |
| --- | --- | --- | --- | --- |
| 60 | 1.765086e-26 | 4.041887e-04 | yes | 4.031887e-04 |
| 61 | 1.617744e-26 | 3.704487e-04 | yes | 3.694487e-04 |
| 62 | 1.515582e-26 | 3.470546e-04 | yes | 3.460546e-04 |
| 63 | 1.211770e-26 | 2.774843e-04 | yes | 2.764843e-04 |
| 64 | 1.054716e-26 | 2.415205e-04 | yes | 2.405205e-04 |
| 65 | 6.539127e-27 | 1.497401e-04 | yes | 1.487401e-04 |
| 66 | 9.423895e-28 | 2.157987e-05 | yes | 2.057987e-05 |
| 67 | 2.154193e-28 | 4.932908e-06 | yes | 3.932908e-06 |
| 68 | 1.085121e-28 | 2.484830e-06 | yes | 1.484830e-06 |
| 69 | 3.172759e-31 | 7.265331e-09 | **no** | -9.927347e-07 |
| 70 | 7.264364e-37 | 1.663474e-14 | **no** | -1.000000e-06 |
| 71 | 4.976238e-37 | 1.139514e-14 | **no** | -1.000000e-06 |
| 72 | 4.144115e-37 | 9.489649e-15 | **no** | -1.000000e-06 |
| 73 | 3.113395e-37 | 7.129394e-15 | **no** | -1.000000e-06 |
| 74 | 2.275584e-37 | 5.210882e-15 | **no** | -1.000000e-06 |
| 75 | 1.597290e-37 | 3.657650e-15 | **no** | -1.000000e-06 |

Focus: λ₆₇/λ_max=`4.932908e-06`, λ₆₈/λ_max=`2.484830e-06`, λ₆₉/λ_max=`7.265331e-09`.
Margin of 68 above cutoff = `1.484830e-06`; margin of 69 below = `9.927347e-07`; gap 68−69 = `2.477565e-06`.
Relative drop of λ₆₈ needed to cross the cutoff: `0.597558`.
MATLAB-eig-flip verdict (gap argument, not parity): **not_borderline** (plausible solver-only 68→67: False).

SPoC `pca_X_var_explained` (AIRI does not pass it; default 1):

- default pca=1 selects **68** components (= numerical rank after `min(n, r)`).
- interval that would yield *exactly* 67 before `min(., r)`: pca in (0.9999979816751061, 0.99999932257837] yields find()=67 before min(., r=68).
- cumulative variance at 66 / 67: `0.999997981675` / `0.999999322578`.

Extra:

- `alternate_pair_mode`: `unique_unordered`
- `cxx_directed_vs_unique_max_abs`: `2.204051907791789e-39`
- `n_pairs_alternate_mode`: `15`
- `note_pairs`: `For a symmetric pair matrix (Gram or cov of ±Δ), directed duplication does not change Cxx. Rank is a property of Cxx, not of z or bootstrap.`

## Diagnostic Cxx (labeled extras; not the two owned paths)

### `airi_window_filt15_matlab_cov`

- Pairs: `airi_directed` (30 pair matrices).
- Pair matrix: `matlab_cov`.
- Window: 99.0…999.0 ms, T=901.
- Bandpass: butter(3) 0.25–15.0 Hz. Note: Labeled extra for the OSF filename filt15. Not the committed AIRI highCutOff=20 path. Applied to the 480 used trials only (unused trials never enter condition averages)..
- λ_max = `4.281383e-23`, cutoff = λ_max·1e-6 = `4.281383e-29`.
- **Numerical rank at 1e-6 (numpy eigh): 68**.
- `whiten_from_covariance(..., pca_var_explained=1)` rows: 68.
- Solver ranks: numpy_eigh=68, scipy_eigh=68, scipy_eig_real=68 (agree).
- max |numpy eigh − scipy eigh| = `8.816208e-39`.
- Cxx asymmetry ‖C−Cᵀ‖_F / ‖C‖_F = `0`.

Indices 60–75 (1-based, descending λ):

| i (1-based) | eigenvalue | λᵢ / λ_max | λᵢ > λ_max·1e-6 | margin (ratio − 1e-6) |
| --- | --- | --- | --- | --- |
| 60 | 1.139988e-26 | 2.662662e-04 | yes | 2.652662e-04 |
| 61 | 1.066226e-26 | 2.490377e-04 | yes | 2.480377e-04 |
| 62 | 9.793203e-27 | 2.287392e-04 | yes | 2.277392e-04 |
| 63 | 8.204489e-27 | 1.916317e-04 | yes | 1.906317e-04 |
| 64 | 6.679901e-27 | 1.560220e-04 | yes | 1.550220e-04 |
| 65 | 4.323087e-27 | 1.009741e-04 | yes | 9.997408e-05 |
| 66 | 6.948999e-28 | 1.623073e-05 | yes | 1.523073e-05 |
| 67 | 1.454659e-28 | 3.397638e-06 | yes | 2.397638e-06 |
| 68 | 7.231727e-29 | 1.689110e-06 | yes | 6.891100e-07 |
| 69 | 2.095914e-31 | 4.895413e-09 | **no** | -9.951046e-07 |
| 70 | 3.400055e-36 | 7.941487e-14 | **no** | -9.999999e-07 |
| 71 | 2.313482e-36 | 5.403585e-14 | **no** | -9.999999e-07 |
| 72 | 2.013777e-36 | 4.703567e-14 | **no** | -1.000000e-06 |
| 73 | 1.670524e-36 | 3.901832e-14 | **no** | -1.000000e-06 |
| 74 | 1.018685e-36 | 2.379335e-14 | **no** | -1.000000e-06 |
| 75 | 6.268517e-37 | 1.464134e-14 | **no** | -1.000000e-06 |

Focus: λ₆₇/λ_max=`3.397638e-06`, λ₆₈/λ_max=`1.689110e-06`, λ₆₉/λ_max=`4.895413e-09`.
Margin of 68 above cutoff = `6.891100e-07`; margin of 69 below = `9.951046e-07`; gap 68−69 = `1.684215e-06`.
Relative drop of λ₆₈ needed to cross the cutoff: `0.407972`.
MATLAB-eig-flip verdict (gap argument, not parity): **not_borderline** (plausible solver-only 68→67: False).

SPoC `pca_X_var_explained` (AIRI does not pass it; default 1):

- default pca=1 selects **68** components (= numerical rank after `min(n, r)`).
- interval that would yield *exactly* 67 before `min(., r)`: pca in (0.9999985677531575, 0.9999995234870686] yields find()=67 before min(., r=68).
- cumulative variance at 66 / 67: `0.999998567753` / `0.999999523487`.

Extra:

- `alternate_pair_mode`: `unique_unordered`
- `cxx_directed_vs_unique_max_abs`: `1.4693679385278594e-39`
- `n_pairs_alternate_mode`: `15`
- `note_pairs`: `For a symmetric pair matrix (Gram or cov of ±Δ), directed duplication does not change Cxx. Rank is a property of Cxx, not of z or bootstrap.`
- `role`: `filename filt15 hypothesis; do not replace airi_executable`

### `airi_window_filtered_unscaled_gram`

- Pairs: `unique_unordered` (15 pair matrices).
- Pair matrix: `unscaled_gram`.
- Window: 99.0…999.0 ms, T=901.
- Bandpass: butter(3) 0.25–20.0 Hz. Note: scipy filtfilt; AIRI MATLAB Signal Processing Toolbox filtfilt is not bit-exact.
- λ_max = `4.191142e-20`, cutoff = λ_max·1e-6 = `4.191142e-26`.
- **Numerical rank at 1e-6 (numpy eigh): 68**.
- `whiten_from_covariance(..., pca_var_explained=1)` rows: 68.
- Solver ranks: numpy_eigh=68, scipy_eigh=68, scipy_eig_real=68 (agree).
- max |numpy eigh − scipy eigh| = `1.203706e-35`.
- Cxx asymmetry ‖C−Cᵀ‖_F / ‖C‖_F = `0`.

Indices 60–75 (1-based, descending λ):

| i (1-based) | eigenvalue | λᵢ / λ_max | λᵢ > λ_max·1e-6 | margin (ratio − 1e-6) |
| --- | --- | --- | --- | --- |
| 60 | 1.814337e-23 | 4.328980e-04 | yes | 4.318980e-04 |
| 61 | 1.562096e-23 | 3.727136e-04 | yes | 3.717136e-04 |
| 62 | 1.436680e-23 | 3.427896e-04 | yes | 3.417896e-04 |
| 63 | 1.344491e-23 | 3.207934e-04 | yes | 3.197934e-04 |
| 64 | 1.061838e-23 | 2.533529e-04 | yes | 2.523529e-04 |
| 65 | 6.070065e-24 | 1.448308e-04 | yes | 1.438308e-04 |
| 66 | 1.016638e-24 | 2.425683e-05 | yes | 2.325683e-05 |
| 67 | 2.258681e-25 | 5.389178e-06 | yes | 4.389178e-06 |
| 68 | 1.042801e-25 | 2.488107e-06 | yes | 1.488107e-06 |
| 69 | 2.908388e-28 | 6.939368e-09 | **no** | -9.930606e-07 |
| 70 | 3.160409e-33 | 7.540688e-14 | **no** | -9.999999e-07 |
| 71 | 2.248898e-33 | 5.365836e-14 | **no** | -9.999999e-07 |
| 72 | 2.167033e-33 | 5.170506e-14 | **no** | -9.999999e-07 |
| 73 | 1.152368e-33 | 2.749532e-14 | **no** | -1.000000e-06 |
| 74 | 8.805484e-34 | 2.100975e-14 | **no** | -1.000000e-06 |
| 75 | 3.656192e-34 | 8.723617e-15 | **no** | -1.000000e-06 |

Focus: λ₆₇/λ_max=`5.389178e-06`, λ₆₈/λ_max=`2.488107e-06`, λ₆₉/λ_max=`6.939368e-09`.
Margin of 68 above cutoff = `1.488107e-06`; margin of 69 below = `9.930606e-07`; gap 68−69 = `2.481167e-06`.
Relative drop of λ₆₈ needed to cross the cutoff: `0.598088`.
MATLAB-eig-flip verdict (gap argument, not parity): **not_borderline** (plausible solver-only 68→67: False).

SPoC `pca_X_var_explained` (AIRI does not pass it; default 1):

- default pca=1 selects **68** components (= numerical rank after `min(n, r)`).
- interval that would yield *exactly* 67 before `min(., r)`: pca in (0.9999978718968894, 0.9999993265392316] yields find()=67 before min(., r=68).
- cumulative variance at 66 / 67: `0.999997871897` / `0.999999326539`.

Extra:

- `alternate_pair_mode`: `airi_directed`
- `cxx_directed_vs_unique_max_abs`: `1.88079096131566e-36`
- `n_pairs_alternate_mode`: `30`
- `note_pairs`: `For a symmetric pair matrix (Gram or cov of ±Δ), directed duplication does not change Cxx. Rank is a property of Cxx, not of z or bootstrap.`
- `role`: `isolate D2 on the AIRI window+filter`

### `airi_window_unfiltered_matlab_cov`

- Pairs: `airi_directed` (30 pair matrices).
- Pair matrix: `matlab_cov`.
- Window: 99.0…999.0 ms, T=901.
- Bandpass: none.
- λ_max = `4.842659e-23`, cutoff = λ_max·1e-6 = `4.842659e-29`.
- **Numerical rank at 1e-6 (numpy eigh): 68**.
- `whiten_from_covariance(..., pca_var_explained=1)` rows: 68.
- Solver ranks: numpy_eigh=68, scipy_eigh=68, scipy_eig_real=68 (agree).
- max |numpy eigh − scipy eigh| = `1.175494e-38`.
- Cxx asymmetry ‖C−Cᵀ‖_F / ‖C‖_F = `0`.

Indices 60–75 (1-based, descending λ):

| i (1-based) | eigenvalue | λᵢ / λ_max | λᵢ > λ_max·1e-6 | margin (ratio − 1e-6) |
| --- | --- | --- | --- | --- |
| 60 | 2.704650e-26 | 5.585051e-04 | yes | 5.575051e-04 |
| 61 | 2.279201e-26 | 4.706507e-04 | yes | 4.696507e-04 |
| 62 | 2.056275e-26 | 4.246169e-04 | yes | 4.236169e-04 |
| 63 | 2.004018e-26 | 4.138259e-04 | yes | 4.128259e-04 |
| 64 | 1.529681e-26 | 3.158761e-04 | yes | 3.148761e-04 |
| 65 | 9.795750e-27 | 2.022804e-04 | yes | 2.012804e-04 |
| 66 | 1.291861e-27 | 2.667668e-05 | yes | 2.567668e-05 |
| 67 | 2.935543e-28 | 6.061840e-06 | yes | 5.061840e-06 |
| 68 | 1.738156e-28 | 3.589259e-06 | yes | 2.589259e-06 |
| 69 | 5.478981e-31 | 1.131399e-08 | **no** | -9.886860e-07 |
| 70 | 1.339770e-38 | 2.766600e-16 | **no** | -1.000000e-06 |
| 71 | 9.145653e-39 | 1.888560e-16 | **no** | -1.000000e-06 |
| 72 | 7.669065e-39 | 1.583647e-16 | **no** | -1.000000e-06 |
| 73 | 6.805795e-39 | 1.405384e-16 | **no** | -1.000000e-06 |
| 74 | 6.727946e-39 | 1.389308e-16 | **no** | -1.000000e-06 |
| 75 | 6.471142e-39 | 1.336279e-16 | **no** | -1.000000e-06 |

Focus: λ₆₇/λ_max=`6.061840e-06`, λ₆₈/λ_max=`3.589259e-06`, λ₆₉/λ_max=`1.131399e-08`.
Margin of 68 above cutoff = `2.589259e-06`; margin of 69 below = `9.886860e-07`; gap 68−69 = `3.577945e-06`.
Relative drop of λ₆₈ needed to cross the cutoff: `0.721391`.
MATLAB-eig-flip verdict (gap argument, not parity): **not_borderline** (plausible solver-only 68→67: False).

SPoC `pca_X_var_explained` (AIRI does not pass it; default 1):

- default pca=1 selects **68** components (= numerical rank after `min(n, r)`).
- interval that would yield *exactly* 67 before `min(., r)`: pca in (0.9999972530717076, 0.9999989763928013] yields find()=67 before min(., r=68).
- cumulative variance at 66 / 67: `0.999997253072` / `0.999998976393`.

Extra:

- `alternate_pair_mode`: `unique_unordered`
- `cxx_directed_vs_unique_max_abs`: `1.4693679385278594e-39`
- `n_pairs_alternate_mode`: `15`
- `note_pairs`: `For a symmetric pair matrix (Gram or cov of ±Δ), directed duplication does not change Cxx. Rank is a property of Cxx, not of z or bootstrap.`
- `role`: `isolate D8 (bandpass) from D6 (window)`

### `paper_window_matlab_cov`

- Pairs: `unique_unordered` (15 pair matrices).
- Pair matrix: `matlab_cov`.
- Window: -500.0…1000.0 ms, T=1501.
- Bandpass: none.
- λ_max = `4.006586e-23`, cutoff = λ_max·1e-6 = `4.006586e-29`.
- **Numerical rank at 1e-6 (numpy eigh): 68**.
- `whiten_from_covariance(..., pca_var_explained=1)` rows: 68.
- Solver ranks: numpy_eigh=68, scipy_eigh=68, scipy_eig_real=68 (agree).
- max |numpy eigh − scipy eigh| = `8.816208e-39`.
- Cxx asymmetry ‖C−Cᵀ‖_F / ‖C‖_F = `0`.

Indices 60–75 (1-based, descending λ):

| i (1-based) | eigenvalue | λᵢ / λ_max | λᵢ > λ_max·1e-6 | margin (ratio − 1e-6) |
| --- | --- | --- | --- | --- |
| 60 | 3.587483e-26 | 8.953965e-04 | yes | 8.943965e-04 |
| 61 | 3.357013e-26 | 8.378737e-04 | yes | 8.368737e-04 |
| 62 | 3.141037e-26 | 7.839686e-04 | yes | 7.829686e-04 |
| 63 | 2.796125e-26 | 6.978821e-04 | yes | 6.968821e-04 |
| 64 | 2.154580e-26 | 5.377597e-04 | yes | 5.367597e-04 |
| 65 | 1.826848e-26 | 4.559613e-04 | yes | 4.549613e-04 |
| 66 | 2.559673e-27 | 6.388664e-05 | yes | 6.288664e-05 |
| 67 | 4.699403e-28 | 1.172920e-05 | yes | 1.072920e-05 |
| 68 | 2.568249e-28 | 6.410068e-06 | yes | 5.410068e-06 |
| 69 | 6.498204e-31 | 1.621880e-08 | **no** | -9.837812e-07 |
| 70 | 9.783439e-39 | 2.441839e-16 | **no** | -1.000000e-06 |
| 71 | 7.633462e-39 | 1.905229e-16 | **no** | -1.000000e-06 |
| 72 | 7.122346e-39 | 1.777660e-16 | **no** | -1.000000e-06 |
| 73 | 6.541617e-39 | 1.632716e-16 | **no** | -1.000000e-06 |
| 74 | 5.930470e-39 | 1.480180e-16 | **no** | -1.000000e-06 |
| 75 | 5.900434e-39 | 1.472684e-16 | **no** | -1.000000e-06 |

Focus: λ₆₇/λ_max=`1.172920e-05`, λ₆₈/λ_max=`6.410068e-06`, λ₆₉/λ_max=`1.621880e-08`.
Margin of 68 above cutoff = `5.410068e-06`; margin of 69 below = `9.837812e-07`; gap 68−69 = `6.393849e-06`.
Relative drop of λ₆₈ needed to cross the cutoff: `0.843995`.
MATLAB-eig-flip verdict (gap argument, not parity): **not_borderline** (plausible solver-only 68→67: False).

SPoC `pca_X_var_explained` (AIRI does not pass it; default 1):

- default pca=1 selects **68** components (= numerical rank after `min(n, r)`).
- interval that would yield *exactly* 67 before `min(., r)`: pca in (0.9999952695302368, 0.9999983256101116] yields find()=67 before min(., r=68).
- cumulative variance at 66 / 67: `0.99999526953` / `0.99999832561`.

Extra:

- `alternate_pair_mode`: `airi_directed`
- `cxx_directed_vs_unique_max_abs`: `1.1020259538958945e-39`
- `n_pairs_alternate_mode`: `30`
- `note_pairs`: `For a symmetric pair matrix (Gram or cov of ±Δ), directed duplication does not change Cxx. Rank is a property of Cxx, not of z or bootstrap.`
- `role`: `isolate D2 (Gram vs cov) on the paper window`

## Author-saved A1

- File: `.reproduction_data/meg/topo_face_vs_tool_correct_filt15.mat`
- SHA-256: `b18be3e159164846c0e9d82e3d7dd62e1f01e53d00b511b10f11bd1f8b3b7328`
- Shape: **204 × 67** (`comps_order`=[1, 2, 3, 4]).
- Column ‖·‖₂: min `6.544470e-13`, median `1.227430e-12`, max `3.469181e-12`.
- SVD numerical rank of A1 itself at 1e-6 relative: 67.

**Subspace vs local AIRI-executable Haufe patterns (facevstool, no bootstrap)** (local n=68, compared dim=67): min cosine `0.8675`, max principal angle `0.5206` rad. Leading-column |Pearson|: c1 |r|=0.985, c2 |r|=0.9541, c3 |r|=0.9756, c4 |r|=0.9311.

**Subspace vs local paper-faithful Haufe patterns (facevstool Gram, no bootstrap)** (local n=68, compared dim=67): min cosine `0.5675`, max principal angle `0.9673` rad. Leading-column |Pearson|: c1 |r|=0.3065, c2 |r|=0.3314, c3 |r|=0.8951, c4 |r|=0.3556.

A1 column count is the whitening size of the MATLAB SPoC run that saved this file (D17: committed script returns before save). It is not by itself a reason to truncate a local Cxx rank to 67.

## AIRI / SPoC clone inspection

- AIRI pin: `15bc19cdc76989da202714b257f6de4d26a42c51`.
- SPoC pin: `18e4754aec1411160fd5b7ef0db852f1e0a87d90`.
- Main-script `spoc(...)` kwargs: `Xspoc, z, 'n_bootstrapping_iterations',1000`.
- `highCutOff` in main script: `20.0`.
- `pca_X_var_explained` / `filt15` / `highCutOff` mentions in AIRI `.m`: main: highCutOff ×2; source_loc: filt15 ×1.
- Source-loc loads `topo_face_vs_tool_correct_filt15` (not produced by a vanilla run of the committed main script).
- Plot `cfg.ylim=[15 20]` present: True (FieldTrip display limits, not a 15 Hz Butterworth).

## What this does not claim

- MATLAB `eig` / `filtfilt` bit-exact parity (MATLAB is not in this environment).
- That 67 is “the correct” rank to impose on the Python reconstruction.
- Any change to MEG GEP *p*-values (those tracks are frozen).

## Environment

- Python packages: `{'h5py': '3.16.0', 'matplotlib': '3.11.1', 'numpy': '2.4.4', 'redisca': '0.1.0', 'scikit-learn': '1.9.0', 'scipy': '1.18.1'}`.
- MATLAB: `None`.
- Captured: `2026-09-04T22:21:52.892106+00:00`.
