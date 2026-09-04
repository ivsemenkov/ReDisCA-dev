# AIRI MATLAB forensic report (sensor-space)

Track: isolated MATLAB forensic / reproduction.  
Python ReDisCA (`src/redisca`) was not imported, modified, or used as an oracle.

**Literal MATLAB execution of `Redisca_tools_faces_3_random_norm_correct.m` did not occur.** MATLAB is not installed on this VM. That is a stated stop condition. Everything below that depends on a live MATLAB `spoc` call is therefore **not numerically available**. Scripts that would produce those numbers are in `scripts/` and `instrumented/`.

---

## 1. MATLAB / toolbox environment

| Item | Value |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS (Noble), Linux 6.12.94+ x86_64 |
| Host | Cloud Agent VM (`cursor`), 4× Xeon, 15 GiB RAM |
| MATLAB binary | **not found** (`which matlab` empty; no `/usr/local/MATLAB`, `/opt/MATLAB`) |
| MATLAB version | **unavailable** |
| MATLAB toolboxes | **unavailable** |
| Signal Processing Toolbox | **unavailable** (`butter` / `filtfilt` not executable) |
| FieldTrip | **not installed** (would be needed only *after* SPoC for `ft_topoplotER` / `prepare4topoNMG`) |
| BBCI toolbox | cloned for inspection only: `bbci/bbci_public` @ `2e6fe9481537dcfee702e74544191dcf737f02ce` |
| GNU Octave | not installed; **not used** as a MATLAB substitute |
| `which -all spoc` etc. | **not executed** (no MATLAB) |

Full OS dump: `environment.txt`.

The main script’s scientific core needs MATLAB `butter`/`filtfilt` (Signal Processing Toolbox) and `spoc`. Plotting needs FieldTrip and an undocumented `prepare4topoNMG`. Per task rules, plotting-only failures after SPoC would not block forensics; here the blocker is earlier: MATLAB itself.

---

## 2. Exact AIRI commit and script hashes

| Item | Value |
| --- | --- |
| Repo | https://github.com/AIRI-Institute/ReDisCA |
| Commit | `15bc19cdc76989da202714b257f6de4d26a42c51` (2024-11-20, “Create LICENSE”) |
| Main script | `Redisca_tools_faces_3_random_norm_correct.m` |
| Git blob | `f5e339c2945cc70d1f7686b7edb347c87c08c587` |
| SHA-256 | `44af60c421bbcc6321c5e65f73fedf8f2a9cd81dd776f094935585cdf7ab17f2` |
| Source-loc script SHA-256 | `e7270939bb8fe052d23189b471dfde8f31d8b902456e272746983caad178dcb9` |
| Original copy | `original/AIRI-ReDisCA/` (byte-identical to the pinned commit; **never edited**) |

This environment’s ReDisCA-dev baseline is `32c672f65932773359f2feeaa902782086c63c1d`. Forensic files live only under `forensics/airi_matlab/`.

---

## 3. Exact MEG data files and hashes

OSF project: https://osf.io/8rk67/ (7 files). No extraction/conversion.

### Files the main script actually `load`s

| File | Bytes | SHA-256 | Matches OSF API |
| --- | --- | --- | --- |
| `MEG_AD_run1.mat` | 1,243,214,548 | `0eca2756c9190ce637a3e14abd24e7cf975d758d3ccea03107963e8b5841a4f6` | yes |
| `ibfctfprespm8_AD_run1_raw_tsss_mc.mat` | 62,701 | `87890337c385e81c718c421d7be35e54423ca9ceb985e047b276b02018334950` | yes |

Expected filenames **are present**. No substitution.

### Contents (inspection only; not a ReDisCA run)

- `MEG_AD_run1.mat`: MATLAB v7.3, variable `d`, MATLAB shape **(207, 1501, 880)** = channels × time × trials. Script uses `d(1:204,:,:)` (204 planar gradiometers). Extra channels: 2 EOG + 1 Other.
- SPM header: `Fsample = 1000`, `Nsamples = 1501`, `timeOnset = -0.5` s, 880 trials.
- Condition counts after the script’s `bValid` / label rules: **80, 80, 80, 80, 80, 80** (face1/2, tool1/2, nons1/2). Matches the paper’s “80 epochs” per subcategory.
- Paper ERF size “204 × 1500” vs data **204 × 1501**. One extra sample is the usual 500 ms pre + 1000 ms post at 1 kHz including both endpoints.

### OSF files not loaded by the sensor-space script

| File | OSF SHA-256 | Downloaded | Why unused here |
| --- | --- | --- | --- |
| `ibfctfprespm8_AD_run1_raw_tsss_mc.dat` | `d609567ff25eb88055fa26713d7debc4a6c359770835c148fe358bcb97c408e8` | no | 1.09 GB SPM binary; script never loads it |
| `topo_face_vs_tool_correct_filt15.mat` | `b18be3e159164846c0e9d82e3d7dd62e1f01e53d00b511b10f11bd1f8b3b7328` | yes | source-loc input; `A1` is 204×67, `comps_order = [1 2 3 4]` |
| `headmodel_surf_os_meg.mat` | `a365912cae29c3ddda7be90b4bb3830f4ce081e7d4de1206d0c1406985ec439c` | no | source loc |
| `results_sLORETA_MEG_GRAD_MEG_MAG_KERNEL_150924_1824.mat` | `794043eb34f588a14186b297721d78e71ac9a08187938f611bdf0a0e0a92a1d3` | no | source loc |
| `tess_cortex_pial_low.mat` | `40502997c4c21d89a4c7ea207ab77c1c458a005d74e7cb78e6e0e2beb578cad1` | no | source loc |

Large MEG files are **not committed**.

Author-saved `topo_face_vs_tool_correct_filt15.mat` is **not** treated as a substitute for a local SPoC run. It is an OSF artifact whose internal timestamp is 2024-09-04 (GLNXA64). `A1` column Euclidean norms are ~1e-12, i.e. pattern scale is tiny; sign/scale conventions will matter in any later comparison.

---

## 4. Exact SPoC / BBCI files MATLAB would resolve

AIRI README: install SPoC from [bbci_public `external`](https://github.com/bbci/bbci_public/tree/master/external).

What is actually in `bbci_public` @ `2e6fe948`:

- `external/` contains **only** `README.md` telling the user to run `bbci_import_dependencies()`.
- SPoC is **not** vendored.
- `misc/bbci_import_dependencies.m` case `'ssd+spoc'` downloads  
  `https://github.com/svendaehne/matlab_SPoC/archive/master.zip`  
  into `external/ssd+spoc`.
- `processing/proc_spoc.m` is a BBCI **wrapper** that calls `spoc(dat.x, dat.y)` and then **re-sorts by `abs(lambda)`**. The AIRI main script calls `spoc` directly, so this wrapper is **not** on the AIRI executable path unless someone used `proc_spoc` instead.

Pinned stock files used as the **reference environment** (SHA-256):

| File | SHA-256 |
| --- | --- |
| `SPoC/spoc.m` | `0979006739b43b9d74e3a9321f3fc232374930b69fc69045a02d6e49173eadd2` |
| `utils/whiten_data.m` | `6c12f2cbe4ba2bb46f40d25a9d18d5ea3bfe0667f85d5de855e561bfc28defce` |
| `utils/create_Cxxz.m` | `7d8cd5e964da92811c27a36405dc1b74e1ea43bd3e317430c25343f36a80c711` |
| `utils/random_phase_surrogate.m` | `0229289ee0bb0a87d9f5cce31360e0b846f8a7acc08212ab58f8fd4ddc69431f` |
| `utils/propertylist2struct.m` | `fc70782003cc27d0f7439f08b1237c5ebb3e0d26955fa98d9a50e280644ec686` |
| `utils/set_defaults.m` | `0416538a6aa865d298fa3feb6703edb8cabef8d04eec277d89057e7b54a26f2d` |
| `utils/get_var_features.m` | `1830daff9b6e926c4e38332b8b159b10f51e9960ee087552e08588fe74a578c0` |

`which -all` could not be run. On a MATLAB machine, `scripts/record_matlab_environment.m` records those paths and hashes of the resolved files.

---

## 5. Match to pinned stock SPoC `18e4754`

| Environment | Result |
| --- | --- |
| **pinned stock-SPoC reference** | Exactly commit `18e4754aec1411160fd5b7ef0db852f1e0a87d90`. Last commit on `master`. |
| **literal AIRI/BBCI** (if imported *today*) | Byte-identical to pinned stock: `origin/master` of `matlab_SPoC` is `18e4754`. BBCI downloads **master.zip**, not a pinned SHA. |
| **literal AIRI/BBCI** (historical author machine) | **Unknown.** No MATLAB session, no frozen `external/ssd+spoc`. If they imported after 2016-04-04, they almost certainly got this same tree. |

`spoc.m` at this commit:

- Standardizes `z` with MATLAB `std` (**sample SD, N−1**).
- `Cxx = mean_e cov(X_e)` — MATLAB `cov` **temporally demeans** and divides by **T−1**.
- `Cxxe(:,:,e) = cov(X_e) - Cxx` (epoch covariances **mean-centered**).
- `Cxxz = create_Cxxz(Cxxe, z)` = `(Cxxe_vec * z') / Ne`.
- Whitening: `eig(Cxx)`, drop eigenvalues `≤ ev(1)*1e-6`, optional PCA variance cutoff default 1 (no extra PCA reduction).
- Filters normalized so `w' * Cxx * w = 1`.
- `A = Cxx * W / (W' * Cxx * W)`.
- Bootstrap: **`random_phase_surrogate`**, not `randperm(z)` (the permutation line is commented out).
- Null statistic: `max(abs(lambda_values_s))` over components; p = fraction of surrogate maxima ≥ |λ_obs|.

`proc_spoc.m` (BBCI wrapper) is **semantically different** from stock `spoc.m` (re-sorts by `|lambda|` and does not expose p-values). AIRI does not call it.

---

## 6. Did the literal AIRI main script run through SPoC?

**No.** Stop condition: MATLAB unavailable. No new methodological choice was invented to force a run. Octave was not substituted.

Instrumented copy  
`instrumented/Redisca_tools_faces_3_random_norm_correct_instrumented.m`  
is ready to run through `spoc(..., 'n_bootstrapping_iterations', 1000)` and then **return before plotting**.

---

## 7. Shapes / hashes of saved major intermediates

**Not produced** (no MATLAB). Expected shapes from the pinned source + data headers:

| Variable | Expected MATLAB shape / content |
| --- | --- |
| `mx{1..6}` | each 204 × 1501 |
| `D` | 6 × 6, `facevstool` (see §8) |
| `Xspoc` | 901 × 204 × 30 (SPoC layout: T × channels × pairs) |
| `z` | 1 × 30 (or 30 × 1) |
| `W1`,`A1` | 204 × r, `r` = whitened rank (≤ 204; OSF topo file has 67 columns) |
| `Cxx1`,`Cxxz1` | 204 × 204 |
| `Cxxe1` | 204 × 204 × 30 |
| `trange` | `600:1500` (length 901) |
| `idxTrial{k}` | length 80 each |

When MATLAB is available, save as `-v7.3` (`airi_literal_after_spoc.mat`) for h5py.

Static JSON (no SPoC): `results/literal/static_reconstruction.json`, `results/literal/data_inspection.json`.

---

## 8. Deterministic outputs (static, from source + headers)

These do **not** require MATLAB execution. They reconstruct what the pinned script **will** pass to `spoc` for RDM `'facevstool'` (the default: `ThRDMArr(2)`).

### Pair construction (literal AIRI)

Double loop `i_cnd = 1:6`, `j_cnd = 1:6`, skip `i==j`.

- **Number of pairs: 30**
- **Both (i,j) and (j,i) are present**
- Order (1-based):  
  `(1,2),(1,3),(1,4),(1,5),(1,6),(2,1),(2,3),(2,4),(2,5),(2,6),(3,1),(3,2),(3,4),(3,5),(3,6),(4,1),(4,2),(4,3),(4,5),(4,6),(5,1),(5,2),(5,3),(5,4),(5,6),(6,1),(6,2),(6,3),(6,4),(6,5)`

Paper Table 1 / §2: SPoC epoch index enumerates **upper-triangular** pairs `i > j` with `k = (j-1)C + i`. That is **15 unique unordered pairs**, not 30 directed pairs. This is a **paper vs executable** discrepancy, not a reconstruction error.

### Default `D` (`facevstool`, then `D = D+D'`)

```
0.0  0.1  1.0  1.0  0.5  0.5
0.1  0.0  1.0  1.0  0.5  0.5
1.0  1.0  0.0  0.1  0.5  0.5
1.0  1.0  0.1  0.0  0.5  0.5
0.5  0.5  0.5  0.5  0.0  0.1
0.5  0.5  0.5  0.5  0.1  0.0
```

This matches the paper’s **non-binary** MEG RDM (Fig. 16): within-category 0.1, face–tool 1, each meaningful vs nonsense 0.5. It is **not** the face-only detector of Fig. 12a / Fig. 13.

### Exact `z` (before SPoC standardization)

```
0.1, 1, 1, 0.5, 0.5,
0.1, 1, 1, 0.5, 0.5,
1, 1, 0.1, 0.5, 0.5,
1, 1, 0.1, 0.5, 0.5,
0.5, 0.5, 0.5, 0.5, 0.1,
0.5, 0.5, 0.5, 0.5, 0.1
```

| Statistic | Value |
| --- | --- |
| mean(z) | 0.5533333333333333 |
| sample SD (MATLAB `std`, N−1) | 0.3148435115761625 |
| population SD (N) | 0.3095516470998373 |
| standardized z (sample SD) | three levels: ±1.43986875, ±1.41869421, ±0.16939632 |

SPoC will re-standardize with sample SD. Variant 4 exists to test population SD **after** a literal run.

### Rank / eigenvalues / W,A checks / Cxx reconstruction errors

**Not available** without MATLAB `spoc`. Independent audit script: `scripts/run_independent_audit.m`.

Whitening rank **will** use `ev_sorted(1) * 10^-6` in `whiten_data.m`. OSF `A1` having 67 columns is a **hint** of rank 67 after SSS/tSSS, not a local measurement.

### `trange`

`trange = 600:1500` (1-based). With SPM `timeOnset = -0.5` s and fs = 1000 Hz:

- index 1 → −500 ms  
- index 600 → **+99 ms**  
- index 1500 → **+999 ms**  
- index 1501 (unused by `trange`) → +1000 ms  

The script’s later plot axis `linspace(-536, 964, size(mx{1},2))` is **inconsistent** with SPM `timeOnset = -0.5` (−500…+1000 ms). If that linspace were used, sample 600 would be +63 ms. **Do not mix the two clocks.** SPM metadata is the file-native clock.

Paper §4.2.2: ReDisCA applied to the **entire 1500 ms** window. The executable uses **~99–999 ms** (901 samples), not the prestimulus. Another paper/code discrepancy.

### Filter

`butter(3, [0.25, 20]/500)` then `filtfilt` per trial. Nyquist assumed 500 Hz ⇒ fs = 1000 Hz. Paper MEG section does **not** quote this 0.25–20 Hz Butterworth. Treat it as **executable pipeline**, not a printed MEG methods number.

---

## 9. SPoC p-values for seeds 1, 2, 3

**Not produced.** `n_bootstrapping_iterations = 1000` is wired in the instrumented script. `scripts/run_seeds.m` would save `p_values1`, `lambda_values1`, and the surrogate `max|λ|` vector via `spoc_save_surrogates.m` (isolated copy; stock `spoc.m` not overwritten).

Null remains **random-phase surrogates**. Condition-label permutation of `z` is still commented out in stock `spoc.m`.

---

## 10. Comparison with paper MEG results

Primary PDF: published NeuroImage 301:120868 (25 pages), SHA-256 `018478c993ab34b46b732b2b00a573a9e342101e6ae80e190f97a904967a3208`.  
Not supplied in the workspace; downloaded from megmoscow.ru. bioRxiv preprint also fetched; **not** used as the comparison source.

Only quantities that can actually be identified:

| Quantity | Paper | MATLAB/AIRI executable | Direct comparison? |
| --- | --- | --- | --- |
| Dataset | Kozunov et al. 2018, subject “AD”, first run | OSF `MEG_AD_run1` + SPM labels `ibfctfprespm8_AD_run1_*` | yes: same study / subject tag |
| Subcategories | face1/2, tool1/2, nons1/2 | same six `idxTrial` | yes |
| Epochs per subcategory | 80; 480 total | 80 each after `bValid` | **match** |
| Sensors | 204 planar gradiometers | `d(1:204,:,:)` of 207 | **match** (3 extras unused) |
| ERF size | 204 × 1500 | 204 × 1501 | 1-sample endpoint difference |
| Time window for ReDisCA | “entire 1500 ms at once” | `trange=600:1500` ≈ 99–999 ms | **mismatch** |
| Pair set | Table 1: unique upper-tri pairs | 30 directed `i≠j` | **mismatch** |
| Default RDM in “main script” | paper shows three category RDMs (Fig. 12) then a non-binary RDM (Fig. 16) | code default `RDM = ThRDMArr(2)` = `'facevstool'` = Fig. 16-style | default executable is the **non-binary** RDM, not Fig. 13 face detector |
| Component p-values (Fig. 13–17 titles) | printed in figures; no numeric table in text | not run | cannot compare numbers |
| Face component peak ~160 ms, tools ~210 ms, meaning ~160 ms | qualitative | would need `W'*mx` time series | plot/sign/scale ambiguity |
| Time-series asterisks | permutation of **epoch labels**, Nmc-style, FWER over time | original script `Nmc=100` `randperm` on pooled trials **after** SPoC | different test from SPoC `p_values1`; not executed here |
| MUSIC / cortex maps | Fig. 18 | source-loc script **not run** | out of scope |

Sign of `W`/`A` is arbitrary (eigenvectors). Topography in the original script is `sqrt(topo(odd)^2+topo(even)^2)` on magnetometer-like slots — **unsigned planar RMS**, so pattern sign is already discarded in plots. Eigenvalue scale follows SPoC `cov` / `Ne` / sample-SD `z`; paper writes unscaled correlation / covariance of RDM entries. **Do not rescale to force agreement** until a MATLAB `lambda_values1` vector exists.

---

## 11. Controlled variants

Scripts are in `scripts/run_variants.m`. **Not executed** (need literal `airi_literal_after_spoc.mat`).

| # | One change | Purpose |
| --- | --- | --- |
| 1 | unique `i < j` pairs (15) vs directed 30 | paper Table 1 vs AIRI loop |
| 2 | uncentered Gram `X'X/T` instead of `cov` | printed difference-correlation vs MATLAB `cov` |
| 3 | sum vs mean scaling of `Cxx`/`Cxxz` | eigenvalue scale; filters should be invariant |
| 4 | population vs sample SD of `z` | only if scale remains unexplained |

No grid search. Combinations not run.

---

## 12. Pair-order sensitivity diagnostic

`scripts/run_order_sensitivity.m`. **Not executed.** Labeled as a **methodological diagnostic**, not a paper reproduction.

Hypothesis from stock `random_phase_surrogate`: `z` is treated as an ordered sequence (`fft`/`ifft`). Reordering the 30 identical pair observations should leave `Cxx`/`Cxxz` (hence `W`,`A`,`λ`) unchanged up to numerical tolerance, but can change surrogate `p_values`. The diagnostic must not be used to “fix” inference in this task.

---

## 13. Source / code / paper discrepancies found

1. **Pairs:** paper unique upper triangle vs AIRI all directed `i≠j` (30 vs 15). Because `D` is symmetric, `z(i,j)=z(j,i)` and `Xspoc(:,:,ji) = -Xspoc(:,:,ij)`, so `cov` of the reverse pair equals `cov` of the forward pair. Directed duplication **repeats each unique covariance twice** and **does not add a new scatter observation**. It **does** change `Ne` in `Cxxz = .../Ne` and in z-standardization (`std` over 30 vs 15 copies). Filters may still span a similar subspace; eigenvalues and p-values need the unique-pair variant.
2. **Time window:** paper “entire 1500 ms” vs `trange=600:1500`.
3. **Covariance:** paper unscaled correlation of difference time series vs MATLAB `cov` (temporal demean, /T−1).
4. **SPoC null:** stock code random-phase surrogates; paper MEG **time-series** asterisks are epoch-label permutations **after** filters exist. SPoC `p_values1` in figure titles are **not** described as epoch-label permutation.
5. **Default RDM:** main script default is `'facevstool'` (Fig. 16 geometry), while the first MEG figures (13–15) use category-specific RDMs also coded in the same file (`'face'`, `'tool'`, `'meaning'`).
6. **Plot time axis** `linspace(-536,964,…)` vs SPM `timeOnset=-0.5`.
7. **Dead code:** original `return` at line 281 makes the later `save topo_*` unreachable. OSF topo file was saved by some other run (`filt15` in the filename vs script `highCutOff=20`).
8. **Undocumented plotting helpers:** `prepare4topoNMG` is not in the AIRI repo. Not a SPoC blocker.
9. **BBCI README vs reality:** SPoC is not in `external/` until an extra download of **unpinned** `matlab_SPoC` master.

No AIRI code was “corrected” to match the paper.

---

## 14. Files created on the forensic branch

All under `forensics/airi_matlab/` (plus this repo’s gitignore entries). `src/redisca/` untouched.

See directory layout in `README.md`. Notable:

- `original/AIRI-ReDisCA/` — pinned sources  
- `vendor/pinned_stock_SPoC/` — stock SPoC @ 18e4754  
- `instrumented/` — instrumented AIRI script + `spoc_save_surrogates.m`  
- `scripts/*.m` — environment, literal, audit, seeds, variants, order diagnostic  
- `reports/forensic_report.md` (this file)  
- `provenance.json`, `environment.txt`  
- `paper/published_neuroimage.txt` — extracted published PDF text  
- `results/literal/*.json` — data inspection + static D/z reconstruction  

**Not committed:** `data/MEG_AD_run1.mat` (~1.2 GB) and other OSF binaries; paper PDFs (hashed in provenance).

---

## 15. What prevented literal execution

1. **MATLAB is not installed** in this Cloud Agent environment (hard stop).
2. Therefore Signal Processing Toolbox, `which -all`, and `spoc` were not executed.
3. BBCI `external/` cannot be observed as a **running** MATLAB path; it can only be reconstructed as “would download matlab_SPoC master.zip”.
4. Paper PDF was **not** pre-supplied in the workspace; the published PDF was obtained from megmoscow.ru and hashed.
5. Source localization was **not** attempted (in scope).
6. No undocumented dependency was invented to replace MATLAB.

When MATLAB is available, run the scripts in `README.md` **without** changing preprocessing, pairs, RDM, `trange`, or the SPoC null, then fill `results/literal/` and re-run the audit/seeds/variants.

---

## Paper PDF note

The workspace did not contain the NeuroImage PDF. Published PDF SHA-256  
`018478c993ab34b46b732b2b00a573a9e342101e6ae80e190f97a904967a3208`  
(25 pages). Extracted text: `paper/published_neuroimage.txt`.
