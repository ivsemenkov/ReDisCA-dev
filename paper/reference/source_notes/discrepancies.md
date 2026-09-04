# Verified paper vs AIRI vs SPoC vs library discrepancies

Each item was re-checked against the published NeuroImage text, the
pinned AIRI commit `15bc19c`, stock SPoC `18e4754`, and library
`5a5c865`. The forensics branch
`cursor/forensics-airi-matlab-0370` was used only as a donor of
hashes/URLs/file lists; numerical claims below were re-verified on
the local clones and OSF files.

## D1. Pair set: triangular vs directed

- **Paper:** unique upper triangle (prose, Table 1, Eq. 6–7). Eq. 5
  is sloppy (`i,j = 1…C` without \(i>j\)).
- **AIRI:** double loop, skip `i==j` only → 30 directed pairs for
  \(C=6\) (15 unique).
- **SPoC:** agnostic; sees 30 epochs.
- **Library:** unique \(i<j\) only.

For a symmetric \(D\), `Cxx` is unchanged by duplication. Sample-SD
z-scoring uses \(N-1\), so `z` and therefore \(\lambda\) **scale**.
Filter rays stay in the same 1-D subspaces (source-faithful unit
test). Report eigenvalues/p-values only with the pair mode named.

## D2. Pair matrix: unscaled Gram vs MATLAB cov

- **Paper Eq. 4:** \(R_{ij}=(X_i-X_j)(X_i-X_j)^\top\) — no demean,
  no \(1/(T-1)\).
- **AIRI→SPoC:** `cov(Xi'-Xj')` — demean each channel, divide by
  \(T-1\).
- **Library:** `demean_time=True` (default) demeans, omits \(T-1\);
  `False` is the printed Gram.

\(1/(T-1)\) is a global scale and cancels in the GEP if applied to
every pair. **Demeaning does not cancel** and is a real estimator
difference. N170 \(T=100\)–150 ms and MEG \(T=901\) vs 1501 are
large enough that centering can move \(\lambda\) and patterns.

## D3. Target standardization (not a discrepancy)

MATLAB `std` / SPoC / library `ddof=1` all use sample SD. Paper
“standardized” is unspecified. **Do not treat this as a bug.** The
pair-duplication effect (D1) still changes the SD because \(N\)
changes.

## D4. Aggregation: paper sum vs SPoC/library mean

- **Paper Eq. 7:** \(\bar R_d=\sum_{i<j}\tilde d^{ij}(R_{ij}-\bar R)\)
  (sum). Prose says “average”.
- **SPoC `create_Cxxz`:** divide by \(N_e\) (mean).
- **Library:** `np.mean`.

Filter directions invariant to this global scale; \(\lambda\) and any
p-value that uses \(\lambda\) as the statistic are not. Prefer to
quote **patterns/filters** when comparing paths, and quote \(\lambda\)
only within one aggregation convention.

## D5. Inference: condition-label permutation vs random-phase

- **Paper §2.3:** permute **condition labels**; surrogate GEPs;
  \(p=\) fraction of surrogate eigenvalues exceeding the original.
  \(B\) unspecified.
- **SPoC live code:** `random_phase_surrogate(z)`; null
  \(\max|\lambda|\); \(p=count/B\) (0 possible). `randperm(z)` is
  commented out.
- **AIRI:** calls stock SPoC with `n_bootstrapping_iterations=1000`
  → random-phase, not paper permutation.
- **Library:** no inference.

A second MEG time-series test is **also** discrepant (D5b):

- **Paper:** permute subcategory labels of epochs, surrogate
  averages, apply fixed filters, **FWER max-stat over time**.
- **AIRI:** `Nmc=100`, pool class1∪class2, **random half-split**,
  channel-std normalized data, pointwise `pplus`/`pminus`, no FWER.

## D6. MEG analysis window

- **Paper:** entire 1500 ms, 204 × 1500, −500…+1000 ms.
- **Data file:** 207 × **1501** × 880; first 204 planars; true time
  −500…+1000 ms at 1 kHz.
- **AIRI:** `trange=600:1500` → 901 samples, **99…999 ms**.
- **AIRI plot axis:** `linspace(-536,964,N)` does not match SPM time.

Paper-faithful MEG fits must use the full epoch (or a documented
truncation). AIRI-executable fits must use 99–999 ms.

## D7. AIRI default RDM vs paper figure order

- **AIRI default:** `ThRDMArr(2) = 'facevstool'` with 0.1 / 0.5 / 1
  geometry. That is the **Fig. 16 non-binary** relation (face vs tool
  far, both midway from nonsense), **not** Fig. 12a/b binary
  category detectors.
- **Paper MEG story order:** Fig. 12a face, 12b tool, 12c meaning,
  then Fig. 16 non-binary.

Within-category fill is **0.1** in AIRI, not 0. Paper figures are
images; exact 0 vs 0.1 cannot be read from the extracted text.
Tracks must emit both the AIRI numeric matrices and a 0/1 binary
variant when comparing to figures.

## D8. MEG bandpass (AIRI only)

AIRI `butter(3,[0.25,20]/500); filtfilt` per trial. Paper MEG
section states no such filter. Kozunov 2018 recorded 0.1–330 Hz
with tSSS. Paper-faithful vs AIRI-executable MEG must be labeled
separately.

## D9. Patterns: \(W^{-1}\) vs Haufe

Paper assumes full-rank square \(W\) and \(A=W^{-1}\). SPoC and the
library use Haufe \(A=Cxx W (W^\top Cxx W)^{-1}\). AIRI MEG rank is
67 < 204, so invert-\(W\) is undefined. Use Haufe for all executable
paths; treat invert-\(W\) as a paper special case that does not apply
to these datasets.

## D10. Whitening / rank

SPoC and the library both use relative tolerance \(10^{-6}\) on the
leading eigenvalue of \(\bar R\)/`Cxx`. MATLAB `eig` vs
`scipy.linalg.eigh` still differ. Not a methods discrepancy;
stochastic-numeric only.

## D11. N170 ICA gloss vs ERP CORE subject `"1"`

Paper: “three ICA components corresponding to ocular and cardiac
artifacts”. ERP CORE `ICA_Components_N170.xlsx` (OSF pfde9, SHA-256
`23373a2b7aae80e7b01abfdc523fb1d04fbc6f41fc48c090f7e840534224cf85`):
subject `"1"` removes **components 2 and 7 only**. Across 40
subjects the modal count is 1, not 3; scripts say **ocular**, not
cardiac. Use the official ERP CORE list, not the paper’s “3”.

## D12. N170 has no AIRI code

AIRI repo cannot reproduce Figs. 7–11. Those figures are
paper-described + ERP CORE pipeline only.

## D13. Simulations have no AIRI code and no named forward model

Figs. 3–6 cannot be bit-reproduced from AIRI. The public AD
`tess_cortex_pial_low` (5002 vertices) + overlapping-spheres `Gain`
is the only candidate forward model in the data dump. Using it is a
**documented hypothesis**, not a paper statement. Do not substitute
fsaverage and call it reproduction.

## D14. Fig. 5 caption vs body (paper-internal)

Caption: \(C=6\); panels (a–e) described as ReDisCA-specific
correlations then localization errors. Body: \(C=5\); (a) three-method
error histograms; (b) stacked ReDisCA metrics; (c,d) SNR 0.2.
BioRxiv overlays: SNR 0.4 and 0.2. Workers must record which
interpretation they plotted against.

## D15. Source localization: paper MUSIC vs AIRI sLORETA of component 4

Paper Fig. 18: MUSIC (Eq. 14) of the Fig. 17 **subspace**, individual
MRI forward model, free orientation. AIRI default: constrained
sLORETA kernel × `A1(:,4)` of the **facevstool** run, plus a
questionable MAG/GRAD index vector (`1:3:304`, `2:3:305`). An AIRI
`'music'` branch exists but is not the default and still uses a
single topography.

## D16. Sample count 1500 vs 1501

Paper prints 204 × 1500. OSF `MEG_AD_run1.mat` is 204 planars ×
**1501** samples (and 3 extra non-MEG channels). Use 1501 and
document the off-by-one; do not silently drop the last sample unless
matching AIRI `trange` (which already drops sample 1501).

## D17. AIRI `return` before `save`

The main script `return`s before `save topo_*`. OSF
`topo_face_vs_tool_correct_filt15.mat` is an author-saved `A1`, not
an output of running the committed script as-is. Source-loc that
loads this file is not a substitute for a local SPoC run. The
filename `filt15` is not explained by the committed 0.25–20 Hz
filter.

## D18. Dead RDM name `toolvsface`

Listed in `ThRDMArr` but not implemented in the `D` switch. Selecting
it yields the zero matrix.

## Starting hypotheses — verdicts

| # | Hypothesis | Verdict |
| --- | --- | --- |
| 1 | Pairs: paper triangular vs AIRI \(i\neq j\) | **Confirmed** (D1) |
| 2 | Pair matrices: paper Gram vs MATLAB cov | **Confirmed** (D2); \(1/(T-1)\) cancels, demeaning does not |
| 3 | Target std: MATLAB sample SD / library `ddof=1` | **Same convention** (D3), not a bug |
| 4 | Aggregation: paper sum vs SPoC/library mean | **Confirmed** (D4); scale-only for filters |
| 5 | Inference: paper label permutation vs SPoC random-phase, \(\max|\lambda|\), \(p=count/B\) | **Confirmed** (D5); AIRI uses stock SPoC bootstrap |
| 6 | MEG window: paper 1500 ms vs AIRI `600:1500` | **Confirmed** (D6); true cropped window 99–999 ms |
| 7 | AIRI default RDM `facevstool = ThRDMArr(2)` | **Confirmed** (D7); this is Fig. 16-like, not Fig. 12a |
