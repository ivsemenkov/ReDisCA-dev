# Published paper methods (NeuroImage 301, 120868)

Authority: extracted-text file
`.reproduction_data/paper_text/published_neuroimage.txt`
(SHA-256 `2b208a183b7f4cafe959464ded67b8105a094b68e830a04ed7e577e17ffa08a4`).
Do not commit the full text. The September 2024 NeuroImage article is
authoritative. The bioRxiv preprint
(doi `10.1101/2024.02.01.578343`, extracted SHA-256
`92aa3a629bfe4a95748c20dc04e0b71637f9132521f24b5fc8feec870ed4e1ea`)
is used only where the published PDF extraction drops in-figure axis
labels (notably Fig. 4/5 SNR overlays).

Citation: Ossadtchi, A., Semenkov, I., Zhuravleva, A., Kozunov, V.,
Serikov, O., & Voloshina, E. (2024). Representational dissimilarity
component analysis (ReDisCA). *NeuroImage*, 301, 120868.
DOI: https://doi.org/10.1016/j.neuroimage.2024.120868

Data-availability sentence in the paper: code “upon request” at
`ossadtchi@gmail.com`; “the data used in the MS” at
https://osf.io/pfde9/. That OSF node is the ERP CORE N170 component
(parent `thsqg`), not the AIRI MEG/source-model dump. MEG and
Brainstorm-style source-model files are on https://osf.io/8rk67/.

## 1. Problem statement (Section 2.1)

- Data: condition-wise matrices \(X^c_i\) of shape \(N \times T\)
  (channels \(\times\) time), trials \(i = 1,\ldots,I_c\), conditions
  \(c = 1,\ldots,C\).
- Evoked response: \(X^c = \frac{1}{I_c}\sum_i X^c_i\).
- Target: a user-supplied \(C \times C\) theoretical RDM \(D = \{d^{ij}\}\).
- Dissimilarity used throughout: squared Euclidean distance between
  condition-specific time series (Eq. 1). Mahalanobis is mentioned as
  a possible extension, not used.

## 2. Source-space RSA baselines (Section 2.2, Fig. 1)

Four combinations, all spotlight scans over \(M\) cortical vertices:

| Label in figures | Inverse solver | When the inverse is applied |
| --- | --- | --- |
| MNE AV RSA | minimum-norm | on condition averages, then \(d_m^{ij}=\|s_m^i-s_m^j\|^2\) |
| MNE S.T. RSA | minimum-norm | per trial, then average the squared distances |
| BF AV RSA | LCMV beamformer | on condition averages |
| BF S.T. RSA | LCMV beamformer | per trial, then average distances |

RSA score (Eq. 2): Pearson correlation of **standardized upper-triangular**
RDM entries. Standardization: subtract mean, divide by standard
deviation (sample vs population not stated). Multiple-comparison
control for maps: cluster-based permutation of condition labels
(Maris & Oostenveld 2007) — this is described for source-space RSA,
not for ReDisCA.

These four baselines appear only in the simulations (Figs. 4–6). They
are not applied to the real EEG/MEG examples.

## 3. ReDisCA core (Section 2.3, Fig. 2, Table 1)

### 3.1 Pair matrix (printed)

Eq. 4–5 define

\[
R_{ij} = (X^{c_i}-X^{c_j})(X^{c_i}-X^{c_j})^\top
\]

and call this the “unscaled correlation matrix of the sensor-space
time series differences”. There is **no temporal demeaning** and **no**
\(1/(T-1)\) in the printed formula.

Pair index set is internally inconsistent in the paper:

- Prose and Table 1: upper-triangular unique pairs; \(e=(i-1)C+j\)
  with \(i>j\).
- Eq. 5 writes \(i=1,\ldots,C\), \(j=1,\ldots,C\) with no \(i\neq j\)
  guard (diagonal \(R_{ii}=0\)).

Reproduction must treat **unique unordered pairs \(i<j\)** as the
paper-faithful pair set, and treat AIRI \(i\neq j\) directed duplicates
as a separate executable variant.

### 3.2 Aggregation (Eq. 6–7)

- \(\bar R = \frac{2}{C(C-1)}\sum_{i=1}^{C}\sum_{j=i+1}^{C} R_{ij}\)
  (mean of unique pairs).
- \(\bar R_d = \sum_{i=1}^{C}\sum_{j=i+1}^{C} \tilde d^{ij}(R_{ij}-\bar R)\)
  (printed as a **sum**). The surrounding sentence calls this the
  “weighted and centered **average**”. Name and formula disagree.
  A global scale of \(\bar R_d\) changes eigenvalues, not filter
  *directions*, if \(\bar R\) is unchanged.

Target entries \(\tilde d^{ij}\) are standardized. The paper does not
say sample vs population SD.

### 3.3 Estimator

Covariance-maximization SPoC (Dähne et al. 2014): generalized
eigenproblem \(\bar R_d w = \lambda \bar R w\) (Eq. 9–10), largest
algebraic \(\lambda\) first. Constraint \(w^\top \bar R w = 1\).

If \((\bar R_d,\bar R)\) is rank-deficient, “perform the procedure in
the lower dimensional principal space and transform topographies
back”. No numerical tolerance is printed.

### 3.4 Patterns

Paper: if \(W\) is square and full rank, \(A = W^{-1}\), rows of \(A\)
are topographies. This is **not** the Haufe formula and does not apply
when the GEP is solved in a rank-\(K<N\) principal space (as MEG
data require).

Eq. 13 cosine-similarity scan has a typesetting slip: denominator
\(\|g_m\|\|a_1\|\) instead of \(\|a_k\|\). Eq. 14 MUSIC uses subspace
correlation between a free-orientation dipole topography pair and the
\(K\)-dimensional significant-component subspace \(A_K\).

### 3.5 Component inference (paper)

“Permutation testing procedure suggested in SPoC”: surrogate GEPs
from **data with permuted condition labels**, which “destroys the
mutual correspondence between the set of difference correlation
matrices \(R_{ij}\) and the condition pair labels \((i,j)\)”.
Asymptotic \(p\) = fraction of surrogates whose generalized
eigenvalue exceeds the original. The paper does not specify:

- number of permutations \(B\),
- whether the null statistic is \(\max_n|\lambda_n|\) (SPoC) or
  matched-component \(\lambda\),
- \((count+1)/(B+1)\) vs \(count/B\) (the latter can be 0).

This is **not** the same procedure as stock SPoC’s live code
(random-phase surrogates of \(z\); see `stock_spoc.md`).

Filtered time series: \(u^{c\top}_k = w_k^\top X^c\) (Eq. 11).

## 4. Simulations (Sections 2.4, 3, 4.1; Figs. 3–6)

Shared generative recipe (2.4.1):

1. Mixing matrix \(M\) with i.i.d. \(\mathcal N(0,1)\) entries.
2. \(Z \in \mathbb R^{C\times T}\) Gaussian rows, 6th-order Butterworth
   low-pass at **2 Hz**.
3. \(S = MZ\). Ideal RDM \(D_0\) from Eq. 1 on the rows of \(S\).
4. Approximate theoretical RDM \(D = D_0 + \Upsilon_d\) (noise matrix
   unspecified in distribution/scale).
5. Window length \(T = 200\) **ms** (sampling rate not stated; if 1 kHz
   this is 200 samples).
6. Forward-model error: \(\delta \sim \mathcal N(0, \sigma_\delta^2 I)\),
   \(\sigma_\delta = 0.15\|g_{m_0}\|\).
7. Brain noise: 1000 randomly seeded cortical 1/\(f\) sources as in
   Ossadtchi et al. 2018; SNR is the ratio of **root mean powers** of
   the noiseless sensor matrix to the noise matrix (Eq. 16 discussion).
8. Monte Carlo: **100** iterations for both simulation sets.

**Not stated in the paper (blocked for exact numeric reproduction):**

- Which cortical mesh / forward model (individual MRI vs template).
  The only public mesh in the AIRI dump is subject AD
  `tess_cortex_pial_low` (5002 vertices) with overlapping-spheres
  `Gain` — a working hypothesis, not a paper statement.
- Number of sensors \(N\) and which channel types.
- Number of EEG/MEG **trials per condition** \(I_c\). BioRxiv Fig. 4
  overlay text says “100 trials”; that string is ambiguous
  (MC iterations vs \(I_c\)).
- Law of \(\Upsilon_d\).
- Sampling rate; Butterworth implemented as `butter`/`filtfilt` vs
  forward-only.
- Random seeds.

### 4.1 Single-source detection (Fig. 4)

- \(C = 5\) conditions, one source, new \(Z\) and new mesh vertex
  \(m_0\) each MC trial; mixing \(M\) **held fixed** across MC.
- Each trial: \(X^{c,l} = (g_{m_0}+\delta) s^{c\top} + \gamma \Upsilon_x^l\)
  (Eq. 15). New \(\delta\) per MC; new noise realization per trial.
- Metric: ROC via vertex-wise scan \(\rho_m\), sphere radius
  \(r_{\max}=0.01\) m around true location (Eq. 17–18). ReDisCA scan
  uses cosine similarity (Eq. 13) and/or MUSIC (Eq. 14); the ROC
  methods list in the caption is “ReDisCA and four source space RSA
  versions”.
- SNRs: panels (c,d) are **SNR = 0.1**. The higher-SNR pair (a,b) is
  not numbered in the published body. BioRxiv in-figure overlay:
  “single trial SNR 0.2” (a,b) and “0.1” (c,d). Use 0.2/0.1 as the
  figure-label reconstruction, marked preprint-supported.
- Qualitative claim: ReDisCA dominates all four RSA versions; ~85%
  true-positive at near-zero false-alarm at SNR 0.1; source-space RSA
  paradoxically improves at lower SNR in this single-source setting.

### 4.2 Four-source realistic (Figs. 3, 5)

- \(P=4\) sources, minimum separation \(\delta_{\min}=2\) cm.
- \(C=6\) in Section 2.4.3 and in the Fig. 5 **caption**; the Results
  paragraph says the Fig. 5 histograms are for **\(C=5\)**. This is an
  unresolved paper-internal contradiction. Fig. 6 separately varies
  \(C\in\{3,4,5,6\}\).
- Mixing matrices \(M_p\) **fixed** across MC; new locations each MC.
- Observation (Eq. 16): superposition of four \((g_p^{\mathrm{true}}+\delta_p)s_p^{c\top}\)
  plus 1/\(f\) noise.
- Metrics (Section 3.2): argmax-localization error \(\|\hat r-r^{\mathrm{true}}\|\);
  corr(\(a_1, g^{\mathrm{true}}\)); corr(\(w_1, g^{\mathrm{true}}\));
  corr(upper triangle of \(D_p\) vs \(\hat D_p = \{w_p^\top R_{ij} w_p\}\)).
- Fig. 5 caption vs body **disagree on panel identity** (see
  `discrepancies.md`). BioRxiv overlays: SNR 0.4 (top) and 0.2 (bottom).
  Body text: panels (c,d) are SNR 0.2.
- Qualitative: ReDisCA has the largest mass of errors < 1 cm; patterns
  align with true topographies better than weights; empirical RDMs
  well correlated with targets.

### 4.3 Error vs \(C\) (Fig. 6)

Average **median** localization error of four simultaneous sources vs
\(C = 3,4,5,6\) for ReDisCA (dipole fitted to \(a_{p1}\)), BF S.T.,
MNE S.T. Claim: ReDisCA best at all \(C\); mean median error
**< 2 cm at \(C=6\)**. SNR for this figure is not stated.

## 5. N170 EEG (Section 4.2 / 4.2.1; Figs. 7–11)

Dataset: ERP CORE N170 (Kappenman et al., 2021), paper URL
https://osf.io/pfde9/, ethics/data DOI `10.18115/D5JW4R`.
Task: face / car / scrambled face / scrambled car; object vs texture
judgment. Stimuli from Rossion & Caharel (2011).

Paper preprocessing sentence: data “were preprocessed and **three ICA
components corresponding to ocular and cardiac artifacts** were
removed”; ERPs averaged within stimulus types; 40 subjects in the
dataset; analysis uses **first participant, index `"1"`**.

That ICA sentence is a coarse gloss of ERP CORE, not a literal
description of subject `"1"` (verified against
`ICA_Components_N170.xlsx`: subject 1 removes components **2 and 7**
only; counts across 40 subjects are mostly 1–2 ocular components;
ERP CORE scripts label them ocular, not cardiac). See
`provenance/datasets.md`.

Time windows (paper):

| Analysis | Window | Result |
| --- | --- | --- |
| Meaningful vs meaningless | sliding \(T=150\) ms; step **not stated** | Fig. 7 p-map; first component uncorrected \(p<0.05\) continuous segment around **\(t=400\) ms**; occipital pattern; three adjacent windows in Fig. 8 |
| Face-specific | \(T=100\) ms **centered at 200 ms** (i.e. 150–250 ms if symmetric) | Fig. 10; one significant component; right-fusiform-like topography; face burst ~170 ms; empirical–theoretical RDM corr **0.82** |
| Car-specific | applied **at \(t=170\) ms** (duration not restated; 100 ms is the only other real-data \(T\) given besides 150 ms) | Fig. 11; **two** components with \(p<0.01\); RDM corr **> 0.99** |

Theoretical RDMs are **figures only** (Figs. 7a, 9a, 9b). No numeric
matrix is printed. From the prose:

- Fig. 7a: contrast meaningful (face, car) vs meaningless (scrambled).
- Fig. 9a: one condition (face) distinct, the other three similar.
- Fig. 9b: same for cars.

Inference: per-window component \(p\)-values (method/B not stated);
Fig. 7 uses uncorrected \(p<0.05\) on component 1.

No AIRI MATLAB exists for N170.

## 6. MEG (Section 4.2.2; Figs. 12–18)

Dataset: Kozunov et al. (2018), first run, first subject **AD**.
Elekta Neuromag 306. Paper: 6 subcategories (face 1/2, tool 1/2,
nons. 1/2) by splitting each category into two equal parts; **80
epochs each**, 480 total; 6 ERF matrices **204 × 1500** (204 planar
gradiometers; 500 ms pre-stimulus + 1000 ms post). ReDisCA applied to
the **entire 1500 ms** at once. First three statistically significant
components shown.

Theoretical RDMs: Fig. 12a face, 12b tool, 12c meaningful vs
meaningless (numeric entries not printed). Fig. 16a non-binary
geometry: within-category similar; distance(face, tool) larger than
distance(each meaningful, nonsense); 1-D components then place face
and tool on opposite sides of nonsense.

Component \(p\)-values: titles of Figs. 13–15, 17 (procedure: paper
permutation of Section 2.3 / “suggested in SPoC”).

Time-series inference (distinct from component \(p\)): permute
**subcategory labels of individual epochs**, recompute surrogate
averages, apply the **fixed** spatial filters, FWER via **maximum
statistics over the entire time interval**. Red/blue asterisks above
and below traces. Reported onsets (paper):

| Figure | Contrast | Reported onsets / peaks |
| --- | --- | --- |
| 13a | face, comp 1 | differential from **65 ms**; peak **160 ms**; rises again from **311 ms**; central/right occipital |
| 13b | face, comp 2 | parietal; first significance **218 ms** |
| 13c | face, comp 3 | late sustained from **273 ms**; bilateral occipital / some frontal |
| 14a | tool, comp 1 | **210 ms**; central occipito-parietal, left (and compact right) central sulcus |
| 14b | tool, comp 2 | later; mid-occipital and left parietal |
| 14c | tool, comp 3 | from ~**240 ms**; right and left sensory-motor / temporal, right-prevalent |
| 15a–b | meaning, comps 1–2 | ventral then dorsal; meaning vs nonsense from **160 ms** occipito-parietal |
| 15c | meaning, comp 3 | early **182 ms** and late **675 ms**; focal mid-parietal |
| 17a | non-binary | tools vs faces from **202 ms**; left parietal / sensory-motor |
| 17b | non-binary | visual / FG-like late ~260–350 ms for tools |
| 17c | non-binary | face-specific peak ~**160 ms** |

Fig. 18: MUSIC scan (Eq. 14) of the Fig. 17 subspace using the
**individual MRI forward model described in Kozunov et al. 2018**.
Claimed regions: right fusiform, right insula, left IPS, anterior
central gyrus.

Paper MEG preprocessing: none beyond Kozunov and averaging. In
particular the paper does **not** state a 0.25–20 Hz Butterworth
or a cropped `600:1500` window.

## 7. What is not a paper result

- AIRI default RDM `facevstool` and AIRI `trange = 600:1500`.
- AIRI timecourse Monte Carlo (`Nmc=100`, half-split of pooled trials).
- AIRI source-loc default (`precomp` sLORETA of `A1(:,4)`).
- fMRI / ANN applications in the Discussion (future work).
- Cross-validated Euclidean distance (Discussion; future).
