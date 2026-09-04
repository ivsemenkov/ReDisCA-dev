# Track D — N170 empirical–theoretical RDM correlation (Fig. 10 = 0.82?)

Owner: overnight D. Branch: `cursor/paper-n170-rdm-f368`.
Working directory: `/tmp/redisca-worktrees/n170-rdm`.

Question: how could the published Fig. 10 observed–theoretical RDM
correlation of **0.82** have been computed? Current unique+Gram face
window Pearson is **0.99988**; car already matches **>0.99**.

This track tests **only** definitions named by the paper equations,
the Fig. 10/11 description, or executable/reference code. It does not
tune windows, invent Spearman/cosine/RV/channel-subset RSA, or change
the target fill.

Compact JSON: `paper/results/n170/rdm_correlation/summary.json`,
`correlations.json`, `headline_table.json`.

## Source evidence

| Claim | Source |
| --- | --- |
| Eq. 1: \(d_{ij}=\|u_i-u_j\|^2\) | NeuroImage §2.1 |
| Eq. 2: Pearson of **standardized unique** \(i<j\) entries; printed RHS \(\frac{2}{C(C-1)}\sum \tilde d_{ij}\tilde d^m_{ij}\) | §2.2; sample vs population SD unstated |
| Simulation \(\hat D_n=\{w^\top R_{ij} w_n\}\) | §3.2 / Fig. 5 |
| Fig. 10: T=100 ms centered at 200 ms; observed RDM in the **bottom panel**; traces for the **full response**; corr **0.82** | §4.2.1 + Fig. 10 caption |
| Fig. 11: applied at t=170 ms; corr **>0.99**; traces “remarkably similar … over the entire response duration” | §4.2.1 |
| Pair matrix: unscaled Gram, no demean, no \(1/(T-1)\) | Eq. 4–5 |
| Unique unordered pairs | Table 1 / Eq. 6–7 |
| AIRI MEG `Q(c,i)`: instantaneous squared diffs, `corrcoef(rdm_e(:),rdm_t(:))` after `rdm_e=[]; rdm_e(k,l)=...` (grows to \((C-1)\times C\)) | `Redisca_tools_faces_3_random_norm_correct.m` 257–268 |
| Stock SPoC computes **no** RDM correlation | `stock_spoc.md` |
| N170 has **no** AIRI script | D12; `paper_methods.md` |

## Decisions (not silent paper values)

1. Official subject-`"1"` `1_N170_erp_ar.erp`, 28 scalp channels, unique
   pairs. Face window T=100 ms @ 200 ms; car T=100 ms @ 170 ms (control).
2. Paths: library `ReDisCA(demean_time=False)` (paper Gram);
   `source_faithful` unique+`unscaled_gram`; unique+`matlab_cov`.
3. Within=0.1 is a **labeled extra** (AIRI MEG fill). Unique-triangle
   Pearson is affine-equivalent to 0/1 for these two-level RDMs.
4. AIRI instantaneous `Q` is reported **only** at paper-named latencies
   (face 170 / 200 ms; car 170 / 150 ms). No search over time for 0.82.
5. Flattening the whole 4×4 / `triu` including the diagonal is a documented
   **possible misread**, not an endorsed metric.

## Commands

```bash
cd /tmp/redisca-worktrees/n170-rdm
python3 -m pytest paper/reproduction/n170/rdm_correlation/test_rdm_correlation.py -q
python3 paper/reproduction/n170/rdm_correlation/run.py
```

ERP SHA-256 `53e74e931e6f0adaf1e5be4d606d028fcc3e04ee8b066569c2ed2d033d9bbc72`
(same file as `paper/results/n170/fig10_face.json`).

## Table versus paper 0.82 / >0.99

Endorsed RSA score: Pearson of unique \(i<j\) (Eq. 2 `corr`).
Item numbers match the Track D brief.

### Face (Fig. 10) — library unique + unscaled Gram, comp 0

λ1 = **0.88006** (paper fingerprint 0.872). Window n = 26 samples.

| # | Definition | Value | vs 0.82 |
| --- | --- | ---: | ---: |
| 1 | Window traces \(\\\|u_i-u_j\\|^2\), unique Pearson | **0.99988** | +0.180 |
| 2 | Full-epoch traces, same \(w\), unique Pearson | **0.94466** | +0.125 |
| 3 | \(w^\top R_{ij}w\), unscaled Gram (window) | **0.99988** | = (1) |
| 4 | \(w^\top R_{ij}w\), MATLAB-cov (window) | **0.99932** | +0.179 |
| 5 | Pearson after Eq. 2 z-score (sample or population SD) | **0.99988** | = (1); affine-invariant |
| 5′ | **Literal Eq. 2 RHS**, MATLAB sample SD | **0.83323** | +0.013 |
| 5″ | Literal Eq. 2 RHS, population SD | **0.99988** | = (1) |
| 6 | AIRI MEG `corrcoef` of grown \((C-1)\times C\) (window RDM) | **0.99977** | +0.180 |
| 6′ | AIRI notes: square upper-only, zeros below | **0.99977** | +0.180 |
| misread | Flatten whole symmetric 4×4 (`A(:)`) | **0.99980** | +0.180 |
| misread | `triu` including the zero diagonal | **0.99978** | +0.180 |
| extra | Unique Pearson vs within=0.1 target | **0.99988** | identical to 0/1 |
| AIRI analog | Instantaneous unique Pearson @ 171.9 ms | 0.99474 | +0.175 |
| AIRI analog | Instantaneous unique Pearson @ 199.2 ms | 0.97575 | +0.156 |

Identities that hold: (1)=(3) to \(10^{-15}\); (4) equals demeaned-window
Eq. 1 by scale invariance; (5)=(1); 0 vs 0.1 unique Pearson identical.

### Face — source-faithful unique + MATLAB-cov (labeled path), comp 0

λ1 = **0.83915** (matches frozen `demeaned_gram_extra` in `fig10_face.json`).
This is **not** the printed pair matrix.

| Definition | Value | vs 0.82 |
| --- | ---: | ---: |
| Native: \(w^\top R^{\mathrm{cov}}_{ij} w\) unique Pearson (window) | **0.99858** | +0.179 |
| Undemeaned window traces unique Pearson | 0.94819 | +0.128 |
| Full-epoch undemeaned unique Pearson | 0.48907 | −0.331 |
| Eq. 2 sample-SD inner product on undemeaned window | 0.79016 | −0.030 |
| AIRI grown `corrcoef` of window RDM | 0.96481 | +0.145 |
| Instantaneous unique @ 171.9 ms | 0.96350 | +0.144 |
| Instantaneous AIRI grown @ 199.2 ms | **0.82870** | +0.009 |
| Instantaneous `triu`+diag @ 199.2 ms | **0.81586** | −0.004 |

Those last two are the **numerically closest** numbers in this file.
They are **not** a reconstruction: they stack MATLAB-cov (not paper Gram),
instantaneous MEG `Q` (not the windowed bottom-panel RDM), and (for 0.816)
a possible misread of “upper triangular”. The same recipe on the car
matlab_cov filter at 170 ms gives AIRI-grown **0.983**, which is **not**
>0.99. Do not treat 0.83 at one named latency as Fig. 10.

### Car (Fig. 11) — library unique + Gram (control)

| Component | λ | Window unique | Full-epoch unique | Paper |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.88691 | **0.99992** | 0.48594 | >0.99 |
| 1 | 0.79170 | **0.99968** | 0.67960 | >0.99 |

Windowed unique Pearson **does** match >0.99. Full-epoch unique Pearson
**does not**. Eq. 2 sample-SD inner product on the car window is
**0.83327** — the same ~0.83 as the face window — so that reading **cannot**
be how both figures were computed.

Source-faithful Gram matches the library to \(~10^{-15}\) in λ1 and in
every Pearson above.

## What this rules out

1. **A different Pearson indexing of the same windowed 4×4** (AIRI grown
   flatten, full-matrix flatten, `triu` including diagonal). All stay
   ≥0.99977 on the Gram face window.
2. **Eq. 2’s printed inner product with MATLAB sample SD** (0.833). Closest
   paper-formula number to 0.82, but the car control is then also 0.833,
   contradicting >0.99.
3. **Within=0.1 fill.** Unique-triangle Pearson is identical to 0/1.
4. **Canonical-library GEP “overfitting” as a bug.** With C=4 there are
   six unique pairs and a two-level target. ReDisCA maximizes a covariance
   proxy of corr(\(w^\top R_{ij}w, d_{ij}\)) in 28 channels × 26 samples.
   A leading component that isolates one condition from the other three
   is expected to drive that 6-entry correlation to ~1. The library
   matching a 6-entry two-level RDM near-perfectly is the expected
   optimum, not a library bug.
5. **Stock SPoC** does not compute an RDM correlation at all.

## What remains (unresolved)

None of the paper-faithful, windowed, unique-pair Pearson readings of
official ERP CORE subject-1 averages produce 0.82. The Gram window is
0.99988; the Gram full epoch is 0.94466; MATLAB-cov’s native window
metric is 0.99858. The published pair (face 0.82, car >0.99) also has
the **opposite** full-epoch pattern from this file (face 0.945 >
car 0.486).

Classification: **unresolved historical preprocessing / implementation**,
not a canonical-library bug. Likely causes sit outside this track’s
allowed definition list: a different average (ICA / filtering /
channel set / not the official `.erp`), a different displayed empirical
RDM than Eq. 1 on the analysis window, or a figure-panel number that
does not match the Eq. 2 `corr` of the fit that produced the traces.

## Verdict

On official subject-1 `1_N170_erp_ar.erp`, unique pairs, T=100 ms at
200 ms, every correlation the paper or the AIRI/SPoC sources actually
name either (a) stays at **~1** for the face window — including Eq. 1,
\(w^\top R_{ij}w\) Gram, Eq. 2 Pearson after z-score, AIRI
`corrcoef(:)`, and flattening the 4×4 — or (b) is the Eq. 2 printed
inner product **0.833**, which the car control **rules out** because it
would make Fig. 11 ~0.83 rather than >0.99. Full-epoch unique Pearson
(0.945) is closer to 0.82 than the window but still not 0.82, and it
would make the car figure worse, not better. The library GEP matching a
six-entry two-level RDM at 0.99988 is expected. Published Fig. 10
corr=0.82 is therefore **not explained** by a legitimate reading of
Eq. 1–2 / \(\hat D=\{w^\top R_{ij}w\}\) / AIRI RDM `corrcoef` on these
averages.

9 tests in `rdm_correlation/test_rdm_correlation.py` passed, including
regression against the frozen Fig. 10 window Pearson 0.9998814542043535.
