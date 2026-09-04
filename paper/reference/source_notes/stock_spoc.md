# Stock SPoC (pinned clone)

Clone: https://github.com/svendaehne/matlab_SPoC
Commit (verified HEAD): `18e4754aec1411160fd5b7ef0db852f1e0a87d90`
Date: 2016-04-04. Message: `replaced zscore with my_zscore`.

This is the code AIRI actually calls (`spoc(Xspoc, z, ...)`), once BBCI
has downloaded `master.zip`. File SHA-256s verified on the local clone:

| Path | SHA-256 |
| --- | --- |
| `SPoC/spoc.m` | `0979006739b43b9d74e3a9321f3fc232374930b69fc69045a02d6e49173eadd2` |
| `utils/create_Cxxz.m` | `7d8cd5e964da92811c27a36405dc1b74e1ea43bd3e317430c25343f36a80c711` |
| `utils/whiten_data.m` | `6c12f2cbe4ba2bb46f40d25a9d18d5ea3bfe0667f85d5de855e561bfc28defce` |
| `utils/random_phase_surrogate.m` | `0229289ee0bb0a87d9f5cce31360e0b846f8a7acc08212ab58f8fd4ddc69431f` |
| `utils/my_zscore.m` | `2db31d939139dcb64259423b1355e1603f2b83590f38c07c65da468c8f7925cb` |
| `utils/propertylist2struct.m` | `fc70782003cc27d0f7439f08b1237c5ebb3e0d26955fa98d9a50e280644ec686` |
| `utils/set_defaults.m` | `0416538a6aa865d298fa3feb6703edb8cabef8d04eec277d89057e7b54a26f2d` |
| `utils/get_var_features.m` | `1830daff9b6e926c4e38332b8b159b10f51e9960ee087552e08588fe74a578c0` |

`my_zscore.m` is **not** used inside `spoc.m`. Live standardization is
the in-line MATLAB expression below.

## Inputs as AIRI uses them

```
X : (n_samples_per_epoch, n_channels, n_epochs) = (901, 204, 30)
z : length n_epochs = 30  (RDM entries, directed pairs)
```

## Deterministic path (`spoc.m`)

1. **Target standardization** (line 98):
   `z = (z-mean(z(:)))./std(z(:));`
   MATLAB `std` default `w=0` is sample SD, divisor \(N-1\). Same
   convention as NumPy `ddof=1` and as the Python library.

2. **Epoch covariance** (lines 101–108): for each epoch,
   `C_tmp = cov(X_e)` with `X_e` of shape `(T, n_channels)`.
   MATLAB `cov` temporally demeans each channel and divides by
   \(T-1\). Then `Cxx = mean_e C_tmp`.

3. **Mean-free epoch stack** (lines 112–120):
   `Cxxe(:,:,e) = C_tmp - Cxx`.

4. **Weighted average** `create_Cxxz.m`:
   `Cxxz = reshape(Cxxe_vec * z', [Nx,Nx]) / Ne`
   i.e. \(\frac{1}{N_e}\sum_e z_e Cxxe_e\). This is a **mean**, not
   the paper Eq. 7 sum. Global scale of `Cxxz` scales \(\lambda\),
   not the filter rays, if `Cxx` is held fixed.

5. **Whitening** `whiten_data.m` with precomputed `Cxx`:
   - `eig(C)`, sort descending.
   - rank: `tol = ev_sorted(1) * 10^-6`, `r = sum(ev > tol)`.
   - optional PCA variance cutoff `pca_X_var_explained` (AIRI leaves
     default 1, so only numerical rank).
   - `M = diag(D.^ -0.5) * V'` truncated to `n_components` rows.

6. Ordinary eig in whitened space: `eig(M * Cxxz * M')`, sort
   `lambda` descending, `W = M' * W_white`.

7. **Filter normalization**: `W(:,k) /= sqrt(W(:,k)' * Cxx * W(:,k))`.

8. **Patterns (Haufe)**: `A = Cxx * W / (W' * Cxx * W)`.
   Valid for rectangular `W`. This is **not** paper \(A=W^{-1}\).

AIRI MEG `A1` is 204 × 67 because whitening truncated to numerical
rank 67. Paper’s invert-`W` construction is inapplicable here.

## Inference path (AIRI sets `n_bootstrapping_iterations=1000`)

Live code (not the commented line):

```matlab
% z_shuffled = z(randperm(length(z)));          % COMMENTED OUT
[z_shuffled, z_amps] = random_phase_surrogate(z, 'z_amps', z_amps);
Cxxz_s = create_Cxxz(Cxxe, z_shuffled);
% recompute eig in the *same* whitened space M
lambda_samples(k) = max(abs(lambda_values_s));
p_values_lambda(n) = sum(abs(lambda_samples)>=abs(lambda_values(n))) / n_bootstrapping_iterations;
```

`random_phase_surrogate.m`: FFT amplitude spectrum of `z`, random
phases on the positive-frequency bins with conjugate symmetry,
`real(ifft)`. Nyquist bin is given a free phase when \(n\) is even;
after `real(ifft)` that bin’s amplitude is not preserved (the
source-faithful tests document this).

Null statistic is **\(\max_n |\lambda_n|\)** (family-wise).
\(p = count/B\), so **\(p=0\) is possible**. No \(+1\) continuity
correction.

`get_var_features` / `r_samples` are computed inside the bootstrap
loop but **never used** for the returned `p_values_lambda`.

This is **not** the paper’s “permute condition labels / reshuffle the
upper triangle of the theoretical RDM”. Random-phase surrogates
preserve the autocorrelation of the length-30 `z` sequence. That
sequence’s order is AIRI’s directed double loop `(1,2),(1,3),…,(6,5)`,
which has no temporal meaning; preserving its “spectrum” is an
accidental property of stuffing an RDM vector into SPoC’s `z(e)`.

## What SPoC does not do

- It does not know about RDMs, triangular vs directed pairs, or
  EEG/MEG time windows. Those are entirely AIRI’s job when building
  `X` and `z`.
- It does not implement paper Eq. 11–14, MUSIC, or MEG timecourse
  FWER.
- Default bootstrap count is 0; AIRI must pass 1000.

## Correspondence with paper Table 1 (as implemented)

| SPoC | Paper ReDisCA | AIRI executable |
| --- | --- | --- |
| epoch index \(e\) | unique pair \((i,j)\), \(i>j\) | directed \(i\neq j\) (30, not 15) |
| \(C(e)=\mathrm{cov}(X_e)\) | unscaled Gram \(R_{ij}\) | MATLAB `cov` of \((X_i-X_j)^\top\) |
| \(z(e)\) | \(\tilde d^{ij}\) | raw \(D_{ij}\) then SPoC z-score |
| \(\frac{1}{E}\sum C(e)\) | \(\bar R\) mean of unique pairs | mean of 30 directed covs (= mean of 15 unique covs for symmetric \(D\)) |
| `Cxxz` mean of \(z_e(C_e-\bar C)\) | Eq. 7 **sum** of \(\tilde d(R-\bar R)\) | SPoC mean |
| random-phase \(z\) | permute condition labels | random-phase (AIRI) |
