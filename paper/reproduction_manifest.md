# Stage A reproduction manifest (frozen before full-result selection)

This file is the human-readable twin of `reproduction_manifest.json`.
It was written **before** using full experiment results to pick a
reproducing candidate.

## Method (not a branch)

Every ReDisCA fit is `ReDisCA(**AIRI_SPOC_KWARGS).fit(X, rdm)` with

| key | value |
| --- | --- |
| n_components | None |
| demean_time | True |
| divide_by_t_minus_1 | True |
| directed_pairs | True |
| aggregation | mean |
| solver | whitening |
| rank | None |
| rank_tol | 1e-6 |

Library pin: `f657b954da7d48d05b50f6f4dc967595a155f7ae`.

## Seeds

`20240904 20240905 20240906 20240907 20240908`

## External candidates

See the JSON file for exact values and source rationales.

- Simulations: `SIM-P1` (primary MEG-like reconstruction), `SIM-P2` (I_c=80), `SIM-P3` (causal Butterworth).
- N170: official unfiltered ERP and documented 20 Hz low-pass ERP; car centers 170 and 200 ms; sliding steps 25 ms and 1 sample.
- MEG: AIRI executable window/filter; paper full 1501-sample epoch; paper 1500-sample crop. AIRI numeric and binary RDMs where both remain source-supported.
- Source localization: paper MUSIC; AIRI precomp sLORETA; AIRI music with `P=eye(Nsns)`.

These are external analysis branches, not method ablations.
