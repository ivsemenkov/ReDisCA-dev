# Simulations track (Ossadtchi et al. 2024, Figs 3–6)

Owner: this directory and `paper/results/simulations/`.
Do not edit `paper/reproduction/common/` or `src/redisca`.

There is **no AIRI simulation script**. This track reconstructs Section 2.4
from the published paper plus documented assumptions. Results that use the
public AD overlapping-spheres Gain are **`approximate`** and **blocked for
exact published numbers** (D13). fsaverage is never used.

## What is reproduced

| ID | Figure | Output |
| --- | --- | --- |
| `fig03-sim-four-source-rdms` | Fig. 3 | Exemplar 4×(6×6) theoretical RDMs |
| `fig04-single-source-roc` | Fig. 4a,c | ROC, 100 MC, C=5, SNR 0.2 (preprint overlay) and 0.1 (published) |
| `fig04-single-source-traces` | Fig. 4b,d | Exemplar noisy/clean channel-mean ERPs + RDM |
| `fig05-four-source-mc` | Fig. 5 | Errors and ReDisCA correlations; **C=5 and C=6** (D14); SNR 0.4 / 0.2 |
| `fig06-error-vs-C` | Fig. 6 | Mean median error vs C=3,4,5,6 at assumed SNR 0.2 |

Canonical fit: `from redisca import ReDisCA` with unique pairs and
`demean_time=False` (printed Gram). `demean_time=True` is a labeled extra on
Fig. 4.

RSA baselines (Fig. 1): MNE AV, MNE S.T., BF AV, BF S.T. Fig. 5/6 use the two
single-trial methods.

## Commands

From the repository root (after OSF source models are in `.reproduction_data/source_models/`):

```bash
python paper/reproduction/common/download_osf.py source-models
python -m pytest paper/reproduction/simulations/tests -q
python paper/reproduction/simulations/run.py --quick
python paper/reproduction/simulations/run.py
```

`--quick` uses 3 MC, I_c=8, 80 noise sources (smoke only).

Optional RSA reduction (still approximate):

```bash
python paper/reproduction/simulations/run.py --rsa-n-mc 20
```

ReDisCA always uses `--n-mc` (default 100). `--skip-rsa` writes ReDisCA-only
metrics.

## Forward model

Public candidate only: OSF `8rk67` subject-AD `tess_cortex_pial_low` (5002
vertices) + `headmodel_surf_os_meg` Gain `(322, 15006)` overlapping spheres.
Simulations use 204 planar rows and constrained `GridOrient`. This is **not**
a paper statement. See `TRACK_REPORT.md`.

## Outputs

Compact JSON under `paper/results/simulations/`. Every file repeats the
assumed-value table and `status: approximate`. After a full 100-MC run,
numeric tables live in `TRACK_REPORT.md`.
