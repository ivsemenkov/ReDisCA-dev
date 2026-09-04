# Ossadtchi et al. 2024 — paper-branch reproduction

Ossadtchi, A., Semenkov, I., Zhuravleva, A., Kozunov, V., Serikov, O.,
& Voloshina, E. (2024). Representational dissimilarity component analysis
(ReDisCA). *NeuroImage*, 301, 120868.
https://doi.org/10.1016/j.neuroimage.2024.120868

This directory is the scientific reproduction program. It lives on the
permanent `paper` branch. The lightweight public library remains on `main`.

**Status of this README:** the source-audit inventory is in
`reproduction_manifest.md`. Track workers fill `results/`. The
integration worker replaces the status table with evidence-backed
classifications after those tracks finish. Until then, treat figure
statuses as `not yet reproduced`.

## Scope

Reproduce every paper result that can be faithfully reproduced from:

1. the published paper,
2. author-supplied AIRI MATLAB,
3. stock SPoC,
4. public datasets and source-model assets,
5. the current Python `redisca` library.

The goal is not “make plots look similar at any cost”. Useful outcomes include
exact reproduction, qualitative reproduction, partial reproduction with a
documented missing dependency, or a classified discrepancy.

## Environment

```bash
python3 -m pip install -e ".[dev,paper]"
```

MATLAB is not required. Historical AIRI/SPoC numerics are reconstructed in
`paper/reproduction/common/source_faithful.py`. That module must not import
`redisca`. Call it a source-faithful Python reconstruction, not MATLAB parity.

Data cache (gitignored):

```text
.reproduction_data/
```

Override with `REDISCA_REPRODUCTION_DATA`.

## Data download

```bash
python paper/reproduction/common/download_osf.py meg-sensor
python paper/reproduction/common/download_osf.py source-models
python paper/reproduction/n170/download_erpcore.py   # written by the N170 track
```

Official sources:

- Paper dataset landing page: https://osf.io/pfde9/
- AIRI MEG assets: https://osf.io/8rk67/
- ERP CORE N170 (Kappenman et al., 2021): same OSF node `pfde9` plus the
  ERP CORE preprocessing materials referenced by the paper
- AIRI code: https://github.com/AIRI-Institute/ReDisCA
  (`15bc19cdc76989da202714b257f6de4d26a42c51`)
- Stock SPoC: https://github.com/svendaehne/matlab_SPoC
  (`18e4754aec1411160fd5b7ef0db852f1e0a87d90`)

## Commands (one per track)

```bash
python -m pytest paper/reproduction/common/tests tests -q
python paper/reproduction/n170/run.py
python paper/reproduction/meg/run.py
python paper/reproduction/simulations/run.py
python paper/reproduction/source_localization/run.py
```

MEG must be run as separate `paper_faithful` and `airi_executable` paths
(see `reproduction_manifest.md`). Never mix those settings in one figure.

Expected runtime is filled after the first complete run of each track.

## Result locations

| Track | Code | Compact results |
| --- | --- | --- |
| Manifest | `paper/reproduction_manifest.md` | `paper/reproduction_manifest.json` |
| N170 | `paper/reproduction/n170/` | `paper/results/n170/` |
| MEG | `paper/reproduction/meg/` | `paper/results/meg/` |
| Simulations | `paper/reproduction/simulations/` | `paper/results/simulations/` |
| Source localization | `paper/reproduction/source_localization/` | `paper/results/source_localization/` |

## Reproduction status

| Figure / result | Status |
| --- | --- |
| Fig. 1 source-space RSA diagrams | visual/qualitative; methods inventory |
| Fig. 2 ReDisCA diagram | visual/qualitative; methods inventory |
| Fig. 3 simulated multi-source RDMs | not yet reproduced |
| Fig. 4 single-source ROC | not yet reproduced |
| Fig. 5 four-source Monte Carlo | not yet reproduced |
| Fig. 6 localization error vs C | not yet reproduced |
| Fig. 7 N170 meaningful vs meaningless p-map | not yet reproduced |
| Fig. 8 N170 meaningful vs meaningless patterns | not yet reproduced |
| Fig. 9 N170 face/car theoretical RDMs | not yet reproduced |
| Fig. 10 N170 face-specific component | not yet reproduced |
| Fig. 11 N170 car-specific components | not yet reproduced |
| Fig. 12 MEG theoretical RDMs | not yet reproduced |
| Fig. 13 MEG face-specific | not yet reproduced |
| Fig. 14 MEG tool-specific | not yet reproduced |
| Fig. 15 MEG meaningful vs meaningless | not yet reproduced |
| Fig. 16–18 MEG non-binary RDM + MUSIC | not yet reproduced |

Statuses will be one of: `reproduced numerically`, `reproduced qualitatively`,
`approximate`, `blocked by missing source asset`, `paper/code discrepancy`,
`stochastic mismatch`, `not yet reproduced`.

## Paper vs AIRI implementation discrepancies

Starting hypotheses to re-verify from sources (not conclusions):

1. Pairs: paper triangular RDM entries vs AIRI `i != j` directed duplicates.
2. Pair matrices: paper unscaled Gram vs MATLAB `cov` (demean + `/ (T-1)`).
3. Target standardization: MATLAB `std` / library `ddof=1` (same convention).
4. Aggregation: paper sum-like notation vs SPoC weighted average vs library mean.
5. Inference: paper condition-label permutation vs SPoC random-phase + `max|lambda|`.
6. MEG window: paper entire 1500 ms vs AIRI `trange = 600:1500`.
7. AIRI default RDM: `facevstool` (`ThRDMArr(2)`), not necessarily the first paper example.

See `paper/WORKER_CONTRACT.md` for ownership and source-authority rules.
