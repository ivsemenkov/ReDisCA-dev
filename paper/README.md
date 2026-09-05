# Stage A paper reproduction

This tree reproduces Ossadtchi et al., *NeuroImage* 301 (2024) 120868 using
**exactly one** ReDisCA configuration: the AIRI → stock-SPoC settings
implemented by current `main` (`f657b954…`).

Experiments import `redisca.ReDisCA` only through
`paper.reproduction.common.method.make_redisca`. They do not contain a
second ReDisCA implementation.

## Frozen method

```python
AIRI_SPOC_KWARGS = dict(
    n_components=None,
    demean_time=True,
    divide_by_t_minus_1=True,
    directed_pairs=True,
    aggregation="mean",
    solver="whitening",
    rank=None,
    rank_tol=1e-6,
)
```

External analysis choices (preprocessing, windows, RDM fill, inference
procedure) may branch. ReDisCA constructor arguments do not.

The candidate matrix is frozen in `paper/reproduction_manifest.json`
**before** full-result selection.

## Commands

```bash
python -m paper.reproduction test
python -m paper.reproduction download all
python -m paper.reproduction stage-a
```

`--quick` exists for debugging. It is **non-reproduction**. Do not report
reduced-MC/B outputs as paper results.

Data live in gitignored `.reproduction_data/`. Lightweight JSON summaries
are written to `paper/results/`.

## Tracks

| Track | Figures | Entry |
| --- | --- | --- |
| Historical validation | A | `paper/reproduction/validation/` |
| Simulations | 3–6 | `python -m paper.reproduction.simulations.run` |
| N170 EEG | 7–11 | `python -m paper.reproduction.n170.run` |
| MEG | 12–17 | `python -m paper.reproduction.meg.run` |
| Source localization | 18 | `python -m paper.reproduction.source_localization.run` |

Primary component inference is `redisca.random_phase_test` (B=1000).
Paper-described condition-label permutation is a labeled secondary branch
in reproduction code only.
