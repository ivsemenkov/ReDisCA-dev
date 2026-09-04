# Paper-branch worker contract

This file is the ownership map for parallel reproduction work.

Permanent git branch: `paper` (starts at library `main` commit
`5a5c8658452172e4011445c9a394c1cbbd3c5f7e`).

`main` is the lightweight library. Do not modify library semantics on
`paper` merely to improve reproduction. Do not merge `paper` into `main`.

## Directory ownership

| Owner | Paths |
| --- | --- |
| Source audit | `paper/reference/`, `paper/reproduction_manifest.md`, `paper/reproduction_manifest.json` |
| N170 | `paper/reproduction/n170/`, `paper/results/n170/` |
| MEG | `paper/reproduction/meg/`, `paper/results/meg/` |
| Simulations | `paper/reproduction/simulations/`, `paper/results/simulations/` |
| Source localization | `paper/reproduction/source_localization/`, `paper/results/source_localization/` |
| Orchestrator / integration only | `paper/reproduction/common/`, `paper/README.md`, `paper/results_summary.*`, this file |

Workers must not concurrently edit the same files. If a shared helper is
needed, request it rather than racing to edit `paper/reproduction/common/`.

## Source authority

Always distinguish:

- A. what the paper says
- B. what AIRI MATLAB executes
- C. what stock SPoC executes
- D. what the current Python library does
- E. what actually reproduces the published result

Never silently collapse these. When they disagree, record the discrepancy
and test controlled variants.

The forensics branch `cursor/forensics-airi-matlab-0370` is read-only donor
material. Re-verify claims from primary sources. Do not treat forensics as
successful reproduction. Do not merge it wholesale.

## Data policy

Large datasets live under `.reproduction_data/` (gitignored). Commit code,
small tables, manifests, compact metrics, provenance. Do not vendor the
full paper PDF/text, full AIRI, full SPoC, or raw datasets.

## Canonical vs historical paths

Canonical deterministic runs use `from redisca import ReDisCA`.

Historical AIRI/SPoC reconstruction uses
`paper/reproduction/common/source_faithful.py` and must not call `redisca`
internally.

## Coordination contract

Workers must follow `paper/reproduction_manifest.md` (and the JSON
companion) once the source-audit worker has written it.

Integration writes the evidence-backed status table in `paper/README.md`
and the reproduction matrix in `paper/results_summary.md` /
`paper/results_summary.json`. TRACK_REPORT self-labels are inputs, not
final statuses.
