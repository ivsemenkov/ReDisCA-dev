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
| N170 (original track; frozen) | `paper/reproduction/n170/` except the overnight subdirs below; `paper/results/n170/` except overnight subdirs |
| MEG (original track; frozen) | `paper/reproduction/meg/` except overnight subdirs; `paper/results/meg/` except overnight subdirs |
| Simulations | `paper/reproduction/simulations/`, `paper/results/simulations/` |
| Source localization | `paper/reproduction/source_localization/`, `paper/results/source_localization/` |
| Overnight A+B historical N170 | `paper/reproduction/n170/historical/`, `paper/results/n170/historical/`, `paper/reproduction/n170/HISTORICAL_REPORT.md` |
| Overnight C preprocessing | `paper/reproduction/n170/preprocessing_forensics/`, `paper/results/n170/preprocessing/`, `paper/reproduction/n170/PREPROCESSING_REPORT.md` |
| Overnight D RDM correlation | `paper/reproduction/n170/rdm_correlation/`, `paper/results/n170/rdm_correlation/`, `paper/reproduction/n170/RDM_CORR_REPORT.md` |
| Overnight E frozen-candidate apply | `paper/reproduction/n170/historical_apply/`, `paper/reproduction/meg/historical_candidate/`, `paper/results/n170/historical_apply/`, `paper/results/meg/historical_candidate/` |
| Overnight F rank 67/68 | `paper/reproduction/meg/rank_audit/`, `paper/results/meg/rank_audit/`, `paper/reproduction/meg/RANK_AUDIT.md` |
| Orchestrator / integration only | `paper/reproduction/common/`, `paper/README.md`, `paper/results_summary.*`, this file, `paper/HISTORICAL_INVESTIGATION.md` |

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
