# Provenance: AIRI-Institute/ReDisCA

Do not vendor this repository into git. Pin and clone into
`.reproduction_data/upstream/AIRI-ReDisCA`.

| Field | Value |
| --- | --- |
| URL | https://github.com/AIRI-Institute/ReDisCA |
| Pinned commit | `15bc19cdc76989da202714b257f6de4d26a42c51` |
| Commit date | 2024-11-20 20:28:24 +0300 |
| Commit message | Create LICENSE |
| Local verification | `git -C .reproduction_data/upstream/AIRI-ReDisCA rev-parse HEAD` |

Tracked scientific files (only two `.m` scripts plus README/LICENSE):

| File | git blob SHA-1 | SHA-256 |
| --- | --- | --- |
| `Redisca_tools_faces_3_random_norm_correct.m` | `f5e339c2945cc70d1f7686b7edb347c87c08c587` | `44af60c421bbcc6321c5e65f73fedf8f2a9cd81dd776f094935585cdf7ab17f2` |
| `Redisca_source_loc_for_tools_faces_3_random_.m` | | `e7270939bb8fe052d23189b471dfde8f31d8b902456e272746983caad178dcb9` |

README data instruction: download https://osf.io/8rk67/ into `data/`.
README SPoC instruction: BBCI `external/` — see `provenance/spoc.md`.

Missing from the repo (blocked helpers): `prepare4topoNMG.m`,
`show_on_cortex.m`, FieldTrip, Signal Processing Toolbox
(`butter`/`filtfilt`), MATLAB itself.

The forensics branch `cursor/forensics-airi-matlab-0370` isolated the
same commit under `forensics/airi_matlab/original/AIRI-ReDisCA/` and
recorded that **no MATLAB binary was available**, so the AIRI script
was never executed through SPoC in that environment. This audit did
not run MATLAB either. Executable semantics are reconstructed from
source + OSF file headers, not from a MATLAB diary.
