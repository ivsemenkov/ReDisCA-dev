# Source-localization / MUSIC track

Owner: source-localization worker. Do **not** edit `paper/reproduction/common/`,
`paper/reproduction/meg/`, `src/redisca`, or the manifests.

Two labeled paths, never collapsed:

| ID | What | Fig. 18? |
| --- | --- | --- |
| `fig18-meg-music` | Paper Eq. 14 MUSIC of a **locally fitted** Fig. 16/17 non-binary (`facevstool`) subspace. Free-orientation dipole = two leading left singular vectors of each 3-column Gain block vs `A_K`. | **This is the Fig. 18 attempt.** Status is `approximate` (see `TRACK_REPORT.md`). |
| `airi-source-loc-precomp` | AIRI `method='precomp'`: `abs(ImagingKernel(:, megplanarbst) * A1(:,4))`. Author-saved OSF topo (D17). | **No.** |
| `airi-music-literal-P-eye1` | AIRI `method='music'` with `P = eye(size(Nsns,1))` = `eye(1)`. | **No.** Non-executable. |
| `airi-music-eye-Nsns-fix` | Same music branch with `P = eye(Nsns)`. `nRAP=1` ⇒ ordinary MUSIC of whatever `topos` is (AIRI default: one column). | **No.** |
| author-saved `A1(:,1:K)` MUSIC | Eq. 14 on OSF `filt15` columns. | **No.** Not a local Fig. 17 fit (D17). |

Do **not** replace subject AD `Gain` with fsaverage. Cosine-similarity
localization of simulated sources (Eq. 13, Figs. 4–6) belongs to the
simulations track; the scanners are implemented here for reuse.

## Commands

```bash
# from the repository root (MEG + source-model OSF cache already under .reproduction_data/)
PYTHONPATH=src:paper/reproduction python3 paper/reproduction/source_localization/run.py

# optional
PYTHONPATH=src:paper/reproduction python3 paper/reproduction/source_localization/run.py --permutation-b 200
PYTHONPATH=src:paper/reproduction python3 paper/reproduction/source_localization/run.py --skip-meg-fit

python3 -m pytest paper/reproduction/source_localization/tests -q
```

`--permutation-b` is the paper-style **condition-label** permutation count used
to pick the three lowest-p components (Fig. 17 analog). The paper does not
print `B`. Default is 200. `--skip-meg-fit` forces Fig. 18 to `blocked` and
keeps only AIRI / author-saved maps.

If MEG load is too heavy, `run.py` records the error and still writes AIRI
precomp + author-saved scans. That fallback is **not** Fig. 18.

## Assets (gitignored cache)

| File | Role |
| --- | --- |
| `headmodel_surf_os_meg.mat` | `Gain` (322, 15006), `GridLoc` (5002, 3), overlapping spheres |
| `tess_cortex_pial_low.mat` | `Vertices` (5002, 3), `Faces` (9974, 3), Destrieux/Mindboggle scouts |
| `results_sLORETA_MEG_GRAD_MEG_MAG_KERNEL_150924_1824.mat` | constrained `ImagingKernel` (5002, 306) |
| `topo_face_vs_tool_correct_filt15.mat` | author-saved `A1` (204, 67), `comps_order=[1,2,3,4]` (D17) |
| `MEG_AD_run1.mat` + SPM labels | local Fig. 16/17 refit (this folder only; does not edit `meg/`) |

## Indexing (D15)

AIRI `megplanarbst = sort([1:3:304, 2:3:305])`. On **this** OSF kernel,
`Options.ChannelTypes` is `MEG GRAD, MEG GRAD, MEG MAG` repeating, so that
index vector **is** the 204 planars. The hazard is real if the triplet were
`MAG, GRAD, GRAD` (same integers, mixed types). `index_audit.json` records
the verification. MUSIC uses Gain planar rows matching the 204 sensor
topographies.

## Outputs

Compact JSON (committed) under `paper/results/source_localization/`.
Vertex maps as `.npz` and matplotlib 3-view Vertex scatters as `.png`
(gitignored; **not** FieldTrip / `show_on_cortex` screenshot parity).

Missing by construction: `show_on_cortex`, `prepare4topoNMG`, FieldTrip,
individual T1. Do not claim screenshot parity with Fig. 18’s A/P/S/L/R views.

See `TRACK_REPORT.md` for statuses and peak vertices.
