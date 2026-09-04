# Source-localization track report

Ossadtchi et al., *NeuroImage* 301 (2024) 120868. Track owner:
`paper/reproduction/source_localization/`. This file is the evidence-backed
status for `fig18-meg-music` and `airi-source-loc-precomp`. It is not a
screenshot of Fig. 18.

## Status

| ID | Status | Why |
| --- | --- | --- |
| **Fig. 18 MUSIC** (`fig18-meg-music`) | **approximate** | Eq. 14 ran on a **locally fitted** Fig. 16/17-style `facevstool` subspace and the public AD overlapping-spheres `Gain` (5002 vertices, not fsaverage). Not `reproduced`: no FieldTrip/`show_on_cortex` A/P/S/L/R views; paper permutation `B` is unspecified (`B=200` here); Haufe patterns (D9); argmax is **left cuneus / V2**, not the paper’s qualitative right FG / right insula / left IPS / anterior central gyrus (those scouts are elevated but not the peak). |
| **AIRI precomp** (`airi-source-loc-precomp`) | **approximate** (numeric map **reproduced**) | `ctx_map = abs(W @ A1[:,3])` with `W = ImagingKernel[:, megplanarbst]` matches the committed AIRI default. Visual screenshot **blocked** (`show_on_cortex` missing). Topography is the OSF **author-saved** `filt15` file (D17), not a vanilla script `save`. **Not Fig. 18.** |
| AIRI `music` `P = eye(1)` | **blocked** | Non-executable dimension error. Not Fig. 18. |
| AIRI `music` `P = eye(Nsns)` | **approximate** | Executable reconstruction of the obvious fix; `nRAP=1` so RAP never deflates. Still `A1(:,4)` only. Not Fig. 18. |
| Author-saved `A1(:,1:3)` MUSIC | **approximate / not Fig. 18** | Eq. 14 on D17 columns. Must not be labeled as Fig. 18. |

## Commands

```bash
PYTHONPATH=src:paper/reproduction python3 paper/reproduction/source_localization/run.py
PYTHONPATH=src:paper/reproduction python3 paper/reproduction/source_localization/run.py --permutation-b 200
python3 -m pytest paper/reproduction/source_localization/tests -q
```

14 unit tests passed (Eq. 13, Eq. 14 self-dipole = 1, AIRI `P=eye(1)` raises,
`megplanarbst` = 204 `MEG GRAD` on this kernel).

## Forward model (not fsaverage)

| Asset | SHA-256 | Shape / note |
| --- | --- | --- |
| `headmodel_surf_os_meg.mat` | `a365912cae29c3ddda7be90b4bb3830f4ce081e7d4de1206d0c1406985ec439c` | `Gain` (322, 15006); rows 306–321 all-NaN extras; `MEGMethod=os_meg`; `GridLoc` ≡ tess `Vertices` |
| `tess_cortex_pial_low.mat` | `40502997c4c21d89a4c7ea207ab77c1c458a005d74e7cb78e6e0e2beb578cad1` | 5002 vtx, `cortex_5002V`, subject AD; Destrieux / Mindboggle scouts used only as labels |
| sLORETA kernel | `794043eb34f588a14186b297721d78e71ac9a08187938f611bdf0a0e0a92a1d3` | `ImagingKernel` (5002, 306), `nComponents=1`, `Function=sloreta` |
| `topo_face_vs_tool_correct_filt15.mat` | `b18be3e159164846c0e9d82e3d7dd62e1f01e53d00b511b10f11bd1f8b3b7328` | `A1` (204, 67), `comps_order=[1,2,3,4]` (D17) |

Individual T1 is not on OSF. The scan uses this Gain; it does not rebuild a BEM
from MRI.

## D15 — MAG vs GRAD indexing

AIRI `megplanarbst = sort([1:3:304, 2:3:305])` (204 indices).

On **this** kernel, `Options.ChannelTypes` repeats `MEG GRAD, MEG GRAD, MEG MAG`.
The AIRI vector therefore **is** both planars (`204 GRAD, 0 MAG`). Verified in
`index_audit.json`.

The hazard remains: the **same integers** would be MAG + first GRAD if the
triplet were MAG, GRAD, GRAD. Sensor-space `d(1:204)` is already both planars.
MUSIC uses Gain rows `megplanarbst` (204 GRAD). A labeled negative-control
precomp with GRAD1+MAG columns on this file peaks at a **different** vertex
(Mindboggle `inferiorparietal R`, vertex 2864) than the true-planar precomp.

## Fig. 18 analog (local paper-faithful subspace)

Independent of `paper/reproduction/meg/`:

- RDM: AIRI `facevstool` 6×6 (0.1 / 0.5 / 1), Fig. 16 non-binary geometry
- `redisca.ReDisCA(demean_time=False)` — unique pairs, printed unscaled Gram
- full epoch 1501 samples, [−500, +1000] ms, no AIRI Butterworth
- Haufe patterns (rank 68 < 204; invert-`W` undefined, D9)
- `A_K`: three lowest permutation *p* (ties by `|λ|`) = components **0, 1, 2**
- permutation: condition-label shuffle, null `max|λ|`, `B=200`, seed `20240915`
  (paper `B` unspecified)
- scanner: Eq. 14, two left singular vectors of each 3-col Gain block vs `A_K`

**Permutation result:** `p = (0.325, 1, 1, …)` under the max-`|λ|` null. No
component is significant at 0.05. Fig. 17’s “three lowest *p*” analog is still
the three leading eigenvalues. This is **not** a claim that three components
survive the paper test.

### Peak (argmax of subcorr)

| Field | Value |
| --- | --- |
| vertex 0-based / MATLAB 1-based | **117 / 118** |
| `xyz` (Brainstorm Vertices, m) | (−0.06806, −0.00194, 0.06321) |
| subcorr (Eq. 14) | **0.8212** |
| first principal angle | 0.607 rad |
| Mindboggle / Destrieux / Brodmann | **cuneus L** / `G_occipital_sup L` / `V2 L` |
| Structures hemisphere | Cortex L (vertices 1–2501) |

Paper claimed regions (scout **maxima**, not the argmax):

| Scout | max subcorr |
| --- | --- |
| Mindboggle fusiform R | 0.754 |
| Mindboggle insula R | 0.722 |
| Destrieux `G_oc-temp_lat-fusifor R` | 0.731 |
| Destrieux left IPS `S_intrapariet_and_P_trans L` | 0.592 |
| Destrieux `G_precentral` L / R | 0.339 / 0.564 |

Those scouts are **on the map** but are not the peak. No screenshot parity.
JSON: `paper/results/source_localization/fig18_meg_music.json`.

## AIRI precomp (not Fig. 18)

`method='precomp'`, `topos = A1(:,4)` (0-based column 3),
`W = ImagingKernel[:, megplanarbst]`, `ctx_map = abs(W @ topos)`.

| Field | Value |
| --- | --- |
| vertex 0-based / MATLAB | **394 / 395** |
| `xyz` (m) | (−0.05012, 0.00317, 0.04179) |
| max `\|W a\|` | 8.50×10⁻¹¹ (tiny because saved `A1` columns have ~10⁻¹² norms) |
| Mindboggle / Destrieux | **lingual L** / `G_oc-temp_med-Lingual L` |

Numeric map: reproduced. Cortex screenshot: blocked. D17: author-saved topo.

## AIRI `music` branch (not Fig. 18)

Committed MATLAB:

```matlab
P = eye(size(Nsns,1));   % Nsns is a scalar → eye(1)
Gp = P*G;                % dimension error
```

Literal bug: **blocked / non-executable** (`airi_music_literal_bug.json`).

`P = eye(Nsns)` with `nRAP=1` and `topos=A1(:,4)`: peak vertex **294**,
subcorr **0.576**, Mindboggle **lingual L**. Equals Eq. 14 with `K=1` on
component 4. RAP is applied only *after* the first scan, so `nRAP=1` is
ordinary MUSIC.

## Other labeled MUSIC scans (not Fig. 18)

| Scan | Peak vertex | Mindboggle | subcorr |
| --- | --- | --- | --- |
| Author-saved `A1(:,1:3)` | 2595 | lateraloccipital R / Pole_occipital R | 0.823 |
| Local `airi_executable` K=3 (bandpass, 99–999 ms, directed pairs, MATLAB cov) | **2595** (same) | lateraloccipital R | 0.838 |

The local AIRI-executable subspace MUSIC peaks at the **same vertex** as the
author-saved first-three-column MUSIC, which is consistent with D17 being an
AIRI-like `facevstool` fit, not with paper-faithful Fig. 18 (different peak).

Eq. 13 (constrained normals × `A1(:,4)`) is shipped as a scanner demo for the
simulations track. Simulation localization itself is out of scope here.

## What is missing (do not claim)

- `show_on_cortex.m`, `prepare4topoNMG`, FieldTrip `ft_topoplotER`
- Individual T1 / rebuild of Gain
- Screenshot parity with Fig. 18 A/P/S/L/R
- Bit-exact MATLAB `eig` / `svd` / `filtfilt`
- Paper-printed permutation `B` and a three-component significant subspace
  under the max-`|λ|` null used here

Plots under `paper/results/source_localization/figures/` are matplotlib
3-view scatters of tess `Vertices` colored by the scan (gitignored). Axes are
native Brainstorm coordinates, not paper camera views.
