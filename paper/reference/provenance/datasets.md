# Provenance: datasets and source-model assets

Large files live under `.reproduction_data/` (gitignored). Record
hashes and URLs only.

## Paper landing pages

| What the paper prints | What the node actually is |
| --- | --- |
| https://osf.io/pfde9/ “data used in the MS” | ERP CORE **N170** component (title “N170”), parent https://osf.io/thsqg/. Folders: “N170 All Data and Scripts”, “N170 Raw Data BIDS-Compatible”, “N170 Raw Data and Scripts Only”. **No MEG.** |
| Ethics DOI 10.18115/D5JW4R | UC Davis ERP CORE distribution (same project). |
| Code “upon request” ossadtchi@gmail.com | Superseded for practical purposes by GitHub AIRI-Institute/ReDisCA (incomplete vs the paper). |

AIRI README / this reproduction: MEG + source models at
https://osf.io/8rk67/ (title “ReDisCA”, created 2024-11-20).

Kozunov et al. 2018 (Front. Hum. Neurosci. 11:650) describes the MEG
experiment, tSSS, Brainstorm overlapping-spheres forward model, and
5002-vertex cortices. Individual T1 volumes are **not** on 8rk67.

## OSF 8rk67 files (API 2026-09-04, hashes re-verified on disk)

API: `https://api.osf.io/v2/nodes/8rk67/files/osfstorage/?filter[kind]=file`

| File | Bytes | SHA-256 | Download | Used by |
| --- | --- | --- | --- | --- |
| `MEG_AD_run1.mat` | 1243214548 | `0eca2756c9190ce637a3e14abd24e7cf975d758d3ccea03107963e8b5841a4f6` | https://osf.io/download/h9zpq/ | AIRI main script `d` |
| `ibfctfprespm8_AD_run1_raw_tsss_mc.mat` | 62701 | `87890337c385e81c718c421d7be35e54423ca9ceb985e047b276b02018334950` | https://osf.io/download/673e184585d2961fe2886e03/ | SPM trial labels |
| `ibfctfprespm8_AD_run1_raw_tsss_mc.dat` | 1093688640 | `d609567ff25eb88055fa26713d7debc4a6c359770835c148fe358bcb97c408e8` | https://osf.io/download/352dp/ | **not** loaded by AIRI |
| `topo_face_vs_tool_correct_filt15.mat` | 105738 | `b18be3e159164846c0e9d82e3d7dd62e1f01e53d00b511b10f11bd1f8b3b7328` | https://osf.io/download/gxbe5/ | AIRI source-loc `A1` (204×67), `comps_order=[1,2,3,4]` |
| `headmodel_surf_os_meg.mat` | 35957600 | `a365912cae29c3ddda7be90b4bb3830f4ce081e7d4de1206d0c1406985ec439c` | https://osf.io/download/2afzg/ | overlapping spheres `Gain` (322×15006) |
| `results_sLORETA_MEG_GRAD_MEG_MAG_KERNEL_150924_1824.mat` | 13203960 | `794043eb34f588a14186b297721d78e71ac9a08187938f611bdf0a0e0a92a1d3` | https://osf.io/download/673e19715cbaa22c0a75e832/ | constrained sLORETA kernel (5002×306) |
| `tess_cortex_pial_low.mat` | 481530 | `40502997c4c21d89a4c7ea207ab77c1c458a005d74e7cb78e6e0e2beb578cad1` | https://osf.io/download/673e1974b0f7255a4475e61b/ | cortex_5002V, subject AD |

Verified MEG headers: `d` is 207×1501×880 in MATLAB orientation;
channels 1–204 `MEGPLANAR`, then EOG, EOG, STI; 880 SPM trials;
AIRI six-subcategory filter yields 80+80+80+80+80+80.

`download_osf.py` (orchestrator-owned) already pins these SHA-256s.
This audit filled the previously missing download URLs for the kernel,
tess, headmodel, and `.dat`.

## ERP CORE N170 (pfde9)

Scripts (GitHub https://github.com/lucklab/ERP_CORE , local clone
commit `c18b43d70d791ca914d90410afe4ff06d6f7f429`, 2020-08-13):
subject folders named `'1'`…`'40'` matching the paper’s index `"1"`.

Official pipeline (N170 `EEG_ERP_Processing/`):

1. Shift events +26 ms; downsample **1024 → 256 Hz**; average
   reference of the 33 EEG-typed channels; bipolar HEOG/VEOG;
   Butterworth high-pass 0.1 Hz, order 2.
2. ICA prep (break rejection + amplitude windows from
   `ICA_Prep_Values_N170.xlsx`).
3. ICA (`binica`/`runica` on chans 1:31) — script is commented so
   that **original weights** are kept.
4. Remove components listed in `ICA_Components_N170.xlsx`.
5. Epoch **[−200, +800] ms**, baseline [−200, 0].
6. Interpolate + artifact rejection (several xlsx parameter files).
7. Average “good” trials; optional 20 Hz low-pass on ERPs
   (order 8, 48 dB/oct). Bins: Faces / Cars / Scrambled Faces /
   Scrambled Cars, correct responses only (`BDF_N170.txt`).

Subject `"1"` artifact-rejection summary (local
`1_AR_Percentages_N170.csv`): 191 accepted / 49 rejected (20.42%);
bins 52 / 38 / 49 / 52 accepted.

`ICA_Components_N170.xlsx` (OSF All Data package,
https://osf.io/download/f9r7c/ , 9904 bytes, SHA-256
`23373a2b7aae80e7b01abfdc523fb1d04fbc6f41fc48c090f7e840534224cf85`):
subject 1 → components **2, 7**. Not three components; not labeled
cardiac. Prefer the precomputed `1_N170_erp_ar.erp` (or
`_lpfilt`) from “N170 All Data and Scripts/1/” over recomputing ICA.

BIDS raw uses `sub-001` numbering; ERP CORE scripts use `'1'`. Do not
confuse them.

## Simulation forward model — public status

**Not public as a named bundle.** Paper never says “Brainstorm”,
“fsaverage”, or “subject AD” in Section 2.4. The only complete
forward operator on 8rk67 is AD overlapping-spheres `Gain` plus
`tess_cortex_pial_low`. Individual MRI, FreeSurfer recon, and
channel.mat are absent. Simulations are **source-model-dependent**
and **currently blocked** for claiming exact Fig. 4–6 numbers
unless a track explicitly adopts the AD `Gain` as a hypothesis and
labels results `approximate` / `blocked by missing source asset`.

Kozunov 2018: meshes downsampled to 5002 vertices; overlapping
spheres; sLORETA — consistent with the 8rk67 filenames, which is
circumstantial evidence for MEG localization, not for the
simulation mesh.

## Paper PDFs (not committed)

Forensics recorded (not re-hashed here; PDFs were not in
`paper_pdf/` at audit time):

- Published PDF source (institutional copy):
  `https://megmoscow.ru/wp-content/uploads/pubs/10.1016_j.neuroimage.2024.120868.pdf`
  SHA-256 `018478c993ab34b46b732b2b00a573a9e342101e6ae80e190f97a904967a3208`
  (25 pages). Treat as forensics-reported until re-downloaded.
- BioRxiv: `https://www.biorxiv.org/content/10.1101/2024.02.01.578343.full.pdf`
  SHA-256 `e9ffb81aab830febd3b31aca1e72848a62c0b956c057f19231faf9dedc59a394`.

Extracted text used in this audit is pinned in `pins.json`.
