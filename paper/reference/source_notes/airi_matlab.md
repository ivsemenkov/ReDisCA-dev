# AIRI MATLAB (pinned clone)

Clone: https://github.com/AIRI-Institute/ReDisCA
Commit (verified HEAD of the local clone):
`15bc19cdc76989da202714b257f6de4d26a42c51`
Date: 2024-11-20. Message: `Create LICENSE`.

The repository is **MEG-only**. There is no N170 script and no
simulation script. README claims the repo “contains MATLAB code needed
to replicate the results of the paper”; that claim is true only for a
sensor-space MEG path plus a separate source-localization script, and
even those diverge from the published MEG methods (window, filter,
pairs, inference, default RDM).

Verified blob hashes (SHA-256 of file bytes; git blob SHA-1 of the
main script is `f5e339c2945cc70d1f7686b7edb347c87c08c587`):

| File | SHA-256 |
| --- | --- |
| `Redisca_tools_faces_3_random_norm_correct.m` | `44af60c421bbcc6321c5e65f73fedf8f2a9cd81dd776f094935585cdf7ab17f2` |
| `Redisca_source_loc_for_tools_faces_3_random_.m` | `e7270939bb8fe052d23189b471dfde8f31d8b902456e272746983caad178dcb9` |
| `README.md` | `a0b2265a4466d365b3edf68b3f4c579d5591645a1c5625c966ac20945cbbfa04` |

SPoC is **not vendored**. README points at
`github.com/bbci/bbci_public/tree/master/external`. That directory is
empty until `bbci_import_dependencies('ssd+spoc')` downloads
`https://github.com/svendaehne/matlab_SPoC/archive/master.zip`.
Today that zip resolves to stock SPoC commit
`18e4754aec1411160fd5b7ef0db852f1e0a87d90` (repo HEAD). See
`provenance/spoc.md`.

Plotting helpers **not in the repo**: `prepare4topoNMG`,
`ft_topoplotER` (FieldTrip), `show_on_cortex`. Source-loc and topo
figures cannot be regenerated from AIRI sources alone.

## Main script execution order

File: `Redisca_tools_faces_3_random_norm_correct.m`

Hard-coded defaults:

```matlab
ThRDMArr = {'face','facevstool','tool','toolvsface','meaning','meaning1'};
RDM = ThRDMArr(2);          % 'facevstool'
Nmc = 100;
bRandomizeLabels = false;
lowCutOff = 0.25; highCutOff = 20;
trange = 600:1500;
spoc(..., 'n_bootstrapping_iterations', 1000)
```

`bEveryOther` is set `true` and never read.

### Data load

- `data/ibfctfprespm8_AD_run1_raw_tsss_mc.mat` — SPM `D.trials` labels
  only. Companion `.dat` is **not** loaded.
- `data/MEG_AD_run1.mat` — variable `d`.

Verified on the OSF file: MATLAB orientation `(207, 1501, 880)`
(h5py reports the transposed `(880, 1501, 207)`). Channel 1–204 are
`MEGPLANAR`; 205–207 are EOG, EOG, STI. `Fsample=1000`,
`Nsamples=1501`, `timeOnset=-0.5` s (true time \([-500, +1000]\) ms).

The script uses `data_meg.d(1:204,:,:)` for averages (correct planar
subset of this file). It filters **all** 207 channels.

### Trial selection (verified counts)

Labels are 3-digit strings. Keep
`t1==1` or `(t1==2 & t2==0)`. Then:

| Cell | Rule | Count |
| --- | --- | --- |
| face1 | `t2==5 & t3==1` | 80 |
| face2 | `t2==6 & t3==1` | 80 |
| tool1 | `t2==7 & t3==1` | 80 |
| tool2 | `t2==8 & t3==1` | 80 |
| nons1 | `t2==0` | 80 |
| nons2 | `t2==9` | 80 |

Matches the paper’s 80 × 6. 400 other trials in the 880-epoch file
are unused.

### Filter

`[bf,af] = butter(3,[0.25,20]/500); filtfilt` per trial, time as
columns after transpose. Nyquist 500 Hz \(\Rightarrow\) 1000 Hz
sampling, consistent with SPM `Fsample`. **Not described in the
paper.** SciPy `filtfilt` is not a bit-exact MATLAB substitute.

### Condition averages and pairs

`mx{idx} = mean(d(1:204,:,idxTrial{idx}), 3)` — full 1501-sample
averages. Pair loop:

```matlab
for i_cnd = 1:Nconds
  for j_cnd = 1:Nconds
    if i_cnd==j_cnd, continue; end
    Xspoc(:,:,e) = Xi'-Xj';   % (T, 204) with T = 901
    z(e) = D(i_cnd,j_cnd);
```

Directed pairs \(i\neq j\): **30** epochs, not 15 unique pairs.
`trange = 600:1500` on 1-based MATLAB indices: samples 600…1500
inclusive = **901 samples**, true time **99 ms … 999 ms**. The paper
window is the entire \([-500, +1000]\) ms (printed as 1500 samples).

Plot time axis `linspace(-536,964,size(mx{1},2))` does **not** match
the SPM time vector. It is a display bug; it does not change SPoC
inputs.

SPoC then computes MATLAB `cov` of each `(T,N)` epoch (demean +
`/(T-1)`), z-scores `z` with sample SD, and random-phase bootstrap
\(B=1000\). See `stock_spoc.md`.

### Theoretical RDMs (exact AIRI numbers)

After the upper-triangle assignments, `D = D+D'`. Condition order:
face1, face2, tool1, tool2, nons1, nons2. Within-category “same” is
**0.1, not 0**.

**`face`** (Fig. 12a analog): faces vs faces 0.1; faces vs all others
1; tools/nons among themselves 0.1.

**`facevstool`** (default; Fig. 16 analog, **not** a binary
face-vs-tool detector): within-category 0.1; face–tool 1;
face/tool–nons 0.5.

**`tool`**: tools vs faces/nons 1; others 0.1.

**`meaning`**: face/tool block 0.1; those vs nons 1; nons 0.1.

**`meaning1`**: faces 0.1, tools 0.1, everything else 1 (including
face vs tool).

**`toolvsface`** is in `ThRDMArr` but has **no `elseif`** in the `D`
builder; selecting it leaves `D` all zeros after `D+D'`. Dead name.

Class-contrast labels used later for the timecourse test:

| RDM | Class1 | Class2 |
| --- | --- | --- |
| face | 1,2 | 5,6 |
| facevstool | 1,2 | 3,4 |
| tool | 3,4 | 5,6 |
| toolvsface | 3,4 | 1,2 |
| meaning / meaning1 | 1–4 | 5,6 |

### Timecourse Monte Carlo (`Nmc=100`) — not the paper FWER test

After SPoC, the script:

1. Channel-wise `std(d,0,3)` (sample SD over 880 trials) and divides
   **all trials** by that (`data_meg_std`). SPoC itself was fit on
   **unnormalized** averages.
2. Pools `idxClass1 ∪ idxClass2`, `randperm`, splits **half vs half**
   (does not preserve subcategory structure).
3. `d12 = mean(std_data(:,:,idx1)) - mean(std_data(:,:,idx2))`,
   projects the first four filters, 100 times.
4. Observed `dd = W(:,i)'*(meanClass2-meanClass1)` on standardized
   data.
5. `pminus(i,t) = 1 - sum(dd>max(aa,[],2))/Nmc` and a symmetric
   `pplus` vs `min`. Pointwise, not max-T FWER. \(p=0\) is possible.

Paper MEG time-series test: permute subcategory labels, surrogate
*averages*, apply filters, **FWER max-stat over time**.

Asterisks in AIRI plots are these `pplus/pminus < 0.05`. Component
title `p` is SPoC `p_values1` (random-phase, \(B=1000\)).

Empirical RDM-vs-time `Q(c,i)` uses **instantaneous** squared
differences of filtered traces (one time sample), `corrcoef` on the
vectorized upper triangle including zeros on and below the diagonal.
That is not the windowed \(w^\top R_{ij} w\) empirical RDM of the
paper.

`return` precedes the `save topo_*` block, so a vanilla run **does not
write** `topo_face_vs_tool_correct.mat`. The OSF file
`topo_face_vs_tool_correct_filt15.mat` is a pre-saved `A1` (204 × 67)
with `comps_order = [1 2 3 4]`. Rank 67 is SPoC’s numerical
whitening rank of 204-channel `Cxx`, not 204 full-rank filters.

## Source-localization script

File: `Redisca_source_loc_for_tools_faces_3_random_.m`

```matlab
load topo_face_vs_tool_correct_filt15;
topos = A1(:,4);          % fourth component only
method = 'precomp';       % default
nRAP = 1;
hm  = load('data/headmodel_surf_os_meg.mat');
io  = load('data/results_sLORETA_MEG_GRAD_MEG_MAG_KERNEL_150924_1824.mat');
ctx = load('data/tess_cortex_pial_low.mat');
megplanarbst = sort([1:3:304 2:3:305]);   % 204 indices
```

Verified source-model contents:

| File | Role | Shape / note |
| --- | --- | --- |
| `headmodel_surf_os_meg.mat` | Brainstorm overlapping spheres (`MEGMethod='os_meg'`) | `Gain` **(322, 15006)**; `GridLoc` (5002, 3); surface `AD/tess_cortex_pial_low.mat` |
| `tess_cortex_pial_low.mat` | cortex_5002V | Vertices (5002, 3), Faces (9974, 3) — matches Kozunov 2018 “downsampled to 5002 vertices” |
| sLORETA kernel | `Function='sloreta'`, `nComponents=1`, comment `sLORETA: MEG ALL(Constr)` | `ImagingKernel` **(5002, 306)**; `GoodChannel` 1…306; `SourceOrient='fixed'`; `SNR=3` |

Default `'precomp'`: `ctx_map = abs(W * topos)` with
`W = ImagingKernel(:, megplanarbst)`. This is **constrained sLORETA
of one topography**, not paper Fig. 18 MUSIC of a multi-component
subspace.

`'music'` branch exists: per-vertex SVD of 3-orientation `G` columns
to 2D, SVD of topographies, leading singular value, optional RAP.
This is the closest AIRI path to Eq. 14, but the script’s default
does not run it, and `topos = A1(:,4)` is one column.

`'mne'` branch: free-orientation MNE with `lam=0.81` and a specific
noise-cov whitening formula.

**Index hazard:** `megplanarbst = sort([1:3:304 2:3:305])` is
Neuromag **MAG + first GRAD** of each triplet if `Gain`/`ImagingKernel`
use standard 306-channel MAG/GRAD/GRAD order. Sensor-space data
`d(1:204,:,:)` are already **both planars and no mags**. Applying the
same index vector to a 306-column kernel does **not** select “204
planar gradiometers”. This must be treated as a source-loc variant,
not as paper Eq. 14.

`Gain` has 322 rows (306 MEG + extras). Indices up to 305 stay inside
a leading 306 MEG block if that is how the channel file is ordered;
this was not re-verified against a Brainstorm channel file (none on
OSF).
