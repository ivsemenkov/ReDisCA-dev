# N170 preprocessing forensics (Track C)

Owner: overnight C. Branch: `cursor/paper-n170-preproc-f368`.
MATLAB was not used. No MATLAB parity is claimed.

Machine copy: `paper/results/n170/preprocessing/`.
ERP CORE scripts: lucklab/ERP_CORE `c18b43d70d791ca914d90410afe4ff06d6f7f429`.
ICA list: OSF `https://osf.io/download/f9r7c/` SHA-256
`23373a2b7aae80e7b01abfdc523fb1d04fbc6f41fc48c090f7e840534224cf85`.

This track does **not** invent a third ICA component, interpolate P9/P10, add
EOG to change results, or run a combinatorial preprocessing search.

## Q1. Is there a public ERP that matches the paper better than `1_N170_erp_ar.erp`?

**No.**

Paper (published NeuroImage extract, N170 paragraph):

> were preprocessed and three ICA components corresponding to ocular and
> cardiac artifacts were removed from the data. Then, the ERP were computed
> by averaging responses within each of the stimulus types.

That is four condition averages (face / car / scrambled face / scrambled car),
not difference waves.

| File | SHA-256 | `isfilt` | bins | Role |
| --- | --- | --- | --- | --- |
| `1_N170_erp_ar.erp` | `53e74e931e6f0adaf1e5be4d606d028fcc3e04ee8b066569c2ed2d033d9bbc72` | 0 | 4 correct-response parent averages | **Preferred.** Script 7 `pop_averager(..., 'Criterion','good')` |
| `1_N170_erp_ar_lpfilt.erp` | `228b52ad69b9dc9b88f6b4c0b1d32dc778450e0d9aa32850b9ad9a8a61a8b9fe` | 1 | same 4 bins | Same averages after Script 7 20 Hz low-pass |
| `1_N170_erp_ar_diff_waves_lpfilt.erp` | `3e63fbc34c66527ed3d88fb46040fd0df5075a08b81bef8c654a470d3faf0846` | 1 | 9 | Parents (bins 1–4) **bit-identical** to lpfilt, plus 5 derived contrasts |

`1_N170_erp_ar_diff_waves.erp` (unfiltered diff-waves) is **not** in the
subject-1 dump. Diff-waves bins 5–9 (`N170_Diff_Wave.txt`, SHA-256
`2cdd1abc57f8c776858e39dc29aa7bf3d6d08609ca20bbeb19056f3064bde675`) are
not the 4-condition ReDisCA input.

## Q2. Is the 20 Hz lpfilt ERP paper-plausible?

**Plausible as a documented ERP CORE extra. Not a paper statement. Not a silent default.**

The paper N170 section does **not** mention low-pass filtering ERPs. Nearby
“low-pass” in the extract is the **simulation** 2 Hz Butterworth on generated
sources, not EEG.

Script 7 (`7_Average_ERPs.m` SHA-256
`1fc8f307ee297f864a96a458ecbac0ced63fe2d2eda6f57aa0801e44c4ae9b00`) first
saves the unfiltered average, then:

> Apply a low-pass filter (non-causal Butterworth impulse response function,
> 20 Hz half-amplitude cut-off, 48 dB/oct roll-off) to the ERP waveforms

```matlab
ERP = pop_filterp( ERP,  1:35 , 'Cutoff',  20, 'Design', 'butter', 'Filter', 'lowpass', 'Order',  8 );
```

Script 8 plots `*_erp_ar_diff_waves_lpfilt.erp`. Script 12 measures mean
amplitude on **unfiltered** diff-waves and peaks on **lpfilt** diff-waves.
Using lpfilt for ReDisCA would be an ERP CORE plotting/peak convention, not
the printed paper analysis. **No estimator was run on lpfilt.**

## Q3. Channels at the likely analysis stage

Likely analysis file: `1_N170_erp_ar.erp` (35 labels).

**28 scalp EEG (ERPLAB order):**
`FP1 F3 F7 FC3 C3 C5 P3 P7 PO7 PO3 O1 Oz Pz CPz FP2 Fz F4 F8 FC4 FCz Cz C4 C6 P4 P8 PO8 PO4 O2`

**7 EOG / bipolar (dropped for ReDisCA, not added back):**
`HEOG_left HEOG_right VEOG_lower (corr) HEOG (corr) VEOG (uncorr) HEOG (uncorr) VEOG`

**P9 / P10**

| Stage | File SHA-256 (prefix) | P9/P10 |
| --- | --- | --- |
| Raw 33-ch `1_N170.set` | `2abe204dd380…ca483` | **present** (ch9=P9, ch27=P10) |
| After shift `1_N170_shifted.set` | `e7aa5accb0f5…00461` | present |
| After Script 1 reref | `f033fba3dc2c…d8a36` | **absent** |
| Pre-average epoch `.set` | `2792a23f6272…7b4c4` | absent |
| Averaged `.erp` | `53e74e93…bbc72` | absent |

Script 1 `Rereference_Add_Uncorrected_Bipolars_N170.txt` SHA-256
`6b03d660ba824d9f992dd176258ff1ec89faebdfab13e4e20e2576eecab6e051`
skips original ch9 and ch27 (`nch9 = ch10` PO7, `nch26 = ch28` PO8).
Do **not** interpolate P9/P10.

## Q4. Third ICA component? **STOP. None source-supported.**

Paper: “three ICA components corresponding to ocular and cardiac artifacts”.

Official list (`ICA_Components_N170.xlsx`): subject `"1"` → **2, 7**.
Script 4 SHA-256 `b2f5f6d3746641f7328619a19b44dfe8dc3dc74f55d684c542f58926d5c69cb4`
loads that sheet as **ocular** artifacts. Cardiac is not mentioned. Script 3
is commented so original weights are kept; ICA was **not** re-run here.

EEGLAB evidence (subject 1):

| File | `icaweights` | SHA-256 |
| --- | --- | --- |
| `…_ica_weighted.set` | 31 × 31 | `1c297bf0cefac7df9ca0f3964f60130b78abd96dd760dc7d147d75d6d7ec400f` |
| `…_ica_corr.set` | **29 × 31** | `3344a27e8c65123a94ffb1fd7b252ae01b5b341d15c490718d6b4cb1f24e4f6a` |

Dropped `icaweights` rows match components **2 and 7** exactly (1-based).
`gcompreject` is all zeros; Script 4 uses the xlsx, not that mask.
ICA channels 1:31 = 28 scalp + HEOG_left/right + VEOG_lower. **No ECG.**

Across 40 subjects the xlsx count is {1: 15, 2: 12, 3: 10, 4: 3}; modal **1**,
mean ≈ 2.03. Exactly three listed components occur in 10/40 subjects, **not**
subject `"1"`. The paper sentence is a coarse ERP CORE gloss (D11).

Remaining ICs can have nonzero EOG mixing energy (largest leftover EOG-sum is
IC 18). That is **not** a source-supported third component. It was not
selected.

## Q5. Alternative data state run?

**No.** No Track A 12-variant battery. No lpfilt / diff-waves / pre-AR /
all-trials estimator.

Correct-response vs all trials (`1_N170_Eventlist_Bins.txt` SHA-256
`69c6360f51ffdcc49fbc20906d2e8eaee9402a1b5018ecb7bd55866cce8a929e`;
`BDF_N170.txt` SHA-256
`c66a75a5d1cd076c990b05d2f8f111bd3a45bdb0a5c4185797d954b78966c847`):

- 320 stimuli; 244 correct (201), 76 incorrect (202)
- BDF bins 1–4: **correct only**, 61 / 53 / 65 / 61 = 240 epochs
- After AR: 191 good / 49 flagged (`1_AR_Percentages_N170.csv` SHA-256
  `6cd40ab3228310935c0ae63a422ccf85ba911d7f191c68b01d04abf1a5cff483`;
  accepted 52 / 38 / 49 / 52)
- Paper does not mention accuracy gating. Official public `.erp` is
  correct-response **and** artifact-rejected. No public all-trials 4-condition
  `.erp` exists. Pre-AR `.set` is not a precomputed average.

The 12 source-motivated historical variants (car: 170 & 200 ms × unique/directed
× gram/cov; face: 200 ms × 2×2; stock-SPoC random-phase B=1000) are recorded in
`track_a_variants.json` as **not executed**.

## D11 status

**Documented, unresolved.** Keep official subject-1 averages with components
2 and 7 already removed. Do not invent a third IC.

## Commands

```bash
python3 paper/reproduction/n170/preprocessing_forensics/run.py
python3 -m pytest paper/reproduction/n170/preprocessing_forensics/test_forensics.py -q
```
