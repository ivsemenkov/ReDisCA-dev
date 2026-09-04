# N170 track (ERP CORE subject `"1"`)

Owner: N170 worker. Do not edit `paper/reproduction/common/` or `src/redisca`.

Reproduce Ossadtchi et al. 2024 Figs 7–11 from official ERP CORE averages and
the paper methodology. Canonical deterministic fits use
`from redisca import ReDisCA`. Inference is a **separate** condition-label
permutation layer (`inference.py`). Do not treat SPoC random-phase as the
paper N170 test.

## Run

From the repository root (cloud worktree: `/tmp/redisca-worktrees/n170`):

```bash
python paper/reproduction/n170/run.py
python paper/reproduction/n170/run.py --B 1000 --step-ms 25 --seed 20240904
python -m pytest paper/reproduction/n170/test_n170.py -q
```

Compact JSON goes to `paper/results/n170/`. PNG figures are written there too
but are gitignored.

## Data (gitignored cache)

Already present under `.reproduction_data/erpcore/all_data_and_scripts/1/`:

| File | Role |
| --- | --- |
| `1_N170_erp_ar.erp` | **Preferred** official average (manifest). Unfiltered ERPs. |
| `1_N170_erp_ar_lpfilt.erp` | 20 Hz low-pass of the same averages (ERP CORE Script 7). Not the default. |
| `1_AR_Percentages_N170.csv` | Accepted trials: 191 total; 52 / 38 / 49 / 52 by bin. |

Official scripts: `.reproduction_data/upstream/ERP_CORE` @ `c18b43d`.

Do not reuse deleted student N170 examples as an oracle.

## Decisions (not silent paper values)

| Choice | This track | Paper |
| --- | --- | --- |
| Subject | folder `"1"` | first participant index `"1"` |
| Conditions | ERPLAB bins Faces, Cars, Scrambled Faces, Scrambled Cars (correct) | same |
| Epoch / fs | official ERP times ≈ [−199.22, 796.88] ms, 256 Hz, 256 samples | [−200, 800] ms, 256 Hz |
| ICA | already applied in the `.erp`; subject 1 removes **2 and 7** only | “three ocular+cardiac” — **D11**, not invented |
| Channels | **28 scalp EEG** FP1…O2; drop EOG / corr / uncorr bipolars | unnamed. P9/P10 already absent from the ERP |
| `demean_time=False` | paper printed Gram (primary) | Eq. 4 |
| `demean_time=True` | labeled extra, never mixed into the primary plots | not printed |
| Sliding T | 150 ms (meaning) | 150 ms |
| Sliding step | **25 ms** (documented choice) | **not specified** |
| Face window | T=100 ms centered at 200 ms → [150, 250] ms | same |
| Car window | T=100 ms centered at 170 ms → [120, 220] ms | “at t=170 ms”; duration not restated |
| RDMs | binary 0/1; 0.1-within extra (z-score-equivalent here) | figures only |
| Inference | permute condition order of D; exact 24 + Monte Carlo B=1000 | B unspecified; condition labels |
| Random-phase | optional exploratory, labeled, **not** the paper N170 test | D5 |
| RDM corr | Pearson of unique `i<j` of D vs `‖u_i−u_j‖²` | paper face **0.82**, car **>0.99** — report **actual**, do not tune |

## Outputs

- `paper/results/n170/summary.json` — headline numbers vs 0.82 / >0.99
- `fig07_meaning_pmap.json`, `fig08_meaning_windows.json`, `fig09_rdms.json`,
  `fig10_face.json`, `fig11_car.json`
- `fingerprints.json` — compact regression numbers
- `channel_selection.json`, `environment.json`
- `TRACK_REPORT.md` — source evidence, commands, discrepancies, blocked items
