# AIRI MATLAB forensic track (sensor-space ReDisCA)

Isolated from the Python library rewrite. Nothing under `src/redisca/` is
imported, modified, or used as an oracle.

Baseline: `ivsemenkov/ReDisCA-dev` @ `32c672f65932773359f2feeaa902782086c63c1d`.

## Stop condition (this environment)

**MATLAB is not installed.** The literal AIRI main script was therefore **not
executed**. GNU Octave was not used as a substitute. Instrumented copies and
runner scripts are ready for a machine that has MATLAB + Signal Processing
Toolbox (`butter`, `filtfilt`).

What **was** completed without MATLAB:

- pinned AIRI sources (byte-identical)
- OSF MEG data download + SHA-256 verification
- SPM/MEG header inspection (trial labels, shapes)
- BBCI vs pinned stock SPoC dependency forensics
- static reconstruction of `D`, directed pair order, and `z` from the MATLAB source
- paper MEG comparison of identifiable quantities
- instrumented MATLAB scripts for later literal execution

## Layout

```
forensics/airi_matlab/
  README.md
  environment.txt
  provenance.json
  original/AIRI-ReDisCA/          # pinned commit 15bc19c, unmodified
  vendor/pinned_stock_SPoC/      # svendaehne/matlab_SPoC @ 18e4754
  vendor/bbci_public_snippets/     # import + proc_spoc wrappers
  instrumented/                  # copies only; original untouched
  scripts/                       # MATLAB runners
  data/                          # OSF MEG files; gitignored; not committed
  paper/                         # published PDF (gitignored) + extracted text
  results/literal/                # static inspection + placeholders
  reports/forensic_report.md
```

## How to run the literal sensor-space pipeline (when MATLAB exists)

1. Place OSF files in `data/` (already hashed in `provenance.json`):
   - `MEG_AD_run1.mat`
   - `ibfctfprespm8_AD_run1_raw_tsss_mc.mat`
2. From MATLAB:

```matlab
FORENSIC_ROOT = '/absolute/path/to/forensics/airi_matlab';
run(fullfile(FORENSIC_ROOT, 'scripts', 'record_matlab_environment.m'));
FORENSIC_SEED = 1;
run(fullfile(FORENSIC_ROOT, 'scripts', 'run_literal_sensor_space.m'));
run(fullfile(FORENSIC_ROOT, 'scripts', 'run_independent_audit.m'));
run(fullfile(FORENSIC_ROOT, 'scripts', 'run_seeds.m'));            % after literal
run(fullfile(FORENSIC_ROOT, 'scripts', 'run_variants.m'));          % after literal
run(fullfile(FORENSIC_ROOT, 'scripts', 'run_order_sensitivity.m')); % diagnostic
```

Do **not** run `Redisca_source_loc_for_tools_faces_3_random_.m` in this stage.

## Two SPoC environments (do not mix)

| Label | What it is |
| --- | --- |
| **literal AIRI/BBCI environment** | AIRI README + `bbci_import_dependencies('ssd+spoc')` downloading `matlab_SPoC` **master.zip**. Cannot be observed here (no MATLAB). Today, `master` == `18e4754`. |
| **pinned stock-SPoC reference environment** | `svendaehne/matlab_SPoC` commit `18e4754aec1411160fd5b7ef0db852f1e0a87d90`, vendored under `vendor/pinned_stock_SPoC/`. This is what the instrumented runners add to the MATLAB path. |

Critical files and hashes: `provenance.json` and `reports/forensic_report.md`.

## Data

Source: https://osf.io/8rk67/  
Do not commit `data/*.mat`. SHA-256 values match the OSF API.

## Isolation rules

- Do not import `redisca`
- Do not modify `src/redisca/`
- Do not port this pipeline to Python as a MATLAB substitute
- Do not “correct” AIRI code to match the paper before first MATLAB execution
