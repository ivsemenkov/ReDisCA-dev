# Track D: N170 RDM correlation (Fig. 10 corr=0.82)

Do not edit `paper/reproduction/n170/run.py` or `TRACK_REPORT.md`.

```bash
python3 -m pytest paper/reproduction/n170/rdm_correlation/test_rdm_correlation.py -q
python3 paper/reproduction/n170/rdm_correlation/run.py
```

Outputs: `paper/results/n170/rdm_correlation/` and
`paper/reproduction/n170/RDM_CORR_REPORT.md`.
