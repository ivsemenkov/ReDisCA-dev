# MEG track (Kozunov et al.)

Owner: MEG worker. Do not edit `paper/reproduction/common/`.

Reproduce the paper MEG analyses and a separately labeled AIRI-executable path.
Do not mix paper-described and AIRI-executable settings.

```bash
python paper/reproduction/common/download_osf.py meg-sensor
python paper/reproduction/meg/run.py
```

Canonical deterministic fits use `from redisca import ReDisCA`. The AIRI path
uses the source-faithful reconstruction. Compact metrics go to `paper/results/meg/`.
