# Common reproduction harness

Orchestrator-owned. Track workers should request helpers rather than editing
this directory.

Contents:

- `paths.py`, `rng.py`, `hashing.py`, `serialize.py`, `metrics.py`, `provenance.py`
- `source_faithful.py` — independent AIRI/SPoC reconstruction (does not import `redisca`)
- `download_osf.py` — OSF MEG / source-model download with SHA-256 checks
- `download_erpcore.py` — ERP CORE subject-1 processed ERP download
- `meg_io.py`, `erp_io.py` — loaders for cached MEG and ERPLAB files
- `tests/` — unit tests for the reconstruction and comparison metrics

```bash
python -m pytest paper/reproduction/common/tests -q
```
