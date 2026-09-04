# Provenance: stock MATLAB SPoC

Do not vendor this repository into git. Pin and clone into
`.reproduction_data/upstream/matlab_SPoC`.

| Field | Value |
| --- | --- |
| URL | https://github.com/svendaehne/matlab_SPoC |
| Pinned commit | `18e4754aec1411160fd5b7ef0db852f1e0a87d90` |
| Commit date | 2016-04-04 12:19:29 +0200 |
| Commit message | replaced zscore with my_zscore |
| Local verification | `git -C .reproduction_data/upstream/matlab_SPoC rev-parse HEAD` |

AIRI README points at `bbci_public/external`. At BBCI commit
`2e6fe9481537dcfee702e74544191dcf737f02ce` (forensics donor;
re-read `misc/bbci_import_dependencies.m` from that snippet), case
`'ssd+spoc'` downloads

```
https://github.com/svendaehne/matlab_SPoC/archive/master.zip
```

and renames `matlab_SPoC-master` → `external/ssd+spoc`.
`origin/master` of matlab_SPoC is currently the pinned commit
above, so a BBCI import **today** is byte-identical to this pin.
A historical AIRI MATLAB session cannot be observed here (no
MATLAB, no pre-imported `external/ssd+spoc`).

Critical file SHA-256s: `paper/reference/source_notes/stock_spoc.md`
and `dependency_pins/pins.json`.

Paper correspondence is Table 1 of Ossadtchi et al. 2024 and Dähne
et al., NeuroImage 86:111–122, 2014.
