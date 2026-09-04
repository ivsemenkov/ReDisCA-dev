# Source-localization / MUSIC track

Owner: source-localization worker. Do not edit `paper/reproduction/common/`.

Reproduce cosine-similarity localization, representational dissimilarity
subspace MUSIC scans, and MEG cortical maps using the exact forward/source
model assets when they are publicly available.

```bash
python paper/reproduction/common/download_osf.py source-models
python paper/reproduction/source_localization/run.py
```

Do not replace a subject-specific forward model with fsaverage and call that
reproduction. Compact metrics go to `paper/results/source_localization/`.
