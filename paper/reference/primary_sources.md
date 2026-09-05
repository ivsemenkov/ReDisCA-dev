# Primary sources inspected for Stage A

Inspected before writing or running reproduction code.

1. Published paper: Ossadtchi et al., NeuroImage 301 (2024) 120868
   (open PDF: https://megmoscow.ru/wp-content/uploads/pubs/10.1016_j.neuroimage.2024.120868.pdf)
2. bioRxiv preprint 10.1101/2024.02.01.578343 (SNR overlays for Figs 4–5)
3. AIRI executable repository @ `15bc19cdc76989da202714b257f6de4d26a42c51`
   - `Redisca_tools_faces_3_random_norm_correct.m`
   - `Redisca_source_loc_for_tools_faces_3_random_.m`
4. Stock SPoC @ `18e4754aec1411160fd5b7ef0db852f1e0a87d90`
   - `SPoC/spoc.m`, `utils/whiten_data.m`, `utils/create_Cxxz.m`, `utils/random_phase_surrogate.m`
5. ERP CORE @ `c18b43d70d791ca914d90410afe4ff06d6f7f429`
   - N170 Scripts 4 and 7 (ICA removal; unfiltered vs 20 Hz LP averages)
6. OSF datasets: https://osf.io/pfde9/ (N170), https://osf.io/8rk67/ (MEG / source models)

Old `paper` branch `e54dd260` was used as source material for loaders, hashes,
and forensic notes only. `source_faithful.py` is not an experiment dependency.
