## TwoRoom Random Subset Indices

This directory stores the exact episode-index subsets used for the current
TwoRoom low-data experiments, so the same subsets can be reproduced on another
machine without depending on `/data/rxm210041/stablewm/...`.

### Current bundle

- `seed42/pct05_episode_indices.npy`: 500 episode indices for the `5%` run
- `seed42/pct10_episode_indices.npy`: 1000 episode indices for the `10%` run
- `seed42/pct15_episode_indices.npy`: 1500 episode indices for the `15%` run
- `seed42/pct20_episode_indices.npy`: 2000 episode indices for the `20%` run

### Provenance

- Dataset: `TwoRoom`
- Sampling mode: random episode subset
- Subset seed: `42`
- Original run directories:
  - `tworoom_pct05`
  - `tworoom_pct10`
  - `tworoom_pct15`
  - `tworoom_pct20`
- Source of truth at training time: `train.py` writes
  `subset_episode_indices.npy` into each run directory whenever
  `selected_episodes` is active.

### Usage

These files are plain NumPy arrays of episode indices. Future agents can load
them directly and either:

1. rebuild the exact subset HDF5 files, or
2. inject the indices into subset-selection logic for reproducible training.

### Important assumption

These indices are only reproducible if the underlying TwoRoom episode ordering
matches the dataset ordering used for the current experiments.
