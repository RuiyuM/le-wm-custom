## TwoRoom Fixed-Subspace Rank-SIGReg Runs

This directory documents the concrete commands for the next `TwoRoom`
rank-only ablation:

- shared full-rank warmup on one fixed subset
- fit one centered PCA basis family from that warmup
- fork `full / pca-r* / random-r16` branch runs from the same warmup

The scripts below assume:

- the repo has been pulled to a machine with the training environment ready
- `TwoRoom` is available under `$STABLEWM_HOME`
- the subset index files under `repro/tworoom_subset_indices/seed42/` should be
  reused as-is

By default, the launcher reads the already-fixed `seed42` subset files for
`pct05 / pct10 / pct15 / pct20`; it does not resample episodes.

### Main launcher

Use:

`scripts/run_tworoom_rank_ablation.sh`

It supports:

- `warmup`
- `fit-pca`
- `make-random`
- `branch`

### Example: `pct05`

1. Train the shared warmup to `epoch 10` on GPU `0`:

```bash
bash scripts/run_tworoom_rank_ablation.sh warmup pct05 0 10
```

Default warmup run dir:

- `$STABLEWM_HOME/tworoom_rank_seed42_pct05_warmup`

2. Fit the ordered PCA basis family from that warmup:

```bash
bash scripts/run_tworoom_rank_ablation.sh fit-pca pct05 tworoom_rank_seed42_pct05_warmup
```

Default PCA artifact:

- `$STABLEWM_HOME/tworoom_rank_subspaces/seed42/pct05_warmup_pca.pt`

3. Build the random control artifact:

```bash
bash scripts/run_tworoom_rank_ablation.sh make-random pct05
```

Default random artifact:

- `$STABLEWM_HOME/tworoom_rank_subspaces/seed42/pct05_warmup_random_seed0.pt`

4. Fork branch runs from the same warmup:

```bash
bash scripts/run_tworoom_rank_ablation.sh branch pct05 full 0 tworoom_rank_seed42_pct05_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct05 pca-r4 1 tworoom_rank_seed42_pct05_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct05 pca-r8 2 tworoom_rank_seed42_pct05_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct05 pca-r16 3 tworoom_rank_seed42_pct05_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct05 pca-r32 0 tworoom_rank_seed42_pct05_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct05 random-r16 1 tworoom_rank_seed42_pct05_warmup 20
```

These commands keep:

- latent width fixed at `192`
- `pred_loss` unchanged
- only the SIGReg path projected

### Other budgets

Repeat the same pattern for:

- `pct10`
- `pct15`
- `pct20`

using the corresponding subset indices from:

- `repro/tworoom_subset_indices/seed42/pct10_episode_indices.npy`
- `repro/tworoom_subset_indices/seed42/pct15_episode_indices.npy`
- `repro/tworoom_subset_indices/seed42/pct20_episode_indices.npy`

### CPU smoke tests

The launcher also accepts `cpu` as the device value. For example:

```bash
bash scripts/run_tworoom_rank_ablation.sh branch pct05 pca-r4 cpu smoke_ranksig_warmup 3 smoke_ranksig_branch_script
```

When `device=cpu`, the launcher automatically overrides:

- `trainer.accelerator=cpu`
- `trainer.devices=1`
- `trainer.precision=32-true`
- `num_workers=0`

### Important protocol constraints

- Do **not** use a basis estimated from one encoder to regularize a fresh random
  model.
- Do **not** compare branch runs started from different warmup checkpoints
  within the same budget.
- Do **not** change latent width or predictor width for this ablation.
- Do **not** refresh PCA online during training in V1.
