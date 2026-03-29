# Rank-SIGReg Minimal Change Plan

Last updated: 2026-03-29 18:16 CDT

## Goal

Run a new ablation family on `TwoRoom` where each low-data subset (`5%`, `10%`, `15%`, `20%`) is trained with **SIGReg applied only inside a low-dimensional supported subspace**, while leaving the rest of LeWM unchanged.

This is a **fixed-subspace / fixed-rank** ablation family, not yet the full coverage-gated curriculum paper.

## Core Experimental Interpretation

The intended interpretation is:

- keep the LeWM latent width at `192`
- keep the predictor and prediction target at full `192` dimensions
- do **not** shrink the model itself
- only shrink the **geometry seen by SIGReg**

Formally:

- `pred_loss` stays on full `192`-dimensional latent predictions
- `sigreg_loss` is computed on `Q_r^T (z - mu)` where `Q_r` is a fixed `192 x r` basis estimated from the current subset
- the basis must be estimated from the **same encoder coordinate system** used to start the rank-sweep runs

This isolates the effect of **full-rank vs low-rank Gaussianization** without changing model capacity.

## Recommended V1 Protocol

### Per-budget warmup plus branch

For each subset budget independently:

1. Choose a subset of episodes using the existing subset mechanism.
2. Train a **common warmup checkpoint** on that subset using the original full-rank LeWM objective.
3. Use that warmup checkpoint to collect latent embeddings on the same subset.
4. Fit PCA on those warmup latents and save an ordered basis family:
   - `mu_b`
   - `Q_b[:, :4]`
   - `Q_b[:, :8]`
   - `Q_b[:, :16]`
   - ...
5. Resume from the **same warmup checkpoint** and fork multiple runs:
   - `full`
   - `pca-r4`
   - `pca-r8`
   - `pca-r16`
   - `pca-r32`
   - optional `random-r16`

This is the preferred V1 because it keeps the PCA basis aligned with the
encoder/projector coordinate system that is actually being regularized.

### Why this correction matters

Do **not** estimate a PCA basis from one encoder and then apply it to a
different randomly initialized model from scratch.

Reason:

- PCA basis directions are not dataset-intrinsic coordinates
- they depend on the concrete encoder/projector latent coordinate system
- using an official full-data checkpoint to estimate `Q_b` and then regularizing
  a fresh random model with that basis creates a coordinate-system mismatch

That official-checkpoint path is acceptable only as an **exploratory pilot** to
see whether an intermediate-rank effect might exist. It is **not** the main
paper protocol.

### Warmup-anchored interpretation

Even in the corrected V1, `Q_b` is anchored to the warmup checkpoint.
After branching, the encoder continues to update, so this is best described as:

- **warmup-anchored supported-subspace regularization**

This is acceptable for V1 and much cleaner than cross-model basis transfer or
online PCA refresh.

## What Not To Change

Do **not** do any of the following in V1:

- do not change latent dimensionality from `192`
- do not change predictor output dimensionality
- do not change `pred_loss`
- do not rewrite `SIGReg`
- do not apply a basis estimated from one encoder to a different fresh model
- do not update PCA every step
- do not add local-rank or mixture-of-subspaces logic
- do not mix in data curation yet
- do not add residual penalties by default

## Minimal Repo Changes

### 1. `module.py`

Add a small projector module, e.g. `SupportSubspaceProjector`, with:

- buffers:
  - `mean: [D]`
  - `basis: [D, r]`
- forward:
  - input: `(..., D)`
  - output: `(..., r)` via `(z - mean) @ basis`

This module should be simple and non-trainable by default.

No changes are required inside `SIGReg` itself.

### 2. `train.py`

Minimal changes:

- extend config parsing to support an optional `subspace` block
- instantiate a projector only when `subspace.enabled=True`
- load `mean` and `basis` from a saved `.pt` file
- support resuming/forking from a warmup checkpoint
- change the SIGReg call path from:

```python
self.sigreg(emb.transpose(0, 1))
```

to:

```python
emb_reg = emb.transpose(0, 1)
if projector is not None:
    emb_reg = projector(emb_reg)
self.sigreg(emb_reg)
```

Important:

- only the regularization path is projected
- `pred_emb`, `tgt_emb`, and `pred_loss` stay full-dimensional
- the `basis` file used by a run must match that run's warmup checkpoint family

### 3. New script: `scripts/fit_sigreg_subspace.py`

Add a preprocessing script that:

- loads a specified warmup checkpoint
- rebuilds the requested subset using the existing subset logic
- runs the encoder over a configurable number of batches
- collects latents as `[N, 192]`
- computes PCA
- saves:
  - `mean`
  - `basis`
  - `rank`
  - metadata such as subset fraction / seed / checkpoint path

Recommended output format:

- `torch.save({...}, output_path)`

Recommended saved keys:

- `mean`
- `basis`
- `rank`
- `latent_dim`
- `subset_fraction`
- `subset_seed`
- `source_checkpoint`
- `num_latents_used`
- `explained_variance`

The important semantic requirement is:

- `source_checkpoint` must be the **warmup checkpoint from the same budget**
- this script should not default to the official full-data checkpoint for main experiments

### 4. New config group: `config/train/subspace/`

Add:

- `config/train/subspace/off.yaml`
- `config/train/subspace/pca_fixed.yaml`

Recommended fields:

```yaml
enabled: false
rank: null
basis_path: null
center: true
mode: pca_fixed
```

and for `pca_fixed.yaml`:

```yaml
enabled: true
rank: 16
basis_path: ???
center: true
mode: pca_fixed
```

### 5. Optional helper script: `scripts/run_tworoom_rank_ablation.sh`

Not strictly required, but useful for batching commands.

The important point is that each budget should run as:

1. warmup
2. basis fit
3. branch runs from the same warmup

Each branch run should independently specify:

- subset fraction
- subset seed
- subspace rank
- path to the saved PCA basis file
- output subdir
- warmup checkpoint path

## Exact Proposed Training Semantics

For a run with subset fraction `f` and rank `r`:

1. Build subset `S_f`
2. Train a common warmup checkpoint `W_f` on `S_f`
3. Precompute `Q_f` and `mu_f` from `S_f` using `W_f`
4. Fork branch run `R_{f,r}` from `W_f`
5. Compute:

- `pred_loss` on full `192` dimensions
- `sigreg_loss` on projected `r`-dimensional latents only using `Q_f[:, :r]`

This is the smallest change that directly tests the scientific question:

**Does low-rank Gaussianization improve low-data planning without shrinking model capacity?**

Within one budget, all rank branches should share the same ordered PCA family.
Across budgets, estimate separate basis families:

- `5%`: `mu_5`, `Q_5`
- `10%`: `mu_10`, `Q_10`
- `15%`: `mu_15`, `Q_15`
- `20%`: `mu_20`, `Q_20`

## First Ablation Family To Run

Keep it simple first.

For each subset in `{5, 10, 15, 20}%`, run a rank grid from a shared warmup.

Recommended initial rank grid:

- `5%`: `r in {2, 4, 8, 16, 32}`
- `10%`: `r in {4, 8, 16, 32, 64}`
- `15%`: `r in {4, 8, 16, 32, 64}`
- `20%`: `r in {8, 16, 32, 64, 96}`

Always include the existing full-rank LeWM baseline for the same subset.
For the first informative control, add:

- `random-r16` from the same warmup and same subset

Later, if needed to answer the “did you just weaken regularization?” objection,
add:

- `full-weaklambda`

## Why This Is The Minimal Change

This plan avoids:

- changing the model architecture
- changing latent capacity
- changing the prediction path
- changing the official SIGReg implementation
- introducing unstable in-training PCA updates

It only changes **where the regularizer acts**.

The only extra machinery beyond that is the shared warmup needed to keep the
subspace aligned with the model coordinates.

## Deferred Extensions

These are intentionally postponed until the fixed-rank story works:

- coverage-gated rank growth inside a run
- count-based vs coverage-based rank schedules
- residual unsupported-direction penalties
- dynamic data curation
- local / mixture-of-subspaces regularization

## Practical Note

If the first rank grid does **not** improve `TwoRoom` planning at low data, do not immediately add more complexity. First verify:

- PCA basis quality
- warmup/basis coordinate alignment
- subset-specific latent covariance spectrum
- whether `pred_loss` stays stable
- whether rank is too small / too large

Only after fixed-subspace ablations show signal should the project move toward curriculum rank or curation.
