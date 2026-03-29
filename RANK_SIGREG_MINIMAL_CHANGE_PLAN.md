# Rank-SIGReg Minimal Change Plan

Last updated: 2026-03-29 17:31 CDT

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

This isolates the effect of **full-rank vs low-rank Gaussianization** without changing model capacity.

## Recommended V1 Protocol

### Per-subset fixed subspace

For each subset budget independently:

1. Choose a subset of episodes using the existing subset mechanism.
2. Use a **frozen reference encoder** to collect latent embeddings on that subset.
3. Fit PCA on those latents and extract a rank-`r` basis.
4. Train LeWM from scratch on that subset, but apply SIGReg only to the projected latents.

### Recommended reference encoder

Use the official full-data `TwoRoom` LeWM checkpoint as the encoder used to estimate the PCA subspace:

- checkpoint: `/data/rxm210041/stablewm/tworoom/lewm_hf/lewm_hf_object.ckpt`

Why this is the preferred V1:

- avoids noisy PCA from a random encoder
- avoids two-phase warmup/resume logic inside the training loop
- gives stable, subset-specific supported subspaces
- minimizes changes to the current training code

## What Not To Change

Do **not** do any of the following in V1:

- do not change latent dimensionality from `192`
- do not change predictor output dimensionality
- do not change `pred_loss`
- do not rewrite `SIGReg`
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

### 3. New script: `scripts/fit_sigreg_subspace.py`

Add a preprocessing script that:

- loads a checkpoint used as the reference encoder
- builds the requested subset using the existing subset logic
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

The important point is that each run should independently specify:

- subset fraction
- subset seed
- subspace rank
- path to the saved PCA basis file
- output subdir

## Exact Proposed Training Semantics

For a run with subset fraction `f` and rank `r`:

1. Build subset `S_f`
2. Precompute `Q_r` and `mu` from `S_f` using the official checkpoint encoder
3. Train LeWM from scratch on `S_f`
4. Compute:

- `pred_loss` on full `192` dimensions
- `sigreg_loss` on projected `r`-dimensional latents only

This is the smallest change that directly tests the scientific question:

**Does low-rank Gaussianization improve low-data planning without shrinking model capacity?**

## First Ablation Family To Run

Keep it simple first.

For each subset in `{5, 10, 15, 20}%`, run a rank grid.

Recommended initial rank grid:

- `5%`: `r in {2, 4, 8, 16, 32}`
- `10%`: `r in {4, 8, 16, 32, 64}`
- `15%`: `r in {4, 8, 16, 32, 64}`
- `20%`: `r in {8, 16, 32, 64, 96}`

Always include the existing full-rank LeWM baseline for the same subset.

## Why This Is The Minimal Change

This plan avoids:

- changing the model architecture
- changing latent capacity
- changing the prediction path
- changing the official SIGReg implementation
- introducing unstable in-training PCA updates

It only changes **where the regularizer acts**.

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
- subset-specific latent covariance spectrum
- whether `pred_loss` stays stable
- whether rank is too small / too large

Only after fixed-subspace ablations show signal should the project move toward curriculum rank or curation.
