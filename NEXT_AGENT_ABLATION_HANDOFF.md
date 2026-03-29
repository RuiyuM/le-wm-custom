# Next Agent Ablation Handoff

Last updated: 2026-03-29 18:16 CDT

## What The Next Ablation Is

The next planned ablation is:

**subset-specific low-rank SIGReg on TwoRoom**

We want to test whether low-data `TwoRoom` subsets perform better when SIGReg is applied only inside a low-dimensional supported subspace instead of the full `192`-dimensional latent space.

This is **not yet** the full paper method. It is the first clean ablation toward:

- coverage-gated Gaussianization
- supported-subspace Gaussianization
- low-data / low-coverage world model regularization

## Key Research Claim Being Tested

In low-data / low-coverage offline world models, **full-rank Gaussianization may mismatch the transition geometry actually supported by the data**.

The near-term goal is to test this claim in the simplest possible way:

- keep LeWM architecture unchanged
- keep prediction loss unchanged
- change only the dimensionality of the subspace regularized by SIGReg

## Do Not Confuse This With

The next ablation is **not**:

- changing latent width
- changing predictor width
- changing JEPA architecture
- adding online curation
- adding dynamic rank schedules
- adding local intrinsic dimension estimation

It is specifically:

**full model, low-rank regularizer**

## Current Preferred V1 Procedure

For each subset budget independently:

1. choose subset (`5%`, `10%`, `15%`, `20%`)
2. train a shared warmup checkpoint on that subset with the original LeWM objective
3. estimate a PCA subspace from that subset using the warmup checkpoint from the same budget
4. save a fixed `mean + ordered basis family`
5. fork multiple rank runs from the same warmup checkpoint
6. apply SIGReg only on the projected latent in the branch runs

This avoids using a random encoder to define the subspace while also avoiding
the coordinate-system bug from cross-model basis transfer.

## Important Correction: Basis Must Match The Encoder Coordinates

Do **not** use this protocol for the main ablation:

1. estimate `Q` from the official full-data checkpoint
2. start a fresh random model from scratch
3. regularize that fresh model with `Q`

That procedure is not clean because PCA directions depend on the concrete
encoder/projector latent coordinate system that produced them.

The correct main-result protocol is:

- warmup on the target budget
- estimate `mu_b, Q_b` from that warmup
- branch all rank runs from the same warmup checkpoint

The official full-data checkpoint may still be used for quick exploratory
pilots, but not for the main paper table.

## Current Repo State

The repo already contains:

- subset-aware training support in `train.py`
- `TwoRoom` subset experiment helpers
- official checkpoint conversion helper
- current local experiment context in `AGENT_CONTEXT.md`

See also:

- `AGENT_CONTEXT.md`
- `RANK_SIGREG_MINIMAL_CHANGE_PLAN.md`

## Current Running Experiments

There are still local random-subset `TwoRoom` runs in progress for:

- `5%`
- `10%`
- `15%`
- `20%`

These are being stopped at epoch `10` by a background watcher.

This is a separate minimum-validation pass and should not be confused with the future rank-SIGReg ablations.

## Exact Desired Minimal Code Direction

The smallest future code path should do the following:

- add a non-trainable projector module
- load a saved PCA basis from disk
- resume branch runs from a shared warmup checkpoint
- project only the latent tensor used by SIGReg
- leave `pred_loss` on full `192` dimensions

The exact detailed plan is in:

- `RANK_SIGREG_MINIMAL_CHANGE_PLAN.md`

## Recommended Initial Rank Grid

Suggested first-pass grid after the shared warmup:

- `5%`: `2, 4, 8, 16, 32`
- `10%`: `4, 8, 16, 32, 64`
- `15%`: `4, 8, 16, 32, 64`
- `20%`: `8, 16, 32, 64, 96`

Always compare against the full-rank LeWM baseline for the same subset.
Useful first control:

- `random-r16` from the same warmup and same subset

Possible later control:

- `full-weaklambda`

## What To Look For

The desired early signal is:

- low-data `TwoRoom` planning improves for some intermediate rank
- the gain comes without shrinking the model itself
- the result supports the claim that the issue is regularization mismatch, not necessarily missing state information

## If You Need To Continue The Project On Another Server

Priority order:

1. read `AGENT_CONTEXT.md`
2. read `RANK_SIGREG_MINIMAL_CHANGE_PLAN.md`
3. verify the current branch and subset-support changes in `train.py` / `utils.py`
4. implement shared-warmup fixed-subspace rank-SIGReg before touching curriculum rank or data curation

Do not let another agent revert to the older “official checkpoint basis plus
from-scratch rank runs” protocol. That is the main design error already
identified.

## Important Framing

When describing the project, prefer:

- `coverage-gated Gaussianization`
- `supported-subspace Gaussianization`
- `low-data / low-coverage regularizer mismatch`

Avoid describing the project as only:

- `changing the Gaussian prior`
- `generic intrinsic-dimension-aware JEPA`

The scientific target is specifically **world model regularization under low coverage**.
