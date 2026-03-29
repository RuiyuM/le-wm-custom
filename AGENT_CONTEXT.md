# Agent Context

Last updated: 2026-03-29 18:16 CDT

Additional planning docs:

- `RANK_SIGREG_MINIMAL_CHANGE_PLAN.md`
- `NEXT_AGENT_ABLATION_HANDOFF.md`

## What This Project Is Doing

We are using `le-wm-custom` to study **data-efficient LeWorldModel behavior in low-data / low-coverage offline settings**.

The immediate experiment track is:

- Task: `TwoRoom`
- Data budgets: random `5%`, `10%`, `15%`, `20%`
- Baseline: official `TwoRoom` pretrained checkpoint evaluated locally
- Goal: compare low-data subset performance against the official full-data checkpoint result

The current medium-term research direction is:

- Core claim: in **low-data / low-coverage offline world models**, **full-rank Gaussianization can mismatch the transition geometry actually supported by the data**
- Main method direction: **coverage-gated / rank-adaptive SIGReg**
- Important framing: this is **not** just a generic “intrinsic-dimension-aware JEPA regularizer” paper
- Better framing: **supported-subspace Gaussianization for low-data / low-coverage world models**

We are still interested in **data curation**, but it should be treated as a secondary or extension direction unless rank adaptation alone is already strong.

## Current Paper Direction

### Core hypothesis

In low-data / low-coverage offline settings, the available transition data does not support the full ambient latent geometry. Applying SIGReg as a **full-rank isotropic Gaussian prior** can therefore regularize unsupported directions too aggressively and hurt downstream planning.

### Preferred paper framing

- Problem: **regularizer complexity mismatch under low coverage**
- Main method: **coverage-gated rank SIGReg**
- Slogan: **Grow the Support, Grow the Rank**

### What to avoid

- Do **not** frame the paper as only “we changed the Gaussian prior”
- Do **not** over-commit to “intrinsic dimension estimation” unless we truly need that machinery
- Do **not** make the paper depend on data curation if rank adaptation alone is not yet validated

### Recommended phased plan

1. Validate low-data behavior on `TwoRoom` with random subsets.
2. Test **fixed-rank** ablations first using a shared warmup plus branch protocol.
3. Test **count-based rank growth** vs **coverage-based rank growth**.
4. Only then add **coverage-aware data curation** as an extension.

### Important protocol correction for fixed-rank ablations

Do **not** estimate a PCA basis from one encoder and then apply it to a
different randomly initialized model from scratch.

Reason:

- the PCA basis lives in a specific encoder/projector coordinate system
- cross-model basis transfer creates a coordinate mismatch

For the main rank-SIGReg ablations, the correct protocol is:

1. warm up on budget `b`
2. estimate `mu_b, Q_b` from that warmup checkpoint on the same subset
3. fork `full / pca-r4 / pca-r8 / pca-r16 / ...` from that same warmup

The official full-data `TwoRoom` checkpoint can be used for quick exploratory
pilots only, not for the main paper protocol.

## Immediate Experimental Objective

We are currently running local `TwoRoom` subset training jobs and will stop them at **epoch 10** for a fast minimum validation pass.

### Baseline result

Official `TwoRoom` checkpoint local eval:

- File: `/data/rxm210041/stablewm/tworoom/tworoom_results_official_full_50ep.txt`
- `success_rate = 88.0`
- `evaluation_time = 1768.64s` (~29.5 min)

### Running subset jobs

Environment:

- Repo: `/data/rxm210041/le-wm-custom`
- Conda env: `/people/cs/r/rxm210041/.conda/envs/worldmodel121`
- `STABLEWM_HOME=/data/rxm210041/stablewm`
- Dataset: `/data/rxm210041/stablewm/tworoom.h5`
- Official converted checkpoint: `/data/rxm210041/stablewm/tworoom/lewm_hf/lewm_hf_object.ckpt`

Subset runs:

- `pct05`: `/data/rxm210041/stablewm/tworoom_pct05`, PID `1480382`
- `pct10`: `/data/rxm210041/stablewm/tworoom_pct10`, PID `1479145`
- `pct15`: `/data/rxm210041/stablewm/tworoom_pct15`, PID `1479138`
- `pct20`: `/data/rxm210041/stablewm/tworoom_pct20`, PID `1479137`

Reproducibility assets already committed to the repo:

- `repro/tworoom_subset_indices/seed42/pct05_episode_indices.npy`
- `repro/tworoom_subset_indices/seed42/pct10_episode_indices.npy`
- `repro/tworoom_subset_indices/seed42/pct15_episode_indices.npy`
- `repro/tworoom_subset_indices/seed42/pct20_episode_indices.npy`

Subset sizes:

- `5%`: `500` episodes, `37052` clips, ~`260` train steps/epoch
- `10%`: `1000` episodes, `73205` clips, ~`514` train steps/epoch
- `15%`: `1500` episodes, `110439` clips, ~`776` train steps/epoch
- `20%`: `2000` episodes, `146184` clips, ~`1027` train steps/epoch

Observed progress as of this note:

- `pct05`: `epoch 1` checkpoint exists
- `pct10`: `epoch 1` checkpoint exists
- `pct15`: no epoch checkpoint yet
- `pct20`: no epoch checkpoint yet

## Important Runtime Constraints

### Do not restart current runs

These jobs are already running. The current policy is:

- let them continue
- stop them automatically at **epoch 10**
- evaluate the saved model after stopping

### Current background watchers

Checkpoint pruning watcher:

- Script: `/data/rxm210041/le-wm-custom/scripts/prune_object_ckpts.py`
- Behavior: keep only the newest `2` `*_object.ckpt` files per run directory
- Reason: `/data` is nearly full, so we cannot keep all epoch checkpoints

Epoch-stop watcher:

- Script: `/data/rxm210041/le-wm-custom/scripts/stop_at_epoch.py`
- Behavior: when `epoch 10` checkpoint appears, stop the corresponding training PID
- It first sends `SIGTERM`, then `SIGKILL` after a grace period if needed

Do **not** remove these watchers unless you intentionally replace them with something equivalent.

### Why pruning is safe

`stable_worldmodel` loads the **most recent** `*_object.ckpt` from a run directory. Keeping the newest two checkpoints is enough for the current eval workflow.

This is safe for:

- final epoch-10 evaluation
- current low-data comparison experiments

This is **not** safe if you later want exhaustive per-epoch retrospective evaluation of old checkpoints.

## Code Changes Already Made

These local changes are intentional and should not be reverted casually.

### Training / dataset support

- `train.py`
  - added subset support via `+subset.fraction=...`, `+subset.seed=...`, and `+subset.indices_file=...`
  - saves selected episode indices to each run directory
- `utils.py`
  - added `filter_dataset_by_episodes(...)`
  - replaced the previous resize path with a `torchvision.transforms.v2.functional.resize` path to avoid the earlier resize API mismatch

### Utility scripts

- `scripts/convert_hf_tworoom_to_object_ckpt.py`
  - converts the official HF TwoRoom checkpoint into local `*_object.ckpt`
- `scripts/summarize_tworoom_results.py`
  - summarizes subset eval results relative to the full baseline
- `scripts/orchestrate_tworoom_subset_evals.sh`
  - waits for baseline/train completion, runs evals, and writes a summary
- `scripts/prune_object_ckpts.py`
  - keeps only latest object checkpoints
- `scripts/stop_at_epoch.py`
  - stops current jobs once epoch 10 is reached

## Near-Term Research Decision

If another agent is picking up the research direction, the current preferred decision is:

- **Primary contribution**: rank-adaptive / coverage-gated SIGReg
- **Secondary contribution**: data curation, only if the rank story already works

The paper should aim to show:

1. `TwoRoom` low-data failures are not just “representation did not learn state”.
2. Planning can improve when Gaussianization is restricted to the supported subspace.
3. Coverage-gated rank growth is better motivated than count-based rank growth.

## Suggested Next Research Steps

If you are continuing the method work rather than just monitoring jobs, the recommended order is:

1. Finish the current `5/10/15/20%` random-subset epoch-10 validation pass.
2. Implement and test **shared-warmup fixed-rank SIGReg** on `TwoRoom`.
3. Add **coverage-gated rank growth**.
4. Only after that, prototype **coverage-aware subset selection / dynamic curation**.

## Notes For Other Agents

- Keep explanations and edits aligned with the above framing.
- Prefer the phrase **coverage-gated Gaussianization** or **supported-subspace Gaussianization** over generic “change the prior” wording.
- Treat `TwoRoom` as the key low-diversity diagnostic environment.
- Do not assume that better probing implies better planning; this mismatch is part of the actual scientific question.
- Do not revive the older “official checkpoint basis plus from-scratch rank run”
  protocol for the main ablation.
