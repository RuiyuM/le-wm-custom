# Runtime Lessons

## Why Startup Was Slow

- The first delay was environment bring-up, not model training.
- The machine needed:
  repo clone,
  venv creation,
  dependency repair,
  dataset download and extraction,
  cache warmup.
- `datasets==1.1.1` was incompatible with the code path used by `stable_pretraining`; upgrading to a modern `datasets` release was required before training became stable.
- A large single GPU still spent time idle while the dataloaders, caches, and extracted HDF5 dataset settled.

## Issues After Launch

### 1. `fit-pca` failed on the first launch

- Cause:
  `FIT_DEVICE=0` produced `RuntimeError: Invalid device string: '0'`.
- Fix:
  launch with `FIT_DEVICE=cuda:0`.

### 2. `fit-pca` and `eval` used the wrong Python interpreter

- Cause:
  the scheduler resolved the venv symlink to a base interpreter that did not have the project packages.
- Symptoms:
  `ModuleNotFoundError: numpy`
  `ModuleNotFoundError: hydra`
- Fix:
  preserve `PYTHON_BIN=/workspace/le-wm-custom/.venv/bin/python` exactly instead of resolving through the uv install path.

### 3. `fit-pca` underused the GPU

- Cause:
  `fit_sigreg_subspace.py` forced `num_workers=0`, moved embeddings back to CPU with `.detach().cpu()`, and computed covariance and eigendecomposition on CPU.
- Effect:
  CPU-heavy stage, low GPU util, long wall-clock time.
- Fix:
  change the script so `--num-workers` defaults to the loader config rather than hard-forcing `0`.

### 4. Branch runs failed immediately on Hydra override syntax

- Cause:
  the scheduler injected `trainer.default_root_dir=...` for runs whose config did not already contain that key.
- Symptom:
  Hydra complained that `default_root_dir` was not in struct and required `+trainer.default_root_dir=...`.
- Fix:
  use `+trainer.default_root_dir=...`.

### 5. Single-GPU branch concurrency caused OOM

- Cause:
  several warmups, one eval, and multiple branch jobs each consumed roughly `14 GiB`, so launching too many at once exhausted memory.
- Observed victims:
  `tworoom_rank_seed42_pct05_pca-r8_e20`
  `tworoom_rank_seed42_pct05_random-r16_e20`
- Fixes:
  add explicit GPU memory headroom,
  requeue OOM jobs with cooldown,
  stop trusting one-shot utilization samples,
  reserve capacity when the scheduler decides to launch a new job.

### 6. Eval was much slower than expected

- Cause:
  TwoRoom eval uses CEM planning with `num_eval=50` and sparse progress logging.
- Observed clue:
  one `CEM solve time` already took roughly `915` seconds.
- Operational implication:
  eval must be treated as a long GPU consumer and git-persisted immediately on completion.

### 7. Watcher and watchdog management needed hardening

- Cause:
  background process launch methods were fragile and could leave stale or duplicate watchers.
- Fixes:
  adopt live jobs from PID files and outputs,
  keep a separate watchdog loop,
  verify the state file freshness,
  restart the watcher if it dies,
  keep watcher and watchdog launch commands simple and explicit.

### 8. Local `full` branches were unnecessary once an external baseline existed

- Cause:
  the `full` control runs had already completed on another server.
- Operational implication:
  keeping `full` in the local queue would only burn GPU time and slow the rank-ablation branches that still needed this machine.
- Fix:
  remove `full` from the local scheduler variants and keep only `pca-r4`, `pca-r8`, `pca-r16`, `pca-r32`, and `random-r16`.

## What Worked

- Reusing fixed subset indices kept the ablation protocol clean.
- Syncing `fit-pca` and `make-random` artifacts into git made cross-server reuse possible.
- Keeping eval outputs under version control reduced risk from server disconnects.
- TMux plus per-run log files made recovery much easier after watcher restarts.

## Recommended Defaults For Future Runs

- Force `PYTHON_BIN` to the repo venv.
- Force `FIT_DEVICE=cuda:0`.
- Use `+trainer.default_root_dir=...` per run.
- Keep an explicit GPU headroom budget.
- Add OOM cooldown and automatic requeue.
- Push after every eval and after every reusable artifact.
- Assume startup will be bottlenecked by environment readiness before GPU throughput.
