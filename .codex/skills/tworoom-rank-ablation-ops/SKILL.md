---
name: tworoom-rank-ablation-ops
description: Run, monitor, recover, and summarize TwoRoom static-rank ablations when warmup, fit-pca, branch training, eval, and git syncing share limited GPU capacity. Use when Codex needs to launch or babysit `scripts/run_tworoom_rank_ablation.sh`, `repro/tworoom_rank_ablation/seed42_commands.sh`, `watch_pipeline.py`, or `watchdog_loop.sh`, especially on a single GPU or any machine where OOM recovery, tmux orchestration, eval persistence, and incremental git pushes matter.
---

# TwoRoom Rank Ablation Ops

## Overview

Use the existing rank-ablation launchers and subset index files. Preserve the experiment protocol unless the user explicitly asks to change it.

Treat this skill as operations guidance, not model-design guidance. Prefer run wrappers, tmux, watchdogs, and git syncing over changing training logic.

## Core Workflow

1. Verify the repo state and protocol.
2. Verify dataset, Python env, and `STABLEWM_HOME`.
3. Start warmups first.
4. Start `fit-pca`, then `make-random`, then branch runs from the same warmup.
5. Keep logs, PIDs, and queue state on disk.
6. Run eval after training and push results immediately.

## Preflight

- Confirm `origin/main` contains the required commit before launching.
- Reuse the fixed subset files under `repro/tworoom_subset_indices/seed42/`.
- Use `scripts/run_tworoom_rank_ablation.sh` as the single launcher.
- If the user already has a `full` baseline from another machine, remove local `full` branches from the queue instead of recomputing them.
- Keep latent width at `192`, keep `pred_loss` full-width, and only project the SIGReg path.
- Do not mix warmup checkpoints within the same budget.

## Startup Guardrails

- Expect the first launch on a fresh machine to be slow because environment setup dominates:
  dataset download or extraction, venv creation, package repair, and cache warmup.
- Prefer tmux windows per role:
  warmup slots, orchestrator, monitor, and watchdog.
- Write stdout and stderr to log files from the first command.
- Record PID files for every launched job.

## Single-GPU Scheduling

- Start conservative. A single large GPU can still OOM if multiple `~14 GiB` training jobs align with eval.
- Warmups can share a GPU, but branch concurrency must leave explicit headroom.
- `fit-pca` is not a GPU-saturating stage. It often looks CPU-bound because it encodes on GPU and then computes PCA statistics on CPU.
- Use an explicit headroom rule instead of trusting a single `nvidia-smi` sample.
- If GPU util is low and ready jobs exist, allow the scheduler to add work only when free memory plus reserved headroom permit it.

## Known Failure Modes

- Wrong interpreter:
  `fit-pca` or `eval` can fail if the launcher resolves to a Python without `numpy` or `hydra`.
  Force the repo venv with `PYTHON_BIN=/path/to/.venv/bin/python`.
- Wrong device string:
  `FIT_DEVICE=0` fails in `fit_sigreg_subspace.py`; use `cuda:0`.
- Hydra override syntax:
  `trainer.default_root_dir=...` fails when the key is absent; use `+trainer.default_root_dir=...`.
- Concurrent Lightning log collisions:
  separate runs need distinct `default_root_dir` values.
- Over-aggressive branch launches:
  `pct05_pca-r8` and `pct05_random-r16` can OOM when warmups, eval, and multiple branches overlap.
- Silent eval slowness:
  TwoRoom eval uses CEM and can run for hours with sparse logging.

For the detailed incident list and remedies, read `references/runtime-lessons.md`.

## Recovery Rules

- Do not kill healthy training jobs just to restart orchestration.
- If a scheduler process dies, restart the scheduler and re-adopt live jobs from PID files and outputs.
- Detect OOM from log text and requeue the failed job with cooldown instead of marking it permanently failed.
- Keep `fit-pca` and `make-random` artifacts under version control or another persistent path so other machines do not recompute them.
- Push eval outputs immediately after completion because the server may disconnect.

## Monitoring Rules

- Keep a machine-readable queue snapshot such as `pipeline_state.json`.
- Record:
  run status, PID, GPU binding, log path, and latest known outputs.
- Use a second watchdog process to:
  confirm the main watcher is alive,
  confirm the state file is fresh,
  warn when ready jobs exist while the GPU stays idle,
  restart the watcher if it dies.
- Treat the watchdog as supervision, not as the primary scheduler.

## Git Persistence

- Copy eval outputs into the repo before committing.
- Commit only files relevant to results or run infrastructure.
- Push after each eval completion.
- Also sync reusable artifacts such as `pct05_warmup_pca.pt` and `pct05_warmup_random_seed0.pt` when other servers can reuse them.

## Files To Know

- `scripts/run_tworoom_rank_ablation.sh`
- `repro/tworoom_rank_ablation/seed42_commands.sh`
- `repro/tworoom_rank_ablation/watch_pipeline.py`
- `repro/tworoom_rank_ablation/watchdog_loop.sh`
- `repro/tworoom_rank_ablation/evaluate_tworoom_run.sh`
- `repro/tworoom_rank_ablation/README.md`
- `scripts/fit_sigreg_subspace.py`

## References

- Read `references/runtime-lessons.md` for the concrete postmortem from the March 30, 2026 run.
