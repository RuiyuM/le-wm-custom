#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
STABLEWM_HOME = Path(os.environ.get("STABLEWM_HOME", "/workspace/stablewm")).resolve()
PYTHON_BIN = Path(
    os.environ.get("PYTHON_BIN", str(REPO_ROOT / ".venv" / "bin" / "python"))
)
RUN_LOGGED = SCRIPT_DIR / "run_logged.sh"
RUNNER = REPO_ROOT / "scripts" / "run_tworoom_rank_ablation.sh"
EVALUATE = SCRIPT_DIR / "evaluate_tworoom_run.sh"
LOG_DIR = STABLEWM_HOME / "logs" / "tworoom_rank_ablation"
STATE_PATH = LOG_DIR / "pipeline_state.json"
QUEUE_PATH = LOG_DIR / "pipeline_queue.tsv"
RESULTS_ROOT = REPO_ROOT / "repro" / "tworoom_rank_ablation" / "results" / "seed42"

GPU_MAX_JOBS = int(os.environ.get("GPU_MAX_JOBS", "8"))
GPU_LAUNCH_UTIL_MAX = int(os.environ.get("GPU_LAUNCH_UTIL_MAX", "90"))
GPU_MEMORY_HEADROOM_MIB = int(os.environ.get("GPU_MEMORY_HEADROOM_MIB", str(6 * 1024)))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "20"))
GIT_SYNC_ENABLED = os.environ.get("GIT_SYNC_ENABLED", "1") == "1"
GITHUB_USER = os.environ.get("GITHUB_USER", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GIT_PUSH_RETRIES = int(os.environ.get("GIT_PUSH_RETRIES", "5"))
RETRY_COOLDOWN_SECONDS = int(os.environ.get("RETRY_COOLDOWN_SECONDS", "180"))

OOM_PATTERNS = (
    re.compile(r"cuda out of memory", re.IGNORECASE),
    re.compile(r"torch\\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"\\boutofmemoryerror\\b", re.IGNORECASE),
)

BUDGETS = ("pct05", "pct10", "pct15", "pct20")
BASE_VARIANTS = ("pca-r4", "pca-r8", "pca-r16", "pca-r32", "random-r16")
EXTRA_VARIANTS_BY_BUDGET = {
    "pct05": ("pca-r64", "pca-r128"),
}


@dataclass
class Job:
    label: str
    stage: str
    command: list[str]
    deps: list[str] = field(default_factory=list)
    budget: str | None = None
    variant: str | None = None
    run_name: str | None = None
    needs_gpu: bool = False
    priority: int = 0
    status: str = "pending"
    pid: int | None = None
    returncode: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    adopted: bool = False
    retry_count: int = 0
    cooldown_until: float | None = None
    failure_reason: str | None = None

    @property
    def pid_path(self) -> Path:
        return LOG_DIR / f"{self.label}.pid"

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"{self.label}.log"

    def output_path(self) -> Path | None:
        if self.stage == "fit-pca" and self.budget:
            return (
                STABLEWM_HOME
                / "tworoom_rank_subspaces"
                / "seed42"
                / f"{self.budget}_warmup_pca.pt"
            )
        if self.stage == "make-random" and self.budget:
            return (
                STABLEWM_HOME
                / "tworoom_rank_subspaces"
                / "seed42"
                / f"{self.budget}_warmup_random_seed0.pt"
            )
        if self.stage.startswith("eval") and self.run_name:
            return STABLEWM_HOME / self.run_name / "eval_results.txt"
        return None


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def warmup_run_name(budget: str) -> str:
    return f"tworoom_rank_seed42_{budget}_warmup"


def branch_run_name(budget: str, variant: str) -> str:
    return f"tworoom_rank_seed42_{budget}_{variant}_e20"


def variants_for_budget(budget: str) -> tuple[str, ...]:
    return BASE_VARIANTS + EXTRA_VARIANTS_BY_BUDGET.get(budget, ())


def train_extra_args_for(run_name: str) -> str:
    return f"+trainer.default_root_dir={STABLEWM_HOME / run_name}"


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def run_dir(run_name: str) -> Path:
    return STABLEWM_HOME / run_name


def has_epoch_object(run_name: str, epoch: int) -> bool:
    return any(run_dir(run_name).glob(f"*_epoch_{epoch}_object.ckpt"))


def has_any_object(run_name: str) -> bool:
    return any(run_dir(run_name).glob("*_epoch_*_object.ckpt"))


def weights_ckpt_exists(run_name: str) -> bool:
    return (run_dir(run_name) / "lewm_weights.ckpt").exists()


def job_complete(job: Job) -> bool:
    if job.stage == "warmup":
        if not job.run_name:
            return False
        return has_epoch_object(job.run_name, 10) and weights_ckpt_exists(job.run_name) and not pid_alive(job.pid)
    if job.stage == "branch":
        if not job.run_name:
            return False
        return has_epoch_object(job.run_name, 20) and weights_ckpt_exists(job.run_name) and not pid_alive(job.pid)
    if job.stage.startswith("eval"):
        output = job.output_path()
        return bool(output and output.exists())
    output = job.output_path()
    return bool(output and output.exists())


def command_string(command: list[str]) -> str:
    return " ".join(command)


def build_jobs() -> dict[str, Job]:
    jobs: dict[str, Job] = {}
    for budget in BUDGETS:
        budget_bonus = 20 if budget == "pct05" else 0
        warmup_run = warmup_run_name(budget)
        warmup_label = warmup_run
        jobs[warmup_label] = Job(
            label=warmup_label,
            stage="warmup",
            budget=budget,
            run_name=warmup_run,
            command=[
                "env",
                f"TRAIN_EXTRA_ARGS={train_extra_args_for(warmup_run)}",
                "bash",
                str(RUNNER),
                "warmup",
                budget,
                "0",
                "10",
            ],
            needs_gpu=True,
            priority=100 + budget_bonus,
        )
        fit_label = f"{budget}_fit-pca"
        jobs[fit_label] = Job(
            label=fit_label,
            stage="fit-pca",
            budget=budget,
            command=[
                "env",
                "FIT_DEVICE=cuda:0",
                "bash",
                str(RUNNER),
                "fit-pca",
                budget,
                warmup_run,
            ],
            deps=[warmup_label],
            needs_gpu=True,
            priority=90 + budget_bonus,
        )
        random_label = f"{budget}_make-random"
        jobs[random_label] = Job(
            label=random_label,
            stage="make-random",
            budget=budget,
            command=[
                "bash",
                str(RUNNER),
                "make-random",
                budget,
            ],
            deps=[fit_label],
            needs_gpu=False,
            priority=85 + budget_bonus,
        )
        warmup_eval_label = f"eval__{warmup_run}"
        jobs[warmup_eval_label] = Job(
            label=warmup_eval_label,
            stage="eval-warmup",
            budget=budget,
            run_name=warmup_run,
            command=[
                "bash",
                str(EVALUATE),
                warmup_run,
                f"{warmup_run}/eval_results.txt",
            ],
            deps=[warmup_label],
            needs_gpu=True,
            priority=55 + budget_bonus,
        )
        for variant in variants_for_budget(budget):
            branch_run = branch_run_name(budget, variant)
            branch_label = branch_run
            jobs[branch_label] = Job(
                label=branch_label,
                stage="branch",
                budget=budget,
                variant=variant,
                run_name=branch_run,
                command=[
                    "env",
                    f"TRAIN_EXTRA_ARGS={train_extra_args_for(branch_run)}",
                    "bash",
                    str(RUNNER),
                    "branch",
                    budget,
                    variant,
                    "0",
                    warmup_run,
                    "20",
                ],
                deps=[random_label],
                needs_gpu=True,
                priority=80 + budget_bonus,
            )
            eval_label = f"eval__{branch_run}"
            jobs[eval_label] = Job(
                label=eval_label,
                stage="eval-branch",
                budget=budget,
                variant=variant,
                run_name=branch_run,
                command=[
                    "bash",
                    str(EVALUATE),
                    branch_run,
                    f"{branch_run}/eval_results.txt",
                ],
                deps=[branch_label],
                needs_gpu=True,
                priority=50 + budget_bonus,
            )
    return jobs


def gpu_stats() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used, free, util = [int(part.strip()) for part in result.stdout.splitlines()[0].split(",")]
    return {
        "memory_used_mib": used,
        "memory_free_mib": free,
        "memory_total_mib": used + free,
        "util_pct": util,
    }


def estimated_job_memory_mib(job: Job) -> int:
    if job.stage == "fit-pca":
        return 4 * 1024
    if job.stage.startswith("eval"):
        return 2 * 1024
    if job.stage in {"warmup", "branch"}:
        return 14 * 1024
    return 0


def min_free_mib(job: Job) -> int:
    if job.stage == "fit-pca":
        return 8 * 1024
    if job.stage.startswith("eval"):
        return 4 * 1024
    return 6 * 1024


def max_eval_parallelism() -> int:
    return 1


def load_existing_state(jobs: dict[str, Job]) -> None:
    for job in jobs.values():
        job.pid = read_pid(job.pid_path)
        if job_complete(job):
            job.status = "completed"
            job.finished_at = time.time()
            continue
        if pid_alive(job.pid):
            job.status = "running"
            job.adopted = True
            job.started_at = time.time()


def write_state(jobs: dict[str, Job], stats: dict[str, int]) -> None:
    payload: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(REPO_ROOT),
        "stablewm_home": str(STABLEWM_HOME),
        "python_bin": str(PYTHON_BIN),
        "gpu_stats": stats,
        "jobs": {
            label: {
                **asdict(job),
                "command_str": command_string(job.command),
                "log_path": str(job.log_path),
                "pid_path": str(job.pid_path),
            }
            for label, job in sorted(jobs.items())
        },
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = ["label\tstage\tbudget\tvariant\tstatus\tpid\tlog_path"]
    for label, job in sorted(jobs.items()):
        lines.append(
            "\t".join(
                [
                    label,
                    job.stage,
                    job.budget or "",
                    job.variant or "",
                    job.status,
                    str(job.pid or ""),
                    str(job.log_path),
                ]
            )
        )
    QUEUE_PATH.write_text("\n".join(lines) + "\n")


def sync_repo_results_snapshot(jobs: dict[str, Job]) -> list[Path]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    state_copy = RESULTS_ROOT / "pipeline_state.json"
    queue_copy = RESULTS_ROOT / "pipeline_queue.tsv"
    eval_index = RESULTS_ROOT / "eval_index.tsv"
    shutil.copy2(STATE_PATH, state_copy)
    shutil.copy2(QUEUE_PATH, queue_copy)

    lines = ["run_name\tstage\tbudget\tvariant\tstatus\trepo_result_path"]
    for job in sorted(jobs.values(), key=lambda item: item.label):
        if not job.stage.startswith("eval"):
            continue
        repo_path = RESULTS_ROOT / (job.run_name or "") / "eval_results.txt"
        lines.append(
            "\t".join(
                [
                    job.run_name or "",
                    job.stage,
                    job.budget or "",
                    job.variant or "",
                    job.status,
                    str(repo_path.relative_to(REPO_ROOT)) if repo_path.exists() else "",
                ]
            )
        )
    eval_index.write_text("\n".join(lines) + "\n")
    return [state_copy, queue_copy, eval_index]


def repo_sync_destination(job: Job) -> Path | None:
    output = job.output_path()
    if not output:
        return None
    if job.stage in {"fit-pca", "make-random"}:
        return RESULTS_ROOT / "artifacts" / output.name
    if job.stage.startswith("eval") and job.run_name:
        return RESULTS_ROOT / job.run_name / "eval_results.txt"
    return None


def github_push_url() -> str | None:
    if not GITHUB_USER or not GITHUB_TOKEN:
        return None
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote = result.stdout.strip()
    if remote.startswith("https://github.com/"):
        slug = remote.removeprefix("https://github.com/")
    elif remote.startswith("git@github.com:"):
        slug = remote.removeprefix("git@github.com:")
    else:
        return None
    if slug.endswith(".git"):
        slug = slug[:-4]
    return f"https://{quote(GITHUB_USER, safe='')}:{quote(GITHUB_TOKEN, safe='')}@github.com/{slug}.git"


def sync_job_output_to_git(job: Job, jobs: dict[str, Job]) -> None:
    if not GIT_SYNC_ENABLED:
        return
    output = job.output_path()
    destination = repo_sync_destination(job)
    if not output or not output.exists() or not destination:
        log(f"Git sync skipped for {job.label}: job output missing")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, destination)
    tracked_paths = [destination, *sync_repo_results_snapshot(jobs)]

    add_cmd = ["git", "-C", str(REPO_ROOT), "add", *[str(path) for path in tracked_paths]]
    subprocess.run(add_cmd, check=True)

    diff_cmd = ["git", "-C", str(REPO_ROOT), "diff", "--cached", "--quiet", "--", *[str(path) for path in tracked_paths]]
    diff = subprocess.run(diff_cmd)
    if diff.returncode == 0:
        log(f"Git sync skipped for {job.label}: no staged changes")
        return
    if diff.returncode != 1:
        raise RuntimeError(f"git diff --cached failed for {job.label} with rc={diff.returncode}")

    subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "config",
            "user.name",
            GITHUB_USER or "codex",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "config",
            "user.email",
            f"{(GITHUB_USER or 'codex')}@users.noreply.github.com",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "commit",
            "--no-gpg-sign",
            "-m",
            f"results: sync output for {job.label}",
            "--",
            *[str(path) for path in tracked_paths],
        ],
        check=True,
    )

    push_url = github_push_url()
    if not push_url:
        log(f"Git sync committed {job.label}, but push skipped because GitHub credentials are unavailable")
        return

    for attempt in range(1, GIT_PUSH_RETRIES + 1):
        push = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "push", push_url, "HEAD:main"]
        )
        if push.returncode == 0:
            log(f"Pushed synced output for {job.label}")
            return
        log(f"Push attempt {attempt}/{GIT_PUSH_RETRIES} failed for {job.label}")
        time.sleep(15)

    log(f"Leaving committed-but-unpushed output for {job.label}; next sync will retry")


def refresh_finished_jobs(
    jobs: dict[str, Job], processes: dict[str, subprocess.Popen[str]]
) -> None:
    for label, process in list(processes.items()):
        job = jobs[label]
        rc = process.poll()
        if rc is None:
            continue
        job.returncode = rc
        job.finished_at = time.time()
        job.pid = read_pid(job.pid_path) or job.pid
        if rc == 0 and job_complete(job):
            job.status = "completed"
            log(f"Completed {label}")
            if job.stage.startswith("eval") or job.stage in {"fit-pca", "make-random"}:
                sync_job_output_to_git(job, jobs)
        else:
            handle_failed_job(job)
        processes.pop(label, None)


def refresh_adopted_jobs(jobs: dict[str, Job]) -> None:
    for job in jobs.values():
        if job.status != "running" or not job.adopted:
            continue
        job.pid = read_pid(job.pid_path) or job.pid
        if pid_alive(job.pid):
            continue
        job.finished_at = time.time()
        if job_complete(job):
            job.status = "completed"
            log(f"Completed adopted job {job.label}")
            if job.stage.startswith("eval") or job.stage in {"fit-pca", "make-random"}:
                sync_job_output_to_git(job, jobs)
        else:
            handle_failed_job(job)


def refresh_completed_outputs(jobs: dict[str, Job]) -> None:
    for job in jobs.values():
        if job.status != "pending":
            continue
        if job_complete(job):
            job.status = "completed"
            job.finished_at = time.time()
            log(f"Detected completed output for pending job {job.label}")
            if job.stage.startswith("eval") or job.stage in {"fit-pca", "make-random"}:
                sync_job_output_to_git(job, jobs)


def log_has_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(errors="ignore")[-20000:]
    except OSError:
        return False
    return any(pattern.search(text) for pattern in OOM_PATTERNS)


def ready_jobs(jobs: dict[str, Job]) -> list[Job]:
    ready: list[Job] = []
    now = time.time()
    for job in jobs.values():
        if job.status != "pending":
            continue
        if job.cooldown_until is not None and job.cooldown_until > now:
            continue
        if all(jobs[dep].status == "completed" for dep in job.deps):
            ready.append(job)
    ready.sort(key=lambda job: (-job.priority, job.label))
    return ready


def active_counts(jobs: dict[str, Job]) -> tuple[int, int]:
    gpu_jobs = 0
    eval_jobs = 0
    for job in jobs.values():
        if job.status != "running":
            continue
        if job.needs_gpu:
            gpu_jobs += 1
        if job.stage.startswith("eval"):
            eval_jobs += 1
    return gpu_jobs, eval_jobs


def launch_job(job: Job, processes: dict[str, subprocess.Popen[str]]) -> None:
    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(STABLEWM_HOME)
    env["PYTHON_BIN"] = str(PYTHON_BIN)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Launching {job.label}: {command_string(job.command)}")
    process = subprocess.Popen(
        ["bash", str(RUN_LOGGED), job.label, *job.command],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    job.status = "running"
    job.started_at = time.time()
    job.pid = process.pid
    job.returncode = None
    job.failure_reason = None
    processes[job.label] = process


def should_launch_gpu_job(job: Job, jobs: dict[str, Job], stats: dict[str, int]) -> bool:
    active_gpu_jobs, active_eval_jobs = active_counts(jobs)
    if active_gpu_jobs >= GPU_MAX_JOBS:
        return False
    if stats["util_pct"] >= GPU_LAUNCH_UTIL_MAX and active_gpu_jobs > 0:
        return False
    if stats["memory_free_mib"] < min_free_mib(job):
        return False
    projected_used = stats["memory_used_mib"] + estimated_job_memory_mib(job)
    if projected_used + GPU_MEMORY_HEADROOM_MIB > stats["memory_total_mib"]:
        return False
    if job.stage.startswith("eval") and active_eval_jobs >= max_eval_parallelism():
        return False
    return True


def reserve_gpu_capacity(stats: dict[str, int], job: Job) -> dict[str, int]:
    reserved = estimated_job_memory_mib(job)
    total = stats["memory_total_mib"]
    used = min(total, stats["memory_used_mib"] + reserved)
    free = max(0, total - used)
    return {
        **stats,
        "memory_used_mib": used,
        "memory_free_mib": free,
    }


def handle_failed_job(job: Job) -> None:
    job.finished_at = time.time()
    if log_has_oom(job.log_path):
        job.status = "pending"
        job.retry_count += 1
        job.cooldown_until = time.time() + RETRY_COOLDOWN_SECONDS
        job.failure_reason = "oom"
        job.pid = None
        log(
            f"OOM for {job.label}; requeueing with cooldown {RETRY_COOLDOWN_SECONDS}s "
            f"(retry {job.retry_count})"
        )
        return
    job.status = "failed"
    job.failure_reason = "failed"
    log(f"Failed {job.label} rc={job.returncode}")


def all_done(jobs: dict[str, Job]) -> bool:
    return all(job.status in {"completed", "failed"} for job in jobs.values())


def install_signal_handlers(processes: dict[str, subprocess.Popen[str]]) -> None:
    def handler(signum: int, _frame: Any) -> None:
        log(f"Received signal {signum}; forwarding SIGTERM to child sessions")
        for process in processes.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    processes: dict[str, subprocess.Popen[str]] = {}
    install_signal_handlers(processes)
    load_existing_state(jobs)
    log(f"repo_root={REPO_ROOT}")
    log(f"stablewm_home={STABLEWM_HOME}")
    log(f"python_bin={PYTHON_BIN}")
    log(f"gpu_max_jobs={GPU_MAX_JOBS}")

    while True:
        stats = gpu_stats()
        refresh_finished_jobs(jobs, processes)
        refresh_adopted_jobs(jobs)
        refresh_completed_outputs(jobs)

        for job in ready_jobs(jobs):
            if job.needs_gpu:
                if should_launch_gpu_job(job, jobs, stats):
                    launch_job(job, processes)
                    stats = reserve_gpu_capacity(stats, job)
            else:
                launch_job(job, processes)

        write_state(jobs, stats)

        if all_done(jobs):
            failures = [job.label for job in jobs.values() if job.status == "failed"]
            if failures:
                log(f"Pipeline finished with failures: {', '.join(failures)}")
                return 1
            log("Pipeline completed successfully")
            return 0

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
