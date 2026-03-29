#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path


EPOCH_CKPT_RE = re.compile(r".*_epoch_(\d+)_object\.ckpt$")


@dataclass
class Job:
    run_dir: Path
    pid: int
    label: str
    last_seen_epoch: int = -1
    finished: bool = False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def latest_object_epoch(run_dir: Path) -> int:
    best = -1
    for path in run_dir.glob("*_object.ckpt"):
        match = EPOCH_CKPT_RE.fullmatch(path.name)
        if match:
            best = max(best, int(match.group(1)))
    return best


def stop_process(pid: int, label: str, grace_seconds: int) -> str:
    if not pid_alive(pid):
        return "already-exited"

    print(f"[{label}] target reached, sending SIGTERM to pid={pid}")
    os.kill(pid, signal.SIGTERM)

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not pid_alive(pid):
            return "sigterm"
        time.sleep(2)

    if pid_alive(pid):
        print(f"[{label}] pid={pid} still alive after {grace_seconds}s, sending SIGKILL")
        os.kill(pid, signal.SIGKILL)

    deadline = time.time() + 30
    while time.time() < deadline:
        if not pid_alive(pid):
            return "sigkill"
        time.sleep(1)

    return "still-alive"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stop running training jobs once a target epoch checkpoint exists."
    )
    parser.add_argument(
        "--watch",
        nargs=3,
        action="append",
        metavar=("RUN_DIR", "PID", "LABEL"),
        required=True,
        help="Run directory, PID, and short label for a training job.",
    )
    parser.add_argument("--target-epoch", type=int, default=10)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--grace-seconds", type=int, default=120)
    args = parser.parse_args()

    jobs = [
        Job(run_dir=Path(run_dir).expanduser().resolve(), pid=int(pid), label=label)
        for run_dir, pid, label in args.watch
    ]

    print(
        f"watching {len(jobs)} jobs for epoch>={args.target_epoch} "
        f"every {args.interval}s"
    )

    while True:
        active = 0
        for job in jobs:
            if job.finished:
                continue

            max_epoch = latest_object_epoch(job.run_dir)
            if max_epoch != job.last_seen_epoch:
                print(
                    f"[{job.label}] latest_object_epoch={max_epoch} "
                    f"pid_alive={pid_alive(job.pid)}"
                )
                job.last_seen_epoch = max_epoch

            if max_epoch >= args.target_epoch:
                result = stop_process(job.pid, job.label, args.grace_seconds)
                print(f"[{job.label}] stop result={result}")
                job.finished = True
                continue

            if not pid_alive(job.pid):
                print(
                    f"[{job.label}] pid={job.pid} exited before reaching "
                    f"epoch {args.target_epoch}"
                )
                job.finished = True
                continue

            active += 1

        if active == 0:
            print("all watched jobs completed")
            return 0

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
