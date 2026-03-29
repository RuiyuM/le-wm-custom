#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def prune_run(run_dir: Path, keep: int) -> int:
    ckpts = sorted(
        run_dir.glob("*_object.ckpt"),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    removed = 0
    for path in ckpts[keep:]:
        try:
            path.unlink()
            print(f"removed {path}")
            removed += 1
        except FileNotFoundError:
            pass
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep only the newest object checkpoints in each run directory."
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories to prune.")
    parser.add_argument("--keep", type=int, default=2, help="How many object checkpoints to keep per run.")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between prune passes while jobs are alive.")
    parser.add_argument("--pid", type=int, action="append", default=[], help="Training PID to watch. Stops when all watched PIDs exit.")
    args = parser.parse_args()

    stop = False

    def _handle_signal(signum, _frame):
        nonlocal stop
        stop = True
        print(f"received signal {signum}, finishing after final prune")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    run_dirs = [Path(p).expanduser().resolve() for p in args.run_dirs]

    while True:
        total_removed = 0
        for run_dir in run_dirs:
            total_removed += prune_run(run_dir, args.keep)

        live_pids = [pid for pid in args.pid if pid_alive(pid)]
        print(
            f"pass complete: removed={total_removed} live_pids={live_pids} "
            f"time={time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if stop or (args.pid and not live_pids):
            break

        if not args.pid:
            break

        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
