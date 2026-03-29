#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path


SUCCESS_RATE_RE = re.compile(r"success_rate':\s*([0-9]+(?:\.[0-9]+)?)")


def extract_success_rate(path: Path) -> float:
    text = path.read_text()
    matches = SUCCESS_RATE_RE.findall(text)
    if not matches:
        raise ValueError(f"Could not find success_rate in {path}")
    return float(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize TwoRoom success rates and relative percentages."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Add a result entry like: --result 5pct /path/to/file.txt",
    )
    args = parser.parse_args()

    baseline = extract_success_rate(args.baseline)
    print(f"{'label':<12} {'success_rate':>12} {'pct_of_full':>12}")
    print("-" * 38)
    print(f"{'full':<12} {baseline:>12.2f} {100.0:>12.2f}")

    for label, raw_path in args.result:
        path = Path(raw_path)
        value = extract_success_rate(path)
        pct_of_full = 0.0 if baseline == 0 else 100.0 * value / baseline
        print(f"{label:<12} {value:>12.2f} {pct_of_full:>12.2f}")


if __name__ == "__main__":
    main()
