"""Run timeframe-specific training scripts for TFT experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

TRAIN_SCRIPTS = {
    "15min": WORKSPACE_ROOT / "experiments/experiment_15min_op_v3/train_15min_op_v3.py",
    "4hr": WORKSPACE_ROOT / "experiments/experiment_4hr_op_v3/train_4hr_op_v3.py",
    "1day": WORKSPACE_ROOT / "experiments/experiment_1day_op_v3/train_1day_op_v3.py",
    "1week": WORKSPACE_ROOT / "experiments/experiment_1week_op_v3/train_1week_op_v3.py",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeframe", choices=list(TRAIN_SCRIPTS.keys()))
    p.add_argument("--all", action="store_true", help="Run all timeframe training scripts sequentially")
    return p.parse_args()


def run_script(script: Path) -> int:
    if not script.exists():
        print(f"Missing training script: {script}")
        return 1
    print(f"\n[RUN] {script}")
    return subprocess.call([sys.executable, str(script)], cwd=str(WORKSPACE_ROOT))


def main() -> None:
    args = parse_args()
    if not args.all and not args.timeframe:
        raise SystemExit("Use --timeframe <tf> or --all")

    targets = list(TRAIN_SCRIPTS.values()) if args.all else [TRAIN_SCRIPTS[args.timeframe]]

    for script in targets:
        rc = run_script(script)
        if rc != 0:
            raise SystemExit(rc)

    print("\nTraining execution completed.")


if __name__ == "__main__":
    main()
