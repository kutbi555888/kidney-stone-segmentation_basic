"""Asosiy bosqichlarni ketma-ket run qilish uchun convenience script."""
from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\nRUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    cfg = "configs/config.yaml"
    if not args.skip_download:
        run([py, "src/download_data.py", "--config", cfg])
    run([py, "src/prepare_data.py", "--config", cfg])
    run([py, "src/visualize_sample.py", "--config", cfg])
    train_cmd = [py, "src/train.py", "--config", cfg]
    if args.resume:
        train_cmd.append("--resume")
    run(train_cmd)
    run([py, "src/evaluate.py", "--config", cfg])


if __name__ == "__main__":
    main()
