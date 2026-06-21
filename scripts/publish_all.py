from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis step if cards already exist",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Publish DPO detector evasion results and analysis",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = sys.executable

    if not args.skip_analysis:
        run_step("Analyze results", [python, str(root / "scripts/analyze_results.py")])

    run_step("Push dataset to Hugging Face", [python, str(root / "scripts/push_dataset_hf.py")])
    run_step("Push model to Hugging Face", [python, str(root / "scripts/push_model_hf.py")])
    run_step("Push results to GitHub", [python, str(root / "scripts/push_results_github.py"), "--message", args.message])


if __name__ == "__main__":
    main()
