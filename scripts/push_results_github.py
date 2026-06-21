from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import subprocess
from pathlib import Path


def git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=cwd, check=check, text=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--message",
        type=str,
        default="Ubuntu Jupyter: DPO detector evasion results",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    repo_root = args.repo_root
    github_name = os.environ.get("GITHUB_NAME")
    github_email = os.environ.get("GITHUB_EMAIL")
    if github_name:
        git(["config", "user.name", github_name], repo_root)
    if github_email:
        git(["config", "user.email", github_email], repo_root)

    results_root = repo_root / "results"
    for subdir in ("plots", "metrics", "monitoring"):
        (results_root / subdir).mkdir(parents=True, exist_ok=True)

    git(["add", "results/plots/", "results/metrics/", "results/monitoring/"], repo_root)
    status = git(["status", "--porcelain", "results/"], repo_root, check=False)
    if not status.stdout.strip():
        print("No changes in results/ to commit")
        return

    commit = git(["commit", "-m", args.message], repo_root, check=False)
    if commit.returncode != 0:
        print(commit.stderr.strip() or commit.stdout.strip())
        return

    git(["push", "origin", "HEAD"], repo_root)
    print("Pushed results to GitHub")


if __name__ == "__main__":
    main()
