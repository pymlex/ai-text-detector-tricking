from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import torch

from utils.config_loader import load_env_file
from utils.paths import CHECKPOINTS_DIR, MONITORING_DIR, PLOTS_DIR, ensure_result_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO training against Oculus detector")
    parser.add_argument(
        "--step",
        choices=["prepare", "preferences", "train", "evaluate", "plot", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file("project")
    ensure_result_dirs()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.step in {"prepare", "all"}:
        from data.prepare import download_and_filter_dataset

        download_and_filter_dataset()

    if args.step in {"preferences", "all"}:
        from preferences.build_dpo_dataset import build_preference_dataset

        build_preference_dataset(device=device)

    if args.step in {"train", "all"}:
        from training.dpo_train import train_dpo

        train_dpo(device=device)

    if args.step in {"evaluate", "all"}:
        from evaluation.evaluate import evaluate_model

        final_model = CHECKPOINTS_DIR / "final"
        evaluate_model(final_model, device=device, output_tag="final")

    if args.step in {"plot", "all"}:
        from plotting.figures import plot_training_summary

        plot_training_summary(MONITORING_DIR, PLOTS_DIR)


if __name__ == "__main__":
    main()
