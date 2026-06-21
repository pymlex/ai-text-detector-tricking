from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from huggingface_hub import HfApi, login

from constants import HF_MODEL_REPO
from utils.config_loader import require_env
from utils.paths import CARDS_DIR, CHECKPOINTS_DIR, METRICS_DIR, PLOTS_DIR


def _upload_assets(api: HfApi, repo_id: str, assets: list[tuple[Path, str]]) -> None:
    for local_path, remote_name in assets:
        if not local_path.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"assets/{remote_name}",
            repo_id=repo_id,
            repo_type="model",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=CHECKPOINTS_DIR / "final",
    )
    args = parser.parse_args()

    token = require_env("HF_TOKEN")
    login(token=token)
    api = HfApi()

    api.create_repo(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    api.upload_folder(
        folder_path=str(args.model_path),
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )

    card_path = CARDS_DIR / "model_card.md"
    if not card_path.exists():
        raise RuntimeError("Missing model card. Run `python scripts/analyze_results.py` first.")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )

    _upload_assets(
        api,
        HF_MODEL_REPO,
        [
            (PLOTS_DIR / "analysis" / "evaluation_summary.png", "evaluation_summary.png"),
            (PLOTS_DIR / "analysis" / "score_distributions.png", "score_distributions.png"),
            (PLOTS_DIR / "analysis" / "training_monitor_analysis.png", "training_monitor_analysis.png"),
            (PLOTS_DIR / "training_summary.png", "training_summary.png"),
            (PLOTS_DIR / "detector_probability_hist_validation.png", "detector_probability_hist_validation.png"),
            (PLOTS_DIR / "detector_probability_hist_test.png", "detector_probability_hist_test.png"),
            (METRICS_DIR / "evaluation_report.json", "evaluation_report.json"),
            (METRICS_DIR / "base_evaluation_report.json", "base_evaluation_report.json"),
        ],
    )
    print(f"Pushed model to https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
