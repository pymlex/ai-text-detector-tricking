from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi, login

from constants import HF_DATASET_REPO
from utils.config_loader import require_env
from utils.paths import CARDS_DIR, PLOTS_DIR, PREFERENCES_DIR


def _upload_assets(api: HfApi, repo_id: str, repo_type: str, assets: list[tuple[Path, str]]) -> None:
    for local_path, remote_name in assets:
        if not local_path.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"assets/{remote_name}",
            repo_id=repo_id,
            repo_type=repo_type,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=PREFERENCES_DIR / "dpo_hf_dataset")
    args = parser.parse_args()

    token = require_env("HF_TOKEN")
    login(token=token)
    api = HfApi()

    api.create_repo(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    dataset = Dataset.load_from_disk(str(args.dataset_path))
    dataset.push_to_hub(HF_DATASET_REPO, private=False)

    card_path = CARDS_DIR / "dataset_card.md"
    if not card_path.exists():
        raise RuntimeError("Missing dataset card. Run `python scripts/analyze_results.py` first.")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )

    _upload_assets(
        api,
        HF_DATASET_REPO,
        "dataset",
        [
            (PLOTS_DIR / "logit_margin_probe_hist.png", "logit_margin_probe_hist.png"),
            (PLOTS_DIR / "logit_chosen_rejected_hist.png", "logit_chosen_rejected_hist.png"),
            (PLOTS_DIR / "analysis" / "logit_probe_summary.png", "logit_probe_summary.png"),
        ],
    )
    print(f"Pushed dataset to https://huggingface.co/datasets/{HF_DATASET_REPO}")


if __name__ == "__main__":
    main()
