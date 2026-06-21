from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi, login

from constants import HF_DATASET_REPO
from utils.config_loader import load_env_file, require_env
from utils.paths import PREFERENCES_DIR


def main() -> None:
    load_env_file("project")
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

    readme = Path("templates/dataset_card.md").read_text(encoding="utf-8")
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Pushed dataset to https://huggingface.co/datasets/{HF_DATASET_REPO}")


if __name__ == "__main__":
    main()
