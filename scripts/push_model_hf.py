from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from huggingface_hub import HfApi, login

from constants import HF_MODEL_REPO
from utils.config_loader import load_env_file, require_env
from utils.paths import CHECKPOINTS_DIR


def main() -> None:
    load_env_file("project")
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

    readme = Path("templates/model_card.md").read_text(encoding="utf-8")
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )
    print(f"Pushed model to https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
