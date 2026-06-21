from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from constants import DATASET_ID, MAX_TOKENS, MODEL_ID, TEXT_COLUMN
from utils.paths import DATA_DIR, PLOTS_DIR, ensure_result_dirs


def count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    """Count tokens for one string without special tokens."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def download_and_filter_dataset(cache_dir: Path | None = None) -> DatasetDict:
    """Load the Spanish abstracts dataset and filter by token length."""
    ensure_result_dirs()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    dataset = load_dataset(DATASET_ID, cache_dir=str(cache_dir) if cache_dir else None)

    frames = []
    for split_name, split_ds in dataset.items():
        frame = split_ds.to_pandas()
        frame["split"] = split_name
        frames.append(frame)

    full_frame = pd.concat(frames, ignore_index=True)
    full_frame[TEXT_COLUMN] = full_frame[TEXT_COLUMN].fillna("").astype(str)
    full_frame["token_length"] = full_frame[TEXT_COLUMN].map(
        lambda text: count_tokens(tokenizer, text)
    )
    filtered = full_frame[full_frame["token_length"] <= MAX_TOKENS].copy()

    print(f"Rows after filtering: {len(filtered):,}")
    print(f"Removed rows: {len(full_frame) - len(filtered):,}")

    summary = filtered["token_length"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    ).to_frame().T
    summary["sum"] = filtered["token_length"].sum()
    summary["n"] = len(filtered)
    print(summary)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(filtered["token_length"], bins=70)
    axis.set_title("Distribution of token lengths in resumen")
    axis.set_xlabel("Token length")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.5)
    fig.tight_layout()
    plot_path = PLOTS_DIR / "token_length_distribution.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    split_frames = {
        split_name: filtered[filtered["split"] == split_name][[TEXT_COLUMN]].reset_index(drop=True)
        for split_name in dataset.keys()
    }
    dataset_dict = DatasetDict(
        {split_name: Dataset.from_pandas(frame, preserve_index=False) for split_name, frame in split_frames.items()}
    )

    local_path = DATA_DIR / "filtered_abstracts"
    dataset_dict.save_to_disk(str(local_path))
    print(f"Saved filtered dataset: {local_path}")
    return dataset_dict


def load_filtered_splits() -> DatasetDict:
    """Load filtered splits from disk."""
    local_path = DATA_DIR / "filtered_abstracts"
    return DatasetDict.load_from_disk(str(local_path))
