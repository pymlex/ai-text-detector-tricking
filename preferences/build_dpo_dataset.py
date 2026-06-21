from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import MODEL_ID, PARAPHRASES_PER_TEXT, TEXT_COLUMN
from data.prepare import load_filtered_splits
from detector.scoring import OculusDetector
from generation.paraphrase import generate_paraphrases
from generation.prompts import build_paraphrase_prompt
from preferences.pairing import select_preference_pair
from schemas.preferences import PreferenceBuildStats, PreferencePair
from utils.paths import PREFERENCES_DIR, ensure_result_dirs


PARAPHRASE_CACHE_PATH = PREFERENCES_DIR / "paraphrase_cache.jsonl"


def _load_paraphrase_cache(train_texts: list[str]) -> list[list[str]] | None:
    if not PARAPHRASE_CACHE_PATH.exists():
        return None
    rows: list[dict] = []
    with PARAPHRASE_CACHE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if len(rows) != len(train_texts):
        return None
    groups = [row["paraphrases"] for row in rows]
    originals = [row["original_text"] for row in rows]
    if originals != list(train_texts):
        return None
    return groups


def _save_paraphrase_cache(train_texts: list[str], paraphrase_groups: list[list[str]]) -> None:
    with PARAPHRASE_CACHE_PATH.open("w", encoding="utf-8") as handle:
        for original_text, paraphrases in zip(train_texts, paraphrase_groups, strict=True):
            handle.write(
                json.dumps(
                    {"original_text": original_text, "paraphrases": paraphrases},
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_preference_dataset(device: torch.device | None = None) -> Path:
    """Generate paraphrases, score with Oculus, and save DPO pairs."""
    ensure_result_dirs()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_filtered_splits()
    train_texts = splits["train"][TEXT_COLUMN]

    paraphrase_groups = _load_paraphrase_cache(train_texts)
    if paraphrase_groups is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            model = model.to(device)
        model.eval()

        paraphrase_groups = generate_paraphrases(
            model=model,
            tokenizer=tokenizer,
            original_texts=train_texts,
            device=device,
            num_samples=PARAPHRASES_PER_TEXT,
        )
        _save_paraphrase_cache(train_texts, paraphrase_groups)
        del model
        del tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"Loaded cached paraphrases: {PARAPHRASE_CACHE_PATH}")

    flat_paraphrases = [item for group in paraphrase_groups for item in group]
    detector = OculusDetector(device=device)
    flat_logits = detector.batch_logits(flat_paraphrases)

    pairs: list[PreferencePair] = []
    skipped_tie = 0
    skipped_empty = 0
    cursor = 0
    for original_text, paraphrases in zip(train_texts, paraphrase_groups, strict=True):
        prompt = build_paraphrase_prompt(original_text)
        chunk_logits = flat_logits[cursor : cursor + PARAPHRASES_PER_TEXT]
        cursor += PARAPHRASES_PER_TEXT
        pair = select_preference_pair(original_text, prompt, paraphrases, chunk_logits)
        if pair is None:
            if not paraphrases[0].strip() or not paraphrases[1].strip():
                skipped_empty += 1
            else:
                skipped_tie += 1
            continue
        pairs.append(pair)

    frame = pd.DataFrame([pair.model_dump() for pair in pairs])
    csv_path = PREFERENCES_DIR / "dpo_preferences.csv"
    jsonl_path = PREFERENCES_DIR / "dpo_preferences.jsonl"
    frame.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    hf_dataset = Dataset.from_pandas(
        frame[["prompt", "chosen", "rejected"]],
        preserve_index=False,
    )
    hf_path = PREFERENCES_DIR / "dpo_hf_dataset"
    hf_dataset.save_to_disk(str(hf_path))

    stats = PreferenceBuildStats(
        train_rows=len(train_texts),
        pairs_built=len(pairs),
        pairs_skipped_tie=skipped_tie,
        pairs_skipped_empty=skipped_empty,
    )
    stats_path = PREFERENCES_DIR / "build_stats.json"
    stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
    print(stats.model_dump_json(indent=2))
    print(f"Saved preferences: {csv_path}")
    return csv_path
