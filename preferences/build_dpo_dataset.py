from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import (
    MODEL_ID,
    PARAPHRASES_PER_TEXT,
    PREFERENCE_PROB_MARGIN,
    TEXT_COLUMN,
)
from data.prepare import load_filtered_splits
from detector.scoring import OculusDetector
from generation.paraphrase import generate_paraphrases
from generation.prompts import build_paraphrase_prompt
from schemas.preferences import PreferenceBuildStats, PreferencePair
from utils.paths import PREFERENCES_DIR, ensure_result_dirs


def _select_pair(
    original_text: str,
    prompt: str,
    paraphrases: list[str],
    logits: list[float],
    probabilities: list[float],
) -> PreferencePair | None:
    if len(paraphrases) < 2:
        return None
    if not paraphrases[0].strip() or not paraphrases[1].strip():
        return None
    if abs(probabilities[0] - probabilities[1]) < PREFERENCE_PROB_MARGIN:
        return None
    if probabilities[0] <= probabilities[1]:
        chosen_idx, rejected_idx = 0, 1
    else:
        chosen_idx, rejected_idx = 1, 0
    return PreferencePair(
        prompt=prompt,
        chosen=paraphrases[chosen_idx],
        rejected=paraphrases[rejected_idx],
        chosen_logit=logits[chosen_idx],
        rejected_logit=logits[rejected_idx],
        chosen_probability=probabilities[chosen_idx],
        rejected_probability=probabilities[rejected_idx],
        original_text=original_text,
    )


def build_preference_dataset(device: torch.device | None = None) -> Path:
    """Generate paraphrases, score with Oculus, and save DPO pairs."""
    ensure_result_dirs()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_filtered_splits()
    train_texts = splits["train"][TEXT_COLUMN]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
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

    flat_paraphrases = [item for group in paraphrase_groups for item in group]
    detector = OculusDetector(device=device)
    flat_logits = detector.batch_logits(flat_paraphrases)
    flat_probs = torch.sigmoid(torch.tensor(flat_logits, dtype=torch.float32)).numpy().tolist()

    pairs: list[PreferencePair] = []
    skipped_tie = 0
    skipped_empty = 0
    cursor = 0
    for original_text, paraphrases in zip(train_texts, paraphrase_groups, strict=True):
        prompt = build_paraphrase_prompt(original_text)
        chunk_logits = flat_logits[cursor : cursor + PARAPHRASES_PER_TEXT]
        chunk_probs = flat_probs[cursor : cursor + PARAPHRASES_PER_TEXT]
        cursor += PARAPHRASES_PER_TEXT
        pair = _select_pair(original_text, prompt, paraphrases, chunk_logits, chunk_probs)
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
