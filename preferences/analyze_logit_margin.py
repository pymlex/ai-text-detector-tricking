from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import ANALYZE_MARGIN_SAMPLES, GENERATION_BATCH_SIZE, MODEL_ID, PARAPHRASES_PER_TEXT, TEXT_COLUMN
from data.prepare import load_filtered_splits
from detector.scoring import OculusDetector
from generation.paraphrase import generate_paraphrases
from preferences.pairing import logit_abs_margin
from utils.paths import PLOTS_DIR, PREFERENCES_DIR, ensure_result_dirs


def analyze_logit_margin(
    device: torch.device,
    n_samples: int = ANALYZE_MARGIN_SAMPLES,
) -> Path:
    """Generate paraphrase pairs for ``n_samples`` texts and plot logit gap histogram."""
    ensure_result_dirs()
    print(f"Probe samples: {n_samples}, generation batch size: {GENERATION_BATCH_SIZE}")
    splits = load_filtered_splits()
    sample_texts = splits["train"][TEXT_COLUMN][:n_samples]

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
        original_texts=sample_texts,
        device=device,
        num_samples=PARAPHRASES_PER_TEXT,
    )
    del model
    del tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    flat_paraphrases = [item for group in paraphrase_groups for item in group]
    detector = OculusDetector(device=device)
    flat_logits = detector.batch_logits(flat_paraphrases)
    logit_array = np.array(flat_logits, dtype=np.float64).reshape(n_samples, PARAPHRASES_PER_TEXT)
    prob_array = torch.sigmoid(torch.tensor(flat_logits, dtype=torch.float64)).numpy().reshape(
        n_samples, PARAPHRASES_PER_TEXT
    )

    rows: list[dict] = []
    abs_logit_gaps: list[float] = []
    chosen_logits: list[float] = []
    rejected_logits: list[float] = []
    for row_idx, (original_text, paraphrases) in enumerate(
        zip(sample_texts, paraphrase_groups, strict=True)
    ):
        row_logits = logit_array[row_idx].tolist()
        row_probs = prob_array[row_idx].tolist()
        abs_gap = logit_abs_margin(row_logits)
        abs_logit_gaps.append(abs_gap)
        if row_logits[0] <= row_logits[1]:
            chosen_logit, rejected_logit = row_logits[0], row_logits[1]
        else:
            chosen_logit, rejected_logit = row_logits[1], row_logits[0]
        chosen_logits.append(chosen_logit)
        rejected_logits.append(rejected_logit)
        rows.append(
            {
                "original_text": original_text,
                "paraphrase_0": paraphrases[0],
                "paraphrase_1": paraphrases[1],
                "logit_0": row_logits[0],
                "logit_1": row_logits[1],
                "chosen_logit": chosen_logit,
                "rejected_logit": rejected_logit,
                "abs_logit_diff": abs_gap,
                "prob_0": row_probs[0],
                "prob_1": row_probs[1],
                "abs_prob_diff": abs(row_probs[0] - row_probs[1]),
            }
        )

    frame = pd.DataFrame(rows)
    csv_path = PREFERENCES_DIR / "logit_margin_probe.csv"
    frame.to_csv(csv_path, index=False)

    gaps = np.array(abs_logit_gaps, dtype=np.float64)
    summary = pd.Series(gaps).describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(summary)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(gaps, bins=40)
    axis.set_title(f"Absolute logit gap over {n_samples} paraphrase pairs")
    axis.set_xlabel("|logit_0 - logit_1|")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.5)
    fig.tight_layout()
    plot_path = PLOTS_DIR / "logit_margin_probe_hist.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    chosen_array = np.array(chosen_logits, dtype=np.float64)
    rejected_array = np.array(rejected_logits, dtype=np.float64)
    bin_edges = np.linspace(
        min(chosen_array.min(), rejected_array.min()),
        max(chosen_array.max(), rejected_array.max()),
        41,
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(chosen_array, bins=bin_edges, alpha=0.6, label="chosen")
    axis.hist(rejected_array, bins=bin_edges, alpha=0.6, label="rejected")
    axis.set_title(f"Chosen vs rejected logit distributions, n={n_samples}")
    axis.set_xlabel("Detector logit")
    axis.set_ylabel("Count")
    axis.legend()
    axis.grid(alpha=0.5)
    fig.tight_layout()
    subset_plot_path = PLOTS_DIR / "logit_chosen_rejected_hist.png"
    fig.savefig(subset_plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved plot: {subset_plot_path}")
    return plot_path
