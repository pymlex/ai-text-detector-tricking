from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from constants import (
    CHECKPOINT_FRACTION,
    DPO_BETA,
    DPO_EPOCHS,
    DPO_GRADIENT_ACCUMULATION_STEPS,
    DPO_LEARNING_RATE,
    DPO_LOGGING_STEPS,
    DPO_PER_DEVICE_BATCH_SIZE,
    DPO_WARMUP_RATIO,
    MODEL_ID,
    MONITORING_FRACTION,
)
from evaluation.metrics import load_preference_dataset
from training.callbacks import DetectorMonitorCallback, MonitorConfig
from utils.paths import CHECKPOINTS_DIR, ensure_result_dirs


def _format_preference_dataset(dataset: Dataset) -> Dataset:
    """Convert plain-text DPO rows into Qwen chat messages."""
    def _map_row(row: dict) -> dict:
        return {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }

    return dataset.map(_map_row)


def train_dpo(device: torch.device | None = None) -> Path:
    """Run DPO fine-tuning on the fixed preference dataset."""
    ensure_result_dirs()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preference_dataset = load_preference_dataset()
    train_dataset = _format_preference_dataset(preference_dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)

    steps_per_epoch = max(
        1,
        len(train_dataset)
        // (DPO_PER_DEVICE_BATCH_SIZE * DPO_GRADIENT_ACCUMULATION_STEPS),
    )
    save_steps = max(1, int(steps_per_epoch * CHECKPOINT_FRACTION))
    monitor_steps = max(1, int(steps_per_epoch * MONITORING_FRACTION))

    training_args = DPOConfig(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=DPO_EPOCHS,
        per_device_train_batch_size=DPO_PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=DPO_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=DPO_LEARNING_RATE,
        beta=DPO_BETA,
        bf16=device.type == "cuda",
        fp16=False,
        lr_scheduler_type="cosine",
        warmup_ratio=DPO_WARMUP_RATIO,
        logging_steps=DPO_LOGGING_STEPS,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=8,
        remove_unused_columns=False,
        report_to="none",
        max_length=512,
        max_prompt_length=512,
    )

    monitor_callback = DetectorMonitorCallback(
        tokenizer=tokenizer,
        device=device,
        monitor_config=MonitorConfig(eval_every_steps=monitor_steps),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[monitor_callback],
    )
    trainer.train()

    final_dir = CHECKPOINTS_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved final model: {final_dir}")
    return final_dir
