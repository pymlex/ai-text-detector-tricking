from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, TrainingArguments

from constants import TEXT_COLUMN
from data.prepare import load_filtered_splits
from detector.scoring import OculusDetector
from generation.paraphrase import generate_paraphrases
from utils.paths import MONITORING_DIR, ensure_result_dirs


@dataclass
class MonitorConfig:
    eval_every_steps: int
    validation_sample_size: int | None = None


class DetectorMonitorCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        monitor_config: MonitorConfig,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.monitor_config = monitor_config
        ensure_result_dirs()
        splits = load_filtered_splits()
        validation_texts = splits["validation"][TEXT_COLUMN]
        if monitor_config.validation_sample_size is not None:
            validation_texts = validation_texts[: monitor_config.validation_sample_size]
        self.validation_texts = validation_texts
        self.detector = OculusDetector(device=device)

    def on_step_end(self, args: TrainingArguments, state, control, model=None, **kwargs):
        if state.global_step == 0:
            return control
        if state.global_step % self.monitor_config.eval_every_steps != 0:
            return control
        if model is None:
            return control

        model.eval()
        paraphrases = [
            group[0]
            for group in generate_paraphrases(
                model=model,
                tokenizer=self.tokenizer,
                original_texts=self.validation_texts,
                device=self.device,
                num_samples=1,
            )
        ]
        logits = np.array(self.detector.batch_logits(paraphrases), dtype=np.float64)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        payload = {
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            "mean_logit": float(logits.mean()),
            "mean_probability": float(probabilities.mean()),
            "n_samples": int(len(probabilities)),
        }
        output_path = MONITORING_DIR / f"monitor_step_{state.global_step:06d}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        model.train()
        return control
