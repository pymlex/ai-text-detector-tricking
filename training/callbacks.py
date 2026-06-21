from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, TrainingArguments

from constants import DPO_MONITOR_EVERY_STEPS, TEXT_COLUMN
from data.prepare import load_filtered_splits
from detector.scoring import OculusDetector
from generation.paraphrase import generate_paraphrases
from utils.paths import MONITORING_DIR, ensure_result_dirs


@dataclass
class MonitorConfig:
    eval_every_steps: int = DPO_MONITOR_EVERY_STEPS
    validation_sample_size: int | None = None


class DetectorMonitorCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        monitor_config: MonitorConfig | None = None,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.monitor_config = monitor_config or MonitorConfig()
        ensure_result_dirs()
        splits = load_filtered_splits()
        validation_texts = splits["validation"][TEXT_COLUMN]
        if self.monitor_config.validation_sample_size is not None:
            validation_texts = validation_texts[: self.monitor_config.validation_sample_size]
        self.validation_texts = validation_texts
        self.detector = OculusDetector(device=device)

    def _should_evaluate(self, step: int) -> bool:
        if step == 0:
            return True
        interval = self.monitor_config.eval_every_steps
        return step > 0 and step % interval == 0

    def _run_detector_eval(self, model: PreTrainedModel, step: int, epoch: float) -> None:
        model.eval()
        paraphrases = [
            group[0]
            for group in generate_paraphrases(
                model=model,
                tokenizer=self.tokenizer,
                original_texts=self.validation_texts,
                device=self.device,
                num_samples=1,
                show_progress=False,
            )
        ]
        logits = np.array(
            self.detector.batch_logits(paraphrases, show_progress=False),
            dtype=np.float64,
        )
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        payload = {
            "step": int(step),
            "epoch": float(epoch),
            "mean_logit": float(logits.mean()),
            "mean_probability": float(probabilities.mean()),
            "n_samples": int(len(probabilities)),
        }
        output_path = MONITORING_DIR / f"monitor_step_{step:06d}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[valid] step={step:4d} | epoch={epoch:5.3f} | "
            f"mean_logit={payload['mean_logit']:.4f} | "
            f"mean_prob={payload['mean_probability']:.4f} | "
            f"n={payload['n_samples']}",
            flush=True,
        )
        model.train()

    def on_train_begin(self, args: TrainingArguments, state, control, model=None, **kwargs):
        if model is None:
            return control
        if self._should_evaluate(0):
            self._run_detector_eval(model, step=0, epoch=0.0)
        return control

    def on_step_end(self, args: TrainingArguments, state, control, model=None, **kwargs):
        if model is None:
            return control
        step = int(state.global_step)
        if not self._should_evaluate(step):
            return control
        if step == 0:
            return control
        epoch = float(state.epoch or 0.0)
        self._run_detector_eval(model, step=step, epoch=epoch)
        return control
