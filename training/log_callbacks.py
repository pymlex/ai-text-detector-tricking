from __future__ import annotations

from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainingArguments


def log_line(message: str) -> None:
    """Print one log line without breaking active tqdm bars."""
    tqdm.write(message)


class CompactDPOLogCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return control
        if "loss" not in logs:
            return control

        epoch = float(logs.get("epoch", state.epoch or 0.0))
        loss = float(logs["loss"])
        line = (
            f"[train] step={state.global_step:4d} | epoch={epoch:5.3f} | "
            f"loss={loss:.4f}"
        )
        if "rewards/accuracies" in logs:
            line += f" | reward_acc={float(logs['rewards/accuracies']):.4f}"
        if "rewards/margins" in logs:
            line += f" | reward_margin={float(logs['rewards/margins']):.4f}"
        if "learning_rate" in logs:
            line += f" | lr={float(logs['learning_rate']):.2e}"
        log_line(line)
        return control


class SuppressPrinterCallback(TrainerCallback):
    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return control
        trainer.callback_handler.callbacks = [
            callback
            for callback in trainer.callback_handler.callbacks
            if callback.__class__.__name__ != "PrinterCallback"
        ]
        return control
