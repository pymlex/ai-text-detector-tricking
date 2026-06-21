from __future__ import annotations

from transformers import TrainerCallback, TrainingArguments


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
        print(line, flush=True)
        return control
