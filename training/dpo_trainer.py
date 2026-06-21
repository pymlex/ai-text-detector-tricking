from __future__ import annotations

from trl import DPOTrainer


class QuietDPOTrainer(DPOTrainer):
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        self.state.log_history.append(dict(logs))
        for callback in self.callback_handler.callbacks:
            if callback.__class__.__name__ == "PrinterCallback":
                continue
            control = callback.on_log(self.args, self.state, self.control, logs)
            if control is not None:
                self.control = control
