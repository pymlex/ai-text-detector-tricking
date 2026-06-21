from __future__ import annotations

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from detector.detector_arch import DesklibAIDetectionModel
from utils.config_loader import env_int, env_str


DETECTOR_MODEL_ID = env_str("DETECTOR_MODEL_ID", "danibor/oculus-v2.0-multilingual")
DETECTOR_BATCH_SIZE = env_int("DETECTOR_BATCH_SIZE", 32)
DETECTOR_MAX_LENGTH = env_int("DETECTOR_MAX_LENGTH", 512)


class OculusDetector:
    def __init__(self, device: torch.device, model_id: str = DETECTOR_MODEL_ID):
        self.device = device
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = DesklibAIDetectionModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def batch_logits(self, texts: list[str], show_progress: bool = True) -> list[float]:
        """Return raw detector logits for a list of texts."""
        logits_out: list[float] = []
        batch_range = range(0, len(texts), DETECTOR_BATCH_SIZE)
        iterator = tqdm(batch_range, desc="detect") if show_progress else batch_range
        for start in iterator:
            chunk = texts[start : start + DETECTOR_BATCH_SIZE]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=DETECTOR_MAX_LENGTH,
                padding="max_length",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs)["logits"].squeeze(-1)
            logits_out.extend(logits.detach().cpu().numpy().tolist())
        return logits_out

    def batch_probabilities(self, texts: list[str]) -> list[float]:
        """Return sigmoid probabilities for AI-generated class."""
        logits = self.batch_logits(texts)
        tensor = torch.tensor(logits, dtype=torch.float32)
        return torch.sigmoid(tensor).numpy().tolist()
