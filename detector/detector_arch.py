from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled)
        return {"logits": logits}
