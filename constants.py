from __future__ import annotations

from utils.config_loader import env_float, env_int, env_str


MODEL_ID = env_str("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DETECTOR_MODEL_ID = env_str("DETECTOR_MODEL_ID", "danibor/oculus-v2.0-multilingual")
DATASET_ID = env_str("DATASET_ID", "Flaglab/academic-knowledge-abstracts-es")
TEXT_COLUMN = env_str("TEXT_COLUMN", "resumen")

MAX_TOKENS = env_int("MAX_TOKENS", 512)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 512)
GENERATION_TEMPERATURE = env_float("GENERATION_TEMPERATURE", 0.7)
GENERATION_BATCH_SIZE = env_int("GENERATION_BATCH_SIZE", 8)

PREFERENCE_PROB_MARGIN = env_float("PREFERENCE_PROB_MARGIN", 0.05)
PARAPHRASES_PER_TEXT = env_int("PARAPHRASES_PER_TEXT", 2)

DPO_EPOCHS = env_int("DPO_EPOCHS", 2)
DPO_LEARNING_RATE = env_float("DPO_LEARNING_RATE", 1e-5)
DPO_BETA = env_float("DPO_BETA", 0.1)
DPO_PER_DEVICE_BATCH_SIZE = env_int("DPO_PER_DEVICE_BATCH_SIZE", 32)
DPO_GRADIENT_ACCUMULATION_STEPS = env_int("DPO_GRADIENT_ACCUMULATION_STEPS", 1)
DPO_WARMUP_RATIO = env_float("DPO_WARMUP_RATIO", 0.1)
DPO_LOGGING_STEPS = env_int("DPO_LOGGING_STEPS", 10)

HF_DATASET_REPO = env_str("HF_DATASET_REPO", "pymlex/ai-generated-texts")
HF_MODEL_REPO = env_str("HF_MODEL_REPO", "pymlex/Qwen2.5-0.5B-Human")

DETECTION_THRESHOLD = env_float("DETECTION_THRESHOLD", 0.5)
MONITORING_FRACTION = env_float("MONITORING_FRACTION", 0.1)
CHECKPOINT_FRACTION = env_float("CHECKPOINT_FRACTION", 0.5)

PARAPHRASE_PROMPT_TEMPLATE = (
    "Parafrasea el siguiente texto manteniendo el significado, "
    "sin añadir comentarios ni explicaciones adicionales. "
    "Texto original: {original_text}"
)
