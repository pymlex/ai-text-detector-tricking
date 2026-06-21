from __future__ import annotations

from pathlib import Path

from utils.config_loader import env_str


RESULTS_DIR = Path(env_str("RESULTS_DIR", "results"))
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
PREFERENCES_DIR = RESULTS_DIR / "preferences"
MONITORING_DIR = RESULTS_DIR / "monitoring"
HF_CACHE_DIR = Path(env_str("HF_CACHE_DIR", "hf_cache"))


def ensure_result_dirs() -> None:
    """Create standard artefact directories under ``RESULTS_DIR``."""
    for path in (
        RESULTS_DIR,
        DATA_DIR,
        PLOTS_DIR,
        METRICS_DIR,
        CHECKPOINTS_DIR,
        PREFERENCES_DIR,
        MONITORING_DIR,
        HF_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
