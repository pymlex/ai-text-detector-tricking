from __future__ import annotations

from pathlib import Path

from analysis.cards import render_dataset_card, render_model_card
from analysis.collect import collect_analysis_snapshot, save_analysis_snapshot
from analysis.narrative import render_analysis_markdown
from plotting.analysis_figures import build_analysis_plots
from utils.paths import ANALYSIS_DIR, CARDS_DIR, ensure_result_dirs


def run_analysis() -> Path:
    """Collect metrics, build analysis plots, and render publication cards."""
    ensure_result_dirs()
    snapshot = collect_analysis_snapshot()
    save_analysis_snapshot(snapshot)
    build_analysis_plots(snapshot)

    analysis_md = render_analysis_markdown(snapshot)
    analysis_path = ANALYSIS_DIR / "ANALYSIS.md"
    analysis_path.write_text(analysis_md, encoding="utf-8")

    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    model_card_path = CARDS_DIR / "model_card.md"
    dataset_card_path = CARDS_DIR / "dataset_card.md"
    model_card_path.write_text(render_model_card(snapshot), encoding="utf-8")
    dataset_card_path.write_text(render_dataset_card(snapshot), encoding="utf-8")

    print(f"Saved analysis: {analysis_path}")
    print(f"Saved model card: {model_card_path}")
    print(f"Saved dataset card: {dataset_card_path}")
    return analysis_path
