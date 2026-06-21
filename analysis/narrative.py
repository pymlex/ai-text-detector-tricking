from __future__ import annotations

from analysis.collect import AnalysisSnapshot
from analysis.metrics_table import render_evaluation_metrics_table


def render_analysis_markdown(snapshot: AnalysisSnapshot) -> str:
    pref = snapshot.preference_stats
    lines = [
        "# Results analysis",
        "",
        "## Preference dataset",
        "",
        f"Train abstracts: {pref.train_rows}. DPO pairs retained: {pref.pairs_built}. "
        f"Pairs skipped by logit margin |z1 - z2| < {pref.logit_margin_threshold:g}: "
        f"{pref.pairs_skipped_tie}. Empty paraphrases: {pref.pairs_skipped_empty}.",
        "",
    ]

    if snapshot.logit_probe is not None:
        probe = snapshot.logit_probe
        lines.extend(
            [
                "## Logit margin probe",
                "",
                f"Probe size: {probe.n_samples}. "
                f"Mean |z1 - z2|: {probe.mean_abs_logit_diff:.4f}. "
                f"Median: {probe.median_abs_logit_diff:.4f}. "
                f"IQR: [{probe.p25_abs_logit_diff:.4f}, {probe.p75_abs_logit_diff:.4f}]. "
                f"Maximum gap: {probe.max_abs_logit_diff:.4f}.",
                "",
                "![Logit probe summary](../plots/analysis/logit_probe_summary.png)",
                "",
            ]
        )

    monitor = snapshot.training_monitor
    if monitor.step0_mean_probability is not None and monitor.final_mean_probability is not None:
        delta_prob = monitor.final_mean_probability - monitor.step0_mean_probability
        lines.extend(
            [
                "## Training monitor",
                "",
                f"Validation subset mean AI probability at step 0: "
                f"{monitor.step0_mean_probability:.4f}. "
                f"At final logged step: {monitor.final_mean_probability:.4f}. "
                f"Change: {delta_prob:+.4f}.",
                "",
            ]
        )
    if monitor.final_loss is not None:
        lines.append(f"Final logged DPO loss: {monitor.final_loss:.4f}.")
        lines.append("")
        lines.append("![Training monitor analysis](../plots/analysis/training_monitor_analysis.png)")
        lines.append("")

    if snapshot.evaluation is not None or snapshot.base_evaluation is not None:
        threshold = (
            snapshot.evaluation.threshold
            if snapshot.evaluation is not None
            else snapshot.base_evaluation.threshold
        )
        lines.extend(
            [
                "## Evaluation",
                "",
                "Paraphrases from the base and fine-tuned models are scored by Oculus. "
                "Ground-truth label is AI-generated. Threshold on detector probability: "
                f"{threshold:.1f}.",
                "",
                render_evaluation_metrics_table(snapshot),
                "",
                "Lower mean probability and MCC near zero indicate weaker detector response "
                "on model paraphrases under the AI-positive labelling convention.",
                "",
                "![Evaluation summary](../plots/analysis/evaluation_summary.png)",
                "",
                "![Score distributions](../plots/analysis/score_distributions.png)",
                "",
            ]
        )

    return "\n".join(lines)
