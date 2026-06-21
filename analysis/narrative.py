from __future__ import annotations

from analysis.collect import AnalysisSnapshot
from schemas.evaluation import SplitMetrics


def _format_split_metrics(name: str, metrics: SplitMetrics) -> str:
    return (
        f"| {name} | {metrics.n_samples} | {metrics.mean_probability:.4f} | "
        f"{metrics.mean_logit:.4f} | {metrics.accuracy:.4f} | {metrics.mcc:.4f} | "
        f"{metrics.roc_auc:.4f} | {metrics.f1:.4f} |"
    )


def render_analysis_markdown(snapshot: AnalysisSnapshot) -> str:
    pref = snapshot.preference_stats
    lines = [
        "# Results analysis",
        "",
        "## Preference dataset",
        "",
        f"Train abstracts: {pref.train_rows}. DPO pairs retained: {pref.pairs_built}. "
        f"Pairs skipped by logit margin $|z_1-z_2|<{pref.logit_margin_threshold:g}$: "
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
                f"Mean $|z_1-z_2|$: {probe.mean_abs_logit_diff:.4f}. "
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
    if monitor.final_loss is not None or monitor.final_reward_accuracy is not None:
        monitor_parts = []
        if monitor.final_loss is not None:
            monitor_parts.append(f"Final logged DPO loss: {monitor.final_loss:.4f}.")
        if monitor.final_reward_accuracy is not None:
            monitor_parts.append(
                f"Final reward accuracy: {monitor.final_reward_accuracy:.4f}."
            )
        lines.append(" ".join(monitor_parts))
        lines.append("")
        lines.append("![Training monitor analysis](../plots/analysis/training_monitor_analysis.png)")
        lines.append("")

    if snapshot.evaluation is not None:
        report = snapshot.evaluation
        lines.extend(
            [
                "## Final evaluation",
                "",
                "Paraphrases from the fine-tuned model are scored by Oculus. "
                "Ground-truth label is AI-generated. Threshold on detector probability: "
                f"{report.threshold:.1f}.",
                "",
                "| Split | n | mean prob | mean logit | accuracy | MCC | ROC-AUC | F1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                _format_split_metrics("validation", report.validation),
                _format_split_metrics("test", report.test),
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
