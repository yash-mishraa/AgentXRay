"""Report generation for AgentXRay experiments.

Produces machine-readable JSON and human-readable Markdown reports from
actually executed experiment data. No results are hardcoded.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentxray.models import ExperimentReport, StochasticExperimentReport
from agentxray.statistics import estimate_two_proportion_power


def format_json_report(report: ExperimentReport) -> str:
    """Format an ExperimentReport into an indented JSON string."""
    return report.model_dump_json(indent=2)


def format_markdown_report(report: ExperimentReport) -> str:
    """Format an ExperimentReport into a concise Markdown summary."""
    lines: list[str] = [
        f"# AgentXRay Experiment Report: {report.experiment_name}",
        "",
        f"> **Generated at**: {report.timestamp.isoformat()}  ",
        f"> **Access Mode**: `{report.config.mode.value}`  ",
        f"> **Trial Budget Cap**: `{report.config.max_trials}`  ",
        f"> **Execution Timeout**: `{report.config.timeout_seconds}s`  ",
        f"> **Total Target Invocations**: `{report.total_target_calls}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        report.summary,
        "",
        "## 2. Benchmark Comparison Matrix",
        "",
        "| Task Name | Method | Discovered Trigger | 1-Minimal? | Calls | Distractors Left | Matches Ground Truth |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for eval_item in report.evaluations:
        cond = eval_item.minimizer_condition
        cond_str = ", ".join(f"`{e}`" for e in cond.subset_elements) if cond.subset_elements else "*(none)*"
        lines.append(
            f"| **{eval_item.task_name}** | **Delta Debugging** | {cond_str} | "
            f"{'Yes' if cond.is_one_minimal else 'No'} | {cond.target_calls_used} | "
            f"{len(eval_item.excess_elements)} | {'**MATCH**' if eval_item.matches_ground_truth else 'FAIL'} |"
        )
        for base in eval_item.baseline_comparisons:
            base_str = ", ".join(f"`{e}`" for e in base.discovered_elements) if base.discovered_elements else "*(none)*"
            lines.append(
                f"| | {base.baseline_name} | {base_str} | "
                f"{'Yes' if base.minimality_check_passed else 'No'} | {base.target_calls} | "
                f"{base.irrelevant_elements_remaining} | {'MATCH' if base.matches_ground_truth else 'FAIL'} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Task Evaluations",
        "",
    ])

    for eval_item in report.evaluations:
        gt_str = ", ".join(f"`{e}`" for e in eval_item.ground_truth_trigger) if eval_item.ground_truth_trigger else "*(None - Null Case)*"
        init_str = ", ".join(f"`{e}`" for e in eval_item.initial_elements)
        lines.extend([
            f"### Task: {eval_item.task_name}",
            f"- **Description**: {eval_item.description}",
            f"- **Initial Elements ({len(eval_item.initial_elements)})**: {init_str}",
            f"- **Planted Ground Truth**: {gt_str}",
            f"- **Minimizer Precision**: `{eval_item.precision * 100:.1f}%`",
            f"- **Minimizer Recall**: `{eval_item.recall * 100:.1f}%`",
            f"- **Excess Distractors Retained**: {', '.join(f'`{e}`' for e in eval_item.excess_elements) if eval_item.excess_elements else '0'}",
            f"- **Status Message**: {eval_item.minimizer_condition.status_message}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 4. Limitations and Reproducibility Notice",
        "",
    ])
    for lim in report.limitations:
        lines.append(f"* {lim}")

    lines.append("")
    return "\n".join(lines)


def write_reports(
    report: ExperimentReport,
    json_path: Path | str,
    md_path: Path | str,
) -> None:
    """Write both JSON and Markdown reports to disk."""
    json_p = Path(json_path)
    md_p = Path(md_path)

    json_p.parent.mkdir(parents=True, exist_ok=True)
    md_p.parent.mkdir(parents=True, exist_ok=True)

    json_p.write_text(format_json_report(report), encoding="utf-8")
    md_p.write_text(format_markdown_report(report), encoding="utf-8")


def format_stochastic_json_report(report: StochasticExperimentReport) -> str:
    """Format a StochasticExperimentReport into an indented JSON string."""
    return report.model_dump_json(indent=2)


def format_stochastic_markdown_report(report: StochasticExperimentReport) -> str:
    """Format a StochasticExperimentReport into a detailed Markdown report with uncertainty bounds."""
    lines: list[str] = [
        f"# AgentXRay Stochastic Experiment Report: {report.experiment_name}",
        "",
        f"> **Generated at**: {report.timestamp.isoformat()}  ",
        f"> **Access Mode**: `{report.config.mode.value}`  ",
        f"> **Trial Budget Cap**: `{report.config.max_trials}`  ",
        f"> **Trials per Candidate Subset**: `{report.trials_per_candidate}`  ",
        f"> **Execution Timeout**: `{report.config.timeout_seconds}s`  ",
        f"> **Total Target Invocations**: `{report.total_target_calls}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        report.summary,
        "",
        "## 2. Benchmark Comparison Matrix",
        "",
        "| Task Name | Method | Discovered Trigger | Failure Rate (95% CI) | Minimality Status | Calls | Distractors Left | Matches Ground Truth |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for eval_item in report.evaluations:
        cond = eval_item.minimizer_condition
        cond_str = ", ".join(f"`{e}`" for e in cond.subset_elements) if cond.subset_elements else "*(none)*"
        cand_ci = f"{cond.candidate_evaluation.failure_rate:.2f} [{cond.candidate_evaluation.confidence_interval[0]:.2f}, {cond.candidate_evaluation.confidence_interval[1]:.2f}]"
        lines.append(
            f"| **{eval_item.task_name}** | **Delta Debugging** | {cond_str} | "
            f"{cand_ci} | `{cond.minimality_status.value}` | {cond.target_calls_used} | "
            f"{len(eval_item.excess_elements)} | {'**MATCH**' if eval_item.matches_ground_truth else 'FAIL'} |"
        )
        for base in eval_item.baseline_comparisons:
            base_str = ", ".join(f"`{e}`" for e in base.discovered_elements) if base.discovered_elements else "*(none)*"
            base_ci = f"{base.candidate_evaluation.failure_rate:.2f} [{base.candidate_evaluation.confidence_interval[0]:.2f}, {base.candidate_evaluation.confidence_interval[1]:.2f}]"
            lines.append(
                f"| | {base.baseline_name} | {base_str} | "
                f"{base_ci} | `{base.minimality_status.value}` | {base.target_calls} | "
                f"{base.irrelevant_elements_remaining} | {'MATCH' if base.matches_ground_truth else 'FAIL'} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Statistical & Uncertainty Evaluations",
        "",
    ])

    for eval_item in report.evaluations:
        gt_str = ", ".join(f"`{e}`" for e in eval_item.ground_truth_trigger) if eval_item.ground_truth_trigger else "*(None - Null Case)*"
        init_str = ", ".join(f"`{e}`" for e in eval_item.initial_elements)
        cond = eval_item.minimizer_condition

        # Theoretical statistical power
        power = estimate_two_proportion_power(
            p1=eval_item.trigger_probability,
            p2=eval_item.base_probability,
            n_per_group=report.trials_per_candidate,
            alpha=0.05,
        )

        lines.extend([
            f"### Task: {eval_item.task_name}",
            f"- **Description**: {eval_item.description}",
            f"- **Initial Elements ({len(eval_item.initial_elements)})**: {init_str}",
            f"- **Planted Ground Truth**: {gt_str}",
            f"- **Ground Truth Probabilities**: Base $P={eval_item.base_probability:.2f}$, Triggered $P={eval_item.trigger_probability:.2f}$ (Delta: ${eval_item.trigger_probability - eval_item.base_probability:.2f}$)",
            f"- **Statistical Power (at n={report.trials_per_candidate}, alpha=0.05)**: `{power * 100:.1f}%`",
            f"- **Minimizer Precision**: `{eval_item.precision * 100:.1f}%`",
            f"- **Minimizer Recall**: `{eval_item.recall * 100:.1f}%`",
            f"- **Minimality Classification**: `{cond.minimality_status.value}`",
            f"- **Final Candidate Failure Rate**: `{cond.candidate_evaluation.failure_rate:.2f}` (95% Wilson CI: `[{cond.candidate_evaluation.confidence_interval[0]:.2f}, {cond.candidate_evaluation.confidence_interval[1]:.2f}]`, $k={cond.candidate_evaluation.failures}/{cond.candidate_evaluation.trials}$)",
            f"- **Status Narrative**: {cond.status_message}",
        ])

        if cond.removal_evaluations:
            lines.append("- **Single-Element Removal Uncertainty Checks**:")
            for elem, rem_eval in cond.removal_evaluations.items():
                lines.append(
                    f"  - Without `{elem}`: failure rate `{rem_eval.failure_rate:.2f}` "
                    f"(95% CI: `[{rem_eval.confidence_interval[0]:.2f}, {rem_eval.confidence_interval[1]:.2f}]`, $k={rem_eval.failures}/{rem_eval.trials}$)"
                )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. Limitations and Reproducibility Notice",
        "",
    ])
    for lim in report.limitations:
        lines.append(f"* {lim}")

    lines.append("")
    return "\n".join(lines)


def write_stochastic_reports(
    report: StochasticExperimentReport,
    json_path: Path | str,
    md_path: Path | str,
) -> None:
    """Write both JSON and Markdown stochastic experiment reports to disk."""
    json_p = Path(json_path)
    md_p = Path(md_path)

    json_p.parent.mkdir(parents=True, exist_ok=True)
    md_p.parent.mkdir(parents=True, exist_ok=True)

    json_p.write_text(format_stochastic_json_report(report), encoding="utf-8")
    md_p.write_text(format_stochastic_markdown_report(report), encoding="utf-8")
