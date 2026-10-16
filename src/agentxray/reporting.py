"""Report generation for AgentXRay experiments.

Produces machine-readable JSON and human-readable Markdown reports from
actually executed experiment data. No results are hardcoded.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentxray.models import (
    ExecutionTier,
    ExperimentReport,
    ModelPilotExperimentReport,
    StochasticExperimentReport,
)
from agentxray.statistics import compare_proportions, estimate_two_proportion_power


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
        cand_ci = (
            f"{cond.candidate_evaluation.failure_rate:.2f} [{cond.candidate_evaluation.confidence_interval[0]:.2f}, {cond.candidate_evaluation.confidence_interval[1]:.2f}]"
            if cond.subset_elements
            else f"N/A (base: {cond.candidate_evaluation.failure_rate:.2f})"
        )
        lines.append(
            f"| **{eval_item.task_name}** | **Delta Debugging** | {cond_str} | "
            f"{cand_ci} | `{cond.minimality_status.value}` | {cond.target_calls_used} | "
            f"{len(eval_item.excess_elements)} | {'**MATCH**' if eval_item.matches_ground_truth else 'FAIL'} |"
        )
        for base in eval_item.baseline_comparisons:
            base_str = ", ".join(f"`{e}`" for e in base.discovered_elements) if base.discovered_elements else "*(none)*"
            base_ci = (
                f"{base.candidate_evaluation.failure_rate:.2f} [{base.candidate_evaluation.confidence_interval[0]:.2f}, {base.candidate_evaluation.confidence_interval[1]:.2f}]"
                if base.discovered_elements
                else f"N/A (base: {base.candidate_evaluation.failure_rate:.2f})"
            )
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

        # Theoretical statistical power (approximate design calculation)
        power = estimate_two_proportion_power(
            p1=eval_item.trigger_probability,
            p2=eval_item.base_probability,
            n_per_group=report.trials_per_candidate,
            alpha=0.05,
            alternative="two-sided",
        )

        lines.extend([
            f"### Task: {eval_item.task_name}",
            f"- **Description**: {eval_item.description}",
            f"- **Initial Elements ({len(eval_item.initial_elements)})**: {init_str}",
            f"- **Planted Ground Truth**: {gt_str}",
            f"- **Ground Truth Probabilities**: Base $P={eval_item.base_probability:.2f}$, Triggered $P={eval_item.trigger_probability:.2f}$ (Delta: ${eval_item.trigger_probability - eval_item.base_probability:.2f}$)",
            f"- **Statistical Power (at n={report.trials_per_candidate}, alpha=0.05)**: `{power * 100:.1f}%` *(Cohen's arcsine theoretical design calculation, not empirical power)*",
            f"- **Minimizer Precision**: `{eval_item.precision * 100:.1f}%`",
            f"- **Minimizer Recall**: `{eval_item.recall * 100:.1f}%`",
            f"- **Minimality Classification**: `{cond.minimality_status.value}`",
            f"- **Final Candidate Failure Rate**: `{cond.candidate_evaluation.failure_rate:.2f}` (95% Wilson CI: `[{cond.candidate_evaluation.confidence_interval[0]:.2f}, {cond.candidate_evaluation.confidence_interval[1]:.2f}]`, $k={cond.candidate_evaluation.failures}/{cond.candidate_evaluation.trials}$)",
            f"- **Status Narrative**: {cond.status_message}",
        ])

        if cond.removal_evaluations:
            lines.append("- **Single-Element Removal Uncertainty Checks & Risk Differences**:")
            for elem, rem_eval in cond.removal_evaluations.items():
                comp = compare_proportions(
                    k1=cond.candidate_evaluation.failures,
                    n1=cond.candidate_evaluation.trials,
                    k2=rem_eval.failures,
                    n2=rem_eval.trials,
                    alpha=0.05,
                )
                lines.append(
                    f"  - Without `{elem}`: failure rate `{rem_eval.failure_rate:.2f}` "
                    f"(95% Wilson CI: `[{rem_eval.confidence_interval[0]:.2f}, {rem_eval.confidence_interval[1]:.2f}]`, $k={rem_eval.failures}/{rem_eval.trials}$); "
                    f"Risk Difference $\\Delta={comp.difference:+.2f}$ (95% Newcombe Hybrid CI: `[{comp.difference_ci[0]:.2f}, {comp.difference_ci[1]:.2f}]`, "
                    f"$z={comp.z_statistic:.2f}$, two-sided $p={comp.p_value:.4f}$, Fisher exact $p={comp.fisher_exact_p_value:.4f}$)"
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


def format_model_pilot_json_report(report: ModelPilotExperimentReport) -> str:
    """Format a ModelPilotExperimentReport into an indented JSON string."""
    return report.model_dump_json(indent=2)


def format_model_pilot_markdown_report(report: ModelPilotExperimentReport) -> str:
    """Format a ModelPilotExperimentReport into a detailed Markdown report."""
    tier_label = f"[{report.execution_tier.value.upper()}]"
    lines: list[str] = [
        f"# AgentXRay Model-Backed Agent Pilot Report: {tier_label} {report.experiment_name}",
        "",
        f"> **Generated at**: {report.timestamp.isoformat()}  ",
        f"> **Execution Tier**: `{report.execution_tier.value.upper()}`  ",
        f"> **Provider**: `{report.provider_name}` | **Model**: `{report.model_name}`  ",
        f"> **Trials per Candidate Subset ($n$)**: `{report.trials_per_candidate}` | **Total Target Invocations**: `{report.total_target_calls}`  ",
        "",
        "---",
        "",
        "## 1. Experimental Protocol & Epistemic Boundaries",
        "",
        "### A. Validation Tier Distinction",
        "- **Software Validation**: In-memory unit and property tests verifying tool isolation, token accounting, and search invariants.",
        f"- **Synthetic Validation (`FAKE_PROVIDER`)**: {'Active in this report' if report.execution_tier.value == 'fake_provider' else 'Not active (simulated)'}. Rule-based heuristic simulation testing runner coordination without neural model drift.",
        f"- **Real-Model Observations (`LIVE_MODEL`)**: {'Active in this report' if report.execution_tier.value == 'live_model' else 'Not active (no live calls made)'}. Empirical outputs from an external language model subjected to black-box intervention experiments.",
        "",
        "### B. Research Question & Pre-Specified Hypotheses",
        "- **Research Question**: Can AgentXRay's black-box delta debugging algorithm isolate the 1-minimal failure-inducing condition in a model-backed agent using observable tool interactions alone, without access to model weights or hidden prompts?",
        "- **Hypotheses**:",
        "  * $H_1$ (Trigger Recovery): Delta debugging isolates the planted reference trigger with precision $\\ge 0.80$ and recall $= 1.0$ under active intervention.",
        "  * $H_0$ (False Discovery Resistance): Under clean baseline control (no failure trigger), delta debugging reports zero failure triggers (minimality status: `not_applicable`).",
        "  * $H_2$ (1-Minimality Verification): Every single-element removal from the isolated trigger produces a statistically significant drop in failure rate ($q < 0.05$ under Benjamini-Hochberg FDR control).",
        "",
        "### C. Epistemic Separation & Ground-Truth Methodology",
        "- **Observable Evaluator**: The outcome evaluator (`ToolCallingTaskEvaluator`) inspects ONLY runtime interaction traces (tool invocations, argument schemas, execution outputs, and final response text). It has zero access to ground truth.",
        "- **Reference Ground Truth**: Planted triggers serve as independently specified intervention hypotheses; empirical ground truth is established only if the model's observed failure rate under intervention is statistically separated from control.",
        "- **Separation of Protocol vs. Results**: Planned budgets ($n$ trials, request caps, spending limits) are fixed a priori; observed usage, failure rates, and uncertainty intervals are empirical outputs.",
        "",
        "---",
        "",
        "## 2. Executive Summary & Cost Accounting",
        "",
        report.summary,
        "",
        "### Resource Consumption & Budget Summary",
        "",
        "| Metric | Observed Usage | Configured Limit | Status |",
        "| :--- | :---: | :---: | :---: |",
    ]

    bs = report.budget_summary
    req_used = bs.get("total_requests", 0)
    req_max = bs.get("max_requests", "N/A")
    lines.append(f"| **Model Requests** | {req_used} | {req_max} | `{'OK' if req_used <= req_max else 'EXCEEDED'}` |")

    spend_used = bs.get("estimated_spend_usd", 0.0)
    spend_max = bs.get("max_spend_usd", 0.0)
    lines.append(f"| **Estimated Spend (USD)** | `${spend_used:.4f}` | `${spend_max:.4f}` | `{'OK' if spend_used <= spend_max else 'EXCEEDED'}` |")

    p_tok = bs.get("prompt_tokens", 0)
    c_tok = bs.get("completion_tokens", 0)
    t_tok = bs.get("total_tokens", p_tok + c_tok)
    lines.append(f"| **Prompt Tokens** | {p_tok:,} | N/A | Logged |")
    lines.append(f"| **Completion Tokens** | {c_tok:,} | N/A | Logged |")
    lines.append(f"| **Total Tokens** | {t_tok:,} | N/A | Logged |")
    lines.append(f"| **Retries** | {bs.get('retries', 0)} | N/A | Logged |")

    lines.extend([
        "",
        "*(Note: Estimated spend is calculated strictly from published standard per-token pricing tables; it does not represent an actual invoice.)*",
        "",
        "## 3. Benchmark Comparison Matrix",
        "",
        "| Pilot Task | Method | Discovered Trigger | Minimality Status | Target Calls | Distractors Left | Matches Ground Truth |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    for eval_item in report.evaluations:
        cond = eval_item.minimizer_condition
        trigger_str = ", ".join(f"`{e}`" for e in cond.subset_elements) if cond.subset_elements else "*(None - Null Case)*"
        match_str = "YES" if eval_item.matches_ground_truth else "NO"
        lines.append(
            f"| **{eval_item.task_name}** | Stochastic Delta Debugging | {trigger_str} | "
            f"`{cond.minimality_status.value}` | {cond.target_calls_used} | {len(eval_item.excess_elements)} | **{match_str}** |"
        )

        for b in eval_item.baseline_comparisons:
            b_trig = ", ".join(f"`{e}`" for e in b.discovered_elements) if b.discovered_elements else "*(None - Null Case)*"
            b_match = "YES" if b.matches_ground_truth else "NO"
            lines.append(
                f"| | *{b.baseline_name}* | {b_trig} | `{b.minimality_status.value}` | "
                f"{b.target_calls} | {b.irrelevant_elements_remaining} | {b_match} |"
            )

    lines.extend([
        "",
        "## 4. Detailed Experimental Evaluations",
        "",
    ])

    for eval_item in report.evaluations:
        gt_str = ", ".join(f"`{e}`" for e in eval_item.ground_truth_trigger) if eval_item.ground_truth_trigger else "*(None - Null Case / Control)*"
        init_str = ", ".join(f"`{e}`" for e in eval_item.initial_elements)
        cond = eval_item.minimizer_condition

        prec_str = f"{eval_item.precision * 100:.1f}%" if eval_item.precision is not None else "N/A (Null Case / Control)"
        rec_str = f"{eval_item.recall * 100:.1f}%" if eval_item.recall is not None else "N/A (Null Case / Control)"

        lines.extend([
            f"### Pilot Task: {eval_item.task_name}",
            f"- **Description**: {eval_item.description}",
            f"- **Initial Candidate Elements ({len(eval_item.initial_elements)})**: {init_str}",
            f"- **Planted Ground Truth**: {gt_str}",
            f"- **Minimizer Precision**: `{prec_str}` | **Recall**: `{rec_str}`",
            f"- **Minimality Classification**: `{cond.minimality_status.value}`",
            f"- **Final Candidate Failure Rate**: `{cond.candidate_evaluation.failure_rate:.2f}` (95% Wilson CI: `[{cond.candidate_evaluation.confidence_interval[0]:.2f}, {cond.candidate_evaluation.confidence_interval[1]:.2f}]`, $k={cond.candidate_evaluation.failures}/{cond.candidate_evaluation.trials}$)",
            f"- **Status Narrative**: {cond.status_message}",
        ])

        if cond.removal_evaluations:
            lines.append("- **Single-Element Removal Uncertainty Checks & Risk Differences**:")
            for elem, rem_eval in cond.removal_evaluations.items():
                comp = compare_proportions(
                    k1=cond.candidate_evaluation.failures,
                    n1=cond.candidate_evaluation.trials,
                    k2=rem_eval.failures,
                    n2=rem_eval.trials,
                    alpha=0.05,
                )
                lines.append(
                    f"  - Without `{elem}`: failure rate `{rem_eval.failure_rate:.2f}` "
                    f"(95% Wilson CI: `[{rem_eval.confidence_interval[0]:.2f}, {rem_eval.confidence_interval[1]:.2f}]`, $k={rem_eval.failures}/{rem_eval.trials}$); "
                    f"Risk Difference $\\Delta={comp.difference:+.2f}$ (95% Newcombe Hybrid CI: `[{comp.difference_ci[0]:.2f}, {comp.difference_ci[1]:.2f}]`, "
                    f"$z={comp.z_statistic:.2f}$, two-sided $p={comp.p_value:.4f}$, Fisher exact $p={comp.fisher_exact_p_value:.4f}$)"
                )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 5. Limitations and Reproducibility Notice",
        "",
    ])
    for lim in report.limitations:
        lines.append(f"* {lim}")

    lines.append("")
    return "\n".join(lines)


def write_model_pilot_reports(
    report: ModelPilotExperimentReport,
    json_path: Path | str,
    md_path: Path | str,
) -> None:
    """Write both JSON and Markdown model pilot experiment reports to disk."""
    json_p = Path(json_path)
    md_p = Path(md_path)

    json_p.parent.mkdir(parents=True, exist_ok=True)
    md_p.parent.mkdir(parents=True, exist_ok=True)

    json_p.write_text(format_model_pilot_json_report(report), encoding="utf-8")
    md_p.write_text(format_model_pilot_markdown_report(report), encoding="utf-8")

