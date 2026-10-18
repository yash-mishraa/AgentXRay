"""Report generation for AgentXRay experiments.

Produces machine-readable JSON and human-readable Markdown reports from
actually executed experiment data. No results are hardcoded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from agentxray.models import (
    ExecutionTier,
    ExperimentReport,
    ModelPilotExperimentReport,
    SearchStatus,
    StochasticExperimentReport,
    StochasticMinimalityStatus,
)
from agentxray.statistics import compare_proportions, estimate_two_proportion_power


def _format_trigger_elements(elements: Sequence[str] | tuple[str, ...], is_null: bool) -> str:
    """Format discovered trigger elements in comparison matrix, distinguishing null controls from search misses."""
    if elements:
        return ", ".join(f"`{e}`" for e in elements)
    if is_null:
        return "*(None - Null Case)*"
    return "*(none)*"


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
        is_null = getattr(eval_item, "is_null_case", False) or not bool(eval_item.ground_truth_trigger)
        cond_str = _format_trigger_elements(cond.subset_elements, is_null)
        lines.append(
            f"| **{eval_item.task_name}** | **Delta Debugging** | {cond_str} | "
            f"{'Yes' if cond.is_one_minimal else 'No'} | {cond.target_calls_used} | "
            f"{len(eval_item.excess_elements)} | {'**MATCH**' if eval_item.matches_ground_truth else 'FAIL'} |"
        )
        for base in eval_item.baseline_comparisons:
            base_str = _format_trigger_elements(base.discovered_elements, is_null)
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
        prec_str = f"`{eval_item.precision * 100:.1f}%`" if eval_item.precision is not None else "`N/A (Null Case / Control)`"
        rec_str = f"`{eval_item.recall * 100:.1f}%`" if eval_item.recall is not None else "`N/A (Null Case / Control)`"
        lines.extend([
            f"### Task: {eval_item.task_name}",
            f"- **Description**: {eval_item.description}",
            f"- **Initial Elements ({len(eval_item.initial_elements)})**: {init_str}",
            f"- **Planted Ground Truth**: {gt_str}",
            f"- **Minimizer Precision**: {prec_str}",
            f"- **Minimizer Recall**: {rec_str}",
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
        is_null = getattr(eval_item, "is_null_case", False) or not bool(eval_item.ground_truth_trigger)
        cond_str = _format_trigger_elements(cond.subset_elements, is_null)
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
            base_str = _format_trigger_elements(base.discovered_elements, is_null)
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

        prec_str = f"`{eval_item.precision * 100:.1f}%`" if eval_item.precision is not None else "`N/A (Null Case / Control)`"
        rec_str = f"`{eval_item.recall * 100:.1f}%`" if eval_item.recall is not None else "`N/A (Null Case / Control)`"
        lines.extend([
            f"### Task: {eval_item.task_name}",
            f"- **Description**: {eval_item.description}",
            f"- **Initial Elements ({len(eval_item.initial_elements)})**: {init_str}",
            f"- **Planted Ground Truth**: {gt_str}",
            f"- **Ground Truth Probabilities**: Base $P={eval_item.base_probability:.2f}$, Triggered $P={eval_item.trigger_probability:.2f}$ (Delta: ${eval_item.trigger_probability - eval_item.base_probability:.2f}$)",
            f"- **Statistical Power (at n={report.trials_per_candidate}, alpha=0.05)**: `{power * 100:.1f}%` *(Cohen's arcsine theoretical design calculation, not empirical power)*",
            f"- **Minimizer Precision**: {prec_str}",
            f"- **Minimizer Recall**: {rec_str}",
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
        "  * $H_2$ (1-Minimality Verification): Every single-element removal from the isolated trigger produces a statistically significant drop in failure rate (each removal test $p < \\alpha$ under Berger's (1982) Intersection-Union Test at $\\alpha = 0.05$).",
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
    req_status = "OK" if not isinstance(req_max, (int, float)) or req_used <= req_max else "EXCEEDED"
    lines.append(f"| **Attempted Requests (Total)** | {req_used} | {req_max} | `{req_status}` |")

    succ_req = bs.get("successful_requests", 0)
    failed_req = bs.get("failed_requests", 0)
    retries = bs.get("retries", 0)
    lines.append(f"| **Successful Responses** | {succ_req} | N/A | Logged |")
    lines.append(f"| **Failed Request Invocations** | {failed_req} | N/A | Logged |")
    lines.append(f"| **Retried Invocations** | {retries} | N/A | Logged |")

    spend_used = bs.get("estimated_spend_usd", 0.0)
    spend_max = bs.get("max_spend_usd", "N/A")
    spend_max_str = f"${spend_max:.4f}" if isinstance(spend_max, (int, float)) else "N/A"
    spend_status = "OK" if not isinstance(spend_max, (int, float)) or spend_used <= spend_max else "EXCEEDED"
    lines.append(f"| **Estimated Billable Spend (USD)** | `${spend_used:.4f}` | {spend_max_str} | `{spend_status}` |")

    p_tok = bs.get("prompt_tokens", 0)
    c_tok = bs.get("completion_tokens", 0)
    t_tok = bs.get("total_tokens", p_tok + c_tok)
    lines.append(f"| **Billable Prompt Tokens** | {p_tok:,} | N/A | Logged |")
    lines.append(f"| **Billable Completion Tokens** | {c_tok:,} | N/A | Logged |")
    lines.append(f"| **Total Billable Tokens** | {t_tok:,} | N/A | Logged |")

    algo_usage = bs.get("per_algorithm_usage", [])
    if algo_usage:
        lines.extend([
            "",
            "### Phase & Algorithm Usage Breakdown",
            "",
            "| Task | Algorithm | Discovery Reqs | Verif Reqs | Total Reqs | Discovery Spend | Verif Spend | Total Spend | Verif Status | Evidence Tier (Requested) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for row in algo_usage:
            lines.append(
                f"| `{row.get('task_name', 'N/A')}` | {row.get('algorithm', 'N/A')} | "
                f"{row.get('discovery_requests', 0)} | {row.get('verification_requests', 0)} | "
                f"{row.get('model_requests', 0)} | ${row.get('discovery_spend_usd', 0.0):.4f} | "
                f"${row.get('verification_spend_usd', 0.0):.4f} | ${row.get('estimated_spend_usd', 0.0):.4f} | "
                f"`{row.get('verification_status', 'skipped')}` | `{row.get('evidence_tier') or 'N/A'}` |"
            )

        has_unresolved_confirmatory = any(
            row.get("evidence_tier") == "confirmatory" and (
                not row.get("confirmatory_power_established", False)
                or row.get("sample_size_validation_status") in ("unresolved", "not_implemented")
                or "unresolved" in str(row.get("sample_size_rationale", ""))
            )
            for row in algo_usage
        )
        if has_unresolved_confirmatory:
            lines.extend([
                "",
                "> [!WARNING]",
                "> **Unresolved Confirmatory Sample Size**: Confirmatory verification was requested with an unresolved sample-size/power rationale. Confirmatory statistical power has NOT been established; these results must not be interpreted as adequately powered confirmatory evidence.",
            ])

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
        is_null = getattr(eval_item, "is_null_case", False) or not bool(eval_item.ground_truth_trigger)
        trigger_str = _format_trigger_elements(cond.subset_elements, is_null)
        match_str = "YES" if eval_item.matches_ground_truth else "NO"
        lines.append(
            f"| **{eval_item.task_name}** | Stochastic Delta Debugging | {trigger_str} | "
            f"`{cond.minimality_status.value}` | {cond.target_calls_used} | {len(eval_item.excess_elements)} | **{match_str}** |"
        )

        for b in eval_item.baseline_comparisons:
            b_trig = _format_trigger_elements(b.discovered_elements, is_null)
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

        v_stat = getattr(cond, "verification_status", None) or "skipped"
        e_tier = getattr(cond, "evidence_tier", None) or "N/A"
        min_stat = getattr(cond, "minimality_status", None)
        min_stat_val = min_stat.value if hasattr(min_stat, "value") else str(min_stat)
        ss_rat = getattr(cond, "sample_size_rationale", None) or "N/A"
        val_stat = getattr(cond, "sample_size_validation_status", "unresolved")
        power_est = getattr(cond, "confirmatory_power_established", False)

        task_eval_lines = [
            f"### Pilot Task: {eval_item.task_name}",
            f"- **Description**: {eval_item.description}",
            f"- **Initial Candidate Elements ({len(eval_item.initial_elements)})**: {init_str}",
            f"- **Planted Ground Truth**: {gt_str}",
            f"- **Minimizer Precision**: `{prec_str}` | **Recall**: `{rec_str}`",
            f"- **Minimality Classification**: `{min_stat_val}`",
            f"- **Statistical Minimality Outcome**: `{min_stat_val}`",
            f"- **Verification Status**: `{v_stat}`",
            f"- **Evidence Tier**: `{e_tier}`",
            f"- **Requested Evidence Tier**: `{e_tier}`",
            f"- **Sample Size Rationale**: `{ss_rat}`",
            f"- **Sample Size Rationale Context**: `{ss_rat}`",
            f"- **Rationale Validation Status**: `{val_stat}`",
            f"- **Confirmatory Power Established**: `{'YES' if power_est else 'NO (Unresolved / Unvalidated)'}`",
        ]
        prospective_val = getattr(cond, "prospective_power_validated", False)
        planned_adeq = getattr(cond, "planned_sample_size_adequate", False)
        proto_adh = getattr(cond, "execution_protocol_adherent", True)
        miss_stat = getattr(cond, "missingness_sensitivity_status", "not_applicable")

        if prospective_val or planned_adeq or not proto_adh or miss_stat != "not_applicable":
            task_eval_lines.extend([
                f"- **Prospective Power Validated**: `{'YES' if prospective_val else 'NO'}`",
                f"- **Planned Sample Size Adequate**: `{'YES' if planned_adeq else 'NO'}`",
                f"- **Protocol Adherence**: `{'YES (100% complete)' if proto_adh else 'NO (Protocol deviation / attrition)'}`",
                f"- **Missingness Sensitivity**: `{miss_stat}`",
            ])

        if e_tier == "confirmatory" and (not power_est or val_stat in ("unresolved", "not_implemented")):
            task_eval_lines.append(
                "- **Evidentiary Validity Disclosure**: Confirmatory power has not been established (sample-size rationale is unresolved/not implemented; results should not be interpreted as adequately powered confirmatory evidence)."
            )
        if cond.candidate_evaluation is not None:
            task_eval_lines.append(
                f"- **Final Candidate Failure Rate**: `{cond.candidate_evaluation.failure_rate:.2f}` (95% Wilson CI: `[{cond.candidate_evaluation.confidence_interval[0]:.2f}, {cond.candidate_evaluation.confidence_interval[1]:.2f}]`, $k={cond.candidate_evaluation.failures}/{cond.candidate_evaluation.trials}$)"
            )
        else:
            task_eval_lines.append("- **Final Candidate Failure Rate**: `N/A`")
        task_eval_lines.append(f"- **Status Narrative**: {cond.status_message}")
        lines.extend(task_eval_lines)

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


def _is_budget_exhausted(eval_item: Any) -> bool:
    """Determine whether a task evaluation hit the execution or search budget limit."""
    cond = getattr(eval_item, "minimizer_condition", None)
    status = getattr(cond, "status", None) or getattr(eval_item, "status", None)
    min_status = getattr(cond, "minimality_status", None) or getattr(eval_item, "minimality_status", None)
    return (
        status == SearchStatus.BUDGET_EXHAUSTED
        or getattr(status, "value", None) == SearchStatus.BUDGET_EXHAUSTED.value
        or min_status == StochasticMinimalityStatus.BUDGET_EXHAUSTED
        or getattr(min_status, "value", None) == StochasticMinimalityStatus.BUDGET_EXHAUSTED.value
        or bool(getattr(cond, "budget_exhausted", False))
        or bool(getattr(eval_item, "budget_exhausted", False))
    )


def _is_timed_out(eval_item: Any) -> bool:
    """Determine whether a task evaluation exceeded wall-clock timeout."""
    cond = getattr(eval_item, "minimizer_condition", None)
    status = getattr(cond, "status", None) or getattr(eval_item, "status", None)
    min_status = getattr(cond, "minimality_status", None) or getattr(eval_item, "minimality_status", None)
    return (
        status == SearchStatus.TIMEOUT
        or getattr(status, "value", None) == SearchStatus.TIMEOUT.value
        or min_status == "timeout"
        or bool(getattr(cond, "timed_out", False))
        or bool(getattr(eval_item, "timed_out", False))
    )


def _is_inconclusive(eval_item: Any) -> bool:
    """Determine whether a task evaluation yielded an inconclusive minimality outcome."""
    cond = getattr(eval_item, "minimizer_condition", None)
    min_status = getattr(cond, "minimality_status", None) or getattr(eval_item, "minimality_status", None)
    return (
        min_status == StochasticMinimalityStatus.INCONCLUSIVE
        or min_status == "inconclusive"
        or getattr(min_status, "value", None) == "inconclusive"
    )


def _is_not_minimal(eval_item: Any) -> bool:
    """Determine whether a task evaluation was classified as NOT_MINIMAL."""
    cond = getattr(eval_item, "minimizer_condition", None)
    min_status = getattr(cond, "minimality_status", None) or getattr(eval_item, "minimality_status", None)
    return (
        min_status == StochasticMinimalityStatus.NOT_MINIMAL
        or min_status == "not_minimal"
        or getattr(min_status, "value", None) == "not_minimal"
    )


def _is_observed_only(eval_item: Any) -> bool:
    """Determine whether a task evaluation was classified as OBSERVED_ONLY."""
    cond = getattr(eval_item, "minimizer_condition", None)
    min_status = getattr(cond, "minimality_status", None) or getattr(eval_item, "minimality_status", None)
    return (
        min_status == StochasticMinimalityStatus.OBSERVED_ONLY
        or min_status == "observed_only"
        or getattr(min_status, "value", None) == "observed_only"
    )


def _is_candidate_confirmation_failure(eval_item: Any) -> bool:
    """Determine whether post-discovery candidate verification failed to reproduce target failure."""
    cond = getattr(eval_item, "minimizer_condition", None)
    if cond is None:
        return False
    msg = getattr(cond, "status_message", "")
    return "Candidate failed to reproduce failure rate" in msg


def _is_interrupted(eval_item: Any) -> bool:
    """Determine whether execution was interrupted by budgets or timeouts in discovery or verification."""
    if _is_budget_exhausted(eval_item) or _is_timed_out(eval_item):
        return True
    cond = getattr(eval_item, "minimizer_condition", None)
    v_stat = getattr(cond, "verification_status", None)
    if v_stat in ("budget_exhausted", "timed_out", "partial"):
        return True
    return False


def compute_stochastic_aggregate_metrics(evaluations: Sequence[Any]) -> dict[str, Any]:
    """Compute partitioned aggregate metrics across active failure tasks and control tasks.

    Parameters
    ----------
    evaluations : Sequence[Any]
        Evaluation records (StochasticEvaluationResult or ModelPilotEvaluationResult).

    Returns
    -------
    dict[str, Any]
        Dictionary with aggregate metrics:
        - tasks_evaluated: Total count of evaluated tasks (the Intention-to-Treat denominator).
        - ground_truth_matches: Tasks matching ground truth exactly.
        - statistically_supported_minimality: Tasks achieving STATISTICALLY_SUPPORTED minimality.
        - inconclusive_tasks: Tasks with inconclusive minimality classification.
        - not_minimal_tasks: Tasks with rejected minimality (not 1-minimal).
        - observed_only_tasks: Tasks with observed but unconfirmed minimality.
        - candidate_confirmation_failures: Tasks where candidate failed post-discovery confirmation.
        - interrupted_tasks: Tasks interrupted by budget or timeout.
        - active_tasks: Count of active (non-null) failure tasks.
        - active_mean_precision: Mean precision across active tasks.
        - active_mean_recall: Mean recall across active tasks.
        - null_tasks: Count of null control tasks.
        - null_detection_rate: Proportion of null controls matching ground truth (clean non-failure).
        - null_false_positive_rate: Overall control false-alarm rate (1.0 - null_detection_rate).
        - null_element_false_positives: Count of null controls with non-empty discovered elements.
        - null_spurious_failures: Count of null controls with empty discovered sets that spuriously reproduced failure.
        - budget_exhausted_tasks: Count of evaluated tasks where the minimizer exhausted the trial budget.
        - timed_out_tasks: Count of evaluated tasks where the minimizer exceeded execution timeout.
        - mean_precision: Legacy mean precision across all evaluations where precision is not None.
        - mean_recall: Legacy mean recall across all evaluations where recall is not None.
        - mean_precision_evaluations: Denominator count for mean_precision.
        - mean_recall_evaluations: Denominator count for mean_recall.

    Diagnostic Counter Overlap Notice
    ---------------------------------
    The diagnostic counters `inconclusive_tasks`, `candidate_confirmation_failures`,
    and `interrupted_tasks` represent diagnostic subsets and can OVERLAP:
    - All candidate confirmation failures are inconclusive (`candidate_confirmation_failures`
      is a diagnostic subset of `inconclusive_tasks`).
    - An interrupted task can be classified as inconclusive (counted in both
      `interrupted_tasks` and `inconclusive_tasks`) or not minimal (counted in both
      `interrupted_tasks` and `not_minimal_tasks`).
    Therefore, these diagnostic counts MUST NOT be summed together as if they
    formed a disjoint partition of `tasks_evaluated`. The primary Intention-to-Treat
    (ITT) success rate denominator is always `tasks_evaluated`.
    """
    active_evals = [
        e for e in evaluations
        if (not getattr(e, "is_null_case", False)) and bool(e.ground_truth_trigger)
    ]
    null_evals = [
        e for e in evaluations
        if getattr(e, "is_null_case", False) or not bool(e.ground_truth_trigger)
    ]

    active_precisions = [e.precision for e in active_evals if e.precision is not None]
    active_recalls = [e.recall for e in active_evals if e.recall is not None]
    active_mean_prec = (
        round(sum(active_precisions) / len(active_precisions), 4)
        if active_precisions
        else None
    )
    active_mean_rec = (
        round(sum(active_recalls) / len(active_recalls), 4)
        if active_recalls
        else None
    )

    null_matches = sum(1 for e in null_evals if e.matches_ground_truth)
    null_fps = len(null_evals) - null_matches
    null_element_false_positives = sum(
        1 for e in null_evals if bool(e.minimizer_condition.subset_elements)
    )
    null_spurious_failures = sum(
        1 for e in null_evals
        if (not bool(e.minimizer_condition.subset_elements)) and (not e.matches_ground_truth)
    )
    null_detection_rate = (
        round(null_matches / len(null_evals), 4)
        if null_evals
        else None
    )
    null_fp_rate = (
        round(null_fps / len(null_evals), 4)
        if null_evals
        else None
    )

    valid_precisions = [e.precision for e in evaluations if e.precision is not None]
    valid_recalls = [e.recall for e in evaluations if e.recall is not None]

    matched_count = sum(1 for e in evaluations if e.matches_ground_truth)
    stat_supp_count = sum(
        1 for e in evaluations
        if getattr(e.minimizer_condition, "minimality_status", None) == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    )
    budget_exhausted_tasks = sum(1 for e in evaluations if _is_budget_exhausted(e))
    timed_out_tasks = sum(1 for e in evaluations if _is_timed_out(e))

    inconclusive_tasks = sum(1 for e in evaluations if _is_inconclusive(e))
    not_minimal_tasks = sum(1 for e in evaluations if _is_not_minimal(e))
    observed_only_tasks = sum(1 for e in evaluations if _is_observed_only(e))
    candidate_confirmation_failures = sum(1 for e in evaluations if _is_candidate_confirmation_failure(e))
    interrupted_tasks = sum(1 for e in evaluations if _is_interrupted(e))

    return {
        "tasks_evaluated": len(evaluations),
        "ground_truth_matches": matched_count,
        "statistically_supported_minimality": stat_supp_count,
        "inconclusive_tasks": inconclusive_tasks,
        "not_minimal_tasks": not_minimal_tasks,
        "observed_only_tasks": observed_only_tasks,
        "candidate_confirmation_failures": candidate_confirmation_failures,
        "interrupted_tasks": interrupted_tasks,
        "active_tasks": len(active_evals),
        "active_mean_precision": active_mean_prec,
        "active_mean_recall": active_mean_rec,
        "null_tasks": len(null_evals),
        "null_detection_rate": null_detection_rate,
        "null_false_positive_rate": null_fp_rate,
        "null_element_false_positives": null_element_false_positives,
        "null_spurious_failures": null_spurious_failures,
        "budget_exhausted_tasks": budget_exhausted_tasks,
        "timed_out_tasks": timed_out_tasks,
        "mean_precision": round(sum(valid_precisions) / len(valid_precisions), 4) if valid_precisions else None,
        "mean_recall": round(sum(valid_recalls) / len(valid_recalls), 4) if valid_recalls else None,
        "mean_precision_evaluations": len(valid_precisions),
        "mean_recall_evaluations": len(valid_recalls),
    }


def format_cohort_summary_sentences(metrics: dict[str, Any]) -> list[str]:
    """Generate narrative summary sentences for active and control cohorts from aggregate metrics."""
    parts: list[str] = []
    if metrics.get("active_tasks", 0) > 0:
        prec = metrics.get("active_mean_precision")
        rec = metrics.get("active_mean_recall")
        prec_str = str(prec) if prec is not None else "N/A"
        rec_str = str(rec) if rec is not None else "N/A"
        parts.append(
            f"Active tasks (n={metrics['active_tasks']}): mean precision={prec_str}, mean recall={rec_str}."
        )
    if metrics.get("null_tasks", 0) > 0:
        det_rate = metrics.get("null_detection_rate")
        fp_rate = metrics.get("null_false_positive_rate")
        det_str = str(det_rate) if det_rate is not None else "N/A"
        fp_str = str(fp_rate) if fp_rate is not None else "N/A"
        null_tasks = metrics["null_tasks"]
        null_matches = null_tasks - (metrics.get("null_element_false_positives", 0) + metrics.get("null_spurious_failures", 0))
        elem_fps = metrics.get("null_element_false_positives", 0)
        spur_fails = metrics.get("null_spurious_failures", 0)
        parts.append(
            f"Control tasks (n={null_tasks}): clean null detection rate={det_str} "
            f"({null_matches}/{null_tasks}), false positive rate={fp_str} "
            f"(element false discoveries: {elem_fps}, spurious empty-set failures: {spur_fails})."
        )
    return parts


def format_exploratory_benchmark_markdown(report: Any) -> str:
    """Format an ExploratoryBenchmarkReport into standard GitHub-flavored Markdown."""
    lines: list[str] = [
        f"# {report.experiment_name} — Research Summary",
        "",
        "> **Methodological Status:** Exploratory Benchmark (Finite-Domain Pilot).",
        "> **Inferential Scope:** Descriptive bounds describe the retained benchmark cohort.",
        "> They do not automatically bound target-population risk if infrastructure attrition occurs.",
        "",
        "## 1. Analysis Cohort and Attrition Accounting",
        "",
        "| Cohort Metric | Count |",
        "| :--- | :--- |",
        f"| Total Planned Mechanisms ($N_{{\\text{{total}}}}$) | {report.n_total} |",
        f"| Excluded: Unresolved Infrastructure Failures ($N_{{\\text{{infra}}}}$) | {report.n_infra} |",
        f"| **Retained Analysis Cohort Denominator ($N_{{\\text{{evaluable}}}}$)** | **{report.n_evaluable}** |",
        f"| Jointly Evaluable Complete Pairs ($N_{{\\text{{complete}}}}$) | {report.n_complete} |",
        f"| Missing Diagnoses: Passive Judge (Malformed Output) | {report.n_miss_judge} |",
        f"| Missing Diagnoses: Active DD (Pre-Candidate Exhaustion) | {report.n_miss_activedd} |",
        "",
        "## 2. Primary Endpoint: Per-Mechanism False-Attribution Risk Difference",
        "",
        f"- **Complete-Case Risk Difference ($\\Delta_{{\\text{{Risk, CC}}}}$):** {report.complete_case_risk_difference}",
        f"- **Discordant Pairs:** $b = {report.mcnemar_b}$ (Judge false attribution, ActiveDD correct), $c = {report.mcnemar_c}$ (Judge correct, ActiveDD false attribution)",
        f"- **Total Discordant Pairs ($n_d$):** {report.mcnemar_n_discordant}",
        f"- **Discordant Odds Ratio:** {report.mcnemar_odds_ratio}",
        f"- **Exact Directional McNemar p-value ($H_1: b > c$):** {report.mcnemar_p_value:.5f}",
        f"- **Statistically Significant ($\alpha=0.05$):** {'YES' if report.mcnemar_statistically_significant else 'NO'}",
        "",
        "## 3. Contrast-Specific Non-Parametric Sensitivity Bounds",
        "",
        f"- **Cohort Sensitivity Envelope on $N_{{\\text{{evaluable}}}}$:** $[{report.sensitivity_lower_bound:.4f}, {report.sensitivity_upper_bound:.4f}]$",
        "- **Upper Bound ($\\Delta_{\\text{Risk}}^{\\max}$):** Imputes missing Judge outcomes as 1, missing ActiveDD outcomes as 0 (wholly missing pairs contribute $+1$).",
        "- **Lower Bound ($\\Delta_{\\text{Risk}}^{\\min}$):** Imputes missing Judge outcomes as 0, missing ActiveDD outcomes as 1 (wholly missing pairs contribute $-1$).",
        "- **Wholly Missing Pair Impact:** Mechanisms where both methods lack valid diagnoses widen the sensitivity interval by $2 / N_{\\text{evaluable}}$ and do not cancel.",
        "",
        "## 4. Per-Mechanism Three-Way Diagnostic Results",
        "",
        "| Mechanism ID | Category | Planted Trigger | Active DD ($Y, R$) | Greedy Ablation ($Y, R$) | LLM Judge ($Y, R$) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for ev in getattr(report, "evaluations", ()):
        dd = ev.active_dd_outcome
        gr = ev.greedy_ablation_outcome
        jd = ev.llm_judge_outcome
        gt_str = f"`{list(ev.ground_truth_trigger)}`" if ev.ground_truth_trigger else "*(Null)*"
        dd_str = f"Y={dd.false_attribution_indicator}, R={dd.exact_recovery_indicator}"
        gr_str = f"Y={gr.false_attribution_indicator}, R={gr.exact_recovery_indicator}"
        jd_str = f"Y={jd.false_attribution_indicator}, R={jd.exact_recovery_indicator}"
        lines.append(f"| `{ev.mechanism_id}` | {ev.category} | {gt_str} | {dd_str} | {gr_str} | {jd_str} |")

    lines.extend([
        "",
        "## 5. Secondary Metrics Summary",
        "",
    ])
    sec = getattr(report, "secondary_metrics", {})
    if "pooled_fdr" in sec:
        lines.append(f"- **Discovery-Level Pooled FDR:** Active DD={sec['pooled_fdr'].get('active_dd')}, Greedy={sec['pooled_fdr'].get('greedy_ablation')}, Judge={sec['pooled_fdr'].get('llm_judge')}")
    if "mean_fdp" in sec:
        lines.append(f"- **Mean Per-Mechanism FDP:** Active DD={sec['mean_fdp'].get('active_dd')}, Greedy={sec['mean_fdp'].get('greedy_ablation')}, Judge={sec['mean_fdp'].get('llm_judge')}")
    if "exact_recovery_rate" in sec:
        lines.append(f"- **Exact Recovery Rate ($R_{{\\text{{exact}}}}$):** Active DD={sec['exact_recovery_rate'].get('active_dd')}, Greedy={sec['exact_recovery_rate'].get('greedy_ablation')}, Judge={sec['exact_recovery_rate'].get('llm_judge')}")
    if "mean_requests" in sec:
        lines.append(f"- **Mean Model Requests:** Active DD={sec['mean_requests'].get('active_dd')}, Greedy={sec['mean_requests'].get('greedy_ablation')}, Judge={sec['mean_requests'].get('llm_judge')}")

    lines.extend([
        "",
        "## 6. Scientific Limitations",
        "",
    ])
    for lim in getattr(report, "limitations", ()):
        lines.append(f"- {lim}")

    return "\n".join(lines)


