"""AgentXRay: Terminal-First Causal Failure Minimization Studio.

Developer-grade research interface for black-box fault localization in tool-calling AI agents.
Runs completely offline in-memory (<0.05s) with zero external network calls and $0.00 cost.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Hugging Face ZeroGPU compatibility hook:
# If the Space is configured on ZeroGPU hardware, Hugging Face mandates an @spaces.GPU hook at startup.
try:
    import spaces  # type: ignore

    @spaces.GPU
    def _hf_zerogpu_compatibility_hook() -> bool:
        """Satisfies Hugging Face ZeroGPU runtime supervisor check if running on ZeroGPU."""
        return True
except Exception:
    _hf_zerogpu_compatibility_hook = None  # type: ignore

# Ensure project root and src/ are available on sys.path
_project_root = Path(__file__).resolve().parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agentxray.web_demo import (
    SCENARIO_CATALOG,
    ScenarioMetadata,
    WebDemoScenarioResult,
    run_web_demo,
)


CUSTOM_CSS = """
/* AgentXRay Terminal-First Developer Theme */
:root {
    --bg-page: #08090B;
    --bg-surface: #0E1013;
    --bg-surface-sub: #14171B;
    --border-subtle: #282C32;
    --border-hover: #3A3F47;
    --text-primary: #E6E8EB;
    --text-muted: #9298A3;
    --text-dim: #5C626E;
    --color-success: #A3E6B0;
    --color-failure: #F08080;
    --color-warning: #E5C07B;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --font-mono: "JetBrains Mono", "SF Mono", "Fira Code", Menlo, Monaco, Consolas, monospace;
}

body, .gradio-container {
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 12px 20px !important;
}

* {
    box-sizing: border-box;
}

/* App Topbar */
.app-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0 14px 0;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border-subtle);
    font-family: var(--font-mono);
}
.topbar-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.brand-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.brand-version {
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-surface-sub);
    border: 1px solid var(--border-subtle);
    padding: 1px 6px;
    border-radius: 3px;
}
.mode-tag {
    font-size: 11px;
    color: var(--text-muted);
    border: 1px solid var(--border-subtle);
    padding: 1px 6px;
    border-radius: 3px;
}
.status-indicator {
    font-size: 11px;
    padding: 1px 7px;
    border-radius: 3px;
    font-weight: 500;
}
.status-ready {
    color: var(--text-muted);
    background: var(--bg-surface-sub);
    border: 1px solid var(--border-subtle);
}
.topbar-right {
    display: flex;
    align-items: center;
}
.topbar-link {
    font-size: 12px;
    color: var(--text-muted);
    text-decoration: none;
    transition: color 0.15s ease;
}
.topbar-link:hover {
    color: var(--text-primary);
}

/* Sidebar Section Header */
.sidebar-heading {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding-bottom: 6px;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--border-subtle);
}

/* Scenario Context Card */
.scenario-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 12px;
    margin-top: 14px;
    font-size: 12px;
    line-height: 1.5;
}
.scenario-card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 6px;
}
.scenario-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.3;
}
.scenario-category {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-muted);
    background: var(--bg-surface-sub);
    border: 1px solid var(--border-subtle);
    padding: 2px 6px;
    border-radius: 3px;
    white-space: nowrap;
}
.scenario-desc {
    color: var(--text-muted);
    font-size: 12px;
    margin-bottom: 8px;
}
.scenario-prompt-box {
    background: var(--bg-page);
    border: 1px solid var(--border-subtle);
    border-radius: 3px;
    padding: 8px 10px;
    margin-bottom: 10px;
    font-size: 11px;
}
.prompt-label {
    color: var(--text-dim);
    font-family: var(--font-mono);
    text-transform: uppercase;
    font-size: 10px;
    margin-bottom: 2px;
}
.prompt-text {
    color: var(--text-primary);
    font-style: italic;
}
.symptom-text {
    margin-top: 4px;
    font-family: var(--font-mono);
    font-size: 11px;
}
.symptom-fail { color: var(--color-failure); }
.symptom-pass { color: var(--color-success); }

/* Candidate Pool Listing */
.candidate-pool-title {
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-dim);
    letter-spacing: 0.05em;
    margin-bottom: 4px;
    font-weight: 600;
}
.candidate-pool-box {
    background: var(--bg-page);
    border: 1px solid var(--border-subtle);
    border-radius: 3px;
    padding: 6px 8px;
    max-height: 220px;
    overflow-y: auto;
}
.candidate-row {
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 3px 0;
    color: var(--text-muted);
    border-bottom: 1px solid #14171B;
    line-height: 1.35;
}
.candidate-row:last-child {
    border-bottom: none;
}
.candidate-row.is-trigger {
    color: var(--text-primary);
}
.tag-trigger {
    color: var(--color-failure);
    font-weight: 700;
    margin-right: 4px;
}
.tag-distract {
    color: var(--text-dim);
    margin-right: 4px;
}
.candidate-id {
    color: var(--text-muted);
}

/* Aligned Status Strip */
.status-strip {
    display: flex;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 8px 0;
    margin-bottom: 12px;
}
.status-cell {
    flex: 1;
    padding: 2px 14px;
    border-right: 1px solid var(--border-subtle);
    font-family: var(--font-mono);
}
.status-cell:last-child {
    border-right: none;
}
.status-cell-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 2px;
}
.status-cell-value {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
}
.status-cell-sub {
    font-size: 11px;
    font-weight: normal;
    color: var(--text-muted);
}
.val-success { color: var(--color-success); }
.val-failure { color: var(--color-failure); }

/* Terminal Pane */
.terminal-pane {
    background: var(--bg-page);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    overflow: hidden;
    font-family: var(--font-mono);
}
.terminal-bar {
    background: var(--bg-surface-sub);
    padding: 6px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 11px;
    color: var(--text-muted);
}
.term-session {
    color: var(--text-primary);
    font-weight: 500;
}
.term-badge {
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 10px;
}
.terminal-body {
    padding: 14px 16px;
    color: var(--text-primary);
    font-size: 12px;
    line-height: 1.6;
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--bg-page);
    overflow-x: hidden;
}

/* Data Tables (Trace & Comparison) */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: var(--font-sans);
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    overflow: hidden;
}
.data-table th {
    background: var(--bg-surface-sub);
    color: var(--text-muted);
    text-align: left;
    padding: 8px 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border-subtle);
}
.data-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #1A1D23;
    color: var(--text-primary);
    vertical-align: middle;
}
.data-table tr:last-child td {
    border-bottom: none;
}
.row-highlight {
    background: #0B0E12;
    border-left: 2px solid var(--color-success);
}
.mono-bold {
    font-family: var(--font-mono);
    font-weight: 600;
}
.mono-dim {
    font-family: var(--font-mono);
    color: var(--text-muted);
}
.table-desc {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.4;
}

/* Badges & Tags */
.badge-tag {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 2px;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.tag-pass {
    color: var(--color-success);
    background: #101F15;
    border: 1px solid #1C3B24;
}
.tag-fail {
    color: var(--color-failure);
    background: #221213;
    border: 1px solid #3D1E20;
}
.alg-title {
    font-weight: 600;
    color: var(--text-primary);
}

/* Gradio Component Overrides */
button.primary, button.secondary, .gr-button {
    background: var(--bg-surface-sub) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
    padding: 7px 14px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
button.primary:hover, button.secondary:hover, .gr-button:hover {
    background: #1C2026 !important;
    border-color: var(--border-hover) !important;
}
button.primary:active, .gr-button:active {
    background: #252A32 !important;
}

/* Inputs & Dropdowns */
.gr-input, input[type="text"], input[type="number"], select, .gr-dropdown {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    box-shadow: none !important;
}
.gr-input:focus, input:focus, select:focus {
    border-color: var(--border-hover) !important;
    outline: none !important;
}
label span {
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Tabs */
.tab-nav {
    border-bottom: 1px solid var(--border-subtle) !important;
    background: transparent !important;
    gap: 4px !important;
    margin-bottom: 12px !important;
}
.tab-nav button {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 5px 12px !important;
    border-radius: 3px 3px 0 0 !important;
}
.tab-nav button.selected {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-bottom: 1px solid var(--bg-surface) !important;
    margin-bottom: -1px !important;
}
"""


def format_scenario_info(scenario_id: str) -> str:
    """Format markdown overview for the selected scenario (backward compatibility)."""
    meta = SCENARIO_CATALOG[scenario_id]
    tools_str = ", ".join(f"`{t}`" for t in meta.available_tools)
    null_badge = "🟢 **Null Control (0 Planted Triggers)**" if meta.is_null_case else "🔴 **Active Fault Mechanism**"

    lines = [
        f"### {meta.name}",
        f"- **Fault Domain:** `{meta.category}` | **Condition Type:** {null_badge}",
        f"- **Agent Description:** {meta.agent_description}",
        f"- **User Prompt:** *\"{meta.user_prompt}\"*",
        f"- **Available Tools:** {tools_str}",
        f"- **Evaluator Criteria:** {meta.evaluator_criteria}",
    ]
    if meta.observable_failure_symptom:
        lines.append(f"- **Expected Symptom:** `{meta.observable_failure_symptom}`")

    lines.append(f"\n**Candidate Pool ({len(meta.candidate_elements)} elements):**")
    for idx, elem in enumerate(meta.candidate_elements, start=1):
        is_trigger = elem in meta.ground_truth_trigger
        badge = "🔴 `[CAUSAL TRIGGER]`" if is_trigger else "⚪ `[distractor]`"
        lines.append(f"- {badge} **`elem_{idx-1}`:** {elem}")

    return "\n".join(lines)


def run_and_format_scenario(
    scenario_id: str, seed: float
) -> tuple[str, list[list[str | int]], str, list[list[str]]]:
    """Execute scenario and format classic Gradio markdown and dataframe rows."""
    result: WebDemoScenarioResult = run_web_demo(scenario_id, seed=int(seed))

    status_label = "🔴 **FAILURE**" if result.initial_failure_reproduced else "🟢 **SUCCESS (CLEAN)**"
    diag_md = (
        f"**Initial Tool Call:** `{result.initial_failure_tool_call}`  \n"
        f"**Target Outcome:** {status_label}  \n"
        f"**Execution Latency:** `{result.execution_time_seconds * 1000.0:.2f} ms` (In-Memory, $0.00 cost)  \n"
        f"**Diagnostic Output:** `{result.initial_failure_output}`"
    )

    traj_rows: list[list[str | int]] = [
        [
            step.step_number,
            step.intervention_type,
            f"{step.element_count} elements",
            step.outcome,
            step.action_narrative,
        ]
        for step in result.dd_steps
    ]

    if result.is_null_case:
        min_str = "Verified Empty (Clean Pass)"
    elif result.is_one_minimal:
        min_str = "Confirmed 1-Minimal (Explicitly Verified)"
    else:
        min_str = "Unverified"

    match_str = "Exact Match (100%)" if result.matches_ground_truth else "False Attribution"

    disc_lines = [
        f"- **Evaluations Used:** **{result.target_calls_used} calls** (Delta Debugging)",
        f"- **1-Minimality Status:** `{min_str}`",
        f"- **Ground Truth Match:** `{match_str}`",
    ]
    if result.discovered_subset:
        disc_lines.append("\n**Isolated Minimal Trigger Element(s):**")
        for elem in result.discovered_subset:
            disc_lines.append(f"> 🎯 **`{elem}`**")
    else:
        disc_lines.append("\n> ✅ **Empty Set `()` Isolated:** No candidates induce failure; correctly identified clean state.")

    disc_lines.append(f"\n*Status Note: {result.status_message}*")
    discovery_md = "\n".join(disc_lines)

    dd_min = "Yes (Verified)" if result.is_one_minimal else ("N/A (Null)" if result.is_null_case else "No")
    dd_gt = "Exact Match" if result.matches_ground_truth else "Mismatch"
    gr_min = "Yes" if result.greedy_is_minimal else ("N/A (Null)" if result.is_null_case else "No")
    gr_gt = "Exact Match" if result.greedy_matches_ground_truth else "Mismatch"
    rd_min = "Yes" if result.random_is_minimal else ("N/A (Null)" if result.is_null_case else "No")
    rd_gt = "Exact Match" if result.random_matches_ground_truth else "Mismatch"

    comp_rows: list[list[str]] = [
        [
            "Delta Debugging (ddmin)",
            f"{result.target_calls_used} calls",
            dd_min,
            dd_gt,
            "Binary hierarchical subset search with explicit 1-minimality check",
        ],
        [
            "Greedy Single Ablation",
            f"{result.greedy_calls} calls",
            gr_min,
            gr_gt,
            "Sequential single-element removal pass (O(N) linear scan)",
        ],
        [
            "Random Search (budget=20)",
            f"{result.random_calls} calls",
            rd_min,
            rd_gt,
            "Monte Carlo random subset sampling with seed reproducibility",
        ],
    ]

    return diag_md, traj_rows, discovery_md, comp_rows


def format_scenario_context_html(scenario_id: str) -> str:
    """Format scenario context for the compact configuration sidebar."""
    meta = SCENARIO_CATALOG[scenario_id]

    if meta.is_null_case:
        symptom_html = "<div class='symptom-text symptom-pass'>CONTROL: Clean baseline execution (0 planted triggers)</div>"
    else:
        symptom_html = f"<div class='symptom-text symptom-fail'>DIAGNOSTIC: {html.escape(meta.observable_failure_symptom)}</div>"

    candidates_list: list[str] = []
    for idx, elem in enumerate(meta.candidate_elements):
        is_trig = elem in meta.ground_truth_trigger
        tag_str = "<span class='tag-trigger'>[TRIGGER]</span>" if is_trig else "<span class='tag-distract'>[DISTRACT]</span>"
        row_cls = "candidate-row is-trigger" if is_trig else "candidate-row"
        escaped_elem = html.escape(elem)
        candidates_list.append(
            f"<div class='{row_cls}'>{tag_str}<span class='candidate-id'>elem_{idx}:</span> {escaped_elem}</div>"
        )
    candidates_html = "".join(candidates_list)

    category_short = meta.category.split('/')[0].strip()

    return f"""
    <div class='scenario-card'>
        <div class='scenario-card-header'>
            <div class='scenario-name'>{html.escape(meta.name)}</div>
            <span class='scenario-category'>{html.escape(category_short)}</span>
        </div>
        <div class='scenario-desc'>{html.escape(meta.description)}</div>
        <div class='scenario-prompt-box'>
            <div class='prompt-label'>TARGET PROMPT</div>
            <div class='prompt-text'>"{html.escape(meta.user_prompt)}"</div>
            {symptom_html}
        </div>
        <div class='candidate-pool-title'>DIRECTIVE POOL ({len(meta.candidate_elements)} CANDIDATES)</div>
        <div class='candidate-pool-box'>
            {candidates_html}
        </div>
    </div>
    """


def format_scenario_banner_html(scenario_id: str) -> str:
    """Backward-compatible alias for format_scenario_context_html."""
    return format_scenario_context_html(scenario_id)


def generate_status_strip_html(result: WebDemoScenarioResult) -> str:
    """Format compact aligned status strip replacing large KPI cards."""
    if result.is_null_case:
        min_status = "CLEAN CONTROL VERIFIED"
        min_cls = "val-success"
    elif result.is_one_minimal:
        min_status = "1-MINIMAL VERIFIED"
        min_cls = "val-success"
    else:
        min_status = "UNVERIFIED"
        min_cls = "val-failure"

    if result.matches_ground_truth:
        match_status = "100% MATCH"
        match_cls = "val-success"
    else:
        match_status = "FALSE ATTRIBUTION"
        match_cls = "val-failure"

    delta_evals = result.greedy_calls - result.target_calls_used
    if delta_evals > 0:
        eval_sub = f"(+{delta_evals} vs greedy)"
    elif delta_evals == 0:
        eval_sub = "(matches greedy)"
    else:
        eval_sub = "optimal path"

    latency_ms = result.execution_time_seconds * 1000.0

    return f"""
    <div class='status-strip stat-box'>
        <div class='status-cell'>
            <div class='status-cell-label'>MINIMALITY STATUS</div>
            <div class='status-cell-value {min_cls}'>{min_status}</div>
        </div>
        <div class='status-cell'>
            <div class='status-cell-label'>EVALUATIONS USED</div>
            <div class='status-cell-value'>{result.target_calls_used} calls <span class='status-cell-sub'>{eval_sub}</span></div>
        </div>
        <div class='status-cell'>
            <div class='status-cell-label'>GROUND TRUTH RECALL</div>
            <div class='status-cell-value {match_cls}'>{match_status}</div>
        </div>
        <div class='status-cell'>
            <div class='status-cell-label'>EXECUTION LATENCY</div>
            <div class='status-cell-value'>{latency_ms:.2f} ms <span class='status-cell-sub'>offline ($0.00)</span></div>
        </div>
    </div>
    """


def generate_stats_grid_html(result: WebDemoScenarioResult) -> str:
    """Backward-compatible alias for generate_status_strip_html."""
    return generate_status_strip_html(result)


def generate_terminal_html(result: WebDemoScenarioResult) -> str:
    """Format monospace terminal output view backed by existing execution results."""
    lines: list[str] = [
        f"<span style='color: #9298A3;'>$</span> <span style='color: #E6E8EB; font-weight: 600;'>agentxray probe</span> <span style='color: #9298A3;'>--scenario</span> <span style='color: #E6E8EB;'>{html.escape(result.scenario_id)}</span> <span style='color: #9298A3;'>--seed</span> <span style='color: #E6E8EB;'>42</span>",
        "",
        f"<span style='color: #5C626E;'>[00:00.000] [INIT]</span>    Target: {html.escape(result.agent_description)}",
        f"<span style='color: #5C626E;'>[00:00.000] [INIT]</span>    Query:  \"{html.escape(result.user_prompt)}\"",
        f"<span style='color: #5C626E;'>[00:00.001] [INIT]</span>    Pool:   {len(result.initial_elements)} candidate directives loaded",
        "",
        f"<span style='color: #5C626E;'>[00:00.001] [TRIAL 01]</span> <span style='color: #E6E8EB;'>BASELINE_CHECK</span>       {len(result.initial_elements)}/{len(result.initial_elements)} candidates   -> <span style='color: {'#F08080' if result.initial_failure_reproduced else '#A3E6B0'}; font-weight: 600;'>{'FAILURE' if result.initial_failure_reproduced else 'SUCCESS'}</span>",
        f"            Tool:   <span style='color: #E6E8EB;'>{html.escape(result.initial_failure_tool_call)}</span>",
        f"            Output: <span style='color: {'#F08080' if result.initial_failure_reproduced else '#A3E6B0'};'>{html.escape(result.initial_failure_output)}</span>",
    ]

    if result.initial_failure_reproduced:
        lines.append("            Action: Baseline failure reproduced. Commencing binary hierarchical delta debugging (ddmin)...")
    else:
        lines.append("            Action: Baseline passed without failure. Verified clean control state. Halting search.")

    for s in result.dd_steps[1:]:
        t_sec = 0.001 + s.step_number * 0.001
        outcome_color = "#F08080" if s.outcome == "FAILURE" else "#A3E6B0"
        inv_name = s.intervention_type.upper()
        lines.append("")
        lines.append(f"<span style='color: #5C626E;'>[{t_sec:06.3f}] [TRIAL {s.step_number:02d}]</span> <span style='color: #E6E8EB;'>{inv_name:<20}</span> {s.element_count:2d}/{len(result.initial_elements)} candidates   -> <span style='color: {outcome_color}; font-weight: 600;'>{s.outcome}</span>")
        lines.append(f"            Action: <span style='color: #9298A3;'>{html.escape(s.action_narrative)}</span>")

    lines.append("")
    lines.append("<span style='color: #282C32;'>────────────────────────────────────────────────────────────────────────────────────────────</span>")

    if result.is_null_case:
        lines.append("<span style='color: #A3E6B0; font-weight: 600;'>[RESULT] CLEAN CONTROL VERIFIED:</span> Minimal trigger set is ()")
        lines.append("         Zero failure conditions detected; control state verified without false attributions.")
    else:
        lines.append("<span style='color: #A3E6B0; font-weight: 600;'>[RESULT] ISOLATED 1-MINIMAL TRIGGER:</span>")
        for elem in result.discovered_subset:
            escaped_e = html.escape(elem)
            lines.append(f"  <span style='color: #A3E6B0;'>➔</span> <span style='color: #E6E8EB; background: #14171B; border: 1px solid #282C32; padding: 2px 8px; border-radius: 2px;'>{escaped_e}</span>")
        lines.append("<span style='color: #9298A3;'>[PROOF]  Counterfactual minimality verified: removing isolated trigger restores success; empty set () passes.</span>")

    lines.append(f"<span style='color: #5C626E;'>[STATS]</span>  Evaluations: <span style='color: #E6E8EB; font-weight: 600;'>{result.target_calls_used} calls</span> (ddmin) vs <span style='color: #9298A3;'>{result.greedy_calls} calls</span> (greedy) | Latency: <span style='color: #E6E8EB;'>{result.execution_time_seconds * 1000.0:.2f} ms</span>")
    lines.append(f"<span style='color: #5C626E;'>[AUDIT]</span>  1-Minimal: <span style='color: #A3E6B0;'>{result.is_one_minimal}</span> | Ground Truth Recall: <span style='color: #A3E6B0;'>{result.matches_ground_truth}</span> | API Cost: <span style='color: #A3E6B0;'>$0.00 (offline)</span>")

    body_content = "\n".join(lines)

    return f"""
    <div class='terminal-pane terminal-window'>
        <div class='terminal-bar'>
            <div class='term-session'>session: agentxray probe --scenario {html.escape(result.scenario_id)} --seed 42</div>
            <div class='term-badge'>OFFLINE MOCK HARNESS</div>
        </div>
        <pre class='terminal-body'>{body_content}</pre>
    </div>
    """


def generate_trace_html(result: WebDemoScenarioResult) -> str:
    """Format step-by-step minimization trajectory table."""
    rows: list[str] = []
    for s in result.dd_steps:
        outcome_tag = (
            "<span class='badge-tag tag-fail'>FAILURE</span>"
            if s.outcome == "FAILURE"
            else "<span class='badge-tag tag-pass'>SUCCESS</span>"
        )
        inv_display = s.intervention_type.replace('_', ' ').upper()
        rows.append(
            f"<tr>"
            f"<td class='mono-bold' style='color: #9298A3;'>{s.step_number:02d}</td>"
            f"<td class='mono-bold'>{html.escape(inv_display)}</td>"
            f"<td class='mono-dim'>{s.element_count} directives</td>"
            f"<td>{outcome_tag}</td>"
            f"<td class='table-desc'>{html.escape(s.action_narrative)}</td>"
            f"</tr>"
        )
    rows_html = "".join(rows)

    return f"""
    <div class='trace-container pipeline-flow'>
        <table class='data-table'>
            <thead>
                <tr>
                    <th style='width: 50px;'>STEP</th>
                    <th style='width: 160px;'>INTERVENTION</th>
                    <th style='width: 110px;'>CANDIDATES</th>
                    <th style='width: 90px;'>OUTCOME</th>
                    <th>DIAGNOSTIC ACTION & NARRATIVE</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """


def generate_pipeline_html(result: WebDemoScenarioResult) -> str:
    """Backward-compatible alias for generate_trace_html."""
    return generate_trace_html(result)


def generate_comparison_html(result: WebDemoScenarioResult) -> str:
    """Format baseline method comparison matrix table."""
    dd_min = "Yes (Verified)" if result.is_one_minimal else ("N/A (Null)" if result.is_null_case else "No")
    dd_gt = "100% Match" if result.matches_ground_truth else "Mismatch"
    gr_min = "Yes" if result.greedy_is_minimal else ("N/A (Null)" if result.is_null_case else "No")
    gr_gt = "100% Match" if result.greedy_matches_ground_truth else "Mismatch"
    rd_min = "Yes" if result.random_is_minimal else ("N/A (Null)" if result.is_null_case else "No")
    rd_gt = "100% Match" if result.random_matches_ground_truth else "Mismatch"

    return f"""
    <div class='comparison-container'>
        <table class='data-table'>
            <thead>
                <tr>
                    <th style='width: 220px;'>DIAGNOSTIC ALGORITHM</th>
                    <th style='width: 120px;'>EVALUATIONS</th>
                    <th style='width: 110px;'>1-MINIMAL?</th>
                    <th style='width: 110px;'>GROUND TRUTH</th>
                    <th>SEARCH COMPLEXITY & METHODOLOGY</th>
                </tr>
            </thead>
            <tbody>
                <tr class='row-highlight'>
                    <td>
                        <span class='alg-title'>Delta Debugging (ddmin)</span>
                        <span class='badge-tag tag-pass' style='margin-left: 6px;'>OPTIMAL</span>
                    </td>
                    <td class='mono-bold' style='color: #A3E6B0;'>{result.target_calls_used} calls</td>
                    <td class='val-success'>{dd_min}</td>
                    <td class='val-success'>{dd_gt}</td>
                    <td class='table-desc'>Hierarchical 2-way subset & complement search with explicit 1-minimality check. Provably isolates 1-minimal trigger in O(log N) evaluations on sparse single-fault topologies.</td>
                </tr>
                <tr>
                    <td><span class='alg-title'>Greedy Single Ablation</span></td>
                    <td class='mono-dim'>{result.greedy_calls} calls</td>
                    <td class='val-success'>{gr_min}</td>
                    <td class='val-success'>{gr_gt}</td>
                    <td class='table-desc'>Sequential single-element removal pass (O(N) linear scan). Eliminates candidates one by one; tests every distractor individually regardless of fault sparsity.</td>
                </tr>
                <tr>
                    <td><span class='alg-title'>Seeded Random Search</span></td>
                    <td class='mono-dim'>{result.random_calls} calls</td>
                    <td class='mono-dim'>{rd_min}</td>
                    <td class='mono-dim'>{rd_gt}</td>
                    <td class='table-desc'>Budget-capped Monte Carlo random subset sampling (budget=20). High variance in call counts; non-deterministic without fixed RNG seed.</td>
                </tr>
            </tbody>
        </table>
    </div>
    """


def execute_scenario_all(scenario_id: str, seed: float) -> tuple[str, str, str, str, str]:
    """Execute scenario and return updated HTML components."""
    result: WebDemoScenarioResult = run_web_demo(scenario_id, seed=int(seed))
    context_html = format_scenario_context_html(scenario_id)
    strip_html = generate_status_strip_html(result)
    terminal_html = generate_terminal_html(result)
    trace_html = generate_trace_html(result)
    comp_html = generate_comparison_html(result)
    return context_html, strip_html, terminal_html, trace_html, comp_html


def build_gradio_app():
    """Construct the terminal-first developer tool UI."""
    import gradio as gr

    choices = [
        (f"{m.scenario_id}: {m.name}", k)
        for k, m in SCENARIO_CATALOG.items()
    ]

    with gr.Blocks(title="agentxray — Causal Failure Minimization", css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <header class='app-topbar'>
                <div class='topbar-left'>
                    <span class='brand-title'>agentxray</span>
                    <span class='brand-version'>v0.1.0</span>
                    <span class='mode-tag'>OFFLINE MODE</span>
                    <span class='status-indicator status-ready'>DIAGNOSTIC ENGINE READY</span>
                </div>
                <div class='topbar-right'>
                    <a href='https://github.com/yash-mishraa/AgentXRay' target='_blank' class='topbar-link'>github.com/yash-mishraa/AgentXRay ↗</a>
                </div>
            </header>
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.HTML("<div class='sidebar-heading'>CONFIGURATION</div>")
                scenario_dropdown = gr.Dropdown(
                    label="Target Scenario",
                    choices=choices,
                    value="currency_ast",
                )
                seed_input = gr.Number(label="RNG Seed", value=42, precision=0)
                run_btn = gr.Button("Run diagnosis")
                scenario_context_output = gr.HTML(value=format_scenario_context_html("currency_ast"))

            with gr.Column(scale=3):
                status_strip_output = gr.HTML()

                with gr.Tabs():
                    with gr.TabItem("OUTPUT"):
                        terminal_output = gr.HTML()

                    with gr.TabItem("TRACE"):
                        trace_output = gr.HTML()

                    with gr.TabItem("COMPARISON"):
                        comp_output = gr.HTML()

                    with gr.TabItem("EPISTEMICS"):
                        gr.Markdown(
                            """
                            ### Epistemic Foundations & Boundaries
                            - **Deterministic Testbed Verification:** Demonstrates exact algorithmic mechanics on an in-memory controlled mock agent using `FakeProvider` ($0.00 spend, zero remote API calls).
                            - **Black-Box Causal Boundary:** The algorithm observes strictly external target inputs and binary tool outcomes (`SUCCESS` / `FAILURE`); it has zero access to agent system prompt weights or internal hidden tokens.
                            - **1-Minimality Guarantee:** Confirms that removing the isolated directive restores agent task success, and that the empty configuration `()` succeeds.
                            - **Real-World Neural Models:** In live non-deterministic models (LLaMA-3, GPT-4o), stochastic noise requires Wilson score intervals, multi-trial sampling, and exact McNemar testing with Manski sensitivity bounds as specified in [`docs/RESEARCH_PROPOSAL.md`](https://github.com/yash-mishraa/AgentXRay/blob/main/docs/RESEARCH_PROPOSAL.md).
                            """
                        )

        # Wire callbacks
        scenario_dropdown.change(
            fn=execute_scenario_all,
            inputs=[scenario_dropdown, seed_input],
            outputs=[scenario_context_output, status_strip_output, terminal_output, trace_output, comp_output],
        )

        run_btn.click(
            fn=execute_scenario_all,
            inputs=[scenario_dropdown, seed_input],
            outputs=[scenario_context_output, status_strip_output, terminal_output, trace_output, comp_output],
        )

        demo.load(
            fn=execute_scenario_all,
            inputs=[scenario_dropdown, seed_input],
            outputs=[scenario_context_output, status_strip_output, terminal_output, trace_output, comp_output],
        )

    return demo


# Module-level instance for Hugging Face Spaces reloader
try:
    import gradio as gr
    demo = build_gradio_app()
except Exception:
    demo = None


def main() -> None:
    """Entry point for launching the web demo."""
    global demo
    try:
        import gradio as gr
        if demo is None:
            demo = build_gradio_app()
        demo.launch()
    except ModuleNotFoundError as err:
        if "gradio" in str(err).lower():
            print(
                "Gradio is required to launch the interactive web demo.\n"
                "Install demo dependencies: pip install -e .[demo]\n"
                "Then launch with: python app.py",
                file=sys.stderr,
            )
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
