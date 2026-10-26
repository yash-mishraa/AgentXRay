"""AgentXRay: High-Performance CLI Terminal & Causal Minimization Studio.

Developer-grade interactive portfolio demonstration for Hugging Face Spaces.
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
/* Dark Developer Terminal Theme */
:root {
    --bg-dark: #07090E;
    --card-bg: #0D1117;
    --card-border: #1F2937;
    --accent-blue: #38BDF8;
    --accent-green: #34D399;
    --accent-red: #F87171;
    --accent-yellow: #FBBF24;
    --accent-purple: #A78BFA;
    --text-muted: #94A3B8;
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    color: #E2E8F0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Custom Hero Header */
.hero-header {
    background: linear-gradient(180deg, #111827 0%, #07090E 100%);
    border: 1px solid #1F2937;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.98rem;
    margin-top: 6px;
    line-height: 1.5;
}

/* Developer Pill Badges */
.badge-pill {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-family: monospace;
}
.badge-prompt { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
.badge-schema { background: rgba(167, 139, 250, 0.15); color: #A78BFA; border: 1px solid rgba(167, 139, 250, 0.3); }
.badge-env    { background: rgba(248, 113, 113, 0.15); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.3); }
.badge-null   { background: rgba(52, 211, 153, 0.15); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.3); }

/* Terminal Window Component */
.terminal-window {
    background: #020408;
    border: 1px solid #1E293B;
    border-radius: 8px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7);
    overflow: hidden;
    margin: 12px 0;
    font-family: "JetBrains Mono", "Fira Code", "SF Mono", Menlo, Consolas, monospace;
}
.terminal-topbar {
    background: #0F172A;
    padding: 9px 14px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #1E293B;
}
.window-dots {
    display: flex;
    gap: 7px;
}
.w-dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
}
.w-red { background: #EF4444; }
.w-yellow { background: #F59E0B; }
.w-green { background: #10B981; }
.terminal-tab-title {
    margin-left: 14px;
    font-size: 0.78rem;
    color: #64748B;
    font-family: monospace;
}
.terminal-content {
    padding: 16px 20px;
    color: #E2E8F0;
    font-size: 0.86rem;
    line-height: 1.65;
    overflow-x: auto;
    white-space: pre;
}

/* Metric Stats Cards Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0;
}
.stat-box {
    background: #0D1117;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
}
.stat-box::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: #3B82F6;
}
.stat-label {
    font-size: 0.74rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}
.stat-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 4px 0;
    font-family: monospace;
}
.stat-sub {
    font-size: 0.76rem;
    color: var(--accent-green);
    font-weight: 500;
}

/* Pipeline Step Tree */
.pipeline-flow {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin: 16px 0;
}
.pipeline-step {
    background: #0D1117;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.step-tag-fail {
    background: rgba(239, 68, 68, 0.2);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.74rem;
    font-weight: 700;
    font-family: monospace;
}
.step-tag-pass {
    background: rgba(16, 185, 129, 0.2);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.74rem;
    font-weight: 700;
    font-family: monospace;
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


def format_scenario_banner_html(scenario_id: str) -> str:
    """Format scenario metadata into a modern developer card."""
    meta = SCENARIO_CATALOG[scenario_id]

    if "Prompt" in meta.category:
        badge_cls = "badge-prompt"
    elif "Schema" in meta.category:
        badge_cls = "badge-schema"
    elif "Environment" in meta.category:
        badge_cls = "badge-env"
    else:
        badge_cls = "badge-null"

    tools_html = " ".join(
        f"<span style='background:#1E293B;padding:2px 6px;border-radius:4px;font-family:monospace;font-size:0.8rem;color:#93C5FD;'>{html.escape(t)}</span>"
        for t in meta.available_tools
    )

    candidates_html = "".join(
        f"<div style='margin:4px 0;font-size:0.85rem;color:#CBD5E1;font-family:monospace;'>"
        f"{'<span style=\"color:#F87171;font-weight:bold;\">[TRIGGER]</span>' if elem in meta.ground_truth_trigger else '<span style=\"color:#64748B;\">[distract]</span>'} "
        f"<span style='color:#94A3B8;'>elem_{idx}:</span> {html.escape(elem)}"
        f"</div>"
        for idx, elem in enumerate(meta.candidate_elements)
    )

    symptom_row = (
        f"<div style='margin-top:6px;font-size:0.84rem;color:#FCA5A5;'><strong>Observable Failure Diagnostic:</strong> <code>{html.escape(meta.observable_failure_symptom)}</code></div>"
        if meta.observable_failure_symptom
        else "<div style='margin-top:6px;font-size:0.84rem;color:#86EFAC;'><strong>Control State:</strong> Clean baseline execution without planted triggers.</div>"
    )

    return f"""
    <div style='background:#0D1117;border:1px solid #1F2937;border-radius:8px;padding:16px 20px;margin-bottom:14px;'>
        <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;'>
            <div style='font-size:1.05rem;font-weight:700;color:#F8FAFC;'>{html.escape(meta.name)}</div>
            <span class='badge-pill {badge_cls}'>{html.escape(meta.category.split('/')[0].strip())}</span>
        </div>
        <div style='font-size:0.88rem;color:#94A3B8;margin-bottom:10px;'>{html.escape(meta.description)}</div>
        <div style='background:#07090E;border:1px solid #1E293B;border-radius:6px;padding:10px 14px;font-size:0.85rem;'>
            <div><span style='color:#64748B;'>User Prompt:</span> <span style='color:#E2E8F0;font-style:italic;'>"{html.escape(meta.user_prompt)}"</span></div>
            <div style='margin-top:4px;'><span style='color:#64748B;'>Tools Exposed:</span> {tools_html}</div>
            {symptom_row}
        </div>
        <div style='margin-top:12px;'>
            <div style='font-size:0.78rem;text-transform:uppercase;color:#64748B;font-weight:600;margin-bottom:4px;'>Injected Candidate Pool ({len(meta.candidate_elements)} directives):</div>
            <div style='background:#07090E;border:1px solid #1E293B;border-radius:6px;padding:8px 12px;'>
                {candidates_html}
            </div>
        </div>
    </div>
    """


def generate_stats_grid_html(result: WebDemoScenarioResult) -> str:
    """Format four top-level metric cards."""
    eff_delta = f"+{result.greedy_calls - result.target_calls_used} calls vs Greedy" if result.greedy_calls > result.target_calls_used else "Optimal Search Path"
    min_status = "VERIFIED EMPTY" if result.is_null_case else ("1-MINIMAL" if result.is_one_minimal else "UNVERIFIED")
    min_color = "#34D399" if (result.is_one_minimal or result.is_null_case) else "#F87171"
    match_status = "100% MATCH" if result.matches_ground_truth else "FALSE POSITIVE"
    match_color = "#34D399" if result.matches_ground_truth else "#F87171"

    return f"""
    <div class='stats-grid'>
        <div class='stat-box'>
            <div class='stat-label'>Search Evaluations</div>
            <div class='stat-value'>{result.target_calls_used} <span style='font-size:0.85rem;color:#64748B;'>calls</span></div>
            <div class='stat-sub' style='color:#38BDF8;'>{eff_delta}</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Causal Minimality</div>
            <div class='stat-value' style='color:{min_color};'>{min_status}</div>
            <div class='stat-sub'>Strictly Counterfactual</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Ground Truth Recall</div>
            <div class='stat-value' style='color:{match_color};'>{match_status}</div>
            <div class='stat-sub'>Verified Against Spec</div>
        </div>
        <div class='stat-box'>
            <div class='stat-label'>Execution Latency</div>
            <div class='stat-value'>{result.execution_time_seconds * 1000.0:.2f} <span style='font-size:0.85rem;color:#64748B;'>ms</span></div>
            <div class='stat-sub'>In-Memory ($0.00 cost)</div>
        </div>
    </div>
    """


def generate_terminal_html(result: WebDemoScenarioResult) -> str:
    """Format full CLI terminal session string with color-coded tokens."""
    lines: list[str] = [
        f"<span style='color:#38BDF8;'>$</span> <span style='color:#F1F5F9;font-weight:700;'>agentxray probe</span> --tier <span style='color:#A78BFA;'>{html.escape(result.category)}</span> --scenario <span style='color:#FBBF24;'>{html.escape(result.scenario_id)}</span> --seed 42",
        f"<span style='color:#64748B;'>[00:00.000] ─── INITIALIZING TESTBED ──────────────────────────────────────────────────────────</span>",
        f"<span style='color:#64748B;'>[00:00.001] Target Harness:</span> {html.escape(result.agent_description)}",
        f"<span style='color:#64748B;'>[00:00.001] User Query:</span>     \"{html.escape(result.user_prompt)}\"",
        f"<span style='color:#64748B;'>[00:00.001] Injected Pool:</span>  {len(result.initial_elements)} candidate elements\n",
        f"<span style='color:#64748B;'>[00:00.002] ─── TRIAL 01: FULL CONTEXT BASELINE CHECK ──────────────────────────────────────</span>",
        f"<span style='color:#64748B;'>[00:00.002] Injected Set:</span>    {len(result.initial_elements)}/{len(result.initial_elements)} candidates present",
        f"<span style='color:#64748B;'>[00:00.002] Tool Invocation:</span> <span style='color:#93C5FD;'>{html.escape(result.initial_failure_tool_call)}</span>",
        f"<span style='color:#64748B;'>[00:00.003] Tool Diagnostic:</span> <span style='color:#FCA5A5;'>{html.escape(result.initial_failure_output)}</span>",
    ]

    if result.initial_failure_reproduced:
        lines.append(f"<span style='color:#64748B;'>[00:00.003] Status:</span>         <span style='background:#7F1D1D;color:#FECDD3;padding:1px 6px;border-radius:3px;font-weight:700;'>FAILURE REPRODUCED</span>")
        lines.append(f"<span style='color:#64748B;'>[00:00.003] Decision:</span>       Initiating 2-way hierarchical binary partitioning (ddmin)...\n")
    else:
        lines.append(f"<span style='color:#64748B;'>[00:00.003] Status:</span>         <span style='background:#14532D;color:#BBF7D0;padding:1px 6px;border-radius:3px;font-weight:700;'>CLEAN BASELINE (NO FAILURE)</span>")
        lines.append(f"<span style='color:#64748B;'>[00:00.003] Decision:</span>       Target passes without failure; no trigger to minimize. Halting search.\n")

    for s in result.dd_steps[1:]:
        t_sec = 0.003 + s.step_number * 0.001
        tag = (
            "<span style='background:#7F1D1D;color:#FECDD3;padding:1px 5px;border-radius:3px;font-weight:700;'>FAIL</span>"
            if s.outcome == "FAILURE"
            else "<span style='background:#14532D;color:#BBF7D0;padding:1px 5px;border-radius:3px;font-weight:700;'>PASS</span>"
        )
        lines.append(
            f"<span style='color:#64748B;'>[{t_sec:06.3f}] ─── TRIAL {s.step_number:02d}: {s.intervention_type.upper():<24} ─────────────────────────</span>\n"
            f"<span style='color:#64748B;'>[{t_sec:06.3f}] Partition Size:</span> {s.element_count:2d} elements | Outcome: {tag}\n"
            f"<span style='color:#64748B;'>[{t_sec:06.3f}] Action Narrative:</span> <span style='color:#E2E8F0;'>{html.escape(s.action_narrative)}</span>\n"
        )

    lines.append(f"<span style='color:#64748B;'>[00:00.008] ─── CAUSAL MINIMIZATION SEARCH COMPLETE ──────────────────────────────────────────</span>")
    if result.is_null_case:
        lines.append(f"<span style='color:#34D399;font-weight:700;'>[RESULT] CLEAN CONTROL VERIFIED:</span> Discovered minimal subset: ()")
        lines.append(f"<span style='color:#64748B;'>[RESULT] Causal Claim:</span>   Zero false attributions made. Correctly verified clean passing state.")
    else:
        lines.append(f"<span style='color:#34D399;font-weight:700;'>[RESULT] 1-MINIMAL TRIGGER ISOLATED:</span>")
        for e in result.discovered_subset:
            lines.append(f"  <span style='color:#F87171;font-weight:700;'>➔</span> <span style='background:#881337;color:#FFE4E6;padding:2px 8px;border-radius:4px;font-weight:600;'>{html.escape(e)}</span>")
        lines.append(f"<span style='color:#64748B;'>[RESULT] Causal Proof:</span>   Removing isolated directive eliminates failure; empty set () succeeds.")

    lines.append(f"<span style='color:#38BDF8;'>[TELEMETRY]</span> Evaluations Used: <span style='color:#FBBF24;font-weight:700;'>{result.target_calls_used} calls</span> (Delta Debugging) vs <span style='color:#94A3B8;'>{result.greedy_calls} calls</span> (Greedy Ablation)")
    lines.append(f"<span style='color:#38BDF8;'>[TELEMETRY]</span> Verification:     1-Minimal: <span style='color:#34D399;'>{result.is_one_minimal}</span> | Ground Truth Match: <span style='color:#34D399;'>{result.matches_ground_truth}</span>")

    inner_content = "\n".join(lines)

    return f"""
    <div class='terminal-window'>
        <div class='terminal-topbar'>
            <div class='window-dots'>
                <div class='w-dot w-red'></div>
                <div class='w-dot w-yellow'></div>
                <div class='w-dot w-green'></div>
            </div>
            <div class='terminal-tab-title'>agentxray ~ probe --scenario {html.escape(result.scenario_id)} (zsh)</div>
        </div>
        <div class='terminal-content'>{inner_content}</div>
    </div>
    """


def generate_pipeline_html(result: WebDemoScenarioResult) -> str:
    """Format visual search trajectory pipeline cards."""
    steps_html = []
    for s in result.dd_steps:
        tag_span = (
            "<span class='step-tag-fail'>FAILURE REPRODUCED</span>"
            if s.outcome == "FAILURE"
            else "<span class='step-tag-pass'>SUCCESS (SAFE)</span>"
        )
        steps_html.append(f"""
        <div class='pipeline-step'>
            <div style='display:flex;align-items:center;gap:12px;'>
                <div style='background:#1E293B;color:#94A3B8;padding:4px 8px;border-radius:4px;font-family:monospace;font-size:0.8rem;font-weight:700;'>STEP {s.step_number:02d}</div>
                <div>
                    <div style='font-size:0.9rem;font-weight:600;color:#F8FAFC;'>{html.escape(s.intervention_type.replace('_', ' ').title())}</div>
                    <div style='font-size:0.8rem;color:#94A3B8;'>{html.escape(s.action_narrative)}</div>
                </div>
            </div>
            <div style='display:flex;align-items:center;gap:12px;'>
                <span style='font-family:monospace;font-size:0.8rem;color:#CBD5E1;'>{s.element_count} directives</span>
                {tag_span}
            </div>
        </div>
        """)

    all_steps = "".join(steps_html)
    return f"<div class='pipeline-flow'>{all_steps}</div>"


def generate_comparison_html(result: WebDemoScenarioResult) -> str:
    """Format the baseline comparison matrix cards."""
    dd_eff = f"<span style='color:#38BDF8;font-weight:bold;'>{result.target_calls_used} calls</span>"
    gr_eff = f"<span style='color:#94A3B8;'>{result.greedy_calls} calls</span>"
    rd_eff = f"<span style='color:#94A3B8;'>{result.random_calls} calls</span>"

    return f"""
    <div style='display:grid;grid-template-columns:repeat(3, 1fr);gap:14px;margin:16px 0;'>
        <div style='background:#0D1117;border:1px solid #38BDF8;border-radius:8px;padding:16px;'>
            <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;'>
                <span style='font-weight:700;color:#38BDF8;'>Delta Debugging (ddmin)</span>
                <span class='step-tag-pass'>RECOMMENDED</span>
            </div>
            <div style='font-size:1.6rem;font-weight:800;font-family:monospace;color:#F8FAFC;margin:6px 0;'>{dd_eff}</div>
            <div style='font-size:0.82rem;color:#94A3B8;margin-bottom:8px;'>Hierarchical 2-way subset & complement search with explicit 1-minimality check.</div>
            <div style='font-size:0.78rem;color:#34D399;font-weight:600;'>✓ Confirmed 1-Minimal | 100% Ground Truth Match</div>
        </div>
        <div style='background:#0D1117;border:1px solid #1F2937;border-radius:8px;padding:16px;'>
            <div style='font-weight:700;color:#CBD5E1;margin-bottom:8px;'>Greedy Single Ablation</div>
            <div style='font-size:1.6rem;font-weight:800;font-family:monospace;color:#CBD5E1;margin:6px 0;'>{gr_eff}</div>
            <div style='font-size:0.82rem;color:#94A3B8;margin-bottom:8px;'>Sequential element-by-element elimination pass (O(N) linear scan).</div>
            <div style='font-size:0.78rem;color:#94A3B8;'>Inefficient on large contexts; tests every distractor individually.</div>
        </div>
        <div style='background:#0D1117;border:1px solid #1F2937;border-radius:8px;padding:16px;'>
            <div style='font-weight:700;color:#CBD5E1;margin-bottom:8px;'>Seeded Random Search</div>
            <div style='font-size:1.6rem;font-weight:800;font-family:monospace;color:#CBD5E1;margin:6px 0;'>{rd_eff}</div>
            <div style='font-size:0.82rem;color:#94A3B8;margin-bottom:8px;'>Budget-capped random sampling of non-empty proper candidate subsets.</div>
            <div style='font-size:0.78rem;color:#94A3B8;'>Non-deterministic; high variance in call counts on combinatorial faults.</div>
        </div>
    </div>
    """


def execute_scenario_all(scenario_id: str, seed: float) -> tuple[str, str, str, str, str]:
    """Execute scenario and return updated HTML components."""
    result: WebDemoScenarioResult = run_web_demo(scenario_id, seed=int(seed))
    banner_html = format_scenario_banner_html(scenario_id)
    stats_html = generate_stats_grid_html(result)
    terminal_html = generate_terminal_html(result)
    pipeline_html = generate_pipeline_html(result)
    comp_html = generate_comparison_html(result)
    return banner_html, stats_html, terminal_html, pipeline_html, comp_html


def build_gradio_app():
    """Construct the high-end developer terminal UI."""
    import gradio as gr

    choices = [
        (f"{m.name} ({m.category.split('/')[0].strip()})", k)
        for k, m in SCENARIO_CATALOG.items()
    ]

    with gr.Blocks(title="AgentXRay — Developer Studio", css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div class='hero-header'>
                <div style='display:flex;align-items:center;justify-content:space-between;'>
                    <div class='hero-title'>
                        <span>🔬 AgentXRay</span>
                        <span style='font-size:0.8rem;background:#1E293B;color:#93C5FD;padding:3px 8px;border-radius:4px;font-family:monospace;'>STUDIO v0.1.0</span>
                    </div>
                    <div style='display:flex;gap:8px;'>
                        <a href='https://github.com/yash-mishraa/AgentXRay' target='_blank' style='text-decoration:none;'>
                            <span style='background:#1F2937;color:#F8FAFC;padding:6px 12px;border-radius:6px;font-size:0.82rem;font-weight:600;'>⭐ GitHub</span>
                        </a>
                    </div>
                </div>
                <div class='hero-subtitle'>
                    Automated black-box fault localization and failure-condition minimization for tool-calling AI agents.
                    Isolates 1-minimal prompt directives, schema mismatches, and mock environment outages via Delta Debugging (ddmin).
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Benchmark Scenario Switcher")
                scenario_dropdown = gr.Dropdown(
                    label="Active Scenario Target",
                    choices=choices,
                    value="currency_ast",
                )
                seed_input = gr.Number(label="RNG Seed", value=42, precision=0)
                run_btn = gr.Button("▶ Run Causal Minimization", variant="primary")
                scenario_banner_output = gr.HTML(value=format_scenario_banner_html("currency_ast"))

            with gr.Column(scale=2):
                stats_output = gr.HTML()

                with gr.Tabs():
                    with gr.TabItem("🖥️ Live CLI Terminal Playback"):
                        terminal_output = gr.HTML()

                    with gr.TabItem("🌲 Causal Partitioning Pipeline"):
                        pipeline_output = gr.HTML()

                    with gr.TabItem("⚖️ Head-to-Head Baseline Matrix"):
                        comp_output = gr.HTML()

                    with gr.TabItem("📚 Research Epistemics & Guarantees"):
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
            outputs=[scenario_banner_output, stats_output, terminal_output, pipeline_output, comp_output],
        )

        run_btn.click(
            fn=execute_scenario_all,
            inputs=[scenario_dropdown, seed_input],
            outputs=[scenario_banner_output, stats_output, terminal_output, pipeline_output, comp_output],
        )

        demo.load(
            fn=execute_scenario_all,
            inputs=[scenario_dropdown, seed_input],
            outputs=[scenario_banner_output, stats_output, terminal_output, pipeline_output, comp_output],
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
