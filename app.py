"""AgentXRay: Interactive Portfolio Demonstration (Gradio / Hugging Face Spaces).

Black-box behavioral failure-condition minimization for AI agents via Delta Debugging.
Runs completely offline in-memory (<0.05s) with zero external network calls and $0.00 cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is available on sys.path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agentxray.web_demo import (
    SCENARIO_CATALOG,
    ScenarioMetadata,
    WebDemoScenarioResult,
    run_web_demo,
)


def format_scenario_info(scenario_id: str) -> str:
    """Format markdown overview for the selected scenario."""
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

    lines.append(f"\n**Input Candidate Pool ({len(meta.candidate_elements)} elements):**")
    for idx, elem in enumerate(meta.candidate_elements, start=1):
        is_trigger = elem in meta.ground_truth_trigger
        badge = "🔴 `[CAUSAL TRIGGER]`" if is_trigger else "⚪ `[distractor]`"
        lines.append(f"- {badge} **`elem_{idx-1}`:** {elem}")

    return "\n".join(lines)


def run_and_format_scenario(scenario_id: str, seed: float) -> tuple[str, list[list[str | int]], str, list[list[str]]]:
    """Execute scenario and format all Gradio output components."""
    result: WebDemoScenarioResult = run_web_demo(scenario_id, seed=int(seed))

    # 1. Diagnostic telemetry markdown
    status_label = "🔴 **FAILURE**" if result.initial_failure_reproduced else "🟢 **SUCCESS (CLEAN)**"
    diag_md = (
        f"**Initial Tool Call:** `{result.initial_failure_tool_call}`  \n"
        f"**Target Outcome:** {status_label}  \n"
        f"**Execution Latency:** `{result.execution_time_seconds * 1000.0:.2f} ms` (In-Memory, $0.00 cost)  \n"
        f"**Diagnostic Output:** `{result.initial_failure_output}`"
    )

    # 2. Trajectory table rows
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

    # 3. Discovery outcome markdown
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

    # 4. Comparison table rows
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
            "Hierarchical 2-way subset & complement partitioning with 1-minimality check",
        ],
        [
            "Greedy Ablation",
            f"{result.greedy_calls} calls",
            gr_min,
            gr_gt,
            "Sequential single-candidate omission testing",
        ],
        [
            "Seeded Random Search",
            f"{result.random_calls} calls",
            rd_min,
            rd_gt,
            "Permuted subset sampling with identical random seed",
        ],
    ]

    return diag_md, traj_rows, discovery_md, comp_rows


def build_gradio_app():
    """Construct the Gradio interface blocks."""
    import gradio as gr

    choices = [
        (f"{m.name} ({m.category.split('/')[0].strip()})", k)
        for k, m in SCENARIO_CATALOG.items()
    ]

    with gr.Blocks(title="AgentXRay — Causal Failure Minimization", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔬 AgentXRay: Causal Failure Minimization")
        gr.Markdown(
            "**Automated fault localization for tool-calling AI agents via black-box Delta Debugging (ddmin).**  \n"
            "Systematically isolating 1-minimal failure-inducing subsets of prompt directives, schema keys, and environment configs."
        )
        gr.Markdown(
            """
            > **🔬 Deterministic Research Testbed:**
            > This portfolio demonstration runs 100% offline in-memory using controlled mock environments
            > and `FakeProvider` (**$0.00 API spend, zero remote network calls**).
            > All scenarios execute in **<0.05 seconds**. This interface illustrates exact algorithmic mechanics
            > with verified ground truth; it is not an assertion of live LLM empirical superiority.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Scenario Selection & Controls")
                scenario_dropdown = gr.Dropdown(
                    label="Select Benchmark Scenario",
                    choices=choices,
                    value="currency_ast",
                )
                seed_input = gr.Number(label="Random Seed", value=42, precision=0)
                run_btn = gr.Button("▶ Run Causal Failure Minimization", variant="primary")
                scenario_info_md = gr.Markdown(value=format_scenario_info("currency_ast"))

            with gr.Column(scale=2):
                gr.Markdown("### 1. Observable Initial Agent Execution")
                diag_output = gr.Markdown()

                gr.Markdown("### 2. Delta Debugging Search Trajectory")
                traj_output = gr.Dataframe(
                    headers=["Step", "Intervention Type", "Candidate Count", "Outcome", "Action Taken"],
                    datatype=["number", "str", "str", "str", "str"],
                    interactive=False,
                )

                gr.Markdown("### 3. Discovered Minimal Failure-Inducing Condition")
                discovery_output = gr.Markdown()

                gr.Markdown("### 4. Baseline Method Comparison")
                comp_output = gr.Dataframe(
                    headers=["Diagnostic Algorithm", "Target Calls", "1-Minimal?", "Matches Ground Truth?", "Methodology"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                )

                gr.Markdown("### 5. Epistemic Boundaries & Research Limitations")
                gr.Markdown(
                    """
                    - **Deterministic Testbed Verification:** This demonstration executes against a strictly controlled mock agent
                      using `FakeProvider` and deterministic tool environments ($0.00 cost, zero network calls).
                    - **Black-Box Causal Boundary:** The minimizer observes **only binary target execution outcomes**
                      (`SUCCESS` / `FAILURE`); it has zero access to agent system prompt weights, hidden states, or provider internals.
                    - **Sample Complexity & Power:** While Delta Debugging demonstrates sample-efficiency advantages in deterministic
                      settings (e.g. 4 calls vs 6 calls), claims of statistical superiority on live non-deterministic models (LLaMA, GPT-4)
                      require multi-trial stochastic probing, Wilson confidence bounds, and exact McNemar testing as specified in
                      `docs/RESEARCH_PROPOSAL.md`.
                    """
                )

        # Wire callbacks
        scenario_dropdown.change(
            fn=format_scenario_info,
            inputs=[scenario_dropdown],
            outputs=[scenario_info_md],
        )

        run_btn.click(
            fn=run_and_format_scenario,
            inputs=[scenario_dropdown, seed_input],
            outputs=[diag_output, traj_output, discovery_output, comp_output],
        )

        # Initial load execution
        demo.load(
            fn=run_and_format_scenario,
            inputs=[scenario_dropdown, seed_input],
            outputs=[diag_output, traj_output, discovery_output, comp_output],
        )

    return demo


def main() -> None:
    """Entry point for launching the web demo."""
    try:
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
