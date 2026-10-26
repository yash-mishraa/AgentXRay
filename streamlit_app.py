"""AgentXRay: Interactive Portfolio Demonstration.

Black-box behavioral failure-condition minimization for AI agents via Delta Debugging.
Runs completely offline in-memory (<0.05s) with zero external network calls and $0.00 cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def render_ui() -> None:
    """Render the Streamlit user interface."""
    import streamlit as st

    st.set_page_config(
        page_title="AgentXRay — Causal Failure Minimization",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom restrained styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: #1E293B;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        .disclaimer-card {
            background-color: #F8FAFC;
            border-left: 4px solid #3B82F6;
            padding: 0.85rem 1.1rem;
            border-radius: 4px;
            font-size: 0.92rem;
            color: #1E293B;
            margin-bottom: 1.5rem;
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #0F172A;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 0.35rem;
            margin-top: 1.6rem;
            margin-bottom: 0.85rem;
        }
        .code-pill {
            background-color: #EEF2F6;
            color: #1E293B;
            padding: 0.15rem 0.45rem;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown('<div class="main-header">🔬 AgentXRay: Causal Failure Minimization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Isolating 1-minimal failure-inducing candidate subsets in AI agent prompts, schemas, and environments '
        'using Delta Debugging (ddmin) versus Greedy Ablation and Random Search.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Epistemic Disclosure Banner
    st.markdown(
        """
        <div class="disclaimer-card">
            <strong>🔬 Deterministic Research Testbed:</strong>
            This portfolio demonstration runs 100% offline in-memory using controlled mock environments
            and <code>FakeProvider</code> (<strong>$0.00 API spend, zero network calls</strong>).
            All scenarios execute in <strong>&lt;0.05 seconds</strong>. This interface demonstrates exact
            algorithmic partitioning mechanics on verified ground truth; it is not an assertion of live LLM empirical superiority.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Scenario Selection & Controls
    with st.sidebar:
        st.subheader("⚙️ Scenario Selection")

        scenario_keys = list(SCENARIO_CATALOG.keys())
        scenario_labels = {
            k: f"{SCENARIO_CATALOG[k].name} ({SCENARIO_CATALOG[k].category.split('/')[0].strip()})"
            for k in scenario_keys
        }

        selected_id = st.selectbox(
            "Select Benchmark Scenario",
            options=scenario_keys,
            format_func=lambda k: scenario_labels[k],
            index=0,
            help="Choose a pre-defined synthetic benchmark scenario with verified causal ground truth.",
        )

        selected_meta: ScenarioMetadata = SCENARIO_CATALOG[selected_id]

        st.markdown("---")
        st.subheader("🎯 Execution Parameters")
        seed_val = st.number_input("Random Seed", value=42, min_value=0, max_value=999999, step=1)
        run_button = st.button("▶ Run Causal Minimization", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown(
            "**Research Documentation:**\n"
            "- [Architecture Reference](https://github.com/yashmishra14/AgentXRay)\n"
            "- [Scientific Decisions](https://github.com/yashmishra14/AgentXRay/blob/main/docs/DECISIONS.md)\n"
            "- [Statistical Protocol](https://github.com/yashmishra14/AgentXRay/blob/main/docs/RESEARCH_PROPOSAL.md)"
        )

    # Main Panel: Scenario Details
    st.markdown('<div class="section-title">1. Scenario Configuration & Input Candidates</div>', unsafe_allow_html=True)

    col_meta1, col_meta2 = st.columns([1, 1])
    with col_meta1:
        st.markdown(f"**Scenario Name:** {selected_meta.name}")
        st.markdown(f"**Fault Domain:** `{selected_meta.category}`")
        st.markdown(f"**Agent Description:** {selected_meta.agent_description}")
        st.markdown(f"**User Prompt:** *\"{selected_meta.user_prompt}\"*")
    with col_meta2:
        tools_str = ", ".join(f"`{t}`" for t in selected_meta.available_tools)
        st.markdown(f"**Available Tools:** {tools_str}")
        st.markdown(f"**Evaluator Success Criteria:** {selected_meta.evaluator_criteria}")
        null_badge = "🟢 Null Control (0 Planted Triggers)" if selected_meta.is_null_case else "🔴 Active Fault Mechanism"
        st.markdown(f"**Condition Type:** {null_badge}")
        if selected_meta.observable_failure_symptom:
            st.markdown(f"**Expected Symptom:** `{selected_meta.observable_failure_symptom}`")

    st.markdown(f"**Input Candidate Pool ({len(selected_meta.candidate_elements)} elements):**")
    for idx, elem in enumerate(selected_meta.candidate_elements, start=1):
        is_trigger = elem in selected_meta.ground_truth_trigger
        badge = "🔴 `[CAUSAL TRIGGER]`" if is_trigger else "⚪ `[distractor]`"
        st.markdown(f"- {badge} **`elem_{idx-1}`:** {elem}")

    # Execution Trigger
    if run_button or "last_run_result" not in st.session_state or st.session_state.get("last_run_id") != selected_id:
        result: WebDemoScenarioResult = run_web_demo(selected_id, seed=int(seed_val))
        st.session_state["last_run_result"] = result
        st.session_state["last_run_id"] = selected_id
    else:
        result = st.session_state["last_run_result"]

    st.markdown('<div class="section-title">2. Observable Initial Agent Execution</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Initial Agent Tool Call",
            result.initial_failure_tool_call,
            delta="Observed in Trace",
            delta_color="off",
        )
    with c2:
        status_label = "FAILURE" if result.initial_failure_reproduced else "SUCCESS (CLEAN)"
        st.metric(
            "Target Outcome (Full Input)",
            status_label,
            delta="Reproduced" if result.initial_failure_reproduced else "Zero Fault",
            delta_color="inverse" if result.initial_failure_reproduced else "normal",
        )
    with c3:
        st.metric(
            "Execution Latency",
            f"{result.execution_time_seconds * 1000.0:.2f} ms",
            delta="In-Memory (<0.05s)",
            delta_color="normal",
        )

    st.markdown(f"**Diagnostic Output / Trace Message:** `{result.initial_failure_output}`")

    # Section 3: Minimization Trajectory
    st.markdown('<div class="section-title">3. Delta Debugging Search Trajectory</div>', unsafe_allow_html=True)
    st.markdown(
        "Delta Debugging partitions candidate elements into subsets and complements, zooming into the minimal "
        "failure-inducing configuration before verifying 1-minimality via single-element ablation:"
    )

    trajectory_data = []
    for step in result.dd_steps:
        trajectory_data.append(
            {
                "Step": step.step_number,
                "Intervention Type": step.intervention_type,
                "Candidate Count": f"{step.element_count} elements",
                "Outcome": step.outcome,
                "Action Taken": step.action_narrative,
            }
        )
    st.table(trajectory_data)

    # Section 4: Discovery Results & 1-Minimality
    st.markdown('<div class="section-title">4. Minimal Failure-Inducing Condition Discovery</div>', unsafe_allow_html=True)

    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric(
            "Delta Debugging Evaluations",
            f"{result.target_calls_used} calls",
            delta=f"-{result.greedy_calls - result.target_calls_used} vs Greedy" if result.greedy_calls > result.target_calls_used else "Optimal",
        )
    with col_res2:
        if result.is_null_case:
            min_str = "Verified Empty (Clean Pass)"
        elif result.is_one_minimal:
            min_str = "Confirmed 1-Minimal"
        else:
            min_str = "Unverified"
        st.metric("1-Minimality Status", min_str)
    with col_res3:
        match_str = "Exact Match (100%)" if result.matches_ground_truth else "False Attribution"
        st.metric("Ground Truth Match", match_str)

    if result.discovered_subset:
        st.markdown("**Isolated Minimal Trigger Element(s):**")
        for elem in result.discovered_subset:
            st.info(f"🎯 **`{elem}`**")
    else:
        st.success("✅ **Empty Set `()` Isolated:** No candidates induce failure. Correctly identified clean non-faulty state.")

    st.caption(f"Status Note: {result.status_message}")

    # Section 5: Baseline Method Comparison
    st.markdown('<div class="section-title">5. Baseline Method Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        "Comparison of Delta Debugging against Greedy Single-Element Ablation and Seeded Random Search "
        "under identical testbed conditions and budget caps:"
    )

    dd_min = "✅ Yes (Verified)" if result.is_one_minimal else ("✅ N/A (Null)" if result.is_null_case else "❌ No")
    dd_gt = "✅ Exact Match" if result.matches_ground_truth else "❌ Mismatch"
    gr_min = "✅ Yes" if result.greedy_is_minimal else ("✅ N/A (Null)" if result.is_null_case else "❌ No")
    gr_gt = "✅ Exact Match" if result.greedy_matches_ground_truth else "❌ Mismatch"
    rd_min = "✅ Yes" if result.random_is_minimal else ("✅ N/A (Null)" if result.is_null_case else "❌ No")
    rd_gt = "✅ Exact Match" if result.random_matches_ground_truth else "❌ Mismatch"

    comp_table = [
        {
            "Diagnostic Algorithm": "Delta Debugging (ddmin)",
            "Target Calls": f"{result.target_calls_used} calls",
            "1-Minimal?": dd_min,
            "Matches Ground Truth?": dd_gt,
            "Methodology": "Hierarchical 2-way subset & complement partitioning with explicit minimality check",
        },
        {
            "Diagnostic Algorithm": "Greedy Ablation",
            "Target Calls": f"{result.greedy_calls} calls",
            "1-Minimal?": gr_min,
            "Matches Ground Truth?": gr_gt,
            "Methodology": "Sequential single-candidate omission testing",
        },
        {
            "Diagnostic Algorithm": "Seeded Random Search",
            "Target Calls": f"{result.random_calls} calls",
            "1-Minimal?": rd_min,
            "Matches Ground Truth?": rd_gt,
            "Methodology": "Permuted subset sampling with identical random seed",
        },
    ]
    st.table(comp_table)

    # Section 6: Epistemic Boundaries & Scientific Proof
    st.markdown('<div class="section-title">6. Epistemic Boundaries & Research Limitations</div>', unsafe_allow_html=True)
    st.markdown(
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


def main() -> None:
    """Entry point for running the web demo."""
    try:
        render_ui()
    except ModuleNotFoundError as err:
        if "streamlit" in str(err).lower():
            print(
                "Streamlit is required to launch the interactive web demo.\n"
                "Install optional demo dependencies: pip install -e .[demo]\n"
                "Then launch with: streamlit run app.py",
                file=sys.stderr,
            )
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
