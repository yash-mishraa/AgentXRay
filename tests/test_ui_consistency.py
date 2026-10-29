"""Regression tests for UI diagnosis consistency and result integrity (Phase 4.6K.20A).

Verifies that the Overview tab, Execution Trace, Minimization Steps, and Method Comparison
all derive strictly from authoritative structured execution results without fabricated narratives,
stale caches, or mismatched candidate IDs.
"""

from __future__ import annotations

import html
import pytest

import app
from agentxray.web_demo import (
    SCENARIO_CATALOG,
    WebDemoScenarioResult,
    run_web_demo,
)


class TestUIDiagnosisConsistency:
    """Verify end-to-end consistency across all UI components and scenarios."""

    def test_mock_outage_overview_matches_execution_trace_and_metadata(self) -> None:
        """Specifically audits scenario trace_env_01 to ensure no fabricated payment outage strings."""
        res = run_web_demo("trace_env_01", seed=42)
        meta = SCENARIO_CATALOG["trace_env_01"]

        banner, overview, term, trace, comp = app.execute_scenario_all("trace_env_01", seed=42)

        # 1. Overview must NOT contain fabricated payment gateway or simulate_outage strings
        assert "simulate_outage" not in overview
        assert "payment processing" not in overview
        assert "payment gateway" not in overview
        assert "payment transaction" not in overview
        assert "503 Service Unavailable" not in overview

        # 2. Candidate ID and isolated element text must match exactly across overview and trace
        culprit_text = meta.ground_truth_trigger[0]
        assert culprit_text in res.discovered_subset
        assert "elem_0" in overview
        assert html.escape(culprit_text) in overview
        assert html.escape(culprit_text) in term

        # 3. Observed failure symptom must match real tool diagnostic output
        assert "500 Internal Server Error" in overview
        assert "Mock calculator service temporarily unavailable" in overview
        assert "calculator(expression=" in overview

        # 4. Status badges must reflect verified minimality
        assert "CULPRIT ISOLATED" in overview
        assert "1-MINIMAL VERIFIED" in overview
        assert "100% MATCH" in overview

    def test_all_scenarios_overview_derived_from_structured_result(self) -> None:
        """Every catalog scenario must generate an overview completely aligned with its structured result."""
        for scenario_id, meta in SCENARIO_CATALOG.items():
            res = run_web_demo(scenario_id, seed=42)
            banner, overview, term, trace, comp = app.execute_scenario_all(scenario_id, seed=42)

            # Ensure no legacy fabricated placeholders appear anywhere in the overview
            assert "simulate_outage" not in overview
            assert "weather assistant" not in overview
            assert "customer service agent" not in overview
            assert "payment processing" not in overview

            if res.is_null_case:
                assert "CLEAN CONTROL RUN" in overview
                assert "CLEAN CONTROL VERIFIED" in overview
                assert "empty set" in overview
                assert "()" in overview
            else:
                assert "CULPRIT ISOLATED" in overview
                assert "1-MINIMAL VERIFIED" in overview
                for elem in res.discovered_subset:
                    assert html.escape(elem) in overview
                    assert html.escape(elem) in term

    def test_scenario_switching_state_isolation(self) -> None:
        """Switching between scenarios computes fresh results without state bleeding."""
        switch_sequence = ["trace_env_01", "trace_schema_01", "trace_null_01", "currency_ast"]

        for sc_id in switch_sequence:
            meta = SCENARIO_CATALOG[sc_id]
            banner, overview, term, trace, comp = app.execute_scenario_all(sc_id, seed=42)

            assert html.escape(meta.name) in banner or meta.name in banner
            if meta.is_null_case:
                assert "CLEAN CONTROL RUN" in overview
            else:
                for elem in meta.ground_truth_trigger:
                    assert html.escape(elem) in overview

    def test_seed_propagation_to_result_and_terminal(self) -> None:
        """The RNG seed passed into the runner must propagate to the result and terminal session."""
        for seed_val in (42, 99, 777):
            res = run_web_demo("trace_env_01", seed=seed_val)
            assert res.seed == seed_val

            _, _, term, _, _ = app.execute_scenario_all("trace_env_01", seed=seed_val)
            assert f"--seed {seed_val}" in term
            assert f"session: agentxray probe --scenario trace_env_01 --seed {seed_val}" in term

    def test_unverified_result_cannot_display_success(self) -> None:
        """If a result is unverified or non-reproducible, overview must reflect unverified status."""
        base_res = run_web_demo("trace_env_01", seed=42)

        # Mock an unverified failure outcome
        unverified_res = WebDemoScenarioResult(
            scenario_id=base_res.scenario_id,
            scenario_name=base_res.scenario_name,
            category=base_res.category,
            agent_description=base_res.agent_description,
            user_prompt=base_res.user_prompt,
            available_tools=base_res.available_tools,
            evaluator_criteria=base_res.evaluator_criteria,
            initial_elements=base_res.initial_elements,
            ground_truth_trigger=base_res.ground_truth_trigger,
            is_null_case=False,
            initial_failure_tool_call=base_res.initial_failure_tool_call,
            initial_failure_output=base_res.initial_failure_output,
            initial_failure_code=base_res.initial_failure_code,
            initial_failure_reproduced=True,
            dd_steps=base_res.dd_steps,
            discovered_subset=base_res.discovered_subset,
            is_one_minimal=False,  # Unverified!
            is_failure=True,
            target_calls_used=base_res.target_calls_used,
            matches_ground_truth=False,
            status_message="Minimality check failed: non-monotonic failure",
            greedy_calls=base_res.greedy_calls,
            greedy_discovered=base_res.greedy_discovered,
            greedy_matches_ground_truth=False,
            greedy_is_minimal=False,
            random_calls=base_res.random_calls,
            random_discovered=base_res.random_discovered,
            random_matches_ground_truth=False,
            random_is_minimal=False,
            execution_time_seconds=base_res.execution_time_seconds,
            scientific_disclaimer=base_res.scientific_disclaimer,
            seed=42,
        )

        overview = app.generate_overview_html(unverified_res)
        strip = app.generate_status_strip_html(unverified_res)
        comp = app.generate_comparison_html(unverified_res)

        assert "UNVERIFIED RESULT" in overview
        assert "1-MINIMAL VERIFIED" not in strip
        assert "UNVERIFIED" in strip
        assert "val-failure" in strip
        assert "val-failure" in comp

    def test_method_comparison_css_accuracy(self) -> None:
        """Comparison table must assign val-failure to mismatches and val-success to matches."""
        base_res = run_web_demo("trace_env_01", seed=42)

        # Normal run has all matches
        comp_normal = app.generate_comparison_html(base_res)
        assert "val-success" in comp_normal
        assert "100% Match" in comp_normal

        # Mock result with greedy mismatch
        mismatch_res = WebDemoScenarioResult(
            scenario_id=base_res.scenario_id,
            scenario_name=base_res.scenario_name,
            category=base_res.category,
            agent_description=base_res.agent_description,
            user_prompt=base_res.user_prompt,
            available_tools=base_res.available_tools,
            evaluator_criteria=base_res.evaluator_criteria,
            initial_elements=base_res.initial_elements,
            ground_truth_trigger=base_res.ground_truth_trigger,
            is_null_case=False,
            initial_failure_tool_call=base_res.initial_failure_tool_call,
            initial_failure_output=base_res.initial_failure_output,
            initial_failure_code=base_res.initial_failure_code,
            initial_failure_reproduced=True,
            dd_steps=base_res.dd_steps,
            discovered_subset=base_res.discovered_subset,
            is_one_minimal=True,
            is_failure=True,
            target_calls_used=base_res.target_calls_used,
            matches_ground_truth=True,
            status_message=base_res.status_message,
            greedy_calls=base_res.greedy_calls,
            greedy_discovered=(),
            greedy_matches_ground_truth=False,  # Mismatch!
            greedy_is_minimal=False,
            random_calls=base_res.random_calls,
            random_discovered=base_res.random_discovered,
            random_matches_ground_truth=True,
            random_is_minimal=True,
            execution_time_seconds=base_res.execution_time_seconds,
            scientific_disclaimer=base_res.scientific_disclaimer,
            seed=42,
        )

        comp_mismatch = app.generate_comparison_html(mismatch_res)
        assert "Mismatch" in comp_mismatch
        assert "val-failure" in comp_mismatch
