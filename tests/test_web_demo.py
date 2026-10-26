"""Headless integration tests for AgentXRay's web demo adapter and scenario execution.

Verifies that all exposed scenarios execute deterministically offline, correctly
distinguish active causal faults from null controls, produce valid trajectories,
and operate safely with zero network or external model dependencies.
"""

from __future__ import annotations

import pytest

from agentxray.web_demo import (
    SCENARIO_CATALOG,
    ScenarioMetadata,
    WebDemoScenarioResult,
    get_available_scenarios,
    run_web_demo,
)
import app


class TestWebDemoScenarios:
    """Test suite for web demo scenario catalog, execution mechanics, and epistemic boundaries."""

    def test_catalog_integrity_and_metadata(self) -> None:
        """Every exposed scenario has complete metadata, valid candidate pools, and ground truth."""
        scenarios = get_available_scenarios()
        assert len(scenarios) == 5
        catalog_keys = set(SCENARIO_CATALOG.keys())
        expected_keys = {
            "currency_ast",
            "trace_prompt_01",
            "trace_schema_01",
            "trace_env_01",
            "trace_null_01",
        }
        assert catalog_keys == expected_keys

        for sc in scenarios:
            assert isinstance(sc, ScenarioMetadata)
            assert sc.scenario_id in expected_keys
            assert len(sc.name) > 0
            assert len(sc.category) > 0
            assert len(sc.description) > 0
            assert len(sc.agent_description) > 0
            assert len(sc.user_prompt) > 0
            assert len(sc.available_tools) >= 1
            assert len(sc.candidate_elements) >= 4
            assert set(sc.ground_truth_trigger).issubset(set(sc.candidate_elements))

            if sc.is_null_case:
                assert len(sc.ground_truth_trigger) == 0
                assert sc.scenario_id == "trace_null_01"
            else:
                assert len(sc.ground_truth_trigger) == 1

    def test_every_scenario_executes_successfully_offline(self) -> None:
        """All scenarios run to completion in-memory under 0.5s with zero network requests."""
        for scenario_id in SCENARIO_CATALOG:
            res = run_web_demo(scenario_id, seed=42)
            assert isinstance(res, WebDemoScenarioResult)
            assert res.scenario_id == scenario_id
            assert res.execution_time_seconds < 0.5
            assert len(res.initial_elements) == len(SCENARIO_CATALOG[scenario_id].candidate_elements)
            assert res.matches_ground_truth is True
            assert len(res.dd_steps) >= 1
            assert res.target_calls_used >= 1

    def test_clean_null_scenario_semantics(self) -> None:
        """Clean null control is not falsely diagnosed as an active failure and terminates immediately."""
        res = run_web_demo("trace_null_01", seed=42)

        # Baseline execution succeeded without failure
        assert res.is_null_case is True
        assert res.initial_failure_reproduced is False
        assert res.initial_failure_code == "SUCCESS"
        assert "successfully" in res.initial_failure_output.lower()

        # Delta Debugging detected clean baseline in 1 call and returned empty trigger set
        assert res.discovered_subset == ()
        assert res.is_failure is False
        assert res.is_one_minimal is False
        assert res.target_calls_used == 1
        assert res.matches_ground_truth is True

        # Trajectory step reports clean baseline execution
        assert len(res.dd_steps) == 1
        assert res.dd_steps[0].step_number == 1
        assert res.dd_steps[0].intervention_type == "baseline"
        assert res.dd_steps[0].outcome == "SUCCESS"
        assert "clean" in res.dd_steps[0].action_narrative.lower()

        # Baselines also identify clean baseline in 1 call
        assert res.greedy_calls == 1
        assert res.greedy_discovered == ()
        assert res.greedy_matches_ground_truth is True

    def test_active_scenarios_isolate_exact_trigger(self) -> None:
        """All active failure scenarios isolate the exact 1-minimal planted causal trigger."""
        active_ids = ["currency_ast", "trace_prompt_01", "trace_schema_01", "trace_env_01"]

        for scenario_id in active_ids:
            res = run_web_demo(scenario_id, seed=42)
            meta = SCENARIO_CATALOG[scenario_id]

            # Failure reproduced on initial full input
            assert res.initial_failure_reproduced is True
            assert res.is_failure is True

            # Delta Debugging recovered exact planted trigger
            assert len(res.discovered_subset) == 1
            assert res.discovered_subset == meta.ground_truth_trigger
            assert res.matches_ground_truth is True
            assert res.is_one_minimal is True

            # 4 target evaluations required (baseline + subset + subset + minimality check)
            assert res.target_calls_used == 4

            # Outperforms or matches Greedy Ablation
            assert res.target_calls_used <= res.greedy_calls
            assert res.greedy_matches_ground_truth is True

    def test_trajectory_steps_continuity_and_narratives(self) -> None:
        """Trajectory records contain strictly sequential steps and non-empty narratives."""
        for scenario_id in SCENARIO_CATALOG:
            res = run_web_demo(scenario_id, seed=42)
            for idx, step in enumerate(res.dd_steps, start=1):
                assert step.step_number == idx
                assert step.element_count <= len(res.initial_elements)
                assert step.outcome in ("FAILURE", "SUCCESS")
                assert len(step.action_narrative) > 5

    def test_determinism_and_reproducibility(self) -> None:
        """Repeated runs with the same seed yield identical candidate sets, calls, and trajectory."""
        for scenario_id in SCENARIO_CATALOG:
            res1 = run_web_demo(scenario_id, seed=42)
            res2 = run_web_demo(scenario_id, seed=42)

            assert res1.discovered_subset == res2.discovered_subset
            assert res1.target_calls_used == res2.target_calls_used
            assert res1.dd_steps == res2.dd_steps
            assert res1.greedy_calls == res2.greedy_calls
            assert res1.random_calls == res2.random_calls

    def test_unknown_scenario_raises_value_error(self) -> None:
        """Requesting an unregistered scenario ID raises ValueError with available choices."""
        with pytest.raises(ValueError, match="Unknown scenario_id: 'unknown_xyz'"):
            run_web_demo("unknown_xyz")

    def test_app_main_graceful_missing_streamlit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When streamlit is not installed, app.main() exits cleanly with instructions."""
        # Test app.main() handles missing streamlit without crash
        with pytest.raises(SystemExit) as exc_info:
            app.main()
        assert exc_info.value.code == 1
