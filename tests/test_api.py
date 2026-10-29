"""Unit and integration tests for AgentXRay Public Python SDK Facade (Phase 4.6K.20B).

Verifies package-level imports, candidate validation, evaluator return types,
exception safety, minimality verification, candidate ID tracking, ground-truth auditing,
and quickstart execution.
"""

from __future__ import annotations

import pytest

import agentxray
from agentxray import (
    DiagnosticResult,
    DiagnosticStatus,
    TargetOutcome,
    TargetResult,
    diagnose,
)
from agentxray.web_demo import run_web_demo, SCENARIO_CATALOG


class TestPublicAPIFacade:
    """Test suite for public agentxray.diagnose(...) function."""

    def test_package_level_imports(self) -> None:
        """Verify exports in agentxray top-level package namespace."""
        assert hasattr(agentxray, "diagnose")
        assert hasattr(agentxray, "DiagnosticResult")
        assert hasattr(agentxray, "DiagnosticStatus")
        assert agentxray.diagnose is diagnose
        assert agentxray.DiagnosticResult is DiagnosticResult
        assert agentxray.DiagnosticStatus is DiagnosticStatus

    def test_successful_diagnosis_custom_evaluator_boolean(self) -> None:
        """Evaluator returning boolean (True=failing, False=passing) isolates minimal trigger."""
        candidates = [
            "forbid_tools: do not use external math utilities",
            "json_format: output raw json object",
            "concise: keep answer short",
            "tone: formal tone",
        ]

        def evaluator(active: tuple[str, ...]) -> bool:
            return any("forbid_tools" in c for c in active)

        result = diagnose(
            candidates=candidates,
            evaluator=evaluator,
            max_evaluations=20,
            seed=42,
        )

        assert isinstance(result, DiagnosticResult)
        assert result.status == DiagnosticStatus.VERIFIED_CULPRIT
        assert result.is_verified is True
        assert result.is_one_minimal is True
        assert result.evaluations_used == 4
        assert len(result.isolated_candidates) == 1
        assert "forbid_tools" in result.isolated_candidates[0]
        assert result.isolated_candidate_ids == ("elem_0",)
        assert "elem_0" in result.candidate_map
        assert "elem_3" in result.candidate_map
        assert "elem_0" in result.summary
        assert result.matches_ground_truth is None  # Not supplied

    def test_successful_diagnosis_with_ground_truth(self) -> None:
        """When ground_truth is provided, matches_ground_truth is evaluated."""
        candidates = ["rule_a", "rule_b", "rule_c", "rule_d"]

        def evaluator(active: tuple[str, ...]) -> bool:
            return "rule_c" in active

        result = diagnose(
            candidates=candidates,
            evaluator=evaluator,
            ground_truth=["rule_c"],
        )

        assert result.status == DiagnosticStatus.VERIFIED_CULPRIT
        assert result.is_verified is True
        assert result.matches_ground_truth is True
        assert result.isolated_candidates == ("rule_c",)
        assert result.isolated_candidate_ids == ("elem_2",)

    def test_candidate_alias_elements(self) -> None:
        """The 'elements' parameter works as an exact alias for 'candidates'."""
        candidates = ["alpha", "beta", "gamma", "delta"]

        result = diagnose(
            elements=candidates,
            evaluator=lambda active: "beta" in active,
        )

        assert result.status == DiagnosticStatus.VERIFIED_CULPRIT
        assert result.isolated_candidates == ("beta",)
        assert result.isolated_candidate_ids == ("elem_1",)

    def test_clean_baseline_no_failure(self) -> None:
        """When full candidate pool does not fail, returns NO_FAILURE in 1 evaluation."""
        candidates = ["safe_rule_1", "safe_rule_2", "safe_rule_3"]

        result = diagnose(
            candidates=candidates,
            evaluator=lambda active: False,  # Never fails
        )

        assert result.status == DiagnosticStatus.NO_FAILURE
        assert result.is_verified is False
        assert result.evaluations_used == 1
        assert result.isolated_candidates == ()
        assert result.isolated_candidate_ids == ()
        assert "passed cleanly" in result.summary.lower()

    def test_evaluator_returning_inconclusive(self) -> None:
        """When evaluator returns inconclusive on baseline, reports INCONCLUSIVE."""
        candidates = ["rule_1", "rule_2"]

        result = diagnose(
            candidates=candidates,
            evaluator=lambda active: None,
        )

        assert result.status == DiagnosticStatus.INCONCLUSIVE
        assert result.is_verified is False
        assert "inconclusive" in result.summary.lower()

    def test_evaluator_raising_exception_safe_reporting(self) -> None:
        """When evaluator raises an unexpected exception, captures error without crashing."""
        candidates = ["directive_1", "directive_2"]

        def faulty_evaluator(active: tuple[str, ...]) -> bool:
            raise ConnectionResetError("Mock downstream service dropped connection")

        result = diagnose(
            candidates=candidates,
            evaluator=faulty_evaluator,
        )

        assert result.status == DiagnosticStatus.ERROR
        assert result.is_verified is False
        assert result.error_message is not None
        assert "ConnectionResetError" in result.error_message
        assert "ConnectionResetError" in result.summary

    def test_structured_target_result_and_tuple_returns(self) -> None:
        """Evaluator can return TargetResult or tuple[bool, str]."""
        candidates = ["opt_a", "opt_b"]

        # 1. TargetResult return
        res1 = diagnose(
            candidates=candidates,
            evaluator=lambda active: TargetResult(
                outcome=TargetOutcome.FAILURE if "opt_a" in active else TargetOutcome.SUCCESS,
                output_message="Target syntax crash",
            ),
        )
        assert res1.status == DiagnosticStatus.VERIFIED_CULPRIT
        assert res1.isolated_candidates == ("opt_a",)

        # 2. tuple[bool, str] return
        res2 = diagnose(
            candidates=candidates,
            evaluator=lambda active: ("opt_b" in active, "Diagnostic details"),
        )
        assert res2.status == DiagnosticStatus.VERIFIED_CULPRIT
        assert res2.isolated_candidates == ("opt_b",)

    def test_budget_exhaustion(self) -> None:
        """When max_evaluations is too low to verify, reports BUDGET_EXHAUSTED."""
        candidates = ["c1", "c2", "c3", "c4", "c5", "c6"]

        result = diagnose(
            candidates=candidates,
            evaluator=lambda active: "c6" in active,
            max_evaluations=2,  # Too small for 4+ trials needed
        )

        assert result.status == DiagnosticStatus.BUDGET_EXHAUSTED
        assert result.is_verified is False
        assert result.evaluations_used <= 2

    def test_invalid_candidate_inputs_raise_clear_exceptions(self) -> None:
        """Invalid candidate pools raise ValueError or TypeError with descriptive messages."""
        # Neither candidates nor elements
        with pytest.raises(ValueError, match="Either 'candidates' or 'elements' must be provided"):
            diagnose(evaluator=lambda a: True)

        # Empty candidates
        with pytest.raises(ValueError, match="Candidate pool cannot be empty"):
            diagnose(candidates=[], evaluator=lambda a: True)

        # Non-string candidate
        with pytest.raises(TypeError, match="must be a string"):
            diagnose(candidates=["valid", 123], evaluator=lambda a: True)  # type: ignore

        # Duplicate candidates
        with pytest.raises(ValueError, match="must be unique.*Found duplicate"):
            diagnose(candidates=["rule_a", "rule_b", "rule_a"], evaluator=lambda a: True)

        # Non-callable evaluator
        with pytest.raises(TypeError, match="'evaluator' must be callable"):
            diagnose(candidates=["rule_a"], evaluator="not_callable")  # type: ignore

        # Ground truth element not in candidates
        with pytest.raises(ValueError, match="not in the candidate pool"):
            diagnose(candidates=["rule_a"], evaluator=lambda a: True, ground_truth=["unknown_rule"])

        # Invalid budget parameters
        with pytest.raises(ValueError, match="positive integer"):
            diagnose(candidates=["rule_a"], evaluator=lambda a: True, max_evaluations=0)

    def test_offline_fixture_execution(self) -> None:
        """Existing offline repository scenario executes cleanly through the SDK facade."""
        meta = SCENARIO_CATALOG["currency_ast"]

        # Run web demo baseline to obtain deterministic behavior
        demo_res = run_web_demo("currency_ast", seed=42)

        def fixture_evaluator(active: tuple[str, ...]) -> bool:
            # Active if the planted trigger element is present
            return any(meta.ground_truth_trigger[0] in c for c in active)

        result = diagnose(
            candidates=meta.candidate_elements,
            evaluator=fixture_evaluator,
            ground_truth=meta.ground_truth_trigger,
            seed=42,
        )

        assert result.status == DiagnosticStatus.VERIFIED_CULPRIT
        assert result.is_verified is True
        assert result.matches_ground_truth is True
        assert result.evaluations_used == demo_res.target_calls_used
        assert result.isolated_candidates == meta.ground_truth_trigger
        assert result.isolated_candidate_ids == ("elem_0",)

    def test_complete_quickstart_example_runnable(self) -> None:
        """Verify that the documented quickstart example executes without error."""
        import examples.diagnose_custom_agent as quickstart
        quickstart.main()
