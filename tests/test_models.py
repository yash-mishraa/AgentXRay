"""Tests for domain models and input validation in agentxray.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentxray.models import (
    AccessMode,
    DiscoveredCondition,
    ExperimentConfig,
    InputCase,
    ProbeTrial,
    SearchStatus,
    TargetOutcome,
    TargetResult,
)


class TestExperimentConfig:
    """Validation tests for ExperimentConfig."""

    def test_valid_config(self) -> None:
        config = ExperimentConfig(
            experiment_name="test_exp",
            max_trials=50,
            timeout_seconds=10.0,
            seed=123,
            mode=AccessMode.CONTROLLED_TESTBED,
        )
        assert config.experiment_name == "test_exp"
        assert config.max_trials == 50
        assert config.timeout_seconds == 10.0
        assert config.seed == 123
        assert config.mode == AccessMode.CONTROLLED_TESTBED

    def test_invalid_max_trials_zero_or_negative(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(experiment_name="test", max_trials=0)
        with pytest.raises(ValidationError):
            ExperimentConfig(experiment_name="test", max_trials=-5)

    def test_invalid_timeout_zero_or_negative(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(experiment_name="test", timeout_seconds=0.0)
        with pytest.raises(ValidationError):
            ExperimentConfig(experiment_name="test", timeout_seconds=-1.0)

    def test_blank_experiment_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(experiment_name="")
        with pytest.raises(ValidationError):
            ExperimentConfig(experiment_name="   ")


class TestInputCase:
    """Validation tests for InputCase."""

    def test_valid_input_case(self) -> None:
        case = InputCase(
            case_id="case_1",
            elements=["param_a", "param_b"],
            description="test case",
        )
        assert case.case_id == "case_1"
        assert case.elements == ("param_a", "param_b")

    def test_empty_elements_allowed(self) -> None:
        case = InputCase(elements=[])
        assert case.elements == ()

    def test_duplicate_elements_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InputCase(elements=["dup", "other", "dup"])

    def test_whitespace_element_cleaned_or_rejected(self) -> None:
        case = InputCase(elements=["  clean_me  ", "other"])
        assert case.elements == ("clean_me", "other")

        empty_case = InputCase(elements=["   ", ""])
        assert empty_case.elements == ()

    def test_without_method(self) -> None:
        case = InputCase(elements=["a", "b", "c"])
        reduced = case.without("b")
        assert reduced.elements == ("a", "c")

        # Removing non-existent element returns copy with same elements
        assert case.without("z").elements == ("a", "b", "c")

        # Removing all elements returns empty InputCase
        single = InputCase(elements=["solo"])
        assert single.without("solo").elements == ()

    def test_subset_method(self) -> None:
        case = InputCase(elements=["a", "b", "c", "d"])
        sub = case.subset(["b", "d"])
        assert sub.elements == ("b", "d")
        assert case.subset([]).elements == ()

        assert case.subset(["not_in_case"]).elements == ()


class TestTargetResult:
    """Validation tests for TargetResult."""

    def test_failure_result(self) -> None:
        res = TargetResult(
            outcome=TargetOutcome.FAILURE,
            output_message="Crash",
            error_code="ERR_CRASH",
            latency_ms=12.5,
        )
        assert res.is_failure is True
        assert res.outcome == TargetOutcome.FAILURE
        assert res.error_code == "ERR_CRASH"

    def test_success_result(self) -> None:
        res = TargetResult(
            outcome=TargetOutcome.SUCCESS,
            output_message="All good",
        )
        assert res.is_failure is False
        assert res.error_code is None
        assert res.latency_ms == 0.0

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TargetResult(outcome=TargetOutcome.SUCCESS, latency_ms=-1.0)


class TestDiscoveredCondition:
    """Tests for DiscoveredCondition model."""

    def test_discovered_condition_fields(self) -> None:
        cond = DiscoveredCondition(
            subset_elements=("a", "b"),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            is_one_minimal=True,
            is_globally_minimal=False,
            verified_by_check=True,
            target_calls_used=14,
            status_message="Verified 1-minimal.",
        )
        assert cond.is_one_minimal is True
        assert cond.is_globally_minimal is False
        assert cond.subset_elements == ("a", "b")
