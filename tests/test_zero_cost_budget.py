"""Focused regression tests for zero-cost local budget semantics (Phase 4.6H.1).

Verifies:
1. Zero cap configuration (max_spend_usd=0.0) is accepted.
2. Negative caps (max_spend_usd < 0.0) are rejected by validation.
3. Zero-cost local requests can proceed within the request cap using approved OllamaProvider on loopback.
4. Positive estimated request cost is rejected under a zero cap.
5. Positive accumulated spend is rejected under a zero cap.
6. Request cap enforcement remains strictly intact under a zero cap.
7. Positive-cap behavior is completely preserved and unchanged.
8. Zero-priced model names cannot bypass zero-cap restrictions when used with unapproved, remote, or mock providers.
9. ControlledAgent and LLMJudgeBaseline integrate safely with zero-cost budget guards.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agentxray.agent_harness.agent import AgentExecutionTrace, ControlledAgent
from agentxray.agent_harness.cost_tracker import (
    CostBudgetExceededError,
    CostTracker,
    ExperimentBudgetConfig,
    IsolatedAlgorithmCostTracker,
    IsolatedPhaseCostTracker,
)
from agentxray.agent_harness.tools import MockToolEnvironment
from agentxray.llm_judge import LLMJudgeBaseline
from agentxray.models import InputCase
from agentxray.providers.base import Message, ModelProvider, ModelResponse, ToolDefinition, UsageMetadata
from agentxray.providers.fake import FakeProvider
from agentxray.providers.ollama_adapter import OllamaProvider
from agentxray.providers.openai_adapter import OpenAIProvider


class TestZeroCapConfigurationValidation:
    """Verifies that max_spend_usd=0.0 is accepted while negative caps are rejected."""

    def test_zero_cap_configuration_accepted(self) -> None:
        cfg = ExperimentBudgetConfig(
            max_model_requests=30,
            max_spend_usd=0.0,
            model_name="llama3.2:3b",
        )
        assert cfg.max_spend_usd == 0.0
        assert cfg.max_model_requests == 30

        tracker = CostTracker(cfg)
        assert tracker.config.max_spend_usd == 0.0
        assert tracker.is_spend_exceeded is False

    def test_negative_spend_cap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentBudgetConfig(max_spend_usd=-0.01)

        with pytest.raises(ValidationError):
            ExperimentBudgetConfig(max_spend_usd=-1.0)


class TestZeroCostLocalExecutionSemantics:
    """Verifies that approved local OllamaProvider requests proceed under max_spend_usd=0.0."""

    def test_zero_cost_local_requests_proceed_within_request_cap(self) -> None:
        prov = OllamaProvider(model_name="llama3.2:3b", base_url="http://127.0.0.1:11434")
        cfg = ExperimentBudgetConfig(
            max_model_requests=3,
            max_spend_usd=0.0,
            model_name="llama3.2:3b",
        )
        tracker = CostTracker(cfg, provider=prov)

        # Pre-request check with projected prompt tokens
        tracker.check_budget_before_request(est_prompt_tokens=500)
        assert tracker.total_requests == 0
        assert tracker.estimated_spend_usd == 0.0
        assert tracker.is_spend_exceeded is False

        # Record usage with zero financial cost
        usage = UsageMetadata(
            prompt_tokens=200,
            completion_tokens=50,
            total_tokens=250,
            estimated_cost_usd=0.0,
        )
        tracker.record_usage(usage, success=True, retries=0)
        assert tracker.total_requests == 1
        assert tracker.successful_requests == 1
        assert tracker.estimated_spend_usd == 0.0
        assert tracker.is_spend_exceeded is False

    def test_request_cap_enforcement_under_zero_cap(self) -> None:
        prov = OllamaProvider(model_name="llama3.2:3b", base_url="http://127.0.0.1:11434")
        cfg = ExperimentBudgetConfig(
            max_model_requests=3,
            max_spend_usd=0.0,
            model_name="llama3.2:3b",
        )
        tracker = CostTracker(cfg, provider=prov)

        # Request 1: OK
        tracker.check_budget_before_request(est_prompt_tokens=100)
        tracker.record_usage(UsageMetadata(prompt_tokens=50, completion_tokens=20))
        assert tracker.total_requests == 1

        # Request 2: OK
        tracker.check_budget_before_request(est_prompt_tokens=100)
        tracker.record_usage(UsageMetadata(prompt_tokens=50, completion_tokens=20))
        assert tracker.total_requests == 2

        # Request 3: OK before request, but hits cap on record_usage
        tracker.check_budget_before_request(est_prompt_tokens=100)
        with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted"):
            tracker.record_usage(UsageMetadata(prompt_tokens=50, completion_tokens=20))
        assert tracker.total_requests == 3

        # Request 4: Blocked before request
        with pytest.raises(CostBudgetExceededError, match="Model request budget exhausted before request"):
            tracker.check_budget_before_request(est_prompt_tokens=100)


class TestZeroCapStrictSpendProtection:
    """Verifies that positive estimated or accumulated costs are strictly rejected under max_spend_usd=0.0."""

    def test_positive_estimated_request_cost_rejected_under_zero_cap(self) -> None:
        prov = OllamaProvider(model_name="llama3.2:3b")
        # Configure model with non-zero custom pricing
        cfg = ExperimentBudgetConfig(
            max_model_requests=10,
            max_spend_usd=0.0,
            model_name="llama3.2:3b",
            custom_pricing_per_1m=(1.00, 2.00),
        )
        tracker = CostTracker(cfg, provider=prov)

        # Pre-request check must reject because pricing is non-zero
        with pytest.raises(CostBudgetExceededError, match="Zero-spend budget.*requires zero pricing"):
            tracker.check_budget_before_request(est_prompt_tokens=200)

    def test_positive_accumulated_spend_rejected_under_zero_cap(self) -> None:
        prov = OllamaProvider(model_name="llama3.2:3b")
        cfg = ExperimentBudgetConfig(
            max_model_requests=10,
            max_spend_usd=0.0,
            model_name="llama3.2:3b",
            custom_pricing_per_1m=(1.00, 2.00),
        )
        tracker = CostTracker(cfg, provider=prov)

        # Post-request recording with positive spend must be rejected
        with pytest.raises(CostBudgetExceededError, match="Estimated spend budget exhausted"):
            tracker.record_usage(UsageMetadata(prompt_tokens=500, completion_tokens=100))


class TestZeroCapProviderProvenanceSafeguards:
    """Verifies that zero-priced model names cannot bypass zero-cap restrictions with unapproved providers."""

    def test_unbound_tracker_rejected_under_zero_cap(self) -> None:
        cfg = ExperimentBudgetConfig(
            max_model_requests=10,
            max_spend_usd=0.0,
            model_name="llama3.2:3b",
        )
        tracker = CostTracker(cfg)  # No provider bound

        with pytest.raises(CostBudgetExceededError, match="Zero-spend budget.*strictly restricted to local execution"):
            tracker.check_budget_before_request(est_prompt_tokens=100)

        with pytest.raises(CostBudgetExceededError, match="Zero-spend budget.*strictly restricted to local execution"):
            tracker.record_usage(UsageMetadata(prompt_tokens=50, completion_tokens=20))

    def test_openai_provider_with_zero_priced_model_name_rejected_under_zero_cap(self) -> None:
        cfg = ExperimentBudgetConfig(
            max_model_requests=10,
            max_spend_usd=0.0,
            model_name="llama3.2:3b",  # Model name matches zero-price pattern, but provider is OpenAI
        )
        openai_prov = OpenAIProvider(model_name="llama3.2:3b", api_key="sk-mock-key")
        tracker = CostTracker(cfg, provider=openai_prov)

        with pytest.raises(CostBudgetExceededError, match="Got provider: 'OpenAIProvider'"):
            tracker.check_budget_before_request(est_prompt_tokens=100)

        with pytest.raises(CostBudgetExceededError, match="Got provider: 'OpenAIProvider'"):
            tracker.record_usage(UsageMetadata(prompt_tokens=50, completion_tokens=20))

    def test_subclassed_or_mock_ollama_rejected_under_zero_cap(self) -> None:
        class SubclassedOllama(OllamaProvider):
            pass

        mock_prov = SubclassedOllama(model_name="llama3.2:3b")
        cfg = ExperimentBudgetConfig(
            max_model_requests=10,
            max_spend_usd=0.0,
            model_name="llama3.2:3b",
        )
        tracker = CostTracker(cfg, provider=mock_prov)

        with pytest.raises(CostBudgetExceededError, match="Got provider: 'SubclassedOllama'"):
            tracker.check_budget_before_request(est_prompt_tokens=100)


class TestPositiveCapTripwirePreservation:
    """Verifies that positive-cap tripwire behavior is completely unchanged."""

    def test_positive_cap_pre_request_and_post_request_tripwires_preserved(self) -> None:
        cfg = ExperimentBudgetConfig(
            max_model_requests=10,
            max_spend_usd=0.0010,  # $0.001 cap
            model_name="gpt-4o-mini",
        )
        tracker = CostTracker(cfg)

        # Pre-request check succeeds when call cost is below cap
        tracker.check_budget_before_request(est_prompt_tokens=100)

        # Record usage below limit ($0.00021 < $0.0010)
        tracker.record_usage(UsageMetadata(prompt_tokens=1000, completion_tokens=100))
        assert tracker.estimated_spend_usd < 0.0010

        # Pre-request check trips when projected call exceeds limit ($0.00021 + $0.0015 >= $0.0010)
        with pytest.raises(CostBudgetExceededError, match="Estimated spend budget would be exceeded before request"):
            tracker.check_budget_before_request(est_prompt_tokens=10_000)

        # Post-request recording trips when usage breaches limit
        with pytest.raises(CostBudgetExceededError, match="Estimated spend budget exhausted"):
            tracker.record_usage(UsageMetadata(prompt_tokens=10_000, completion_tokens=1000))


class TestHarnessAndJudgeIntegrationWithZeroCap:
    """Verifies ControlledAgent and LLMJudgeBaseline integrate safely with zero-cost budget guards."""

    def test_controlled_agent_with_approved_ollama_and_zero_cap(self) -> None:
        prov = OllamaProvider(model_name="llama3.2:3b")
        cfg = ExperimentBudgetConfig(max_model_requests=5, max_spend_usd=0.0, model_name="llama3.2:3b")
        tracker = CostTracker(cfg)
        env = MockToolEnvironment()
        agent = ControlledAgent(provider=prov, mock_env=env, cost_tracker=tracker, max_turns=2)

        # ControlledAgent.__init__ automatically bound prov to tracker
        assert tracker._provider is prov
        assert tracker.is_guaranteed_local_provider() is True

    def test_controlled_agent_blocks_openai_under_zero_cap(self) -> None:
        openai_prov = OpenAIProvider(model_name="llama3.2:3b", api_key="sk-test-key")
        cfg = ExperimentBudgetConfig(max_model_requests=5, max_spend_usd=0.0, model_name="llama3.2:3b")
        tracker = CostTracker(cfg)
        env = MockToolEnvironment()
        agent = ControlledAgent(provider=openai_prov, mock_env=env, cost_tracker=tracker, max_turns=2)

        # Execution halts cleanly with budget_exhausted before dispatch; verify generate is never invoked
        with patch.object(
            openai_prov, "generate", side_effect=AssertionError("Dispatch occurred when provider should be blocked!")
        ) as mock_generate:
            trace = agent.run_session(system_prompt="Test", user_prompt="Hello")
            assert trace.finish_reason == "budget_exhausted"
            assert "Zero-spend budget (max_spend_usd=0.0) is strictly restricted to local execution" in str(
                trace.error_message
            )
            assert mock_generate.call_count == 0

    def test_llm_judge_blocks_unapproved_provider_under_zero_cap(self) -> None:
        openai_prov = OpenAIProvider(model_name="llama3.2:3b", api_key="sk-test-key")
        cfg = ExperimentBudgetConfig(max_model_requests=5, max_spend_usd=0.0, model_name="llama3.2:3b")
        tracker = CostTracker(cfg)
        judge = LLMJudgeBaseline(provider=openai_prov, cost_tracker=tracker)

        inp = InputCase(case_id="c_zero_cap_judge", elements=("elem1", "elem2"))
        trace = AgentExecutionTrace(
            messages=(
                Message(role="user", content="Execute task"),
                Message(role="assistant", content="Failed to execute tool"),
            ),
            final_response="",
            finish_reason="error",
            error_message="Simulation failure",
            turns_used=1,
        )

        # Diagnosis halts cleanly with budget_exhausted before dispatch; verify generate is never invoked
        with patch.object(
            openai_prov, "generate", side_effect=AssertionError("Dispatch occurred when judge should be blocked!")
        ) as mock_generate:
            res = judge.diagnose(inp, trace)
            assert res.is_valid_diagnosis is False
            assert res.termination_reason == "budget_exhausted"
            assert res.attempts_used == 0
            assert mock_generate.call_count == 0

    def test_isolated_algorithm_and_phase_trackers_zero_cap(self) -> None:
        prov = OllamaProvider(model_name="llama3.2:3b")
        global_cfg = ExperimentBudgetConfig(max_model_requests=10, max_spend_usd=0.0, model_name="llama3.2:3b")
        global_tracker = CostTracker(global_cfg, provider=prov)

        algo_cfg = ExperimentBudgetConfig(max_model_requests=5, max_spend_usd=0.0, model_name="llama3.2:3b")
        algo_tracker = IsolatedAlgorithmCostTracker(algo_cfg, global_tracker=global_tracker, enforce_global_spend=True)
        algo_tracker.bind_provider(prov)

        # Check passes within limits
        algo_tracker.check_budget_before_request(est_prompt_tokens=200)
        algo_tracker.record_usage(UsageMetadata(prompt_tokens=100, completion_tokens=20))
        assert algo_tracker.total_requests == 1
        assert global_tracker.total_requests == 1
        assert algo_tracker.estimated_spend_usd == 0.0
        assert global_tracker.estimated_spend_usd == 0.0
