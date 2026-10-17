"""Tests for Phase 2B.1.2: Statistical implementation audit, estimand repair, and Protocol v1.2 freeze."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence
import pytest

from agentxray.agent_harness.agent import ControlledAgent
from agentxray.agent_harness.cost_tracker import CostBudgetExceededError, CostTracker, ExperimentBudgetConfig
from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator
from agentxray.agent_harness.target import ModelBackedAgentTarget
from agentxray.agent_harness.tools import MockToolEnvironment
from agentxray.models import (
    CandidateEvaluation,
    InputCase,
    StochasticMinimalityStatus,
)
from agentxray.providers.base import Message, ModelProvider, ModelResponse, ToolDefinition, UsageMetadata
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.model_pilot.confirmatory_tasks import (
    ACTIVE_MECHANISMS,
    CONFIRMATORY_TASK_REGISTRY,
    MECHANISM_REGISTRY,
    NULL_MECHANISMS,
    get_confirmatory_task,
    get_tasks_by_mechanism,
    list_confirmatory_tasks,
)
from benchmarks.model_pilot.power_and_budget_analysis import (
    asymptotic_wilcoxon_p_value_with_ties,
    exact_wilcoxon_permutation_p_value,
    simulate_wilcoxon_power_detailed,
    wilcoxon_signed_rank_exact_critical_value,
)
from benchmarks.model_pilot.validate_confirmatory_protocol import ConfirmatoryProtocolValidator


from agentxray.providers.fake import FakeProvider


def test_exact_dp_permutation_vs_brute_force() -> None:
    """Verify that DP exact permutation p-value matches brute-force enumeration for m <= 8."""
    ranks = [1.0, 2.0, 3.5, 3.5, 5.0, 6.0]
    signs = [True, True, True, False, True, True]
    m = len(ranks)

    # DP calculation
    p_dp = exact_wilcoxon_permutation_p_value(ranks, signs)

    # Brute-force enumeration of all 2^m sign permutations
    w_obs = sum(r for r, s in zip(ranks, signs) if s)
    total_w = sum(ranks)
    w_min = min(w_obs, total_w - w_obs)

    count = 0
    tot_perms = 1 << m
    for mask in range(tot_perms):
        w = sum(ranks[i] for i in range(m) if (mask >> i) & 1)
        if w <= w_min or w >= total_w - w_min:
            count += 1
    p_brute = min(1.0, count / tot_perms)

    assert pytest.approx(p_dp, abs=1e-6) == p_brute
    assert 0.0 < p_dp <= 1.0


def test_monte_carlo_standard_error_bounds() -> None:
    """Verify that simulate_wilcoxon_power_detailed reports mathematically consistent MCSE."""
    res = simulate_wilcoxon_power_detailed(
        n=12, mu=12.0, sigma=12.0, n_sims=500, dist="normal", p_tie=0.05, seed=123
    )
    p_exact = res["power_exact"]
    mcse_exact = res["mcse_exact"]

    expected_mcse = math.sqrt(p_exact * (1.0 - p_exact) / 500)
    assert pytest.approx(mcse_exact, abs=1e-4) == expected_mcse

    ci_low, ci_high = res["ci_95_exact"]
    assert ci_low <= p_exact <= ci_high
    assert 0.70 < p_exact < 0.95


def test_preflight_budget_hard_stop_with_counting_provider() -> None:
    """Verify that budget is checked before every request and provider is NEVER called after cap."""
    provider = FakeProvider(model_name="fake-counter")
    cap = 3
    config = ExperimentBudgetConfig(
        max_model_requests=cap,
        max_spend_usd=1.00,
        model_name="fake-counter",
    )
    tracker = CostTracker(config)
    env = MockToolEnvironment()
    agent = ControlledAgent(provider=provider, mock_env=env, cost_tracker=tracker, max_turns=2)

    # Call 1
    t1 = agent.run_session(system_prompt="sys", user_prompt="test 1")
    assert provider.request_count == 1
    assert tracker.total_requests == 1
    assert t1.finish_reason == "stop"

    # Call 2
    t2 = agent.run_session(system_prompt="sys", user_prompt="test 2")
    assert provider.request_count == 2
    assert tracker.total_requests == 2

    # Call 3
    t3 = agent.run_session(system_prompt="sys", user_prompt="test 3")
    assert provider.request_count == 3
    assert tracker.total_requests == 3

    # Call 4: Must raise CostBudgetExceededError BEFORE provider.generate is invoked
    with pytest.raises(CostBudgetExceededError):
        tracker.check_budget_before_request()

    t4 = agent.run_session(system_prompt="sys", user_prompt="test 4")
    assert t4.finish_reason == "budget_exhausted"
    # CRITICAL INVARIANT: Provider call count must remain STRICTLY equal to cap
    assert provider.request_count == cap, f"Provider was called {provider.request_count} times, expected strictly {cap}"


def test_minimality_missing_deletion_returns_inconclusive() -> None:
    """Verify that missing single-element deletion evaluation strictly returns INCONCLUSIVE."""
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=5,
        failure_threshold=0.50,
    )
    current_cand = ("c1", "c2", "c3")
    current_eval = CandidateEvaluation(
        candidate_elements=current_cand,
        trials=5,
        failures=5,
        failure_rate=1.0,
        confidence_interval=(0.56, 1.0),
    )
    # Only c1 and c2 are evaluated; c3 is missing
    partial_removals = {
        "c1": CandidateEvaluation(
            candidate_elements=("c2", "c3"), trials=5, failures=0, failure_rate=0.0, confidence_interval=(0.0, 0.44)
        ),
        "c2": CandidateEvaluation(
            candidate_elements=("c1", "c3"), trials=5, failures=0, failure_rate=0.0, confidence_interval=(0.0, 0.44)
        ),
    }

    status, msg = minimizer._classify_minimality(
        current_candidate=current_cand,
        current_eval=current_eval,
        removal_evals=partial_removals,
    )

    assert status == StochasticMinimalityStatus.INCONCLUSIVE
    assert "Missing removal evaluations" in msg
    assert "c3" in msg


def test_mechanism_registry_and_clustering_integrity() -> None:
    """Verify that the 16 tasks map cleanly into exactly 8 mechanisms (6 active, 2 null)."""
    assert len(MECHANISM_REGISTRY) == 8
    assert len(ACTIVE_MECHANISMS) == 6
    assert len(NULL_MECHANISMS) == 2

    by_mech = get_tasks_by_mechanism()
    assert len(by_mech) == 8

    # Check active mechanisms
    for mid in ACTIVE_MECHANISMS:
        tasks = by_mech[mid]
        assert len(tasks) == 2, f"Mechanism {mid} must have exactly 2 tasks"
        ks = {t.k for t in tasks}
        assert ks == {4, 8}, f"Mechanism {mid} must have K=4 and K=8, got {ks}"

        t4 = [t for t in tasks if t.k == 4][0]
        t8 = [t for t in tasks if t.k == 8][0]

        # Verify underlying mechanism identity
        assert t4.user_prompt == t8.user_prompt
        assert t4.required_tool == t8.required_tool
        assert t4.planted_trigger_elements == t8.planted_trigger_elements
        assert t4.expected_failure_classification == "FAIL_TOOL_EXEC"
        assert t8.expected_failure_classification == "FAIL_TOOL_EXEC"

    # Check null mechanisms
    for mid in NULL_MECHANISMS:
        tasks = by_mech[mid]
        assert len(tasks) == 2
        for t in tasks:
            assert t.is_null_case is True
            assert len(t.planted_trigger_elements) == 0
            assert t.expected_failure_classification == "CLEAN_PASS"


def test_protocol_v1_2_validator_passes_and_fails_closed(tmp_path: Path) -> None:
    """Verify that ConfirmatoryProtocolValidator passes on v1.2 and fails closed on corrupted specs."""
    cfg_path = Path("benchmarks/configs/phase_2b1_exploratory_config_v1_2.json")
    proto_path = Path("docs/research/phase_2b1_exploratory_protocol_v1_2.md")

    validator = ConfirmatoryProtocolValidator(config_path=cfg_path, protocol_path=proto_path)
    summary = validator.validate()

    assert summary["validation_status"] == "VALIDATION_PASSED"
    assert summary["all_checks_passed"] is True
    assert summary["total_discrepancies"] == 0
    assert "mechanism_ids_valid" in summary["checks_passed"]
    assert "all_mechanisms_verified_and_paired" in summary["checks_passed"]

    # Test fail-closed on corrupted mechanism count
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["mechanism_ids"] = cfg["mechanism_ids"][:5]  # Corrupt to 5
    corrupt_cfg = tmp_path / "corrupt_config_v1_2.json"
    corrupt_cfg.write_text(json.dumps(cfg), encoding="utf-8")

    fail_validator = ConfirmatoryProtocolValidator(config_path=corrupt_cfg, protocol_path=proto_path)
    fail_summary = fail_validator.validate()
    assert fail_summary["validation_status"] == "VALIDATION_FAILED"
    assert any("mechanism_count" in d for d in fail_summary["discrepancies"])
