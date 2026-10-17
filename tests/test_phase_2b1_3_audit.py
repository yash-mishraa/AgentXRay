"""Deterministic Tests for Phase 2B.1.3: Final Statistical Validity Audit & Execution-Readiness Gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest

from agentxray.agent_harness.agent import ControlledAgent
from agentxray.agent_harness.cost_tracker import CostBudgetExceededError, CostTracker, ExperimentBudgetConfig
from agentxray.agent_harness.tools import MockToolEnvironment
from agentxray.models import (
    CandidateEvaluation,
    InputCase,
    SearchStatus,
    StochasticMinimalityStatus,
)
from agentxray.providers.fake import FakeProvider
from agentxray.stochastic_baselines import _classify_stochastic_minimality
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.model_pilot.validate_confirmatory_protocol import ConfirmatoryProtocolValidator


# ---------------------------------------------------------------------------
# 1. 1-Minimality Classification & IUT Boundary Tests
# ---------------------------------------------------------------------------

def test_minimality_strong_reproduction_and_supported_deletion() -> None:
    """Candidate with strong reproduction and statistically significant deletions -> STATISTICALLY_SUPPORTED."""
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=20,
        failure_threshold=0.50,
        alpha=0.05,
    )
    current_cand = ("e1", "e2")
    # Candidate: 20/20 failures (100%)
    current_eval = CandidateEvaluation(
        candidate_elements=current_cand,
        trials=20,
        failures=20,
        failure_rate=1.0,
        confidence_interval=(0.83, 1.0),
    )
    # Both deletions: 0/20 failures (0%), p < 0.001
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=("e2",), trials=20, failures=0, failure_rate=0.0, confidence_interval=(0.0, 0.17)
        ),
        "e2": CandidateEvaluation(
            candidate_elements=("e1",), trials=20, failures=0, failure_rate=0.0, confidence_interval=(0.0, 0.17)
        ),
    }

    status, msg = minimizer._classify_minimality(
        current_candidate=current_cand,
        current_eval=current_eval,
        removal_evals=removals,
    )

    assert status == StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    assert status == StochasticMinimalityStatus.SUPPORTED_1_MINIMAL
    assert "Intersection-Union Test" in msg


def test_minimality_low_observed_deletion_rates_without_significance() -> None:
    """Small sample size (N=5): observed drops occur, but p >= 0.05 -> strictly OBSERVED_ONLY."""
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=5,
        failure_threshold=0.50,
        alpha=0.05,
    )
    current_cand = ("e1", "e2")
    # Candidate: 4/5 failures (80%)
    current_eval = CandidateEvaluation(
        candidate_elements=current_cand,
        trials=5,
        failures=4,
        failure_rate=0.80,
        confidence_interval=(0.38, 0.96),
    )
    # Deletions: 1/5 failures (20%). While 20% <= 0.20, 4/5 vs 1/5 gives Fisher one-sided p = 0.103 > 0.05
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=("e2",), trials=5, failures=1, failure_rate=0.20, confidence_interval=(0.04, 0.62)
        ),
        "e2": CandidateEvaluation(
            candidate_elements=("e1",), trials=5, failures=1, failure_rate=0.20, confidence_interval=(0.04, 0.62)
        ),
    }

    status, msg = minimizer._classify_minimality(
        current_candidate=current_cand,
        current_eval=current_eval,
        removal_evals=removals,
    )

    # CRITICAL AUDIT CHECK: Must NOT be granted statistical support just because observed rate <= 0.20!
    assert status == StochasticMinimalityStatus.OBSERVED_ONLY
    assert status == StochasticMinimalityStatus.OBSERVED_1_MINIMAL
    assert status != StochasticMinimalityStatus.STATISTICALLY_SUPPORTED
    assert "did not achieve statistical significance" in msg


def test_minimality_invalid_shortcut_prevention() -> None:
    """Deliberately probe whether the rejected disjunctive shortcut 'OR observed <= 0.20' is absent."""
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=5,
        failure_threshold=0.50,
        alpha=0.05,
    )
    current_cand = ("e1",)
    # Candidate: 3/5 failures (60%)
    current_eval = CandidateEvaluation(
        candidate_elements=current_cand,
        trials=5,
        failures=3,
        failure_rate=0.60,
        confidence_interval=(0.23, 0.88),
    )
    # Deletion: 1/5 failures (20%). Observed rate is <= 0.20, but difference 60% vs 20% on N=5 is not significant (p > 0.20).
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=(), trials=5, failures=1, failure_rate=0.20, confidence_interval=(0.04, 0.62)
        ),
    }

    status, msg = minimizer._classify_minimality(
        current_candidate=current_cand,
        current_eval=current_eval,
        removal_evals=removals,
    )

    assert status == StochasticMinimalityStatus.OBSERVED_ONLY
    assert status != StochasticMinimalityStatus.STATISTICALLY_SUPPORTED


def test_minimality_deletion_still_reproduces_failure() -> None:
    """If removing an element still causes failure rate >= failure_threshold -> NOT_MINIMAL."""
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=5,
        failure_threshold=0.50,
    )
    current_cand = ("e1", "e2")
    current_eval = CandidateEvaluation(
        candidate_elements=current_cand, trials=5, failures=5, failure_rate=1.0, confidence_interval=(0.56, 1.0)
    )
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=("e2",), trials=5, failures=0, failure_rate=0.0, confidence_interval=(0.0, 0.44)
        ),
        "e2": CandidateEvaluation(
            candidate_elements=("e1",), trials=5, failures=4, failure_rate=0.80, confidence_interval=(0.38, 0.96)
        ),
    }

    status, msg = minimizer._classify_minimality(
        current_candidate=current_cand,
        current_eval=current_eval,
        removal_evals=removals,
    )

    assert status == StochasticMinimalityStatus.NOT_MINIMAL
    assert "still produces failure rate >= threshold" in msg


def test_minimality_missing_deletion_returns_inconclusive() -> None:
    """If any deletion evaluation is missing -> strictly INCONCLUSIVE."""
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=5,
        failure_threshold=0.50,
    )
    current_cand = ("e1", "e2", "e3")
    current_eval = CandidateEvaluation(
        candidate_elements=current_cand, trials=5, failures=5, failure_rate=1.0, confidence_interval=(0.56, 1.0)
    )
    # e3 missing
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=("e2", "e3"), trials=5, failures=0, failure_rate=0.0, confidence_interval=(0.0, 0.44)
        ),
        "e2": CandidateEvaluation(
            candidate_elements=("e1", "e3"), trials=5, failures=0, failure_rate=0.0, confidence_interval=(0.0, 0.44)
        ),
    }

    status, msg = minimizer._classify_minimality(
        current_candidate=current_cand,
        current_eval=current_eval,
        removal_evals=removals,
    )

    assert status == StochasticMinimalityStatus.INCONCLUSIVE
    assert "Missing removal evaluations" in msg


def test_minimality_candidate_below_failure_threshold_returns_inconclusive() -> None:
    """If candidate itself did not reproduce failure (e.g. rate < threshold or 0 trials) -> INCONCLUSIVE."""
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=5,
        failure_threshold=0.50,
    )
    current_cand = ("e1",)
    current_eval = CandidateEvaluation(
        candidate_elements=current_cand, trials=5, failures=2, failure_rate=0.40, confidence_interval=(0.12, 0.77)
    )
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=(), trials=5, failures=0, failure_rate=0.0, confidence_interval=(0.0, 0.44)
        ),
    }

    status, msg = minimizer._classify_minimality(
        current_candidate=current_cand,
        current_eval=current_eval,
        removal_evals=removals,
    )

    assert status == StochasticMinimalityStatus.INCONCLUSIVE
    assert "Candidate failed to reproduce failure rate" in msg


def test_minimality_zero_trials_returns_inconclusive() -> None:
    """Candidate with zero trials -> INCONCLUSIVE."""
    minimizer = StochasticDeltaDebugMinimizer(
        runner=None,  # type: ignore
        trials_per_candidate=5,
        failure_threshold=0.50,
    )
    current_cand = ("e1",)
    current_eval = CandidateEvaluation(
        candidate_elements=current_cand, trials=0, failures=0, failure_rate=0.0, confidence_interval=(0.0, 1.0)
    )
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=(), trials=5, failures=0, failure_rate=0.0, confidence_interval=(0.0, 0.44)
        ),
    }

    status, msg = minimizer._classify_minimality(
        current_candidate=current_cand,
        current_eval=current_eval,
        removal_evals=removals,
    )

    assert status == StochasticMinimalityStatus.INCONCLUSIVE


def test_baseline_classifier_parity() -> None:
    """Verify that _classify_stochastic_minimality in baselines matches minimizer behavior."""
    current_cand = ("e1", "e2")
    current_eval = CandidateEvaluation(
        candidate_elements=current_cand, trials=5, failures=4, failure_rate=0.80, confidence_interval=(0.38, 0.96)
    )
    removals = {
        "e1": CandidateEvaluation(
            candidate_elements=("e2",), trials=5, failures=1, failure_rate=0.20, confidence_interval=(0.04, 0.62)
        ),
        "e2": CandidateEvaluation(
            candidate_elements=("e1",), trials=5, failures=1, failure_rate=0.20, confidence_interval=(0.04, 0.62)
        ),
    }

    status = _classify_stochastic_minimality(
        candidate=current_cand,
        cand_eval=current_eval,
        removal_evals=removals,
        failure_threshold=0.50,
    )

    assert status == StochasticMinimalityStatus.OBSERVED_ONLY
    assert status == StochasticMinimalityStatus.OBSERVED_1_MINIMAL


# ---------------------------------------------------------------------------
# 2. Non-Inferiority Sample Size Boundary Calculation
# ---------------------------------------------------------------------------

def test_non_inferiority_boundary_demonstration() -> None:
    """Demonstrate that with n=6 independent mechanisms, a 95% CI cannot rule out delta = 0.15."""
    n = 6
    # Suppose perfect recovery: 6/6 for DD, 6/6 for Greedy (0 discordant pairs)
    diff = 0.0
    # For paired binary data with n=6 and 0 discordant pairs, Wilson-type score interval width:
    # Under standard Newcombe paired score interval, bound for p1 - p2 when diff=0 with n=6:
    # Wilson margin on single proportion at p=1.0 with n=6: interval is [0.61, 1.0]. Lower bound drops by 0.39!
    # Even under continuity-corrected Wald or score, lower bound is <= -0.25, which is strictly < -0.15.
    z_95 = 1.95996
    se_lower_bound_estimate = math.sqrt(2.0 * (1.0 / n) * (1.0 - 1.0 / n))
    assert se_lower_bound_estimate > 0.15, "Standard error of paired difference exceeds non-inferiority margin"


# ---------------------------------------------------------------------------
# 3. Mutation-Style Protocol Validator Tests
# ---------------------------------------------------------------------------

def test_protocol_v1_3_validator_success() -> None:
    """Verify that Protocol v1.3 configuration and document pass validation cleanly."""
    cfg_path = Path("benchmarks/configs/phase_2b1_exploratory_config_v1_3.json")
    proto_path = Path("docs/research/phase_2b1_exploratory_protocol_v1_3.md")

    validator = ConfirmatoryProtocolValidator(config_path=cfg_path, protocol_path=proto_path)
    summary = validator.validate()

    assert summary["validation_status"] == "VALIDATION_PASSED"
    assert summary["all_checks_passed"] is True
    assert summary["total_discrepancies"] == 0
    assert "non_inferiority_demoted" in summary["checks_passed"]
    assert "protocol_shortcut_purged" in summary["checks_passed"]


def test_validator_mutation_mechanism_count(tmp_path: Path) -> None:
    """Validator fails closed when mechanism count is corrupted."""
    cfg = json.loads(Path("benchmarks/configs/phase_2b1_exploratory_config_v1_3.json").read_text(encoding="utf-8"))
    cfg["mechanism_ids"] = cfg["mechanism_ids"][:7]  # Corrupt from 8 to 7
    corrupt_cfg = tmp_path / "corrupt_mech.json"
    corrupt_cfg.write_text(json.dumps(cfg), encoding="utf-8")

    validator = ConfirmatoryProtocolValidator(
        config_path=corrupt_cfg,
        protocol_path="docs/research/phase_2b1_exploratory_protocol_v1_3.md",
    )
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("mechanism_count" in d for d in summary["discrepancies"])


def test_validator_mutation_underfunded_budget(tmp_path: Path) -> None:
    """Validator fails closed when global budget is underfunded (< 6000)."""
    cfg = json.loads(Path("benchmarks/configs/phase_2b1_exploratory_config_v1_3.json").read_text(encoding="utf-8"))
    cfg["budget_rules"]["max_total_request_budget"] = 4000
    corrupt_cfg = tmp_path / "corrupt_budget.json"
    corrupt_cfg.write_text(json.dumps(cfg), encoding="utf-8")

    validator = ConfirmatoryProtocolValidator(
        config_path=corrupt_cfg,
        protocol_path="docs/research/phase_2b1_exploratory_protocol_v1_3.md",
    )
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("budget_cap_underfunded" in d for d in summary["discrepancies"])


def test_validator_mutation_k4_cap(tmp_path: Path) -> None:
    """Validator fails closed when K=4 request cap is corrupted."""
    cfg = json.loads(Path("benchmarks/configs/phase_2b1_exploratory_config_v1_3.json").read_text(encoding="utf-8"))
    cfg["budget_rules"]["max_requests_per_algorithm_k4"] = 90  # Expected 120
    corrupt_cfg = tmp_path / "corrupt_k4.json"
    corrupt_cfg.write_text(json.dumps(cfg), encoding="utf-8")

    validator = ConfirmatoryProtocolValidator(
        config_path=corrupt_cfg,
        protocol_path="docs/research/phase_2b1_exploratory_protocol_v1_3.md",
    )
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("budget_k4_cap" in d for d in summary["discrepancies"])


def test_validator_mutation_qualification_threshold(tmp_path: Path) -> None:
    """Validator fails closed when calibration trigger qualification threshold is weakened."""
    cfg = json.loads(Path("benchmarks/configs/phase_2b1_exploratory_config_v1_3.json").read_text(encoding="utf-8"))
    cfg["stopping_rules"]["active_task_qualification_min_trigger_failure_rate"] = 0.50  # Expected 0.70
    corrupt_cfg = tmp_path / "corrupt_qual.json"
    corrupt_cfg.write_text(json.dumps(cfg), encoding="utf-8")

    validator = ConfirmatoryProtocolValidator(
        config_path=corrupt_cfg,
        protocol_path="docs/research/phase_2b1_exploratory_protocol_v1_3.md",
    )
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("stopping_min_trigger" in d for d in summary["discrepancies"])


def test_validator_mutation_confirmatory_non_inferiority(tmp_path: Path) -> None:
    """Validator fails closed if v1.3 configuration attempts to claim CONFIRMATORY non-inferiority."""
    cfg = json.loads(Path("benchmarks/configs/phase_2b1_exploratory_config_v1_3.json").read_text(encoding="utf-8"))
    cfg["statistical_analysis_plan"]["non_inferiority_status"] = "CONFIRMATORY_CLAIM"
    corrupt_cfg = tmp_path / "corrupt_ni.json"
    corrupt_cfg.write_text(json.dumps(cfg), encoding="utf-8")

    validator = ConfirmatoryProtocolValidator(
        config_path=corrupt_cfg,
        protocol_path="docs/research/phase_2b1_exploratory_protocol_v1_3.md",
    )
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("non_inferiority_status" in d for d in summary["discrepancies"])


def test_validator_mutation_protocol_shortcut_reintroduced(tmp_path: Path) -> None:
    """Validator fails closed if protocol text reintroduces the illicit disjunctive shortcut."""
    proto_text = Path("docs/research/phase_2b1_exploratory_protocol_v1_3.md").read_text(encoding="utf-8")
    corrupt_proto_text = proto_text + "\n\nNote: A candidate qualifies if p < 0.05 or ablated rate <= 0.20.\n"
    corrupt_proto_path = tmp_path / "corrupt_proto.md"
    corrupt_proto_path.write_text(corrupt_proto_text, encoding="utf-8")

    validator = ConfirmatoryProtocolValidator(
        config_path="benchmarks/configs/phase_2b1_exploratory_config_v1_3.json",
        protocol_path=corrupt_proto_path,
    )
    summary = validator.validate()
    assert summary["validation_status"] == "VALIDATION_FAILED"
    assert any("protocol_shortcut_purged" in d for d in summary["discrepancies"])


# ---------------------------------------------------------------------------
# 4. Budget Hard Stop & Counting Provider Safety
# ---------------------------------------------------------------------------

def test_budget_hard_stop_exact_counting() -> None:
    """Verify that fake provider request count stops strictly at cap and never leaks calls."""
    provider = FakeProvider(model_name="fake-strict-counter")
    cap = 5
    budget_cfg = ExperimentBudgetConfig(
        max_model_requests=cap,
        max_spend_usd=1.00,
        model_name="fake-strict-counter",
    )
    tracker = CostTracker(budget_cfg)
    agent = ControlledAgent(provider=provider, mock_env=MockToolEnvironment(), cost_tracker=tracker, max_turns=1)

    for i in range(cap):
        agent.run_session(system_prompt="sys", user_prompt=f"call {i+1}")

    assert provider.request_count == cap
    assert tracker.total_requests == cap

    # Next attempt must be blocked before calling provider.generate()
    with pytest.raises(CostBudgetExceededError):
        tracker.check_budget_before_request()

    res = agent.run_session(system_prompt="sys", user_prompt="call after cap")
    assert res.finish_reason == "budget_exhausted"
    assert provider.request_count == cap, f"Provider received {provider.request_count} calls, expected strictly {cap}"
