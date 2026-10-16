"""Tests for the StochasticSyntheticTarget."""

from __future__ import annotations

import pytest

from agentxray.models import InputCase, TargetOutcome
from benchmarks.synthetic.stochastic_target import StochasticSyntheticTarget


def test_stochastic_target_deterministic_with_seed():
    """Verify target produces reproducible Bernoulli sequence under fixed target_seed."""
    def rule_fn(elements: set[str]) -> float:
        return 0.50

    case = InputCase(case_id="c1", elements=("param1",))

    t1 = StochasticSyntheticTarget(rule_fn=rule_fn, target_seed=123)
    outcomes1 = [t1.execute(case).outcome for _ in range(20)]

    t2 = StochasticSyntheticTarget(rule_fn=rule_fn, target_seed=123)
    outcomes2 = [t2.execute(case).outcome for _ in range(20)]

    assert outcomes1 == outcomes2


def test_stochastic_target_reset_reseeds_correctly():
    """Verify reset() restores PRNG sequence."""
    def rule_fn(elements: set[str]) -> float:
        return 0.70

    case = InputCase(case_id="c1", elements=("param1",))

    target = StochasticSyntheticTarget(rule_fn=rule_fn, target_seed=42)
    seq1 = [target.execute(case).outcome for _ in range(15)]
    assert target.call_count == 15

    target.reset()
    assert target.call_count == 0
    seq2 = [target.execute(case).outcome for _ in range(15)]
    assert seq1 == seq2


def test_stochastic_target_boundary_probabilities():
    """Verify target behavior at P=0.0 and P=1.0."""
    case = InputCase(case_id="c1", elements=("param1",))

    always_fail = StochasticSyntheticTarget(rule_fn=lambda el: 1.0)
    for _ in range(10):
        assert always_fail.execute(case).outcome == TargetOutcome.FAILURE

    always_succeed = StochasticSyntheticTarget(rule_fn=lambda el: 0.0)
    for _ in range(10):
        assert always_succeed.execute(case).outcome == TargetOutcome.SUCCESS


def test_stochastic_target_invalid_probability_raises():
    """Verify ValueError is raised if rule returns invalid probability."""
    case = InputCase(case_id="c1", elements=("param1",))
    invalid_target = StochasticSyntheticTarget(rule_fn=lambda el: 1.5)
    with pytest.raises(ValueError, match="invalid probability"):
        invalid_target.execute(case)
