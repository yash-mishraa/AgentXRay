"""Architectural isolation tests.

Verifies that the probing algorithms and runner have zero import dependencies
or leakage from ground_truth.py.
"""

from __future__ import annotations

import inspect
import sys
import pytest

import agentxray.baselines
import agentxray.minimizer
import agentxray.models
import agentxray.runner
from benchmarks.synthetic.target import SyntheticTarget


class TestArchitecturalIsolation:
    """Verifies that ground truth does not leak into the algorithm's decision path."""

    def test_minimizer_module_does_not_import_ground_truth(self) -> None:
        minimizer_source = inspect.getsource(agentxray.minimizer)
        assert "ground_truth" not in minimizer_source
        assert "GroundTruthSpec" not in minimizer_source
        assert "evaluate_ground_truth" not in minimizer_source

    def test_runner_module_does_not_import_ground_truth(self) -> None:
        runner_source = inspect.getsource(agentxray.runner)
        assert "ground_truth" not in runner_source
        assert "GroundTruthSpec" not in runner_source
        assert "evaluate_ground_truth" not in runner_source

    def test_baselines_module_does_not_import_ground_truth(self) -> None:
        baselines_source = inspect.getsource(agentxray.baselines)
        assert "benchmarks.synthetic.ground_truth" not in baselines_source
        assert "GroundTruthSpec" not in baselines_source
        assert "evaluate_ground_truth" not in baselines_source

    def test_stochastic_minimizer_does_not_import_ground_truth(self) -> None:
        import agentxray.stochastic_minimizer
        source = inspect.getsource(agentxray.stochastic_minimizer)
        assert "ground_truth" not in source
        assert "StochasticGroundTruthSpec" not in source
        assert "evaluate_stochastic_ground_truth" not in source

    def test_stochastic_baselines_does_not_import_ground_truth(self) -> None:
        import agentxray.stochastic_baselines
        source = inspect.getsource(agentxray.stochastic_baselines)
        assert "benchmarks.synthetic" not in source
        assert "StochasticGroundTruthSpec" not in source
        assert "evaluate_stochastic_ground_truth" not in source

    def test_target_exposes_no_public_rule_attributes(self) -> None:
        def sample_rule(elements: set[str]) -> tuple[bool, str, str | None]:
            return (False, "ok", None)

        target = SyntheticTarget(rule_fn=sample_rule, name="IsolationTarget")
        public_methods = [m for m in dir(target) if not m.startswith("_")]

        assert "execute" in public_methods
        assert "call_count" in public_methods
        assert "reset_call_count" in public_methods
        assert "name" in public_methods

        # Ensure no rule inspectability
        assert "rule_fn" not in public_methods
        assert "ground_truth" not in public_methods
        assert "trigger" not in public_methods

    def test_stochastic_target_exposes_no_public_rule_attributes(self) -> None:
        from benchmarks.synthetic.stochastic_target import StochasticSyntheticTarget

        def sample_prob_rule(elements: set[str]) -> float:
            return 0.50

        target = StochasticSyntheticTarget(rule_fn=sample_prob_rule, name="ProbIsolationTarget")
        public_methods = [m for m in dir(target) if not m.startswith("_")]

        assert "execute" in public_methods
        assert "call_count" in public_methods
        assert "reset" in public_methods
        assert "name" in public_methods

        # Ensure no rule or PRNG inspectability
        assert "rule_fn" not in public_methods
        assert "ground_truth" not in public_methods
        assert "trigger" not in public_methods
        assert "rng" not in public_methods
