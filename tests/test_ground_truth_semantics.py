"""Targeted regression tests for multi-trigger partial recovery and null-case semantics.

Enforces Phase 4.2B semantics across deterministic, stochastic, and model-pilot evaluators,
as well as downstream domain models, report generators, and aggregators.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest

from agentxray.models import (
    CandidateEvaluation,
    DiscoveredCondition,
    EvaluationResult,
    ExecutionTier,
    ExperimentConfig,
    ExperimentReport,
    ModelPilotEvaluationResult,
    ModelPilotExperimentReport,
    SearchStatus,
    StochasticDiscoveredCondition,
    StochasticEvaluationResult,
    StochasticExperimentReport,
    StochasticMinimalityStatus,
)
from agentxray.reporting import (
    format_json_report,
    format_markdown_report,
    format_model_pilot_json_report,
    format_model_pilot_markdown_report,
    format_stochastic_json_report,
    format_stochastic_markdown_report,
)
from benchmarks.model_pilot.ground_truth import (
    ModelPilotGroundTruthSpec,
    evaluate_model_pilot_ground_truth,
)
from benchmarks.synthetic.ground_truth import GroundTruthSpec, evaluate_ground_truth
from benchmarks.synthetic.scaling_benchmark import (
    AlgorithmRunResult,
    ScalingBenchmarkSuite,
    ScalingInstanceResult,
    export_benchmark_results,
)
from benchmarks.synthetic.stochastic_ground_truth import (
    StochasticGroundTruthSpec,
    evaluate_stochastic_ground_truth,
)


# ---------------------------------------------------------------------------
# Test Fixtures / Helper Specs
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_trigger_spec() -> GroundTruthSpec:
    """Multi-trigger benchmark spec with primary and alternative valid triggers."""
    return GroundTruthSpec(
        task_id="multi_trig_task",
        name="Multi-Trigger Task",
        description="Task with primary trigger and two valid alternatives.",
        trigger_elements=("t_a", "t_b"),
        alternative_valid_triggers=(("alt_c", "alt_d"), ("alt_e",)),
    )


@pytest.fixture
def stochastic_multi_trigger_spec() -> StochasticGroundTruthSpec:
    """Stochastic multi-trigger spec."""
    return StochasticGroundTruthSpec(
        task_id="stoch_multi_trig_task",
        name="Stochastic Multi-Trigger Task",
        description="Stochastic task with primary and alternative triggers.",
        trigger_elements=("t_a", "t_b"),
        base_probability=0.05,
        trigger_probability=0.85,
        alternative_valid_triggers=(("alt_c", "alt_d"), ("alt_e",)),
    )


@pytest.fixture
def model_pilot_multi_trigger_spec() -> ModelPilotGroundTruthSpec:
    """Model-pilot multi-trigger spec."""
    return ModelPilotGroundTruthSpec(
        task_id="pilot_multi_trig_task",
        name="Pilot Multi-Trigger Task",
        description="Model pilot reference task with alternative hypotheses.",
        trigger_elements=("t_a", "t_b"),
        alternative_valid_triggers=(("alt_c", "alt_d"), ("alt_e",)),
    )


# ---------------------------------------------------------------------------
# 1-7: Alternative-Trigger Evaluation Semantics
# ---------------------------------------------------------------------------

class TestAlternativeTriggerEvaluationSemantics:
    """Comprehensive test matrix for multi-trigger evaluation across all three evaluators."""

    def test_case_1_exact_primary_trigger_recovery(self, multi_trigger_spec: GroundTruthSpec) -> None:
        """Case 1: Exact recovery of the primary trigger."""
        discovered = ("t_a", "t_b")
        matches, prec, rec, excess, missing = evaluate_ground_truth(multi_trigger_spec, discovered)

        assert matches is True
        assert prec == 1.0
        assert rec == 1.0
        assert excess == ()
        assert missing == ()

    def test_case_2_exact_alternative_trigger_recovery(self, multi_trigger_spec: GroundTruthSpec) -> None:
        """Case 2: Exact recovery of an alternative trigger."""
        discovered = ("alt_c", "alt_d")
        matches, prec, rec, excess, missing = evaluate_ground_truth(multi_trigger_spec, discovered)

        assert matches is True
        assert prec == 1.0
        assert rec == 1.0
        assert excess == ()
        assert missing == ()

        # Test second alternative trigger as well
        discovered_singleton = ("alt_e",)
        m2, p2, r2, e2, mis2 = evaluate_ground_truth(multi_trigger_spec, discovered_singleton)
        assert m2 is True
        assert p2 == 1.0
        assert r2 == 1.0
        assert e2 == ()
        assert mis2 == ()

    def test_case_3_alternative_trigger_plus_distractor(self, multi_trigger_spec: GroundTruthSpec) -> None:
        """Case 3: An alternative trigger plus a distractor.

        Discovered: {'alt_c', 'alt_d', 'distractor_x'}
        Against primary ('t_a', 't_b'): TP=0, FP=3, FN=2 -> F1=0.0
        Against alt 1 ('alt_c', 'alt_d'): TP=2, FP=1, FN=0 -> Prec=2/3 (0.6667), Rec=1.0, F1=0.8
        Against alt 2 ('alt_e'): TP=0, FP=3, FN=1 -> F1=0.0
        Selected: alt 1 ('alt_c', 'alt_d').
        """
        discovered = ("alt_c", "alt_d", "distractor_x")
        matches, prec, rec, excess, missing = evaluate_ground_truth(multi_trigger_spec, discovered)

        assert matches is False
        assert prec == pytest.approx(0.6667, abs=1e-4)
        assert rec == 1.0
        assert excess == ("distractor_x",)
        assert missing == ()

    def test_case_4_primary_trigger_plus_distractor(self, multi_trigger_spec: GroundTruthSpec) -> None:
        """Case 4: A primary trigger plus a distractor.

        Discovered: {'t_a', 't_b', 'distractor_y'}
        Against primary ('t_a', 't_b'): TP=2, FP=1, FN=0 -> Prec=2/3 (0.6667), Rec=1.0, F1=0.8
        Against alternatives: F1=0.0
        Selected: primary.
        """
        discovered = ("t_a", "t_b", "distractor_y")
        matches, prec, rec, excess, missing = evaluate_ground_truth(multi_trigger_spec, discovered)

        assert matches is False
        assert prec == pytest.approx(0.6667, abs=1e-4)
        assert rec == 1.0
        assert excess == ("distractor_y",)
        assert missing == ()

    def test_case_5_partial_overlap_multiple_triggers(self) -> None:
        """Case 5: A discovery that partially overlaps multiple candidate triggers.

        Primary: ('t_a', 'shared')
        Alt 1: ('shared', 'alt_b', 'alt_c')
        Discovered: ('shared', 'alt_b')

        Against Primary: TP=1, FP=1, FN=1 -> Prec=1/2=0.5, Rec=1/2=0.5, F1=0.5
        Against Alt 1:   TP=2, FP=0, FN=1 -> Prec=2/2=1.0, Rec=2/3=0.6667, F1=4/5=0.8
        Selected candidate must be Alt 1 due to higher F1 (0.8 > 0.5).
        """
        spec = GroundTruthSpec(
            task_id="overlap_task",
            name="Overlap Task",
            description="",
            trigger_elements=("t_a", "shared"),
            alternative_valid_triggers=(("shared", "alt_b", "alt_c"),),
        )
        discovered = ("shared", "alt_b")
        matches, prec, rec, excess, missing = evaluate_ground_truth(spec, discovered)

        assert matches is False
        assert prec == 1.0
        assert rec == pytest.approx(0.6667, abs=1e-4)
        assert excess == ()
        assert missing == ("alt_c",)

    def test_case_6_discovery_matching_none(self, multi_trigger_spec: GroundTruthSpec) -> None:
        """Case 6: A discovery matching none of the triggers.

        Discovered: ('distractor_x', 'distractor_y')
        All candidates yield TP=0, Prec=0, Rec=0, F1=0.
        Candidate sizes: primary=2, alt1=2, alt2=1.
        Size tie-break: alt2 has size 1 (< 2), so alt2 is selected!
        Excess: ('distractor_x', 'distractor_y'), Missing: ('alt_e',)
        """
        discovered = ("distractor_x", "distractor_y")
        matches, prec, rec, excess, missing = evaluate_ground_truth(multi_trigger_spec, discovered)

        assert matches is False
        assert prec == 0.0
        assert rec == 0.0
        assert excess == ("distractor_x", "distractor_y")
        assert missing == ("alt_e",)

    def test_case_7_unequal_sized_triggers_and_f1_tie_handling(self) -> None:
        """Case 7: Unequal-sized valid triggers and deterministic F1 tie handling.

        Tests all deterministic tie-breaking levels:
        7A: F1 equal -> Recall breaks tie.
        7B: F1 equal, Recall equal -> Smaller size |T| breaks tie.
        7C: F1 equal, Recall equal, Size equal -> Candidate index (primary first) breaks tie.
        """
        # 7A: Recall breaks F1 tie
        # Discovered size = 4: {'k1', 'k2', 'd1', 'd2'}
        # Primary: {'k1', 'k2', 'p1'} (size 3) -> TP=2, FP=2, FN=1 -> Prec=0.5, Rec=2/3=0.6667, F1=4/7≈0.5714
        # Alt 1: {'k1', 'k2', 'd1', 'd2', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'} (size 10)
        #         -> TP=4, FP=0, FN=6 -> Prec=1.0, Rec=4/10=0.4000, F1=8/14=4/7≈0.5714
        # Both have F1 = 4/7. Primary has higher recall (0.6667 > 0.4000). Primary must be selected.
        spec_7a = GroundTruthSpec(
            task_id="tie_7a",
            name="Tie 7A",
            description="",
            trigger_elements=("k1", "k2", "p1"),
            alternative_valid_triggers=(("k1", "k2", "d1", "d2", "x1", "x2", "x3", "x4", "x5", "x6"),),
        )
        discovered_7a = ("k1", "k2", "d1", "d2")
        matches_a, prec_a, rec_a, excess_a, missing_a = evaluate_ground_truth(spec_7a, discovered_7a)

        assert matches_a is False
        assert prec_a == 0.5
        assert rec_a == pytest.approx(0.6667, abs=1e-4)
        assert excess_a == ("d1", "d2")
        assert missing_a == ("p1",)

        # 7B: Size breaks F1 & Recall tie
        # Discovered: {'unrelated'} -> TP=0 for all. F1=0, Rec=0.
        # Primary has size 3 ('p1', 'p2', 'p3')
        # Alt has size 1 ('a1',)
        # Alt must be selected because |Alt| = 1 < |Primary| = 3.
        spec_7b = GroundTruthSpec(
            task_id="tie_7b",
            name="Tie 7B",
            description="",
            trigger_elements=("p1", "p2", "p3"),
            alternative_valid_triggers=(("a1",),),
        )
        discovered_7b = ("unrelated",)
        matches_b, prec_b, rec_b, excess_b, missing_b = evaluate_ground_truth(spec_7b, discovered_7b)

        assert matches_b is False
        assert prec_b == 0.0
        assert rec_b == 0.0
        assert excess_b == ("unrelated",)
        assert missing_b == ("a1",)

        # 7C: Order of appearance breaks F1, Recall, and Size tie
        # Discovered: {'unrelated'} -> TP=0 for all.
        # Primary: {'p1', 'p2'} (size 2, index 0)
        # Alt:     {'a1', 'a2'} (size 2, index 1)
        # Primary must be selected due to earlier index (0 < 1).
        spec_7c = GroundTruthSpec(
            task_id="tie_7c",
            name="Tie 7C",
            description="",
            trigger_elements=("p1", "p2"),
            alternative_valid_triggers=(("a1", "a2"),),
        )
        discovered_7c = ("unrelated",)
        matches_c, prec_c, rec_c, excess_c, missing_c = evaluate_ground_truth(spec_7c, discovered_7c)

        assert matches_c is False
        assert prec_c == 0.0
        assert rec_c == 0.0
        assert excess_c == ("unrelated",)
        assert missing_c == ("p1", "p2")

    def test_case_8_tie_between_alternative_triggers(self) -> None:
        """Case 8: Deterministic tie-break between two distinct alternative candidates.

        Primary: ('p1', 'p2', 'p3') (size 3)
        Alt 1: ('shared_alt', 'alt_a') (size 2, candidate index 1)
        Alt 2: ('shared_alt', 'alt_b') (size 2, candidate index 2)
        Discovered: ('shared_alt', 'distractor')

        Against Primary: TP=0, FP=2, FN=3 -> F1=0.0
        Against Alt 1:   TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5, F1=0.5, size=2, index=1
        Against Alt 2:   TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5, F1=0.5, size=2, index=2

        Both Alt 1 and Alt 2 tie on F1 (0.5), Recall (0.5), and Size (2).
        Deterministic tie-break favors Alt 1 because candidate index 1 < 2 (-1 > -2).
        Selection of Alt 1 is observable via missing elements ('alt_a',).
        """
        spec = GroundTruthSpec(
            task_id="alt_tie_task",
            name="Alt Tie Task",
            description="",
            trigger_elements=("p1", "p2", "p3"),
            alternative_valid_triggers=(
                ("shared_alt", "alt_a"),
                ("shared_alt", "alt_b"),
            ),
        )
        discovered = ("shared_alt", "distractor")
        matches, prec, rec, excess, missing = evaluate_ground_truth(spec, discovered)

        assert matches is False
        assert prec == 0.5
        assert rec == 0.5
        assert excess == ("distractor",)
        assert missing == ("alt_a",)

    def test_case_9_duplicate_alternative_triggers(self) -> None:
        """Case 9: Evaluation is deterministic and handles duplicate alternative triggers without error.

        Primary: ('t_a', 't_b')
        Alt 1: ('alt_x', 'alt_y') (candidate index 1)
        Alt 2: ('alt_x', 'alt_y') (candidate index 2, duplicate of Alt 1)
        """
        spec = GroundTruthSpec(
            task_id="duplicate_alt_task",
            name="Duplicate Alt Task",
            description="",
            trigger_elements=("t_a", "t_b"),
            alternative_valid_triggers=(
                ("alt_x", "alt_y"),
                ("alt_x", "alt_y"),
            ),
        )

        # Exact match on duplicate alternative
        exact_discovered = ("alt_x", "alt_y")
        m_ex, p_ex, r_ex, e_ex, mis_ex = evaluate_ground_truth(spec, exact_discovered)
        assert m_ex is True
        assert p_ex == 1.0
        assert r_ex == 1.0
        assert e_ex == ()
        assert mis_ex == ()

        # Partial match on duplicate alternative
        partial_discovered = ("alt_x", "distractor")
        m_part, p_part, r_part, e_part, mis_part = evaluate_ground_truth(spec, partial_discovered)
        assert m_part is False
        assert p_part == 0.5
        assert r_part == 0.5
        assert e_part == ("distractor",)
        assert mis_part == ("alt_y",)

    def test_stochastic_and_model_pilot_consistency(
        self,
        stochastic_multi_trigger_spec: StochasticGroundTruthSpec,
        model_pilot_multi_trigger_spec: ModelPilotGroundTruthSpec,
    ) -> None:
        """Verify that stochastic and model-pilot evaluators follow identical multi-trigger logic."""
        discovered = ("alt_c", "alt_d", "distractor_x")

        # Stochastic
        s_matches, s_prec, s_rec, s_excess, s_missing = evaluate_stochastic_ground_truth(
            stochastic_multi_trigger_spec, discovered
        )
        assert s_matches is False
        assert s_prec == pytest.approx(0.6667, abs=1e-4)
        assert s_rec == 1.0
        assert s_excess == ("distractor_x",)
        assert s_missing == ()

        # Model pilot
        p_matches, p_prec, p_rec, p_excess, p_missing = evaluate_model_pilot_ground_truth(
            model_pilot_multi_trigger_spec, discovered
        )
        assert p_matches is False
        assert p_prec == pytest.approx(0.6667, abs=1e-4)
        assert p_rec == 1.0
        assert p_excess == ("distractor_x",)
        assert p_missing == ()


# ---------------------------------------------------------------------------
# Boundary Tests: Empty Discovery on Non-Null Tasks
# ---------------------------------------------------------------------------

class TestEmptyDiscoveryOnNonNullTask:
    """Verify that an empty discovered set on an active (non-null) task reports
    matches=False, precision=0.0, recall=0.0, excess=(), missing=sorted(true_trigger).
    """

    def test_deterministic_non_null_task_empty_discovery(self) -> None:
        """Deterministic active task with empty discovery returns 0.0 precision/recall."""
        spec = GroundTruthSpec(
            task_id="active_det_task",
            name="Active Deterministic Task",
            description="Active failure-inducing task",
            trigger_elements=("t_x", "t_y"),
            is_null_case=False,
        )
        cond = DiscoveredCondition(
            subset_elements=(),
            status=SearchStatus.NO_TRIGGER_FOUND,
            is_failure=False,
            is_one_minimal=False,
            target_calls_used=10,
            status_message="No trigger isolated",
        )
        matches, prec, rec, excess, missing = evaluate_ground_truth(spec, cond)
        assert matches is False
        assert prec == 0.0
        assert rec == 0.0
        assert excess == ()
        assert missing == ("t_x", "t_y")

        # Also test raw empty tuple input
        m2, p2, r2, e2, mis2 = evaluate_ground_truth(spec, ())
        assert m2 is False
        assert p2 == 0.0
        assert r2 == 0.0
        assert e2 == ()
        assert mis2 == ("t_x", "t_y")

    def test_stochastic_non_null_task_empty_discovery(self) -> None:
        """Stochastic active task with empty discovery returns 0.0 precision/recall."""
        spec = StochasticGroundTruthSpec(
            task_id="active_stoch_task",
            name="Active Stochastic Task",
            description="",
            trigger_elements=("s_1", "s_2"),
            base_probability=0.05,
            trigger_probability=0.90,
            is_null_case=False,
        )
        cond = StochasticDiscoveredCondition(
            subset_elements=(),
            status=SearchStatus.NO_TRIGGER_FOUND,
            is_failure=False,
            minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
            candidate_evaluation=CandidateEvaluation(
                candidate_elements=(),
                trials=10,
                failures=0,
                failure_rate=0.0,
                confidence_interval=(0.0, 0.28),
            ),
            target_calls_used=10,
        )
        matches, prec, rec, excess, missing = evaluate_stochastic_ground_truth(spec, cond)
        assert matches is False
        assert prec == 0.0
        assert rec == 0.0
        assert excess == ()
        assert missing == ("s_1", "s_2")

        # Also test raw empty tuple input
        m2, p2, r2, e2, mis2 = evaluate_stochastic_ground_truth(spec, ())
        assert m2 is False
        assert p2 == 0.0
        assert r2 == 0.0
        assert e2 == ()
        assert mis2 == ("s_1", "s_2")

    def test_model_pilot_non_null_task_empty_discovery(self) -> None:
        """Model pilot active task with empty discovery returns 0.0 precision/recall."""
        spec = ModelPilotGroundTruthSpec(
            task_id="active_pilot_task",
            name="Active Pilot Task",
            description="",
            trigger_elements=("p_a", "p_b"),
            is_null_case=False,
        )
        cond = StochasticDiscoveredCondition(
            subset_elements=(),
            status=SearchStatus.NO_TRIGGER_FOUND,
            is_failure=False,
            minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
            candidate_evaluation=CandidateEvaluation(
                candidate_elements=(),
                trials=5,
                failures=0,
                failure_rate=0.0,
                confidence_interval=(0.0, 0.43),
            ),
            target_calls_used=5,
        )
        matches, prec, rec, excess, missing = evaluate_model_pilot_ground_truth(spec, cond)
        assert matches is False
        assert prec == 0.0
        assert rec == 0.0
        assert excess == ()
        assert missing == ("p_a", "p_b")

        # Also test raw empty tuple input
        m2, p2, r2, e2, mis2 = evaluate_model_pilot_ground_truth(spec, ())
        assert m2 is False
        assert p2 == 0.0
        assert r2 == 0.0
        assert e2 == ()
        assert mis2 == ("p_a", "p_b")


# ---------------------------------------------------------------------------
# 8: Null-Case Precision & Recall Semantics (None Policy)
# ---------------------------------------------------------------------------

class TestNullCaseSemantics:
    """Verify null cases return (True, None, None, (), ()) and handle false discoveries."""

    def test_deterministic_null_case_clean(self) -> None:
        """Correct deterministic null identification returns True and None for undefined metrics."""
        null_spec = GroundTruthSpec(
            task_id="null_task",
            name="Null Task",
            description="Control condition with no trigger",
            trigger_elements=(),
            is_null_case=True,
        )
        cond = DiscoveredCondition(
            subset_elements=(),
            status=SearchStatus.NO_TRIGGER_FOUND,
            is_failure=False,
            is_one_minimal=False,
            target_calls_used=5,
            status_message="No trigger found",
        )
        matches, prec, rec, excess, missing = evaluate_ground_truth(null_spec, cond)

        assert matches is True
        assert prec is None
        assert rec is None
        assert excess == ()
        assert missing == ()

    def test_deterministic_null_case_empty_set_spurious_failure(self) -> None:
        """Empty set discovery on null task that falsely reproduces failure reports False, None, None, (), ()."""
        null_spec = GroundTruthSpec(
            task_id="null_task",
            name="Null Task",
            description="",
            trigger_elements=(),
            is_null_case=True,
        )
        cond = DiscoveredCondition(
            subset_elements=(),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            is_one_minimal=False,
            target_calls_used=5,
            status_message="Spurious failure reported on empty set",
        )
        matches, prec, rec, excess, missing = evaluate_ground_truth(null_spec, cond)

        assert matches is False
        assert prec is None
        assert rec is None
        assert excess == ()
        assert missing == ()

    def test_deterministic_null_case_false_positive(self) -> None:
        """False positive discovery on null task reports False, precision=0.0, and recall=None."""
        null_spec = GroundTruthSpec(
            task_id="null_task",
            name="Null Task",
            description="",
            trigger_elements=(),
            is_null_case=True,
        )
        cond = DiscoveredCondition(
            subset_elements=("false_alarm",),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            is_one_minimal=False,
            target_calls_used=10,
            status_message="False trigger isolated",
        )
        matches, prec, rec, excess, missing = evaluate_ground_truth(null_spec, cond)

        assert matches is False
        assert prec == 0.0
        assert rec is None
        assert excess == ("false_alarm",)
        assert missing == ()

    def test_stochastic_null_case_clean(self) -> None:
        """Correct stochastic null identification returns True and None for undefined metrics."""
        stoch_null = StochasticGroundTruthSpec(
            task_id="stoch_null",
            name="Stochastic Null",
            description="Background rate only",
            trigger_elements=(),
            base_probability=0.05,
            trigger_probability=0.05,
            is_null_case=True,
        )
        cond = StochasticDiscoveredCondition(
            subset_elements=(),
            status=SearchStatus.NO_TRIGGER_FOUND,
            is_failure=False,
            minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
            candidate_evaluation=CandidateEvaluation(
                candidate_elements=(),
                trials=10,
                failures=0,
                failure_rate=0.0,
                confidence_interval=(0.0, 0.28),
            ),
            target_calls_used=10,
        )
        matches, prec, rec, excess, missing = evaluate_stochastic_ground_truth(stoch_null, cond)

        assert matches is True
        assert prec is None
        assert rec is None
        assert excess == ()
        assert missing == ()

    def test_stochastic_null_case_empty_set_spurious_failure(self) -> None:
        """Empty set discovery on stochastic null task that falsely reproduces failure reports False, None, None, (), ()."""
        stoch_null = StochasticGroundTruthSpec(
            task_id="stoch_null",
            name="Stochastic Null",
            description="",
            trigger_elements=(),
            base_probability=0.05,
            trigger_probability=0.05,
            is_null_case=True,
        )
        cond = StochasticDiscoveredCondition(
            subset_elements=(),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            minimality_status=StochasticMinimalityStatus.OBSERVED_ONLY,
            candidate_evaluation=CandidateEvaluation(
                candidate_elements=(),
                trials=10,
                failures=8,
                failure_rate=0.8,
                confidence_interval=(0.49, 0.94),
            ),
            target_calls_used=20,
        )
        matches, prec, rec, excess, missing = evaluate_stochastic_ground_truth(stoch_null, cond)

        assert matches is False
        assert prec is None
        assert rec is None
        assert excess == ()
        assert missing == ()

    def test_stochastic_null_case_false_positive(self) -> None:
        """False positive discovery on stochastic null task reports False, precision=0.0, and recall=None."""
        stoch_null = StochasticGroundTruthSpec(
            task_id="stoch_null",
            name="Stochastic Null",
            description="",
            trigger_elements=(),
            base_probability=0.05,
            trigger_probability=0.05,
            is_null_case=True,
        )
        cond = StochasticDiscoveredCondition(
            subset_elements=("spurious_feature",),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            minimality_status=StochasticMinimalityStatus.OBSERVED_ONLY,
            candidate_evaluation=CandidateEvaluation(
                candidate_elements=("spurious_feature",),
                trials=10,
                failures=8,
                failure_rate=0.8,
                confidence_interval=(0.49, 0.94),
            ),
            target_calls_used=20,
        )
        matches, prec, rec, excess, missing = evaluate_stochastic_ground_truth(stoch_null, cond)

        assert matches is False
        assert prec == 0.0
        assert rec is None
        assert excess == ("spurious_feature",)
        assert missing == ()

    def test_model_pilot_null_case_clean(self) -> None:
        """Clean model-pilot null task returns True and None for precision and recall."""
        pilot_null = ModelPilotGroundTruthSpec(
            task_id="pilot_null",
            name="Pilot Null",
            description="Clean null hypothesis",
            trigger_elements=(),
            is_null_case=True,
        )
        cond = StochasticDiscoveredCondition(
            subset_elements=(),
            status=SearchStatus.NO_TRIGGER_FOUND,
            is_failure=False,
            minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
            candidate_evaluation=CandidateEvaluation(
                candidate_elements=(),
                trials=5,
                failures=0,
                failure_rate=0.0,
                confidence_interval=(0.0, 0.43),
            ),
            target_calls_used=5,
        )
        matches, prec, rec, excess, missing = evaluate_model_pilot_ground_truth(pilot_null, cond)
        assert matches is True
        assert prec is None
        assert rec is None
        assert excess == ()
        assert missing == ()

    def test_model_pilot_null_case_empty_set_spurious_failure(self) -> None:
        """Empty set discovery on model-pilot null task with spurious failure reports False, None, None, (), ()."""
        pilot_null = ModelPilotGroundTruthSpec(
            task_id="pilot_null",
            name="Pilot Null",
            description="Clean null hypothesis",
            trigger_elements=(),
            is_null_case=True,
        )
        cond = StochasticDiscoveredCondition(
            subset_elements=(),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            minimality_status=StochasticMinimalityStatus.OBSERVED_ONLY,
            candidate_evaluation=CandidateEvaluation(
                candidate_elements=(),
                trials=5,
                failures=4,
                failure_rate=0.8,
                confidence_interval=(0.37, 0.96),
            ),
            target_calls_used=10,
        )
        matches, prec, rec, excess, missing = evaluate_model_pilot_ground_truth(pilot_null, cond)
        assert matches is False
        assert prec is None
        assert rec is None
        assert excess == ()
        assert missing == ()

    def test_model_pilot_null_case_false_positive(self) -> None:
        """False positive discovery on model-pilot null task returns False, precision=0.0, and recall=None."""
        pilot_null = ModelPilotGroundTruthSpec(
            task_id="pilot_null",
            name="Pilot Null",
            description="Clean null hypothesis",
            trigger_elements=(),
            is_null_case=True,
        )
        cond = StochasticDiscoveredCondition(
            subset_elements=("hallucinated_param",),
            status=SearchStatus.COMPLETE,
            is_failure=True,
            minimality_status=StochasticMinimalityStatus.OBSERVED_ONLY,
            candidate_evaluation=CandidateEvaluation(
                candidate_elements=("hallucinated_param",),
                trials=5,
                failures=4,
                failure_rate=0.8,
                confidence_interval=(0.37, 0.96),
            ),
            target_calls_used=10,
        )
        matches, prec, rec, excess, missing = evaluate_model_pilot_ground_truth(pilot_null, cond)
        assert matches is False
        assert prec == 0.0
        assert rec is None
        assert excess == ("hallucinated_param",)
        assert missing == ()


# ---------------------------------------------------------------------------
# Downstream Models, Reports, and Aggregations
# ---------------------------------------------------------------------------

class TestDownstreamConsumersWithNullSemantics:
    """Verify models, report formatters, and aggregators handle None precision and recall."""

    def test_evaluation_result_model_accepts_none(self) -> None:
        """EvaluationResult accepts precision=None, recall=None and validates json round-trip."""
        res = EvaluationResult(
            task_name="Null Case",
            description="Control",
            initial_elements=("d1", "d2"),
            ground_truth_trigger=(),
            minimizer_condition=DiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                target_calls_used=5,
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
            excess_elements=(),
            missing_elements=(),
        )
        assert res.precision is None
        assert res.recall is None

        # JSON Roundtrip
        data = res.model_dump_json()
        reloaded = EvaluationResult.model_validate_json(data)
        assert reloaded.precision is None
        assert reloaded.recall is None
        assert reloaded.matches_ground_truth is True

    def test_stochastic_evaluation_result_model_accepts_none(self) -> None:
        """StochasticEvaluationResult accepts precision=None, recall=None and validates json round-trip."""
        res = StochasticEvaluationResult(
            task_name="Stochastic Null Case",
            description="Control",
            initial_elements=("d1", "d2"),
            ground_truth_trigger=(),
            base_probability=0.05,
            trigger_probability=0.05,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
                candidate_evaluation=CandidateEvaluation(
                    candidate_elements=(),
                    trials=5,
                    failures=0,
                    failure_rate=0.0,
                    confidence_interval=(0.0, 0.43),
                ),
                target_calls_used=5,
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
            excess_elements=(),
            missing_elements=(),
        )
        assert res.precision is None
        assert res.recall is None

        data = res.model_dump_json()
        reloaded = StochasticEvaluationResult.model_validate_json(data)
        assert reloaded.precision is None
        assert reloaded.recall is None

    def test_algorithm_run_result_accepts_none(self) -> None:
        """AlgorithmRunResult accepts precision=None, recall=None."""
        res = AlgorithmRunResult(
            algorithm_name="test_algo",
            discovered_elements=(),
            target_calls=5,
            search_target_calls=5,
            verification_target_calls=0,
            subsets_examined=1,
            failure_reproduced=False,
            minimality_status="not_applicable",
            is_verified_one_minimal=False,
            matches_ground_truth=True,
            precision=None,
            recall=None,
            budget_exhausted=False,
            runtime_ms=12.3,
        )
        assert res.precision is None
        assert res.recall is None

    def test_deterministic_markdown_report_formatting_with_none(self) -> None:
        """format_markdown_report formats None precision and recall cleanly without crashing."""
        eval_res = EvaluationResult(
            task_name="Null Task",
            description="Non-failing control",
            initial_elements=("param_a",),
            ground_truth_trigger=(),
            minimizer_condition=DiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                target_calls_used=2,
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
        )
        report = ExperimentReport(
            experiment_name="test_null_formatting",
            config=ExperimentConfig(experiment_name="test", max_trials=10),
            total_target_calls=2,
            evaluations=(eval_res,),
            summary="Null test summary",
        )

        md = format_markdown_report(report)
        assert "Minimizer Precision**: `N/A (Null Case / Control)`" in md
        assert "Minimizer Recall**: `N/A (Null Case / Control)`" in md

    def test_stochastic_markdown_report_formatting_with_none(self) -> None:
        """format_stochastic_markdown_report formats None precision and recall cleanly without crashing."""
        eval_res = StochasticEvaluationResult(
            task_name="Stochastic Null Task",
            description="Background noise only",
            initial_elements=("param_a",),
            ground_truth_trigger=(),
            base_probability=0.05,
            trigger_probability=0.05,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
                candidate_evaluation=CandidateEvaluation(
                    candidate_elements=(),
                    trials=10,
                    failures=0,
                    failure_rate=0.0,
                    confidence_interval=(0.0, 0.28),
                ),
                target_calls_used=10,
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
        )
        report = StochasticExperimentReport(
            experiment_name="test_stoch_null_formatting",
            config=ExperimentConfig(experiment_name="test_stoch", max_trials=20),
            trials_per_candidate=10,
            total_target_calls=10,
            evaluations=(eval_res,),
            summary="Stochastic null test summary",
        )

        md = format_stochastic_markdown_report(report)
        assert "Minimizer Precision**: `N/A (Null Case / Control)`" in md
        assert "Minimizer Recall**: `N/A (Null Case / Control)`" in md

    def test_aggregation_excludes_none_values(self) -> None:
        """Aggregations properly exclude None values without raising TypeError or skewing averages."""
        eval1 = EvaluationResult(
            task_name="Task 1",
            description="",
            initial_elements=("a", "b"),
            ground_truth_trigger=("a",),
            minimizer_condition=DiscoveredCondition(
                subset_elements=("a",),
                status=SearchStatus.COMPLETE,
                is_failure=True,
                target_calls_used=4,
            ),
            matches_ground_truth=True,
            precision=1.0,
            recall=1.0,
        )
        eval_null = EvaluationResult(
            task_name="Task Null",
            description="",
            initial_elements=("c",),
            ground_truth_trigger=(),
            minimizer_condition=DiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                target_calls_used=2,
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
        )

        evaluations = [eval1, eval_null]
        valid_precs = [e.precision for e in evaluations if e.precision is not None]
        valid_recs = [e.recall for e in evaluations if e.recall is not None]

        # Valid precision count is 1 (null case excluded)
        assert len(valid_precs) == 1
        assert len(valid_recs) == 1
        mean_prec = sum(valid_precs) / len(valid_precs)
        mean_rec = sum(valid_recs) / len(valid_recs)

        # Average is correctly 1.0, NOT diluted to 0.5 by treating None as 0.0
        assert mean_prec == 1.0
        assert mean_rec == 1.0

    def test_model_pilot_markdown_report_formatting_with_none(self) -> None:
        """format_model_pilot_markdown_report formats clean null cases and false positives correctly."""
        clean_eval = ModelPilotEvaluationResult(
            task_name="Clean Null Pilot",
            description="Control hypothesis",
            initial_elements=("cand_a", "cand_b"),
            ground_truth_trigger=(),
            is_null_case=True,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
                candidate_evaluation=CandidateEvaluation(
                    candidate_elements=(),
                    trials=5,
                    failures=0,
                    failure_rate=0.0,
                    confidence_interval=(0.0, 0.43),
                ),
                target_calls_used=5,
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
        )

        fp_eval = ModelPilotEvaluationResult(
            task_name="False Positive Pilot",
            description="Control hypothesis with spurious failure",
            initial_elements=("cand_a", "cand_b"),
            ground_truth_trigger=(),
            is_null_case=True,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=("cand_a",),
                status=SearchStatus.COMPLETE,
                is_failure=True,
                minimality_status=StochasticMinimalityStatus.OBSERVED_ONLY,
                candidate_evaluation=CandidateEvaluation(
                    candidate_elements=("cand_a",),
                    trials=5,
                    failures=5,
                    failure_rate=1.0,
                    confidence_interval=(0.57, 1.0),
                ),
                target_calls_used=10,
            ),
            matches_ground_truth=False,
            precision=0.0,
            recall=None,
            excess_elements=("cand_a",),
        )

        report = ModelPilotExperimentReport(
            experiment_name="test_model_pilot_reporting",
            execution_tier=ExecutionTier.FAKE_PROVIDER,
            provider_name="fake_provider",
            model_name="fake-model",
            config=ExperimentConfig(experiment_name="test_pilot", max_trials=50),
            budget_summary={"requests": 15, "estimated_spend_usd": 0.0},
            trials_per_candidate=5,
            total_target_calls=15,
            evaluations=(clean_eval, fp_eval),
            aggregate_metrics={
                "tasks_evaluated": 2,
                "ground_truth_matches": 1,
                "statistically_supported_minimality": 0,
                "mean_precision": 0.0,  # 0.0 from fp_eval, clean_eval excluded
                "mean_recall": None,    # both have recall=None
            },
            summary="Model pilot test summary",
        )

        md = format_model_pilot_markdown_report(report)
        assert "# AgentXRay Model-Backed Agent Pilot Report: [FAKE_PROVIDER] test_model_pilot_reporting" in md
        # Clean null case: both precision and recall N/A
        assert "Minimizer Precision**: `N/A (Null Case / Control)` | **Recall**: `N/A (Null Case / Control)`" in md
        # False positive null case: precision is 0.0%, recall is N/A
        assert "Minimizer Precision**: `0.0%` | **Recall**: `N/A (Null Case / Control)`" in md

    def test_model_pilot_json_report_roundtrip(self) -> None:
        """format_model_pilot_json_report serializes and reloads ModelPilotExperimentReport with None metrics."""
        clean_eval = ModelPilotEvaluationResult(
            task_name="Clean Null Pilot",
            description="Control hypothesis",
            initial_elements=("cand_a",),
            ground_truth_trigger=(),
            is_null_case=True,
            minimizer_condition=StochasticDiscoveredCondition(
                subset_elements=(),
                status=SearchStatus.NO_TRIGGER_FOUND,
                is_failure=False,
                minimality_status=StochasticMinimalityStatus.NOT_APPLICABLE,
                candidate_evaluation=CandidateEvaluation(
                    candidate_elements=(),
                    trials=5,
                    failures=0,
                    failure_rate=0.0,
                    confidence_interval=(0.0, 0.43),
                ),
                target_calls_used=5,
            ),
            matches_ground_truth=True,
            precision=None,
            recall=None,
        )
        report = ModelPilotExperimentReport(
            experiment_name="test_pilot_json",
            execution_tier=ExecutionTier.SYNTHETIC,
            provider_name="fake_provider",
            model_name="fake-model",
            config=ExperimentConfig(experiment_name="test_json", max_trials=10),
            budget_summary={},
            trials_per_candidate=5,
            total_target_calls=5,
            evaluations=(clean_eval,),
            summary="JSON roundtrip test",
        )
        json_str = format_model_pilot_json_report(report)
        reloaded = ModelPilotExperimentReport.model_validate_json(json_str)
        assert reloaded.evaluations[0].precision is None
        assert reloaded.evaluations[0].recall is None
        assert reloaded.evaluations[0].matches_ground_truth is True

    def test_scaling_benchmark_csv_export_with_none_and_fp_metrics(self, tmp_path: Path) -> None:
        """export_benchmark_results exports empty string for None and exact float string for defined metrics."""
        clean_run = AlgorithmRunResult(
            algorithm_name="stochastic_delta_debugging",
            discovered_elements=(),
            target_calls=5,
            search_target_calls=5,
            verification_target_calls=0,
            subsets_examined=1,
            failure_reproduced=False,
            minimality_status="not_applicable",
            is_verified_one_minimal=False,
            matches_ground_truth=True,
            precision=None,
            recall=None,
            budget_exhausted=False,
            runtime_ms=10.0,
        )
        fp_run = AlgorithmRunResult(
            algorithm_name="stochastic_greedy_ablation",
            discovered_elements=("spurious_x",),
            target_calls=15,
            search_target_calls=15,
            verification_target_calls=0,
            subsets_examined=3,
            failure_reproduced=True,
            minimality_status="observed_only",
            is_verified_one_minimal=False,
            matches_ground_truth=False,
            precision=0.0,
            recall=None,
            budget_exhausted=False,
            runtime_ms=25.0,
        )
        active_run = AlgorithmRunResult(
            algorithm_name="stochastic_seeded_random",
            discovered_elements=("t1",),
            target_calls=20,
            search_target_calls=10,
            verification_target_calls=10,
            subsets_examined=5,
            failure_reproduced=True,
            minimality_status="statistically_supported",
            is_verified_one_minimal=True,
            matches_ground_truth=True,
            precision=1.0,
            recall=1.0,
            budget_exhausted=False,
            runtime_ms=30.0,
        )

        inst = ScalingInstanceResult(
            family="null_target",
            k=4,
            seed=100,
            initial_elements=("a", "b", "c", "d"),
            ground_truth_trigger=(),
            alternative_valid_triggers=(),
            is_null_case=True,
            dd_result=clean_run,
            greedy_result=fp_run,
            random_result=active_run,
        )

        suite = ScalingBenchmarkSuite(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_instances=1,
            candidate_sizes=(4,),
            target_families=("null_target",),
            trials_per_candidate=5,
            max_trials=100,
            instances=(inst,),
        )

        _, csv_path = export_benchmark_results(suite, tmp_path)
        assert csv_path.exists()

        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        assert len(rows) == 3
        # Clean run: precision and recall are both empty string
        assert rows[0]["algorithm"] == "stochastic_delta_debugging"
        assert rows[0]["precision"] == ""
        assert rows[0]["recall"] == ""

        # False positive run: precision is 0.0, recall is empty string
        assert rows[1]["algorithm"] == "stochastic_greedy_ablation"
        assert rows[1]["precision"] == "0.0"
        assert rows[1]["recall"] == ""

        # Active run: precision is 1.0, recall is 1.0
        assert rows[2]["algorithm"] == "stochastic_seeded_random"
        assert rows[2]["precision"] == "1.0"
        assert rows[2]["recall"] == "1.0"
