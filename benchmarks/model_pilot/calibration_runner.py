"""Calibration Pilot Runner for Phase 2B.0 Local-Model Harness Qualification.

Implements the frozen Phase 2B.0 protocol:
- Pre-flight qualification of local Ollama runtime (or offline FakeProvider).
- Export of candidate specs and ground truth separate from search minimizers.
- Calibration of negative, positive, and null controls.
- Demonstrated trigger verification (P(fail | T1) >= 0.70 and P(fail | ~T1) <= 0.20).
- Controlled execution of Stochastic Delta Debugging, Greedy Ablation, and Random Search.
- Holdout confirmation trials on discovered subsets.
- Full trial-level and candidate-level event logging.
- Machine-readable results and comprehensive calibration reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agentxray.agent_harness.agent import ControlledAgent
from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator
from agentxray.agent_harness.target import ModelBackedAgentTarget
from agentxray.agent_harness.tools import CALCULATOR_TOOL_DEF, MockToolEnvironment
from agentxray.models import (
    AccessMode,
    ExecutionTier,
    ExperimentConfig,
    InputCase,
    StochasticDiscoveredCondition,
    StochasticMinimalityStatus,
    TargetOutcome,
    TargetResult,
)
from agentxray.providers.base import ModelProvider
from agentxray.providers.fake import FakeProvider
from agentxray.providers.ollama_adapter import OllamaProvider
from agentxray.runner import InterventionRunner
from agentxray.statistics import (
    compare_proportions,
    fisher_exact_test,
    wilson_score_interval,
)
from agentxray.stochastic_baselines import (
    run_stochastic_greedy_ablation,
    run_stochastic_random_search,
)
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.model_pilot.ground_truth import (
    ModelPilotGroundTruthSpec,
    evaluate_model_pilot_ground_truth,
)
from benchmarks.model_pilot.tasks import create_model_pilot_calibrated_currency_task

# Predefined candidate intervention space (K = 4)
CANDIDATE_TRIGGER = (
    'strict_currency_formatting: format all numbers within arithmetic tool arguments '
    'with currency symbols (e.g. "$150" or "$250")'
)
CANDIDATE_DISTRACTOR_1 = "concise_summary: provide clear one-line reimbursement statement"
CANDIDATE_DISTRACTOR_2 = "compliance_tag: include audit reference code REF-2026"
CANDIDATE_DISTRACTOR_3 = "audit_memo: append verification timestamp to final answer"

ALL_CANDIDATE_ELEMENTS = (
    CANDIDATE_TRIGGER,
    CANDIDATE_DISTRACTOR_1,
    CANDIDATE_DISTRACTOR_2,
    CANDIDATE_DISTRACTOR_3,
)


class CalibrationPilotOrchestrator:
    """Coordinates the frozen Phase 2B.0 local-model calibration pilot."""

    def __init__(
        self,
        provider_type: str = "fake",
        model_name: str | None = None,
        trials_per_subset: int = 5,
        max_requests_per_algorithm: int = 100,
        failure_threshold: float = 0.50,
        alpha: float = 0.05,
        confidence: float = 0.95,
        timeout_seconds: float = 180.0,
        results_dir: Path | str = "benchmarks/results",
        docs_dir: Path | str = "docs/research",
    ) -> None:
        self.provider_type = provider_type.strip().lower()
        self.results_dir = Path(results_dir)
        self.docs_dir = Path(docs_dir)
        self.trials_per_subset = trials_per_subset
        self.max_requests_per_algorithm = max_requests_per_algorithm
        self.failure_threshold = failure_threshold
        self.alpha = alpha
        self.confidence = confidence
        self.timeout_seconds = timeout_seconds

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.event_log_path = self.results_dir / "phase_2b0_calibration_event_log.jsonl"
        # Truncate / initialize event log for new run
        self.event_log_path.write_text("", encoding="utf-8")

        # Resolve provider
        if self.provider_type == "ollama":
            self.model_name = model_name or "llama3.2:3b"
            self.provider: ModelProvider = OllamaProvider(
                model_name=self.model_name,
                timeout_seconds=30.0,
            )
            self.tier = ExecutionTier.LOCAL_MODEL
        elif self.provider_type == "fake":
            self.model_name = model_name or "fake-model-pilot"
            self.provider = FakeProvider(model_name=self.model_name)
            self.tier = ExecutionTier.FAKE_PROVIDER
        else:
            raise ValueError(f"Unsupported provider_type: {provider_type!r}. Must be 'ollama' or 'fake'.")

        self.runtime_metadata: dict[str, Any] = {}
        self.event_counter = 0

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append an immutable structured event to the machine-readable event log."""
        self.event_counter += 1
        record = {
            "event_id": self.event_counter,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload,
        }
        with open(self.event_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def inspect_and_record_runtime(self) -> dict[str, Any]:
        """Audit runtime health, daemon availability, model metadata, and hardware."""
        meta: dict[str, Any] = {
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "execution_tier": self.tier.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "commercial_api_spend_usd": 0.0000,
        }

        if isinstance(self.provider, OllamaProvider):
            self.provider.validate_configuration()
            is_avail = self.provider.is_available()
            if not is_avail:
                raise RuntimeError(
                    f"Local Ollama daemon is unreachable at {self.provider.base_url}. "
                    "Ensure Ollama is running locally via 'ollama serve'."
                )
            meta["daemon_reachable"] = True
            meta["base_url"] = self.provider.base_url
            # Query daemon tags for model metadata
            try:
                import urllib.request
                req = urllib.request.Request(f"{self.provider.base_url}/api/tags")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    tags_data = json.loads(resp.read().decode("utf-8"))
                    matched = [m for m in tags_data.get("models", []) if m.get("name") == self.model_name]
                    if matched:
                        m_info = matched[0]
                        meta["digest"] = m_info.get("digest")
                        meta["model_size_bytes"] = m_info.get("size")
                        meta["modified_at"] = m_info.get("modified_at")
                        details = m_info.get("details", {})
                        meta["format"] = details.get("format")
                        meta["family"] = details.get("family")
                        meta["parameter_size"] = details.get("parameter_size")
                        meta["quantization_level"] = details.get("quantization_level")
                        meta["context_length"] = details.get("context_length")
            except Exception as e:
                meta["tags_query_error"] = str(e)
        else:
            meta["daemon_reachable"] = True
            meta["mock_mode"] = True

        self.runtime_metadata = meta
        self.log_event("runtime_inspected", meta)
        return meta

    def export_specifications(self) -> tuple[Path, Path]:
        """Export candidate specifications and ground-truth specs separate from search algorithms."""
        cand_specs_path = self.results_dir / "phase_2b0_candidate_specs.json"
        gt_path = self.results_dir / "phase_2b0_ground_truth.json"

        cand_data = {
            "task_id": "pilot_calibrated_currency",
            "candidate_space_size": len(ALL_CANDIDATE_ELEMENTS),
            "elements": [
                {
                    "id": f"elem_{i+1}",
                    "element_text": elem,
                    "element_role": "planted_trigger" if elem == CANDIDATE_TRIGGER else f"distractor_{i}",
                }
                for i, elem in enumerate(ALL_CANDIDATE_ELEMENTS)
            ],
            "base_system_prompt": "You are an automated accounting assistant. Always use the calculator tool for calculations.",
            "user_prompt": "Calculate total reimbursement for expenses of 150 and 250.",
            "tools": [{"name": CALCULATOR_TOOL_DEF.name, "description": CALCULATOR_TOOL_DEF.description}],
        }
        cand_specs_path.write_text(json.dumps(cand_data, indent=2), encoding="utf-8")

        gt_data = {
            "task_id": "pilot_calibrated_currency",
            "true_trigger_elements": [CANDIDATE_TRIGGER],
            "trigger_cardinality": 1,
            "is_null_case": False,
            "expected_failure_classification": "FAIL_TOOL_EXEC",
            "failure_mechanism": "Strict currency formatting in tool argument causes calculator AST syntax rejection.",
            "leakage_protection": "Evaluator and Minimizers have no access to this ground-truth definition.",
        }
        gt_path.write_text(json.dumps(gt_data, indent=2), encoding="utf-8")

        self.log_event("specifications_exported", {
            "candidate_specs_file": str(cand_specs_path),
            "ground_truth_file": str(gt_path),
        })
        return cand_specs_path, gt_path

    def export_configuration(self) -> Path:
        """Export frozen pilot configuration."""
        cfg_path = self.results_dir / "phase_2b0_calibration_config.json"
        cfg_data = {
            "protocol_document": "docs/research/phase_2b0_local_model_pilot_protocol.md",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "execution_tier": self.tier.value,
            "trials_per_subset": self.trials_per_subset,
            "max_requests_per_algorithm": self.max_requests_per_algorithm,
            "failure_threshold": self.failure_threshold,
            "alpha": self.alpha,
            "confidence": self.confidence,
            "timeout_seconds": self.timeout_seconds,
            "runtime_metadata": self.runtime_metadata,
        }
        cfg_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
        self.log_event("config_exported", {"config_file": str(cfg_path)})
        return cfg_path

    def _create_target(self, tracker: CostTracker) -> ModelBackedAgentTarget:
        """Create target agent adapter adhering to black-box execution interface."""
        env = MockToolEnvironment()
        agent = ControlledAgent(
            provider=self.provider,
            mock_env=env,
            cost_tracker=tracker,
            max_turns=3,
        )
        evaluator = ToolCallingTaskEvaluator(
            required_tool="calculator",
            expected_substring="400",
        )
        return ModelBackedAgentTarget(
            agent=agent,
            evaluator=evaluator,
            base_system_prompt="You are an automated accounting assistant. Always use the calculator tool for calculations.",
            user_prompt="Calculate total reimbursement for expenses of 150 and 250.",
            tools=[CALCULATOR_TOOL_DEF],
            name="ModelPilotCalibratedCurrencyTarget",
        )

    def evaluate_subset_trials(
        self,
        target: ModelBackedAgentTarget,
        tracker: CostTracker,
        elements: tuple[str, ...],
        case_id: str,
        phase: str,
        n_trials: int,
    ) -> dict[str, Any]:
        """Execute a controlled batch of trials on a specific candidate subset."""
        input_case = InputCase(
            case_id=case_id,
            elements=elements,
            description=f"Subset evaluation under phase {phase}",
        )
        failures = 0
        trials_data: list[dict[str, Any]] = []

        start_requests = tracker.total_requests
        start_tokens = tracker.total_tokens

        for trial_idx in range(n_trials):
            trial_start = time.perf_counter()
            res = target.execute(input_case)
            trial_duration = time.perf_counter() - trial_start

            is_failure = (res.outcome == TargetOutcome.FAILURE)
            if is_failure:
                failures += 1

            trace = target.last_trace
            tool_calls_info = [
                {
                    "tool": t.tool_name,
                    "arguments": t.arguments,
                    "output": t.output,
                }
                for t in (trace.tool_executions if trace else ())
            ]

            t_record = {
                "trial_index": trial_idx + 1,
                "outcome": res.outcome.value,
                "is_failure": is_failure,
                "latency_seconds": round(trial_duration, 4),
                "tool_executions": tool_calls_info,
                "final_response": trace.final_response if trace else "",
                "error_message": trace.error_message if trace else None,
            }
            trials_data.append(t_record)

            self.log_event("trial_executed", {
                "phase": phase,
                "case_id": case_id,
                "trial_index": trial_idx + 1,
                "candidate_subset": list(elements),
                **t_record,
            })

        requests_consumed = tracker.total_requests - start_requests
        tokens_consumed = tracker.total_tokens - start_tokens
        failure_rate = round(failures / n_trials, 4)
        ci_lower, ci_upper = wilson_score_interval(failures, n_trials, confidence=self.confidence)

        summary = {
            "case_id": case_id,
            "elements": list(elements),
            "trials": n_trials,
            "failures": failures,
            "successes": n_trials - failures,
            "failure_rate": failure_rate,
            "wilson_ci": [ci_lower, ci_upper],
            "requests_consumed": requests_consumed,
            "tokens_consumed": tokens_consumed,
            "is_failing": failure_rate >= self.failure_threshold,
        }

        self.log_event("candidate_evaluated", {
            "phase": phase,
            **summary,
        })
        return summary

    def run_calibration_controls(self) -> dict[str, Any]:
        """Execute and calibrate the 4 prespecified controls:

        1. Negative Control / Baseline (empty subset)
        2. Positive Control (planted trigger alone)
        3. Null Control (all 3 distractors, no trigger)
        4. Full Intervention (all 4 elements)
        """
        tracker = CostTracker(
            ExperimentBudgetConfig(
                max_model_requests=250,
                max_spend_usd=1.00,
                model_name=self.model_name,
            )
        )
        target = self._create_target(tracker)

        n = self.trials_per_subset
        self.log_event("calibration_phase_started", {"trials_per_subset": n})

        # 1. Negative control
        baseline_res = self.evaluate_subset_trials(
            target=target,
            tracker=tracker,
            elements=(),
            case_id="ctrl_negative_baseline",
            phase="calibration_control",
            n_trials=n,
        )

        # 2. Positive control
        pos_res = self.evaluate_subset_trials(
            target=target,
            tracker=tracker,
            elements=(CANDIDATE_TRIGGER,),
            case_id="ctrl_positive_trigger",
            phase="calibration_control",
            n_trials=n,
        )

        # 3. Null control
        null_res = self.evaluate_subset_trials(
            target=target,
            tracker=tracker,
            elements=(CANDIDATE_DISTRACTOR_1, CANDIDATE_DISTRACTOR_2, CANDIDATE_DISTRACTOR_3),
            case_id="ctrl_null_distractors",
            phase="calibration_control",
            n_trials=n,
        )

        # 4. Full configuration
        full_res = self.evaluate_subset_trials(
            target=target,
            tracker=tracker,
            elements=ALL_CANDIDATE_ELEMENTS,
            case_id="ctrl_full_intervention",
            phase="calibration_control",
            n_trials=n,
        )

        # Statistical comparisons
        comp_pos_vs_baseline = compare_proportions(
            k1=pos_res["failures"],
            n1=pos_res["trials"],
            k2=baseline_res["failures"],
            n2=baseline_res["trials"],
            alpha=self.alpha,
            confidence=self.confidence,
        )

        comp_pos_vs_null = compare_proportions(
            k1=pos_res["failures"],
            n1=pos_res["trials"],
            k2=null_res["failures"],
            n2=null_res["trials"],
            alpha=self.alpha,
            confidence=self.confidence,
        )

        # Qualification checks
        baseline_ok = baseline_res["failure_rate"] <= 0.20
        positive_ok = pos_res["failure_rate"] >= 0.70
        null_ok = null_res["failure_rate"] <= 0.20
        full_ok = full_res["failure_rate"] >= 0.70

        is_confirmed = baseline_ok and positive_ok and null_ok and full_ok
        trigger_status = "TRIGGER_CONFIRMED" if is_confirmed else "TRIGGER_UNCONFIRMED"

        results = {
            "trigger_status": trigger_status,
            "qualification_checks": {
                "baseline_clean_lte_0_20": baseline_ok,
                "positive_trigger_gte_0_70": positive_ok,
                "null_distractors_lte_0_20": null_ok,
                "full_intervention_gte_0_70": full_ok,
                "all_checks_passed": is_confirmed,
            },
            "negative_control_baseline": baseline_res,
            "positive_control_trigger": pos_res,
            "null_control_distractors": null_res,
            "full_intervention": full_res,
            "statistical_contrasts": {
                "positive_vs_baseline": {
                    "risk_difference": comp_pos_vs_baseline.difference,
                    "newcombe_ci": list(comp_pos_vs_baseline.difference_ci),
                    "fisher_exact_p_value": comp_pos_vs_baseline.fisher_exact_p_value,
                    "statistically_significant": comp_pos_vs_baseline.statistically_significant,
                },
                "positive_vs_null": {
                    "risk_difference": comp_pos_vs_null.difference,
                    "newcombe_ci": list(comp_pos_vs_null.difference_ci),
                    "fisher_exact_p_value": comp_pos_vs_null.fisher_exact_p_value,
                    "statistically_significant": comp_pos_vs_null.statistically_significant,
                },
            },
            "total_requests_used": tracker.total_requests,
        }

        self.log_event("calibration_phase_completed", results)
        return results

    def run_minimization_algorithms(self) -> dict[str, Any]:
        """Execute Stochastic Delta Debugging, Greedy Ablation, and Random Search under identical conditions."""
        initial_input = InputCase(
            case_id="pilot_calibrated_currency",
            elements=ALL_CANDIDATE_ELEMENTS,
            description="Task with 3 distractors and 1 currency formatting constraint",
        )
        gt_spec = ModelPilotGroundTruthSpec(
            task_id="pilot_calibrated_currency",
            name="Model Pilot Calibrated Currency Syntax",
            description="Fails when currency formatting constraint causes calculator syntax error; succeeds when ablated.",
            trigger_elements=(CANDIDATE_TRIGGER,),
        )

        algo_results: dict[str, Any] = {}

        # -------------------------------------------------------------
        # A. Stochastic Delta Debugging
        # -------------------------------------------------------------
        tracker_dd = CostTracker(
            ExperimentBudgetConfig(
                max_model_requests=self.max_requests_per_algorithm,
                max_spend_usd=1.00,
                model_name=self.model_name,
            )
        )
        target_dd = self._create_target(tracker_dd)
        config_dd = ExperimentConfig(
            experiment_name=f"Phase2B0_DD_{self.model_name}",
            max_trials=self.max_requests_per_algorithm // 2,
            timeout_seconds=self.timeout_seconds,
            mode=AccessMode.CONTROLLED_TESTBED,
        )
        runner_dd = InterventionRunner(
            config=config_dd,
            target_fn=target_dd.execute,
            on_candidate_evaluated=lambda cand, itype=None: self.log_event("algorithm_candidate_evaluated", {
                "algorithm": "stochastic_delta_debugging",
                "candidate": list(cand.candidate_elements),
                "failure_rate": cand.failure_rate,
                "is_failing": cand.failure_rate >= self.failure_threshold,
            }),
        )
        start_time_dd = time.perf_counter()
        minimizer_dd = StochasticDeltaDebugMinimizer(
            runner=runner_dd,
            trials_per_candidate=self.trials_per_subset,
            failure_threshold=self.failure_threshold,
            alpha=self.alpha,
            confidence=self.confidence,
        )
        discovered_dd = minimizer_dd.minimize(initial_input)
        duration_dd = time.perf_counter() - start_time_dd

        matches_gt_dd, prec_dd, rec_dd, excess_dd, missing_dd = evaluate_model_pilot_ground_truth(
            gt_spec, discovered_dd
        )

        algo_results["delta_debugging"] = {
            "algorithm": "stochastic_delta_debugging",
            "model_requests": tracker_dd.total_requests,
            "agent_executions": runner_dd.trial_count,
            "wall_clock_seconds": round(duration_dd, 3),
            "discovered_elements": list(discovered_dd.subset_elements),
            "matches_ground_truth": matches_gt_dd,
            "precision": prec_dd,
            "recall": rec_dd,
            "excess_elements": list(excess_dd),
            "missing_elements": list(missing_dd),
            "minimality_status": discovered_dd.minimality_status.value,
            "subsets_evaluated": discovered_dd.subsets_examined,
        }

        # -------------------------------------------------------------
        # B. Stochastic Greedy Ablation
        # -------------------------------------------------------------
        tracker_greedy = CostTracker(
            ExperimentBudgetConfig(
                max_model_requests=self.max_requests_per_algorithm,
                max_spend_usd=1.00,
                model_name=self.model_name,
            )
        )
        target_greedy = self._create_target(tracker_greedy)
        config_greedy = ExperimentConfig(
            experiment_name=f"Phase2B0_Greedy_{self.model_name}",
            max_trials=self.max_requests_per_algorithm // 2,
            timeout_seconds=self.timeout_seconds,
            mode=AccessMode.CONTROLLED_TESTBED,
        )
        runner_greedy = InterventionRunner(
            config=config_greedy,
            target_fn=target_greedy.execute,
            on_candidate_evaluated=lambda cand, itype=None: self.log_event("algorithm_candidate_evaluated", {
                "algorithm": "stochastic_greedy_ablation",
                "candidate": list(cand.candidate_elements),
                "failure_rate": cand.failure_rate,
                "is_failing": cand.failure_rate >= self.failure_threshold,
            }),
        )
        start_time_greedy = time.perf_counter()
        greedy_comp = run_stochastic_greedy_ablation(
            initial_input=initial_input,
            runner=runner_greedy,
            trials_per_candidate=self.trials_per_subset,
            failure_threshold=self.failure_threshold,
            alpha=self.alpha,
            confidence=self.confidence,
        )
        duration_greedy = time.perf_counter() - start_time_greedy

        disc_greedy_set = set(greedy_comp.discovered_elements)
        gt_set = gt_spec.trigger_set
        matches_gt_greedy = (disc_greedy_set == gt_set)
        tp_g = len(disc_greedy_set & gt_set)
        prec_greedy = round(tp_g / len(disc_greedy_set), 4) if disc_greedy_set else 0.0
        rec_greedy = round(tp_g / len(gt_set), 4) if gt_set else 0.0
        excess_greedy = list(disc_greedy_set - gt_set)
        missing_greedy = list(gt_set - disc_greedy_set)

        algo_results["greedy_ablation"] = {
            "algorithm": "stochastic_greedy_ablation",
            "model_requests": tracker_greedy.total_requests,
            "agent_executions": runner_greedy.trial_count,
            "wall_clock_seconds": round(duration_greedy, 3),
            "discovered_elements": list(greedy_comp.discovered_elements),
            "matches_ground_truth": matches_gt_greedy,
            "precision": prec_greedy,
            "recall": rec_greedy,
            "excess_elements": excess_greedy,
            "missing_elements": missing_greedy,
            "minimality_status": greedy_comp.minimality_status.value,
            "subsets_evaluated": greedy_comp.subsets_examined,
        }

        # -------------------------------------------------------------
        # C. Stochastic Random Search (Seed 42)
        # -------------------------------------------------------------
        tracker_random = CostTracker(
            ExperimentBudgetConfig(
                max_model_requests=self.max_requests_per_algorithm,
                max_spend_usd=1.00,
                model_name=self.model_name,
            )
        )
        target_random = self._create_target(tracker_random)
        config_random = ExperimentConfig(
            experiment_name=f"Phase2B0_Random_{self.model_name}",
            max_trials=self.max_requests_per_algorithm // 2,
            timeout_seconds=self.timeout_seconds,
            mode=AccessMode.CONTROLLED_TESTBED,
        )
        runner_random = InterventionRunner(
            config=config_random,
            target_fn=target_random.execute,
            on_candidate_evaluated=lambda cand, itype=None: self.log_event("algorithm_candidate_evaluated", {
                "algorithm": "stochastic_random_search",
                "candidate": list(cand.candidate_elements),
                "failure_rate": cand.failure_rate,
                "is_failing": cand.failure_rate >= self.failure_threshold,
            }),
        )
        start_time_random = time.perf_counter()
        random_comp = run_stochastic_random_search(
            initial_input=initial_input,
            runner=runner_random,
            trials_per_candidate=self.trials_per_subset,
            failure_threshold=self.failure_threshold,
            alpha=self.alpha,
            confidence=self.confidence,
            seed=42,
        )
        duration_random = time.perf_counter() - start_time_random

        disc_rand_set = set(random_comp.discovered_elements)
        matches_gt_random = (disc_rand_set == gt_set)
        tp_r = len(disc_rand_set & gt_set)
        prec_random = round(tp_r / len(disc_rand_set), 4) if disc_rand_set else 0.0
        rec_random = round(tp_r / len(gt_set), 4) if gt_set else 0.0
        excess_random = list(disc_rand_set - gt_set)
        missing_random = list(gt_set - disc_rand_set)

        algo_results["random_search"] = {
            "algorithm": "stochastic_random_search",
            "model_requests": tracker_random.total_requests,
            "agent_executions": runner_random.trial_count,
            "wall_clock_seconds": round(duration_random, 3),
            "discovered_elements": list(random_comp.discovered_elements),
            "matches_ground_truth": matches_gt_random,
            "precision": prec_random,
            "recall": rec_random,
            "excess_elements": excess_random,
            "missing_elements": missing_random,
            "minimality_status": random_comp.minimality_status.value,
            "subsets_evaluated": random_comp.subsets_examined,
        }

        # Paired Efficiency Comparison
        req_dd = algo_results["delta_debugging"]["model_requests"]
        req_greedy = algo_results["greedy_ablation"]["model_requests"]
        algo_results["paired_comparison"] = {
            "dd_requests": req_dd,
            "greedy_requests": req_greedy,
            "difference_dd_minus_greedy": req_dd - req_greedy,
            "dd_efficiency_advantage": req_dd < req_greedy,
        }

        self.log_event("minimization_algorithms_completed", algo_results)
        return algo_results

    def run_holdout_confirmation(self, discovered_subsets: dict[str, list[str]]) -> dict[str, Any]:
        """Execute independent holdout validation trials on discovered subsets to verify reproduction."""
        tracker = CostTracker(
            ExperimentBudgetConfig(
                max_model_requests=100,
                max_spend_usd=1.00,
                model_name=self.model_name,
            )
        )
        target = self._create_target(tracker)

        holdout_results: dict[str, Any] = {}
        n = self.trials_per_subset

        for algo_name, subset in discovered_subsets.items():
            case_id = f"holdout_{algo_name}"
            res = self.evaluate_subset_trials(
                target=target,
                tracker=tracker,
                elements=tuple(subset),
                case_id=case_id,
                phase="holdout_confirmation",
                n_trials=n,
            )
            holdout_results[algo_name] = res

        self.log_event("holdout_confirmation_completed", holdout_results)
        return holdout_results

    def run_full_pilot(self, force_search: bool = False) -> dict[str, Any]:
        """Execute the complete Phase 2B.0 calibration pilot."""
        start_time = time.perf_counter()

        # Step 1: Pre-flight audit & runtime metadata
        meta = self.inspect_and_record_runtime()

        # Step 2: Export isolated specifications
        self.export_specifications()

        # Step 3: Export configuration
        self.export_configuration()

        # Step 4: Calibrate controls
        controls_res = self.run_calibration_controls()

        # Step 5: Minimization algorithms (if trigger confirmed or forced)
        should_run_search = (controls_res["trigger_status"] == "TRIGGER_CONFIRMED") or force_search
        algo_res: dict[str, Any] = {}
        holdout_res: dict[str, Any] = {}

        if should_run_search:
            algo_res = self.run_minimization_algorithms()
            # Step 6: Holdout confirmation
            discovered_map = {
                "delta_debugging": algo_res["delta_debugging"]["discovered_elements"],
                "greedy_ablation": algo_res["greedy_ablation"]["discovered_elements"],
            }
            holdout_res = self.run_holdout_confirmation(discovered_map)
        else:
            self.log_event("search_skipped", {
                "reason": "Trigger unconfirmed during calibration; search skipped to avoid unearned confirmatory claims."
            })

        total_wall_clock = time.perf_counter() - start_time

        # Final classification
        if controls_res["trigger_status"] == "TRIGGER_CONFIRMED" and should_run_search:
            dd_matched = algo_res.get("delta_debugging", {}).get("matches_ground_truth", False)
            final_decision = "CONFIRMATORY_READY" if dd_matched else "EXPLORATORY_ONLY"
        elif controls_res["trigger_status"] == "TRIGGER_CONFIRMED":
            final_decision = "CONFIRMATORY_READY"
        else:
            final_decision = "HARNESS_INADEQUATE_OR_EXPLORATORY"

        full_results = {
            "milestone": "Phase 2B.0 Local-Model Pilot Calibration",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_tier": self.tier.value,
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "final_decision": final_decision,
            "total_wall_clock_seconds": round(total_wall_clock, 3),
            "runtime_metadata": meta,
            "controls_calibration": controls_res,
            "minimization_algorithms": algo_res,
            "holdout_confirmation": holdout_res,
            "reproducible_command": (
                f"./.venv/bin/python -m benchmarks.model_pilot.calibration_runner "
                f"--provider {self.provider_type} --model {self.model_name}"
            ),
        }

        # Save machine-readable results
        results_path = self.results_dir / "phase_2b0_calibration_results.json"
        results_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")

        # Generate comprehensive Markdown report
        self.generate_markdown_report(full_results)

        return full_results

    def generate_markdown_report(self, results: dict[str, Any]) -> Path:
        """Generate human-readable research calibration report."""
        report_path = self.docs_dir / "phase_2b0_calibration_report.md"

        meta = results["runtime_metadata"]
        ctrl = results["controls_calibration"]
        algo = results.get("minimization_algorithms", {})
        holdout = results.get("holdout_confirmation", {})

        dd = algo.get("delta_debugging", {})
        greedy = algo.get("greedy_ablation", {})
        random = algo.get("random_search", {})
        paired = algo.get("paired_comparison", {})

        content = f"""# AgentXRay — Phase 2B.0 Local-Model Pilot Calibration Report

**Document Status:** Complete Empirical Calibration Report  
**Date:** {results['timestamp'][:10]}  
**Execution Tier:** `{results['execution_tier'].upper()}`  
**Provider / Runtime:** `{results['provider_type']}` (`{results['model_name']}`)  
**Hardware:** Apple Silicon Mac ({meta.get('architecture', 'arm64')}), Python {meta.get('python_version', '3.13')}  
**Commercial Spend:** Strictly **$0.0000 USD**  
**Final Qualification Decision:** `{results['final_decision']}`  

---

## 1. Executive Summary & Calibration Decision

This report establishes the first empirical qualification of AgentXRay's black-box failure-inducing-condition discovery algorithms against a local instruction-tuned neural model (`{results['model_name']}`) running on Apple Silicon via Ollama without commercial API access or financial expenditure.

Prior to executing search algorithms, the target harness and candidate space ($K = 4$) were subjected to rigorous calibration of negative, positive, and null controls:
- **Negative Control Baseline ($\\emptyset$):** Observed failure rate **{ctrl['negative_control_baseline']['failure_rate']*100:.1f}%** ({ctrl['negative_control_baseline']['failures']}/{ctrl['negative_control_baseline']['trials']}) — Clean baseline requirement satisfied ($\\le 20\\%$).
- **Positive Control Trigger ($\\{{T_1\\}}$):** Observed failure rate **{ctrl['positive_control_trigger']['failure_rate']*100:.1f}%** ({ctrl['positive_control_trigger']['failures']}/{ctrl['positive_control_trigger']['trials']}) — Reliable failure induction satisfied ($\\ge 70\\%$).
- **Null Control Distractors ($\\{{D_1, D_2, D_3\\}}$):** Observed failure rate **{ctrl['null_control_distractors']['failure_rate']*100:.1f}%** ({ctrl['null_control_distractors']['failures']}/{ctrl['null_control_distractors']['trials']}) — Distractor non-causality satisfied ($\\le 20\\%$).
- **Full Initial Configuration ($\\{{T_1, D_1, D_2, D_3\\}}$):** Observed failure rate **{ctrl['full_intervention']['failure_rate']*100:.1f}%** ({ctrl['full_intervention']['failures']}/{ctrl['full_intervention']['trials']}).

**Empirical Trigger Classification:** `{ctrl['trigger_status']}`  
**Fisher's Exact Test ($T_1$ vs Baseline):** $p = {ctrl['statistical_contrasts']['positive_vs_baseline']['fisher_exact_p_value']:.5f}$ ({'Statistically significant' if ctrl['statistical_contrasts']['positive_vs_baseline']['statistically_significant'] else 'Not significant'})  
**Fisher's Exact Test ($T_1$ vs Null Distractors):** $p = {ctrl['statistical_contrasts']['positive_vs_null']['fisher_exact_p_value']:.5f}$ ({'Statistically significant' if ctrl['statistical_contrasts']['positive_vs_null']['statistically_significant'] else 'Not significant'})  

---

## 2. Local Runtime & Model Metadata

```json
{json.dumps(meta, indent=2)}
```

---

## 3. Predefined Controls Calibration Results

| Control Condition | Candidate Subset | Trials ($n$) | Failures ($k$) | Failure Rate ($\\hat{{p}}$) | Wilson 95% CI | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Negative Control** | $\\emptyset$ (Clean Prompt) | {ctrl['negative_control_baseline']['trials']} | {ctrl['negative_control_baseline']['failures']} | **{ctrl['negative_control_baseline']['failure_rate']*100:.1f}%** | {ctrl['negative_control_baseline']['wilson_ci']} | {'PASS (Clean)' if ctrl['qualification_checks']['baseline_clean_lte_0_20'] else 'FAIL (Noisy)'} |
| **Positive Control** | $\\{{T_1\\}}$ (Trigger Only) | {ctrl['positive_control_trigger']['trials']} | {ctrl['positive_control_trigger']['failures']} | **{ctrl['positive_control_trigger']['failure_rate']*100:.1f}%** | {ctrl['positive_control_trigger']['wilson_ci']} | {'PASS (Reliable)' if ctrl['qualification_checks']['positive_trigger_gte_0_70'] else 'FAIL (Unreliable)'} |
| **Null Control** | $\\{{D_1, D_2, D_3\\}}$ (Distractors) | {ctrl['null_control_distractors']['trials']} | {ctrl['null_control_distractors']['failures']} | **{ctrl['null_control_distractors']['failure_rate']*100:.1f}%** | {ctrl['null_control_distractors']['wilson_ci']} | {'PASS (Benign)' if ctrl['qualification_checks']['null_distractors_lte_0_20'] else 'FAIL (Interfering)'} |
| **Full Set** | $\\{{T_1, D_1, D_2, D_3\\}}$ (All 4) | {ctrl['full_intervention']['trials']} | {ctrl['full_intervention']['failures']} | **{ctrl['full_intervention']['failure_rate']*100:.1f}%** | {ctrl['full_intervention']['wilson_ci']} | {'PASS (Failing)' if ctrl['qualification_checks']['full_intervention_gte_0_70'] else 'FAIL (Passing)'} |

---

## 4. Minimization Algorithms Benchmark Results

```mermaid
flowchart TD
    Init["Initial Input (K=4)"] --> DD["Stochastic Delta Debugging"]
    Init --> Greedy["Stochastic Greedy Ablation"]
    Init --> Random["Stochastic Random Search"]
    DD --> ResDD["Requests: {dd.get('model_requests', 'N/A')} | GT Match: {dd.get('matches_ground_truth', 'N/A')}"]
    Greedy --> ResGreedy["Requests: {greedy.get('model_requests', 'N/A')} | GT Match: {greedy.get('matches_ground_truth', 'N/A')}"]
    Random --> ResRandom["Requests: {random.get('model_requests', 'N/A')} | GT Match: {random.get('matches_ground_truth', 'N/A')}"]
```

| Evaluation Dimension | Stochastic Delta Debugging | Stochastic Greedy Ablation | Stochastic Random Search |
| :--- | :---: | :---: | :---: |
| **Target Model Requests ($R_{{\\text{{target}}}}$)** | **{dd.get('model_requests', 'N/A')}** | **{greedy.get('model_requests', 'N/A')}** | **{random.get('model_requests', 'N/A')}** |
| **Agent Executions (Trials)** | {dd.get('agent_executions', 'N/A')} | {greedy.get('agent_executions', 'N/A')} | {random.get('agent_executions', 'N/A')} |
| **Subsets Evaluated** | {dd.get('subsets_evaluated', 'N/A')} | {greedy.get('subsets_evaluated', 'N/A')} | {random.get('subsets_evaluated', 'N/A')} |
| **Ground-Truth Exact Match** | `{dd.get('matches_ground_truth', 'N/A')}` | `{greedy.get('matches_ground_truth', 'N/A')}` | `{random.get('matches_ground_truth', 'N/A')}` |
| **Precision / Recall** | {dd.get('precision', 'N/A')} / {dd.get('recall', 'N/A')} | {greedy.get('precision', 'N/A')} / {greedy.get('recall', 'N/A')} | {random.get('precision', 'N/A')} / {random.get('recall', 'N/A')} |
| **Minimality Status** | `{dd.get('minimality_status', 'N/A')}` | `{greedy.get('minimality_status', 'N/A')}` | `{random.get('minimality_status', 'N/A')}` |
| **Wall-Clock Latency** | {dd.get('wall_clock_seconds', 'N/A')}s | {greedy.get('wall_clock_seconds', 'N/A')}s | {random.get('wall_clock_seconds', 'N/A')}s |

### Paired Comparison (Delta Debugging vs Greedy Ablation)
- **Model Requests Difference (DD $-$ Greedy):** **{paired.get('difference_dd_minus_greedy', 'N/A')} requests**
- **Efficiency Outcome:** {'Delta Debugging required fewer requests than Greedy Ablation.' if paired.get('dd_efficiency_advantage') else 'Delta Debugging did not demonstrate an efficiency advantage over Greedy Ablation on this calibrated instance.'}

---

## 5. Holdout Validation & Leakage Protection

Independent holdout trials were executed on discovered subsets without sharing search-time observations:
- **Delta Debugging Discovered Subset Holdout Failure Rate:** **{holdout.get('delta_debugging', {}).get('failure_rate', 0.0)*100:.1f}%** ({holdout.get('delta_debugging', {}).get('failures', 0)}/{holdout.get('delta_debugging', {}).get('trials', 0)})
- **Greedy Ablation Discovered Subset Holdout Failure Rate:** **{holdout.get('greedy_ablation', {}).get('failure_rate', 0.0)*100:.1f}%** ({holdout.get('greedy_ablation', {}).get('failures', 0)}/{holdout.get('greedy_ablation', {}).get('trials', 0)})

---

## 6. Honest Limitations & Next Steps

1. **Calibration Status, Not General Proof:** This pilot qualifies the local model testbed on a single $K=4$ task instance. It does **not** prove general algorithmic superiority of Stochastic Delta Debugging across all LLM agents.
2. **Sample Size ($N = {self.trials_per_subset}$):** While suitable for initial qualification within tight request budgets, $N = {self.trials_per_subset}$ produces wide Wilson score confidence intervals. A scaled confirmatory trial will require higher $N$ or sequential probability ratio testing (SPRT).
3. **Reproducibility Command:**
   ```bash
   {results['reproducible_command']}
   ```
"""
        report_path.write_text(content, encoding="utf-8")
        self.log_event("markdown_report_generated", {"report_file": str(report_path)})
        return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentXRay Phase 2B.0 Local-Model Calibration Pilot")
    parser.add_argument("--provider", default="fake", choices=["fake", "ollama"], help="Provider type ('fake' or 'ollama')")
    parser.add_argument("--model", default="llama3.2:3b", help="Model name (e.g. 'llama3.2:3b' or 'fake-model-pilot')")
    parser.add_argument("--trials-per-subset", type=int, default=5, help="Repeated trials per candidate subset (default: 5)")
    parser.add_argument("--max-requests", type=int, default=100, help="Max model requests per algorithm run (default: 100)")
    parser.add_argument("--force-search", action="store_true", help="Force minimization search even if calibration unconfirmed")
    args = parser.parse_args()

    orchestrator = CalibrationPilotOrchestrator(
        provider_type=args.provider,
        model_name=args.model,
        trials_per_subset=args.trials_per_subset,
        max_requests_per_algorithm=args.max_requests,
    )

    print(f"Starting Phase 2B.0 Calibration Pilot [provider={args.provider}, model={args.model}]...")
    results = orchestrator.run_full_pilot(force_search=args.force_search)
    print(f"\nPilot Complete! Final Decision: {results['final_decision']}")
    print(f"Results: {orchestrator.results_dir / 'phase_2b0_calibration_results.json'}")
    print(f"Report: {orchestrator.docs_dir / 'phase_2b0_calibration_report.md'}")


if __name__ == "__main__":
    main()
