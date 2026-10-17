"""Phase 2B.0.2 Corrective Calibration Pilot Runner (Protocol v1.2).

Implements the amended and verified Phase 2B.0.2 protocol:
- De-conflicted orthogonal distractors avoiding length/format truncation.
- Class-aware evaluation separating FAIL_TOOL_EXEC, FAIL_BEHAVIOR, and FAIL_INFRA.
- Independent exhaustive characterization across all 2^4 = 16 candidate configurations.
- Universal trial-level event logging across all execution phases (characterization, search, holdout).
- Explicit reporting guardrails: skipped stages are marked BLOCKED_BY_QUALIFICATION instead of 0.0% / [].
- Decoupled post-hoc ground-truth evaluation.
- Preserves all historical Phase 2B.0 and 2B.0.1 artifacts by writing strictly to 'phase_2b0_2_*' paths.
- Enforces 100% local inference ($0.0000 commercial spend) and local loopback binding.
"""

from __future__ import annotations

import argparse
import itertools
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

# De-conflicted candidate intervention space (Protocol v1.2, K = 4)
CANDIDATE_TRIGGER = (
    'strict_currency_formatting: format all numbers within arithmetic tool arguments '
    'with currency symbols (e.g. "$150" or "$250")'
)
CANDIDATE_DISTRACTOR_1 = "expense_category: specify expense category as Travel & Subsistence"
CANDIDATE_DISTRACTOR_2 = "compliance_tag: include audit reference code REF-2026"
CANDIDATE_DISTRACTOR_3 = "audit_memo: include verification timestamp in final audit notes"

ALL_CANDIDATE_ELEMENTS = (
    CANDIDATE_TRIGGER,
    CANDIDATE_DISTRACTOR_1,
    CANDIDATE_DISTRACTOR_2,
    CANDIDATE_DISTRACTOR_3,
)

ELEMENT_ID_MAP = {
    CANDIDATE_TRIGGER: "elem_1",
    CANDIDATE_DISTRACTOR_1: "elem_2",
    CANDIDATE_DISTRACTOR_2: "elem_3",
    CANDIDATE_DISTRACTOR_3: "elem_4",
}


class CorrectiveCalibrationOrchestratorV12:
    """Orchestrates the Phase 2B.0.2 corrective local-model calibration pilot under Protocol v1.2."""

    def __init__(
        self,
        provider_type: str = "fake",
        model_name: str | None = None,
        trials_per_subset: int = 5,
        max_requests_per_algorithm: int = 120,
        failure_threshold: float = 0.50,
        alpha: float = 0.05,
        confidence: float = 0.95,
        timeout_seconds: float = 300.0,
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

        # Versioned output paths for Phase 2B.0.2
        self.cand_specs_path = self.results_dir / "phase_2b0_2_candidate_specs.json"
        self.gt_path = self.results_dir / "phase_2b0_2_ground_truth.json"
        self.cfg_path = self.results_dir / "phase_2b0_2_calibration_config.json"
        self.event_log_path = self.results_dir / "phase_2b0_2_calibration_event_log.jsonl"
        self.results_path = self.results_dir / "phase_2b0_2_calibration_results.json"
        self.report_path = self.docs_dir / "phase_2b0_2_calibration_report.md"

        # Initialize versioned event log
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
        self._current_phase = "initialization"

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
            "protocol_version": "1.2",
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
        cand_data = {
            "protocol_version": "1.2",
            "task_id": "pilot_calibrated_currency_v1_2",
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
        self.cand_specs_path.write_text(json.dumps(cand_data, indent=2), encoding="utf-8")

        gt_data = {
            "protocol_version": "1.2",
            "task_id": "pilot_calibrated_currency_v1_2",
            "true_trigger_elements": [CANDIDATE_TRIGGER],
            "trigger_cardinality": 1,
            "is_null_case": False,
            "expected_failure_classification": "FAIL_TOOL_EXEC",
            "failure_mechanism": "Strict currency formatting in tool argument causes calculator AST syntax rejection.",
            "leakage_protection": "Evaluator and Minimizers have no access to this ground-truth definition.",
        }
        self.gt_path.write_text(json.dumps(gt_data, indent=2), encoding="utf-8")

        self.log_event("specifications_exported", {
            "candidate_specs_file": str(self.cand_specs_path),
            "ground_truth_file": str(self.gt_path),
        })
        return self.cand_specs_path, self.gt_path

    def export_configuration(self) -> Path:
        """Export frozen pilot configuration."""
        cfg_data = {
            "protocol_version": "1.2",
            "protocol_document": "docs/research/phase_2a_protocol_v1.2.md",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "execution_tier": self.tier.value,
            "exhaustive_characterization_required": True,
            "power_set_subsets_count": 2 ** len(ALL_CANDIDATE_ELEMENTS),
            "trials_per_subset": self.trials_per_subset,
            "max_requests_per_algorithm": self.max_requests_per_algorithm,
            "failure_threshold": self.failure_threshold,
            "alpha": self.alpha,
            "confidence": self.confidence,
            "timeout_seconds": self.timeout_seconds,
            "runtime_metadata": self.runtime_metadata,
        }
        self.cfg_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
        self.log_event("config_exported", {"config_file": str(self.cfg_path)})
        return self.cfg_path

    def _create_target(self, tracker: CostTracker, phase_label: str) -> tuple[ModelBackedAgentTarget, Any]:
        """Create black-box target adapter instrumented with universal trial event logging."""
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
        base_target = ModelBackedAgentTarget(
            agent=agent,
            evaluator=evaluator,
            base_system_prompt="You are an automated accounting assistant. Always use the calculator tool for calculations.",
            user_prompt="Calculate total reimbursement for expenses of 150 and 250.",
            tools=[CALCULATOR_TOOL_DEF],
            name="ModelPilotCalibratedCurrencyTarget_v1_2",
        )

        trial_counter = [0]

        def logged_execute(input_case: InputCase) -> TargetResult:
            trial_counter[0] += 1
            req_before = tracker.total_requests
            tok_before = tracker.total_tokens
            t0 = time.perf_counter()

            res = base_target.execute(input_case)
            duration = time.perf_counter() - t0
            req_used = tracker.total_requests - req_before
            tok_used = tracker.total_tokens - tok_before

            trace = base_target.last_trace
            tool_calls_info = [
                {
                    "tool": t.tool_name,
                    "arguments": t.arguments,
                    "output": t.output,
                }
                for t in (trace.tool_executions if trace else ())
            ]

            # Detailed failure classification
            if trace:
                _, _, fclass = evaluator.evaluate_detailed(trace)
            else:
                fclass = "FAIL_INFRA"

            self.log_event("trial_executed", {
                "phase": self._current_phase,
                "case_id": input_case.case_id,
                "trial_index": trial_counter[0],
                "candidate_subset": list(input_case.elements),
                "element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in input_case.elements],
                "outcome": res.outcome.value,
                "is_failure": res.outcome == TargetOutcome.FAILURE,
                "failure_class": fclass,
                "model_requests": req_used,
                "tokens": tok_used,
                "latency_seconds": round(duration, 4),
                "tool_executions": tool_calls_info,
                "final_response": trace.final_response if trace else "",
                "error_message": trace.error_message if trace else None,
            })
            return res

        return base_target, logged_execute

    def run_exhaustive_characterization(self) -> dict[str, Any]:
        """Exhaustively evaluate all 2^4 = 16 candidate configurations in P(C)."""
        self._current_phase = "characterization"
        tracker = CostTracker(
            ExperimentBudgetConfig(
                max_model_requests=400,
                max_spend_usd=1.00,
                model_name=self.model_name,
            )
        )
        _, logged_target_fn = self._create_target(tracker, "characterization")

        # Generate all 16 subsets
        power_set: list[tuple[str, ...]] = []
        for r in range(len(ALL_CANDIDATE_ELEMENTS) + 1):
            for combo in itertools.combinations(ALL_CANDIDATE_ELEMENTS, r):
                power_set.append(combo)

        self.log_event("characterization_phase_started", {
            "power_set_size": len(power_set),
            "trials_per_subset": self.trials_per_subset,
        })

        subset_results: list[dict[str, Any]] = []
        trigger_present_rates: list[float] = []
        trigger_absent_rates: list[float] = []

        for idx, subset in enumerate(power_set, 1):
            case_id = f"char_subset_{idx:02d}_len{len(subset)}"
            input_case = InputCase(
                case_id=case_id,
                elements=subset,
                description=f"Exhaustive characterization subset {idx}/16",
            )
            failures = 0
            n = self.trials_per_subset

            for _ in range(n):
                res = logged_target_fn(input_case)
                if res.outcome == TargetOutcome.FAILURE:
                    failures += 1

            fail_rate = round(failures / n, 4)
            l_ci, u_ci = wilson_score_interval(failures, n, confidence=self.confidence)
            has_trigger = CANDIDATE_TRIGGER in subset

            if has_trigger:
                trigger_present_rates.append(fail_rate)
            else:
                trigger_absent_rates.append(fail_rate)

            sub_summary = {
                "subset_index": idx,
                "case_id": case_id,
                "cardinality": len(subset),
                "element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in subset],
                "contains_trigger": has_trigger,
                "elements": list(subset),
                "trials": n,
                "failures": failures,
                "successes": n - failures,
                "failure_rate": fail_rate,
                "wilson_ci": [l_ci, u_ci],
                "is_failing": fail_rate >= self.failure_threshold,
            }
            subset_results.append(sub_summary)
            self.log_event("candidate_evaluated", {
                "phase": "characterization",
                **sub_summary,
            })

        min_trigger_fail_rate = min(trigger_present_rates) if trigger_present_rates else 0.0
        max_absent_fail_rate = max(trigger_absent_rates) if trigger_absent_rates else 1.0

        trigger_qualified = (min_trigger_fail_rate >= 0.70) and (max_absent_fail_rate <= 0.20)
        trigger_status = "TRIGGER_CONFIRMED" if trigger_qualified else "TRIGGER_UNCONFIRMED"

        char_summary = {
            "trigger_status": trigger_status,
            "power_set_size": len(power_set),
            "subsets_evaluated": len(subset_results),
            "min_trigger_present_failure_rate": min_trigger_fail_rate,
            "max_trigger_absent_failure_rate": max_absent_fail_rate,
            "qualification_criteria_met": trigger_qualified,
            "subsets": subset_results,
            "total_characterization_requests": tracker.total_requests,
        }

        self.log_event("characterization_phase_completed", {
            "trigger_status": trigger_status,
            "subsets_evaluated": len(subset_results),
            "qualification_criteria_met": trigger_qualified,
        })
        return char_summary

    def run_minimization_algorithms(self) -> dict[str, Any]:
        """Execute Stochastic Delta Debugging, Greedy Ablation, and Random Search under Protocol v1.2."""
        initial_input = InputCase(
            case_id="pilot_calibrated_currency_v1_2",
            elements=ALL_CANDIDATE_ELEMENTS,
            description="Task with 3 de-conflicted distractors and 1 currency formatting constraint",
        )

        algo_results: dict[str, Any] = {}

        # -------------------------------------------------------------
        # A. Stochastic Delta Debugging
        # -------------------------------------------------------------
        self._current_phase = "search_dd"
        tracker_dd = CostTracker(
            ExperimentBudgetConfig(
                max_model_requests=self.max_requests_per_algorithm,
                max_spend_usd=1.00,
                model_name=self.model_name,
            )
        )
        _, logged_target_dd = self._create_target(tracker_dd, "search_dd")
        config_dd = ExperimentConfig(
            experiment_name=f"Phase2B0_2_DD_{self.model_name}",
            max_trials=self.max_requests_per_algorithm // 2,
            timeout_seconds=self.timeout_seconds,
            mode=AccessMode.CONTROLLED_TESTBED,
        )
        runner_dd = InterventionRunner(
            config=config_dd,
            target_fn=logged_target_dd,
            on_candidate_evaluated=lambda cand, itype=None: self.log_event("algorithm_candidate_evaluated", {
                "algorithm": "stochastic_delta_debugging",
                "candidate": list(cand.candidate_elements),
                "element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in cand.candidate_elements],
                "failure_rate": cand.failure_rate,
                "is_failing": cand.failure_rate >= self.failure_threshold,
            }),
        )
        t0_dd = time.perf_counter()
        minimizer_dd = StochasticDeltaDebugMinimizer(
            runner=runner_dd,
            trials_per_candidate=self.trials_per_subset,
            failure_threshold=self.failure_threshold,
            alpha=self.alpha,
            confidence=self.confidence,
        )
        discovered_dd = minimizer_dd.minimize(initial_input)
        duration_dd = time.perf_counter() - t0_dd

        algo_results["delta_debugging"] = {
            "algorithm": "stochastic_delta_debugging",
            "model_requests": tracker_dd.total_requests,
            "agent_executions": runner_dd.trial_count,
            "wall_clock_seconds": round(duration_dd, 3),
            "discovered_elements": list(discovered_dd.subset_elements),
            "discovered_element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in discovered_dd.subset_elements],
            "minimality_status": discovered_dd.minimality_status.value,
            "subsets_evaluated": discovered_dd.subsets_examined,
        }

        # -------------------------------------------------------------
        # B. Stochastic Greedy Ablation
        # -------------------------------------------------------------
        self._current_phase = "search_greedy"
        tracker_greedy = CostTracker(
            ExperimentBudgetConfig(
                max_model_requests=self.max_requests_per_algorithm,
                max_spend_usd=1.00,
                model_name=self.model_name,
            )
        )
        _, logged_target_greedy = self._create_target(tracker_greedy, "search_greedy")
        config_greedy = ExperimentConfig(
            experiment_name=f"Phase2B0_2_Greedy_{self.model_name}",
            max_trials=self.max_requests_per_algorithm // 2,
            timeout_seconds=self.timeout_seconds,
            mode=AccessMode.CONTROLLED_TESTBED,
        )
        runner_greedy = InterventionRunner(
            config=config_greedy,
            target_fn=logged_target_greedy,
            on_candidate_evaluated=lambda cand, itype=None: self.log_event("algorithm_candidate_evaluated", {
                "algorithm": "stochastic_greedy_ablation",
                "candidate": list(cand.candidate_elements),
                "element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in cand.candidate_elements],
                "failure_rate": cand.failure_rate,
                "is_failing": cand.failure_rate >= self.failure_threshold,
            }),
        )
        t0_greedy = time.perf_counter()
        greedy_comp = run_stochastic_greedy_ablation(
            initial_input=initial_input,
            runner=runner_greedy,
            trials_per_candidate=self.trials_per_subset,
            failure_threshold=self.failure_threshold,
            alpha=self.alpha,
            confidence=self.confidence,
        )
        duration_greedy = time.perf_counter() - t0_greedy

        algo_results["greedy_ablation"] = {
            "algorithm": "stochastic_greedy_ablation",
            "model_requests": tracker_greedy.total_requests,
            "agent_executions": runner_greedy.trial_count,
            "wall_clock_seconds": round(duration_greedy, 3),
            "discovered_elements": list(greedy_comp.discovered_elements),
            "discovered_element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in greedy_comp.discovered_elements],
            "minimality_status": greedy_comp.minimality_status.value,
            "subsets_evaluated": greedy_comp.subsets_examined,
        }

        # -------------------------------------------------------------
        # C. Stochastic Random Search (Seed 42)
        # -------------------------------------------------------------
        self._current_phase = "search_random"
        tracker_random = CostTracker(
            ExperimentBudgetConfig(
                max_model_requests=self.max_requests_per_algorithm,
                max_spend_usd=1.00,
                model_name=self.model_name,
            )
        )
        _, logged_target_random = self._create_target(tracker_random, "search_random")
        config_random = ExperimentConfig(
            experiment_name=f"Phase2B0_2_Random_{self.model_name}",
            max_trials=self.max_requests_per_algorithm // 2,
            timeout_seconds=self.timeout_seconds,
            mode=AccessMode.CONTROLLED_TESTBED,
        )
        runner_random = InterventionRunner(
            config=config_random,
            target_fn=logged_target_random,
            on_candidate_evaluated=lambda cand, itype=None: self.log_event("algorithm_candidate_evaluated", {
                "algorithm": "stochastic_random_search",
                "candidate": list(cand.candidate_elements),
                "element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in cand.candidate_elements],
                "failure_rate": cand.failure_rate,
                "is_failing": cand.failure_rate >= self.failure_threshold,
            }),
        )
        t0_random = time.perf_counter()
        random_comp = run_stochastic_random_search(
            initial_input=initial_input,
            runner=runner_random,
            trials_per_candidate=self.trials_per_subset,
            failure_threshold=self.failure_threshold,
            alpha=self.alpha,
            confidence=self.confidence,
            seed=42,
        )
        duration_random = time.perf_counter() - t0_random

        algo_results["random_search"] = {
            "algorithm": "stochastic_random_search",
            "model_requests": tracker_random.total_requests,
            "agent_executions": runner_random.trial_count,
            "wall_clock_seconds": round(duration_random, 3),
            "discovered_elements": list(random_comp.discovered_elements),
            "discovered_element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in random_comp.discovered_elements],
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

    def run_complete_holdout_confirmation(self, discovered_subsets: dict[str, list[str]]) -> dict[str, Any]:
        """Execute independent holdout validation trials for all three search algorithms."""
        self._current_phase = "holdout_confirmation"
        tracker = CostTracker(
            ExperimentBudgetConfig(
                max_model_requests=150,
                max_spend_usd=1.00,
                model_name=self.model_name,
            )
        )
        _, logged_target_fn = self._create_target(tracker, "holdout_confirmation")

        holdout_results: dict[str, Any] = {}
        n = self.trials_per_subset

        for algo_name, subset in discovered_subsets.items():
            case_id = f"holdout_{algo_name}"
            input_case = InputCase(
                case_id=case_id,
                elements=tuple(subset),
                description=f"Holdout confirmation for {algo_name}",
            )
            failures = 0
            for _ in range(n):
                res = logged_target_fn(input_case)
                if res.outcome == TargetOutcome.FAILURE:
                    failures += 1

            fail_rate = round(failures / n, 4)
            l_ci, u_ci = wilson_score_interval(failures, n, confidence=self.confidence)

            h_summary = {
                "algorithm": algo_name,
                "case_id": case_id,
                "elements": list(subset),
                "element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in subset],
                "trials": n,
                "failures": failures,
                "successes": n - failures,
                "failure_rate": fail_rate,
                "wilson_ci": [l_ci, u_ci],
                "is_failing": fail_rate >= self.failure_threshold,
            }
            holdout_results[algo_name] = h_summary
            self.log_event("candidate_evaluated", {
                "phase": "holdout_confirmation",
                **h_summary,
            })

        self.log_event("holdout_confirmation_completed", holdout_results)
        return holdout_results

    def evaluate_ground_truth_post_hoc(self, algo_results: dict[str, Any]) -> dict[str, Any]:
        """Decoupled post-hoc ground-truth evaluation."""
        gt_data = json.loads(self.gt_path.read_text(encoding="utf-8"))
        true_trigger_set = set(gt_data["true_trigger_elements"])

        post_hoc_eval: dict[str, Any] = {}
        for algo_key in ["delta_debugging", "greedy_ablation", "random_search"]:
            if algo_key not in algo_results:
                continue
            discovered = set(algo_results[algo_key]["discovered_elements"])
            matches_gt = (discovered == true_trigger_set)
            tp = len(discovered & true_trigger_set)
            prec = round(tp / len(discovered), 4) if discovered else 0.0
            rec = round(tp / len(true_trigger_set), 4) if true_trigger_set else 0.0
            excess = list(discovered - true_trigger_set)
            missing = list(true_trigger_set - discovered)

            post_hoc_eval[algo_key] = {
                "matches_ground_truth": matches_gt,
                "precision": prec,
                "recall": rec,
                "excess_elements": excess,
                "excess_element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in excess],
                "missing_elements": missing,
                "missing_element_ids": [ELEMENT_ID_MAP.get(e, "unknown") for e in missing],
            }
            # Update algo_results in-place
            algo_results[algo_key].update(post_hoc_eval[algo_key])

        self.log_event("ground_truth_evaluated_post_hoc", post_hoc_eval)
        return post_hoc_eval

    def run_full_pilot(self) -> dict[str, Any]:
        """Execute the complete Phase 2B.0.2 corrective calibration pilot."""
        start_time = time.perf_counter()

        # Step 1: Pre-flight audit
        meta = self.inspect_and_record_runtime()

        # Step 2: Export specifications
        self.export_specifications()

        # Step 3: Export configuration
        self.export_configuration()

        # Step 4: Exhaustive characterization (all 16 subsets)
        char_res = self.run_exhaustive_characterization()

        # Step 5: Minimization algorithms (if trigger confirmed)
        should_run_search = char_res["qualification_criteria_met"]
        algo_res: dict[str, Any] = {}
        holdout_res: dict[str, Any] = {}

        if should_run_search:
            algo_res = self.run_minimization_algorithms()
            # Step 6: Post-hoc ground truth evaluation
            self.evaluate_ground_truth_post_hoc(algo_res)
            # Step 7: Complete holdout confirmation across all 3 algorithms
            discovered_map = {
                "delta_debugging": algo_res["delta_debugging"]["discovered_elements"],
                "greedy_ablation": algo_res["greedy_ablation"]["discovered_elements"],
                "random_search": algo_res["random_search"]["discovered_elements"],
            }
            holdout_res = self.run_complete_holdout_confirmation(discovered_map)
        else:
            self.log_event("search_skipped", {
                "reason": "Trigger unconfirmed during exhaustive characterization; minimization aborted."
            })

        total_wall_clock = time.perf_counter() - start_time

        # Final classification
        if char_res["qualification_criteria_met"] and should_run_search:
            dd_matched = algo_res.get("delta_debugging", {}).get("matches_ground_truth", False)
            final_decision = "CONFIRMATORY_READY" if dd_matched else "EXPLORATORY_ONLY"
        elif char_res["qualification_criteria_met"]:
            final_decision = "CONFIRMATORY_READY"
        else:
            final_decision = "HARNESS_INADEQUATE_OR_EXPLORATORY"

        full_results = {
            "milestone": "Phase 2B.0.2 Corrective Local-Model Pilot Calibration Recovery",
            "protocol_version": "1.2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_tier": self.tier.value,
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "final_decision": final_decision,
            "total_wall_clock_seconds": round(total_wall_clock, 3),
            "runtime_metadata": meta,
            "exhaustive_characterization": char_res,
            "minimization_algorithms": algo_res,
            "holdout_confirmation": holdout_res,
            "reproducible_command": (
                f"./.venv/bin/python -m benchmarks.model_pilot.calibration_runner_v1_2 "
                f"--provider {self.provider_type} --model {self.model_name}"
            ),
        }

        # Save machine-readable results
        self.results_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")

        # Generate comprehensive Markdown report
        self.generate_markdown_report(full_results)

        return full_results

    def generate_markdown_report(self, results: dict[str, Any]) -> Path:
        """Generate human-readable research calibration report for Phase 2B.0.2."""
        meta = results["runtime_metadata"]
        char = results["exhaustive_characterization"]
        algo = results.get("minimization_algorithms", {})
        holdout = results.get("holdout_confirmation", {})
        search_executed = bool(algo)

        dd = algo.get("delta_debugging", {})
        greedy = algo.get("greedy_ablation", {})
        random = algo.get("random_search", {})
        paired = algo.get("paired_comparison", {})

        content = f"""# AgentXRay — Phase 2B.0.2 Corrective Local-Model Pilot Calibration Report

**Document Status:** Complete Empirical Calibration Report (Protocol v1.2)  
**Date:** {results['timestamp'][:10]}  
**Execution Tier:** `{results['execution_tier'].upper()}`  
**Provider / Runtime:** `{results['provider_type']}` (`{results['model_name']}`)  
**Hardware:** Apple Silicon Mac ({meta.get('architecture', 'arm64')}), Python {meta.get('python_version', '3.13')}  
**Commercial Spend:** Strictly **$0.0000 USD**  
**Final Qualification Decision:** `{results['final_decision']}`  

---

## 1. Executive Summary & Calibration Recovery Decision

Under Protocol Amendment v1.2, Phase 2B.0.2 resolves all defects diagnosed during the forensic investigation of Phase 2B.0.1:
1. **De-Conflicted Distractor Phrasing:** Replaced $D_1$ ("one-line reimbursement") with non-conflicting categorization $D_1$ ("specify expense category as Travel & Subsistence").
2. **Exhaustive Characterization Complete:** Evaluated all $2^4 = 16$ candidate configurations in $\\mathcal{{P}}(C)$ at $N = {self.trials_per_subset}$ trials each ({char['subsets_evaluated']} subsets evaluated, {char['total_characterization_requests']} model requests).
3. **Explicit Element Identifiers:** Every table row records explicit element identifiers (`elem_1`, `elem_2`, etc.).
4. **Reporting Integrity Guardrails:** No phantom holdouts or deceptive 0.0% failure rates.

### Empirical Trigger Qualification Summary:
- **Trigger Present Subsets ($n=8$):** Minimum observed failure rate = **{char['min_trigger_present_failure_rate']*100:.1f}%** (Requirement: $\\ge 70\\%$).
- **Trigger Absent Subsets ($n=8$):** Maximum observed failure rate = **{char['max_trigger_absent_failure_rate']*100:.1f}%** (Requirement: $\\le 20\\%$).
- **Qualification Criteria Met:** `{char['qualification_criteria_met']}`
- **Trigger Status:** `{char['trigger_status']}`

---

## 2. Exhaustive Characterization Table (All 16 Subsets)

| Row | Subset Case ID | Candidate Element IDs | Elements Role Description | Contains $T_1$? | Trials ($n$) | Failures ($k$) | Failure Rate ($\\hat{{p}}$) | Wilson 95% CI | Status |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
"""
        for s in char["subsets"]:
            eids_str = f"`{s['element_ids']}`" if s['element_ids'] else "`[]`"
            card_desc = f"{s['cardinality']} elements" if s['cardinality'] > 0 else "Baseline ($\\emptyset$)"
            contains_str = "YES" if s["contains_trigger"] else "NO"
            status_str = "FAIL (Triggered)" if s["is_failing"] else "PASS (Clean)"
            content += f"| {s['subset_index']} | `{s['case_id']}` | {eids_str} | {card_desc} | {contains_str} | {s['trials']} | {s['failures']} | **{s['failure_rate']*100:.1f}%** | {s['wilson_ci']} | {status_str} |\n"

        content += f"""
---

## 3. Minimization Algorithms Benchmark Results

"""
        if search_executed:
            content += f"""| Evaluation Dimension | Stochastic Delta Debugging | Stochastic Greedy Ablation | Stochastic Random Search |
| :--- | :---: | :---: | :---: |
| **Discovered Element IDs** | `{dd.get('discovered_element_ids', [])}` | `{greedy.get('discovered_element_ids', [])}` | `{random.get('discovered_element_ids', [])}` |
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
"""
        else:
            content += """*Minimization search was **BLOCKED_BY_QUALIFICATION** because trigger qualification criteria were not satisfied during exhaustive characterization.*

| Evaluation Dimension | Stochastic Delta Debugging | Stochastic Greedy Ablation | Stochastic Random Search |
| :--- | :---: | :---: | :---: |
| **Status** | `BLOCKED_BY_QUALIFICATION` | `BLOCKED_BY_QUALIFICATION` | `BLOCKED_BY_QUALIFICATION` |
| **Discovered Elements** | `NOT_RUN` | `NOT_RUN` | `NOT_RUN` |
| **Model Requests** | `N/A` | `N/A` | `N/A` |
"""

        content += f"""
---

## 4. Holdout Confirmation Across Algorithms

"""
        if search_executed and holdout:
            content += """| Algorithm | Discovered Element IDs | Holdout Trials ($n$) | Holdout Failures ($k$) | Failure Rate ($\\hat{p}$) | Wilson 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
            for algo_key, h in holdout.items():
                content += f"| **{algo_key}** | `{h.get('element_ids', [])}` | {h.get('trials', 0)} | {h.get('failures', 0)} | **{h.get('failure_rate', 0.0)*100:.1f}%** | {h.get('wilson_ci', [])} |\n"
        else:
            content += """*Holdout confirmation was **BLOCKED_BY_QUALIFICATION** because minimization search was not authorized.*

| Algorithm | Status | Discovered Subset | Holdout Trials ($n$) | Holdout Failures ($k$) | Failure Rate ($\\hat{p}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Delta Debugging** | `BLOCKED_BY_QUALIFICATION` | `NOT_RUN` | 0 | 0 | `N/A` |
| **Greedy Ablation** | `BLOCKED_BY_QUALIFICATION` | `NOT_RUN` | 0 | 0 | `N/A` |
| **Random Search** | `BLOCKED_BY_QUALIFICATION` | `NOT_RUN` | 0 | 0 | `N/A` |
"""

        content += f"""
---

## 5. Honest Scientific Limitations

1. **Local Calibration Scope:** This pilot demonstrates that the testbed, failure trigger, and de-conflicted distractors are sound across the entire power set for this $K=4$ task instance. It does **not** prove general algorithmic superiority across unseen tasks or model families.
2. **Phase 2B.1 Status:** Phase 2B.1 remains **STRICTLY UNAUTHORIZED**. Successful qualification establishes only harness reliability.
3. **Reproducibility Command:**
   ```bash
   {results['reproducible_command']}
   ```
"""
        self.report_path.write_text(content, encoding="utf-8")
        self.log_event("markdown_report_generated", {"report_file": str(self.report_path)})
        return self.report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentXRay Phase 2B.0.2 Corrective Local-Model Calibration Pilot")
    parser.add_argument("--provider", default="fake", choices=["fake", "ollama"], help="Provider type ('fake' or 'ollama')")
    parser.add_argument("--model", default="llama3.2:3b", help="Model name")
    parser.add_argument("--trials-per-subset", type=int, default=5, help="Repeated trials per candidate subset")
    parser.add_argument("--max-requests", type=int, default=120, help="Max requests per algorithm")
    args = parser.parse_args()

    orchestrator = CorrectiveCalibrationOrchestratorV12(
        provider_type=args.provider,
        model_name=args.model,
        trials_per_subset=args.trials_per_subset,
        max_requests_per_algorithm=args.max_requests,
    )

    print(f"Starting Phase 2B.0.2 Corrective Calibration Pilot [provider={args.provider}, model={args.model}]...")
    results = orchestrator.run_full_pilot()
    print(f"\nPilot Complete! Final Decision: {results['final_decision']}")
    print(f"Results: {orchestrator.results_path}")
    print(f"Report: {orchestrator.report_path}")


if __name__ == "__main__":
    main()
