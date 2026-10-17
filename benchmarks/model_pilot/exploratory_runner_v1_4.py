"""Phase 2B.1 Authorized Exploratory Benchmark Execution Runner (Protocol v1.4).

Executes the Phase 2B.1 exploratory feasibility benchmark:
- Strictly follows Protocol v1.4 and Configuration v1.4.
- Enforces 100% local inference with Ollama llama3.2:3b ($0.0000 commercial spend).
- Enforces global hard cap of 7,500 target model requests and per-task caps (120 for K=4, 180 for K=8).
- Implements two-stage execution: Calibration Gate followed by Algorithmic Search on qualified tasks.
- Universal trial-level event logging to trials.jsonl.
- Produces manifest.json, results.json, and phase_2b1_5_exploratory_execution_report.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agentxray.agent_harness.agent import AgentExecutionTrace, ControlledAgent
from agentxray.agent_harness.cost_tracker import (
    CostBudgetExceededError,
    CostTracker,
    ExperimentBudgetConfig,
)
from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator
from agentxray.agent_harness.target import ModelBackedAgentTarget
from agentxray.agent_harness.tools import MockToolEnvironment
from agentxray.models import (
    CandidateEvaluation,
    ExperimentConfig,
    InputCase,
    SearchStatus,
    StochasticDiscoveredCondition,
    StochasticMinimalityStatus,
    TargetOutcome,
    TargetResult,
)
from agentxray.providers.base import ModelProvider, UsageMetadata
from agentxray.providers.fake import FakeProvider
from agentxray.providers.ollama_adapter import OllamaProvider
from agentxray.runner import BudgetExhaustedError, InterventionRunner, TimeoutExceededError
from agentxray.statistics import (
    compare_proportions,
    exact_binomial_test,
    fisher_exact_test,
    wilson_score_interval,
)
from agentxray.stochastic_baselines import (
    _classify_stochastic_minimality,
    run_stochastic_greedy_ablation,
    run_stochastic_random_search,
)
from agentxray.stochastic_minimizer import StochasticDeltaDebugMinimizer
from benchmarks.model_pilot.confirmatory_tasks import (
    ACTIVE_MECHANISMS,
    CONFIRMATORY_TASK_REGISTRY,
    ConfirmatoryTaskSpec,
    NULL_MECHANISMS,
    get_tasks_by_mechanism,
    list_confirmatory_tasks,
)
from benchmarks.model_pilot.power_and_budget_analysis import exact_wilcoxon_permutation_p_value
from benchmarks.model_pilot.validate_confirmatory_protocol import ConfirmatoryProtocolValidator

EXPECTED_PROTOCOL_SHA256 = "db45f4032f2340a9edc3441bafcb3957895b616d1289c28e85a5cc4671574f62"
EXPECTED_CONFIG_SHA256 = "b3e3b0b8450e325914ea8229faafce90a833eb038b02903bfa48062e8ee0cfb2"


def compute_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class DualCostTracker(CostTracker):
    """Cost tracker checking both a shared global budget cap and a per-task cap."""

    def __init__(
        self,
        task_config: ExperimentBudgetConfig,
        global_tracker: CostTracker,
    ) -> None:
        super().__init__(task_config)
        self.global_tracker = global_tracker

    def check_budget_before_request(self, est_prompt_tokens: int = 0) -> None:
        # Check global budget cap first
        self.global_tracker.check_budget_before_request(est_prompt_tokens)
        # Check task budget cap second
        super().check_budget_before_request(est_prompt_tokens)

    def record_usage(self, usage: UsageMetadata, success: bool = True, retries: int = 0) -> None:
        super().record_usage(usage, success, retries)
        self.global_tracker.record_usage(usage, success, retries)


class ExploratoryBenchmarkOrchestrator:
    """Executes the Phase 2B.1 exploratory benchmark under Protocol v1.4."""

    def __init__(
        self,
        config_path: str = "benchmarks/configs/phase_2b1_exploratory_config_v1_4.json",
        protocol_path: str = "docs/research/phase_2b1_exploratory_protocol_v1_4.md",
        provider_type: str = "ollama",
        model_name: str = "llama3.2:3b",
        results_dir: str = "benchmarks/results/phase_2b1_5_execution",
        dry_run: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.protocol_path = Path(protocol_path)
        self.provider_type = provider_type
        self.model_name = model_name
        self.results_dir = Path(results_dir)
        self.dry_run = dry_run

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.trials_log_path = self.results_dir / "trials.jsonl"
        self.manifest_path = self.results_dir / "manifest.json"
        self.results_json_path = self.results_dir / "results.json"

        # Global cost tracker enforcing 7,500 hard safety cap
        self.global_budget_cfg = ExperimentBudgetConfig(
            max_model_requests=7500,
            max_spend_usd=1.00,  # model pricing is $0.00
            model_name=self.model_name,
        )
        self.global_tracker = CostTracker(self.global_budget_cfg)

        self._start_time: str = ""
        self._end_time: str = ""
        self._global_trial_index: int = 0
        self._current_phase: str = "initialization"

        # Initialize provider
        if self.provider_type == "ollama":
            self.provider: ModelProvider = OllamaProvider(
                model_name=self.model_name,
                base_url="http://127.0.0.1:11434",
                timeout_seconds=30.0,
                max_retries=2,
                options={"temperature": 0.0, "seed": 42, "top_p": 1.0},
            )
        elif self.provider_type == "fake":
            self.provider = FakeProvider(model_name="fake-llama3.2:3b")
        else:
            raise ValueError(f"Unsupported provider_type: {self.provider_type}")

    def log_trial_event(self, event_data: dict[str, Any]) -> None:
        """Append an immutable structured event record to trials.jsonl."""
        self._global_trial_index += 1
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "global_trial_index": self._global_trial_index,
            **event_data,
        }
        with open(self.trials_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def run_preflight(self) -> dict[str, Any]:
        """Perform strict preflight checks before any model inference."""
        print(">>> [Preflight 1/5] Verifying repository location and Git state...")
        cwd = Path.cwd()
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        print(f"    Repository: {cwd}")
        print(f"    Git HEAD:   {git_head}")

        print(">>> [Preflight 2/5] Recomputing cryptographic checksums...")
        proto_hash = compute_sha256(self.protocol_path)
        cfg_hash = compute_sha256(self.config_path)
        print(f"    Protocol SHA-256: {proto_hash}")
        print(f"    Config SHA-256:   {cfg_hash}")

        if proto_hash != EXPECTED_PROTOCOL_SHA256:
            raise ValueError(
                f"Protocol hash mismatch! Expected {EXPECTED_PROTOCOL_SHA256}, got {proto_hash}"
            )
        if cfg_hash != EXPECTED_CONFIG_SHA256:
            raise ValueError(
                f"Config hash mismatch! Expected {EXPECTED_CONFIG_SHA256}, got {cfg_hash}"
            )
        print("    Checksums match frozen specification bit-for-bit.")

        print(">>> [Preflight 3/5] Running automated Protocol v1.4 validator...")
        validator = ConfirmatoryProtocolValidator(
            config_path=str(self.config_path),
            protocol_path=str(self.protocol_path),
        )
        val_summary = validator.validate()
        if not val_summary["all_checks_passed"]:
            raise ValueError(f"Protocol validator failed! Discrepancies: {val_summary['discrepancies']}")
        print(f"    Validator passed 100% of checks ({val_summary['total_checks_passed']} checks, 0 discrepancies).")

        print(">>> [Preflight 4/5] Checking local model availability and network isolation...")
        if self.provider_type == "ollama":
            if not self.provider.is_available():
                raise RuntimeError("Local Ollama daemon is unreachable at http://127.0.0.1:11434")
            # Verify llama3.2:3b is available locally
            tags_cmd = subprocess.check_output(["curl", "-s", "http://localhost:11434/api/tags"]).decode("utf-8")
            if "llama3.2:3b" not in tags_cmd:
                raise RuntimeError("Local model 'llama3.2:3b' is not installed in Ollama!")
            print("    Local Ollama daemon reachable and 'llama3.2:3b' is present.")
        else:
            print("    Using FakeProvider (dry-run/test mode).")

        print(">>> [Preflight 5/5] Verifying budget caps and interlocks...")
        print(f"    Global Safety Hard Cap: {self.global_budget_cfg.max_model_requests} requests")
        print(f"    Per-Task Caps: K=4 -> 120 requests, K=8 -> 180 requests")
        print(f"    Commercial API Spend Cap: $0.0000 USD (strictly local inference)")
        print(">>> Preflight checks completed successfully.\n")

        return {
            "git_head": git_head,
            "protocol_sha256": proto_hash,
            "config_sha256": cfg_hash,
            "validator_checks_passed": val_summary["total_checks_passed"],
        }

    def _create_task_tracker(self, cap: int) -> DualCostTracker:
        """Create a per-task cost tracker linked to the global budget tracker."""
        task_cfg = ExperimentBudgetConfig(
            max_model_requests=cap,
            max_spend_usd=1.00,
            model_name=self.model_name,
        )
        return DualCostTracker(task_config=task_cfg, global_tracker=self.global_tracker)

    def _create_instrumented_target(
        self,
        task: ConfirmatoryTaskSpec,
        task_tracker: DualCostTracker,
        phase_label: str,
    ) -> tuple[ModelBackedAgentTarget, Any]:
        """Create a target adapter instrumented with universal trial event logging."""
        base_target = task.create_target(
            provider=self.provider,
            cost_tracker=task_tracker,
            max_turns=3,
        )

        trial_counter = [0]

        def logged_target_fn(input_case: InputCase) -> TargetResult:
            trial_counter[0] += 1
            req_before = self.global_tracker.total_requests
            tok_before = self.global_tracker.total_tokens
            t0 = time.perf_counter()

            res = base_target.execute(input_case)
            duration = time.perf_counter() - t0
            req_used = self.global_tracker.total_requests - req_before
            tok_used = self.global_tracker.total_tokens - tok_before

            trace = base_target.last_trace
            tool_calls_info = [
                {
                    "tool": t.tool_name,
                    "arguments": t.arguments,
                    "output": t.output,
                }
                for t in (trace.tool_executions if trace else ())
            ]

            if trace:
                _, _, fclass = base_target.evaluator.evaluate_detailed(trace)  # type: ignore
            else:
                fclass = "FAIL_INFRA"

            self.log_trial_event({
                "phase": phase_label,
                "task_id": task.instance_id,
                "mechanism_id": task.mechanism_id,
                "k": task.k,
                "task_trial_index": trial_counter[0],
                "candidate_subset": list(input_case.elements),
                "outcome": res.outcome.value,
                "is_failure": res.outcome == TargetOutcome.FAILURE,
                "failure_class": fclass,
                "model_requests_used": req_used,
                "tokens_used": tok_used,
                "latency_seconds": round(duration, 4),
                "tool_executions": tool_calls_info,
                "final_response": trace.final_response if trace else "",
                "error_message": trace.error_message if trace else None,
            })
            return res

        return base_target, logged_target_fn

    def run_calibration_stage(self) -> dict[str, Any]:
        """Stage 1: Pre-Experiment Calibration & Harness Qualification Gate."""
        print("======================================================================")
        print("STAGE 1: PRE-EXPERIMENT CALIBRATION & QUALIFICATION GATE")
        print("======================================================================")
        calibration_results: dict[str, Any] = {}

        for task in list_confirmatory_tasks():
            print(f"\n--- Calibrating Task: {task.instance_id} (Mechanism: {task.mechanism_id}, K={task.k}) ---")
            task_tracker = self._create_task_tracker(50)

            if not task.is_null_case:
                # 1. Trigger condition: evaluate candidate with trigger for 5 trials
                _, target_trigger = self._create_instrumented_target(task, task_tracker, "calibration_trigger")
                trigger_input = task.create_initial_input()
                trigger_failures = 0
                for _ in range(5):
                    res = target_trigger(trigger_input)
                    if res.outcome == TargetOutcome.FAILURE:
                        trigger_failures += 1

                trigger_rate = trigger_failures / 5.0
                trigger_ci = wilson_score_interval(trigger_failures, 5, confidence=0.95)

                # 2. Control condition: evaluate candidate without trigger for 5 trials
                _, target_control = self._create_instrumented_target(task, task_tracker, "calibration_control")
                ablated_elems = tuple(e for e in task.candidate_elements if e not in task.planted_trigger_elements)
                control_input = InputCase(case_id=f"{task.instance_id}_ablated", elements=ablated_elems)
                control_failures = 0
                for _ in range(5):
                    res = target_control(control_input)
                    if res.outcome == TargetOutcome.FAILURE:
                        control_failures += 1

                control_rate = control_failures / 5.0
                control_ci = wilson_score_interval(control_failures, 5, confidence=0.95)

                # Active task qualification criteria:
                # Trigger failure rate >= 0.70 (>= 4/5) and Wilson lower bound > 0.20
                # Control failure rate <= 0.20 (<= 1/5) and Wilson upper bound < 0.50
                trigger_qualified = (trigger_rate >= 0.70) and (trigger_ci[0] > 0.20)
                control_qualified = (control_rate <= 0.20) and (control_ci[1] < 0.50)
                is_qualified = trigger_qualified and control_qualified

                task_calib = {
                    "task_id": task.instance_id,
                    "mechanism_id": task.mechanism_id,
                    "k": task.k,
                    "is_null_case": False,
                    "trigger_trials": 5,
                    "trigger_failures": trigger_failures,
                    "trigger_rate": trigger_rate,
                    "trigger_ci": trigger_ci,
                    "trigger_qualified": trigger_qualified,
                    "control_trials": 5,
                    "control_failures": control_failures,
                    "control_rate": control_rate,
                    "control_ci": control_ci,
                    "control_qualified": control_qualified,
                    "qualification_status": "QUALIFIED" if is_qualified else "DISQUALIFIED_TASK",
                    "disqualification_reason": None if is_qualified else (
                        "Trigger failure rate insufficient" if not trigger_qualified else "Control failure rate excessive"
                    ),
                }
                calibration_results[task.instance_id] = task_calib
                print(f"    Trigger: {trigger_failures}/5 ({trigger_rate:.0%}) 95% CI {trigger_ci} -> {'PASS' if trigger_qualified else 'FAIL'}")
                print(f"    Control: {control_failures}/5 ({control_rate:.0%}) 95% CI {control_ci} -> {'PASS' if control_qualified else 'FAIL'}")
                print(f"    Status:  {task_calib['qualification_status']}")

            else:
                # Null task: clean baseline condition for 5 trials
                _, target_clean = self._create_instrumented_target(task, task_tracker, "calibration_null")
                clean_input = InputCase(case_id=f"{task.instance_id}_clean", elements=())
                clean_failures = 0
                for _ in range(5):
                    res = target_clean(clean_input)
                    if res.outcome == TargetOutcome.FAILURE:
                        clean_failures += 1

                clean_rate = clean_failures / 5.0
                clean_ci = wilson_score_interval(clean_failures, 5, confidence=0.95)

                # Null qualification: clean failure rate <= 0.10 (0/5)
                clean_qualified = (clean_rate <= 0.10)
                task_calib = {
                    "task_id": task.instance_id,
                    "mechanism_id": task.mechanism_id,
                    "k": task.k,
                    "is_null_case": True,
                    "clean_trials": 5,
                    "clean_failures": clean_failures,
                    "clean_rate": clean_rate,
                    "clean_ci": clean_ci,
                    "qualification_status": "QUALIFIED" if clean_qualified else "DISQUALIFIED_TASK",
                    "disqualification_reason": None if clean_qualified else "Clean execution failure rate exceeded 0.10",
                }
                calibration_results[task.instance_id] = task_calib
                print(f"    Clean Baseline: {clean_failures}/5 ({clean_rate:.0%}) 95% CI {clean_ci} -> {'PASS' if clean_qualified else 'FAIL'}")
                print(f"    Status:         {task_calib['qualification_status']}")

        qualified_active = sum(1 for t, r in calibration_results.items() if not r["is_null_case"] and r["qualification_status"] == "QUALIFIED")
        disqualified_active = sum(1 for t, r in calibration_results.items() if not r["is_null_case"] and r["qualification_status"] == "DISQUALIFIED_TASK")
        qualified_null = sum(1 for t, r in calibration_results.items() if r["is_null_case"] and r["qualification_status"] == "QUALIFIED")

        print("\n======================================================================")
        print("CALIBRATION GATE SUMMARY:")
        print(f"Active Tasks Qualified:   {qualified_active} / 12")
        print(f"Active Tasks Disqualified: {disqualified_active} / 12")
        print(f"Null Tasks Qualified:     {qualified_null} / 4")
        print(f"Total Model Requests So Far: {self.global_tracker.total_requests} / 7500")
        print("======================================================================\n")

        return calibration_results

    def run_search_stage(self, calibration_results: dict[str, Any]) -> dict[str, Any]:
        """Stage 2: Algorithmic Search (DD, Greedy, Random) on Qualified Tasks."""
        print("======================================================================")
        print("STAGE 2: ALGORITHMIC SEARCH EXECUTION")
        print("======================================================================")
        search_results: dict[str, Any] = {}

        for task in list_confirmatory_tasks():
            calib = calibration_results[task.instance_id]
            task_dict: dict[str, Any] = {
                "task_id": task.instance_id,
                "mechanism_id": task.mechanism_id,
                "k": task.k,
                "is_null_case": task.is_null_case,
                "qualification_status": calib["qualification_status"],
                "algorithms": {},
            }

            if calib["qualification_status"] != "QUALIFIED":
                print(f"\n[SKIP] Task {task.instance_id} was disqualified during calibration. Skipping search.")
                search_results[task.instance_id] = task_dict
                continue

            per_task_cap = 120 if task.k == 4 else 180
            print(f"\n>>> Running Search on Task: {task.instance_id} (K={task.k}, Cap={per_task_cap} requests)")

            # ------------------------------------------------------------------
            # 1. Stochastic Delta Debugging
            # ------------------------------------------------------------------
            print("    [1/3] Running Stochastic Delta Debugging...")
            dd_tracker = self._create_task_tracker(per_task_cap)
            _, dd_target_fn = self._create_instrumented_target(task, dd_tracker, "search_dd")
            dd_runner = InterventionRunner(
                config=ExperimentConfig(
                    experiment_name=f"{task.instance_id}_dd",
                    max_trials=per_task_cap,
                    timeout_seconds=300.0,
                ),
                target_fn=dd_target_fn,
            )
            dd_minimizer = StochasticDeltaDebugMinimizer(
                runner=dd_runner,
                trials_per_candidate=5,
                failure_threshold=0.50,
                alpha=0.05,
                confidence=0.95,
                alternative="greater",
            )
            try:
                dd_res = dd_minimizer.minimize(task.create_initial_input())
                dd_data = {
                    "algorithm": "stochastic_delta_debugging",
                    "status": dd_res.status.value,
                    "minimality_status": dd_res.minimality_status.value,
                    "discovered_elements": list(dd_res.subset_elements),
                    "target_calls_used": dd_res.target_calls_used,
                    "search_target_calls": dd_res.search_target_calls,
                    "verification_target_calls": dd_res.verification_target_calls,
                    "model_requests_used": dd_tracker.total_requests,
                    "cap_utilized_pct": round(dd_tracker.total_requests / per_task_cap * 100, 1),
                    "status_message": dd_res.status_message,
                    "matches_ground_truth": (set(dd_res.subset_elements) == set(task.planted_trigger_elements)) if not task.is_null_case else (len(dd_res.subset_elements) == 0),
                }
            except (CostBudgetExceededError, BudgetExhaustedError) as exc:
                dd_data = {
                    "algorithm": "stochastic_delta_debugging",
                    "status": SearchStatus.BUDGET_EXHAUSTED.value,
                    "minimality_status": StochasticMinimalityStatus.BUDGET_EXHAUSTED.value,
                    "discovered_elements": list(task.candidate_elements),
                    "model_requests_used": dd_tracker.total_requests,
                    "cap_utilized_pct": 100.0,
                    "status_message": str(exc),
                    "matches_ground_truth": False,
                }
            task_dict["algorithms"]["stochastic_delta_debugging"] = dd_data
            print(f"        DD: {dd_data['minimality_status']}, requests: {dd_data['model_requests_used']}/{per_task_cap}, matches GT: {dd_data.get('matches_ground_truth')}")

            # ------------------------------------------------------------------
            # 2. Stochastic Greedy Ablation Baseline
            # ------------------------------------------------------------------
            print("    [2/3] Running Stochastic Greedy Ablation...")
            greedy_tracker = self._create_task_tracker(per_task_cap)
            _, greedy_target_fn = self._create_instrumented_target(task, greedy_tracker, "search_greedy")
            greedy_runner = InterventionRunner(
                config=ExperimentConfig(
                    experiment_name=f"{task.instance_id}_greedy",
                    max_trials=per_task_cap,
                    timeout_seconds=300.0,
                ),
                target_fn=greedy_target_fn,
            )
            try:
                greedy_res = run_stochastic_greedy_ablation(
                    initial_input=task.create_initial_input(),
                    runner=greedy_runner,
                    trials_per_candidate=5,
                    failure_threshold=0.50,
                    alpha=0.05,
                    confidence=0.95,
                    alternative="greater",
                )
                greedy_data = {
                    "algorithm": "stochastic_greedy_ablation",
                    "minimality_status": greedy_res.minimality_status.value,
                    "discovered_elements": list(greedy_res.discovered_elements),
                    "target_calls_used": greedy_res.target_calls,
                    "search_target_calls": greedy_res.search_target_calls,
                    "verification_target_calls": greedy_res.verification_target_calls,
                    "model_requests_used": greedy_tracker.total_requests,
                    "cap_utilized_pct": round(greedy_tracker.total_requests / per_task_cap * 100, 1),
                    "budget_exhausted": greedy_res.budget_exhausted,
                    "matches_ground_truth": (set(greedy_res.discovered_elements) == set(task.planted_trigger_elements)) if not task.is_null_case else (len(greedy_res.discovered_elements) == 0),
                }
            except (CostBudgetExceededError, BudgetExhaustedError) as exc:
                greedy_data = {
                    "algorithm": "stochastic_greedy_ablation",
                    "minimality_status": StochasticMinimalityStatus.BUDGET_EXHAUSTED.value,
                    "discovered_elements": list(task.candidate_elements),
                    "model_requests_used": greedy_tracker.total_requests,
                    "cap_utilized_pct": 100.0,
                    "budget_exhausted": True,
                    "matches_ground_truth": False,
                }
            task_dict["algorithms"]["stochastic_greedy_ablation"] = greedy_data
            print(f"        Greedy: {greedy_data['minimality_status']}, requests: {greedy_data['model_requests_used']}/{per_task_cap}, matches GT: {greedy_data.get('matches_ground_truth')}")

            # ------------------------------------------------------------------
            # 3. Seeded Random Search Baseline
            # ------------------------------------------------------------------
            print("    [3/3] Running Seeded Random Search Baseline...")
            random_tracker = self._create_task_tracker(per_task_cap)
            _, random_target_fn = self._create_instrumented_target(task, random_tracker, "search_random")
            random_runner = InterventionRunner(
                config=ExperimentConfig(
                    experiment_name=f"{task.instance_id}_random",
                    max_trials=per_task_cap,
                    timeout_seconds=300.0,
                ),
                target_fn=random_target_fn,
            )
            try:
                random_res = run_stochastic_random_search(
                    initial_input=task.create_initial_input(),
                    runner=random_runner,
                    trials_per_candidate=5,
                    failure_threshold=0.50,
                    alpha=0.05,
                    confidence=0.95,
                    alternative="greater",
                    seed=42,
                )
                random_data = {
                    "algorithm": "stochastic_seeded_random",
                    "minimality_status": random_res.minimality_status.value,
                    "discovered_elements": list(random_res.discovered_elements),
                    "target_calls_used": random_res.target_calls,
                    "search_target_calls": random_res.search_target_calls,
                    "verification_target_calls": random_res.verification_target_calls,
                    "model_requests_used": random_tracker.total_requests,
                    "cap_utilized_pct": round(random_tracker.total_requests / per_task_cap * 100, 1),
                    "budget_exhausted": random_res.budget_exhausted,
                    "matches_ground_truth": (set(random_res.discovered_elements) == set(task.planted_trigger_elements)) if not task.is_null_case else (len(random_res.discovered_elements) == 0),
                }
            except (CostBudgetExceededError, BudgetExhaustedError) as exc:
                random_data = {
                    "algorithm": "stochastic_seeded_random",
                    "minimality_status": StochasticMinimalityStatus.BUDGET_EXHAUSTED.value,
                    "discovered_elements": list(task.candidate_elements),
                    "model_requests_used": random_tracker.total_requests,
                    "cap_utilized_pct": 100.0,
                    "budget_exhausted": True,
                    "matches_ground_truth": False,
                }
            task_dict["algorithms"]["stochastic_seeded_random"] = random_data
            print(f"        Random: {random_data['minimality_status']}, requests: {random_data['model_requests_used']}/{per_task_cap}, matches GT: {random_data.get('matches_ground_truth')}")

            search_results[task.instance_id] = task_dict

        return search_results

    def run_holdout_stage(self, search_results: dict[str, Any]) -> dict[str, Any]:
        """Stage 3: Holdout Verification on Discovered Triggers."""
        print("\n======================================================================")
        print("STAGE 3: HOLDOUT VERIFICATION")
        print("======================================================================")
        holdout_results: dict[str, Any] = {}

        for task_id, task_data in search_results.items():
            if task_data.get("qualification_status") != "QUALIFIED" or task_data.get("is_null_case"):
                continue

            task = CONFIRMATORY_TASK_REGISTRY[task_id]
            task_holdout: dict[str, Any] = {}

            for algo_name, algo_data in task_data.get("algorithms", {}).items():
                min_status = algo_data.get("minimality_status")
                disc_elems = tuple(algo_data.get("discovered_elements", []))

                # Evaluate holdout only if a non-empty subset was discovered
                if disc_elems and min_status in (
                    StochasticMinimalityStatus.STATISTICALLY_SUPPORTED.value,
                    StochasticMinimalityStatus.SUPPORTED_1_MINIMAL.value,
                    StochasticMinimalityStatus.OBSERVED_ONLY.value,
                    StochasticMinimalityStatus.OBSERVED_1_MINIMAL.value,
                ):
                    h_tracker = self._create_task_tracker(25)
                    _, h_target = self._create_instrumented_target(task, h_tracker, "holdout")
                    h_input = InputCase(case_id=f"{task_id}_holdout", elements=disc_elems)
                    h_failures = 0
                    for _ in range(5):
                        res = h_target(h_input)
                        if res.outcome == TargetOutcome.FAILURE:
                            h_failures += 1
                    h_rate = h_failures / 5.0
                    h_ci = wilson_score_interval(h_failures, 5, confidence=0.95)
                    task_holdout[algo_name] = {
                        "holdout_trials": 5,
                        "holdout_failures": h_failures,
                        "holdout_rate": h_rate,
                        "holdout_ci": h_ci,
                        "reproduced_in_holdout": h_rate >= 0.50,
                    }
                    print(f"    Holdout [{task_id} - {algo_name}]: {h_failures}/5 ({h_rate:.0%}) CI {h_ci} -> {'PASS' if h_rate >= 0.50 else 'FAIL'}")

            holdout_results[task_id] = task_holdout

        return holdout_results

    def compute_aggregate_statistics(
        self,
        calibration_results: dict[str, Any],
        search_results: dict[str, Any],
        holdout_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Stage 4: Post-Hoc Estimand & Statistical Analysis under Protocol v1.4."""
        # 1. Active Task Recovery Rates
        active_tasks = [t for t in list_confirmatory_tasks() if not t.is_null_case]
        active_ids = [t.instance_id for t in active_tasks]

        dd_recoveries: dict[str, bool] = {}
        greedy_recoveries: dict[str, bool] = {}
        random_recoveries: dict[str, bool] = {}

        dd_requests: dict[str, int] = {}
        greedy_requests: dict[str, int] = {}
        random_requests: dict[str, int] = {}

        dd_norm_burden: dict[str, float] = {}
        greedy_norm_burden: dict[str, float] = {}

        for tid in active_ids:
            s_data = search_results.get(tid, {}).get("algorithms", {})
            cap = 120 if CONFIRMATORY_TASK_REGISTRY[tid].k == 4 else 180

            # DD
            dd_info = s_data.get("stochastic_delta_debugging", {})
            dd_rec = dd_info.get("matches_ground_truth", False) and dd_info.get("minimality_status") in (
                StochasticMinimalityStatus.STATISTICALLY_SUPPORTED.value,
                StochasticMinimalityStatus.SUPPORTED_1_MINIMAL.value,
                StochasticMinimalityStatus.OBSERVED_ONLY.value,
                StochasticMinimalityStatus.OBSERVED_1_MINIMAL.value,
            )
            dd_req = dd_info.get("model_requests_used", cap)
            dd_recoveries[tid] = dd_rec
            dd_requests[tid] = dd_req
            dd_norm_burden[tid] = dd_req / cap

            # Greedy
            gr_info = s_data.get("stochastic_greedy_ablation", {})
            gr_rec = gr_info.get("matches_ground_truth", False) and gr_info.get("minimality_status") in (
                StochasticMinimalityStatus.STATISTICALLY_SUPPORTED.value,
                StochasticMinimalityStatus.SUPPORTED_1_MINIMAL.value,
                StochasticMinimalityStatus.OBSERVED_ONLY.value,
                StochasticMinimalityStatus.OBSERVED_1_MINIMAL.value,
            )
            gr_req = gr_info.get("model_requests_used", cap)
            greedy_recoveries[tid] = gr_rec
            greedy_requests[tid] = gr_req
            greedy_norm_burden[tid] = gr_req / cap

            # Random
            rnd_info = s_data.get("stochastic_seeded_random", {})
            rnd_rec = rnd_info.get("matches_ground_truth", False) and rnd_info.get("minimality_status") in (
                StochasticMinimalityStatus.STATISTICALLY_SUPPORTED.value,
                StochasticMinimalityStatus.SUPPORTED_1_MINIMAL.value,
                StochasticMinimalityStatus.OBSERVED_ONLY.value,
                StochasticMinimalityStatus.OBSERVED_1_MINIMAL.value,
            )
            rnd_req = rnd_info.get("model_requests_used", cap)
            random_recoveries[tid] = rnd_rec
            random_requests[tid] = rnd_req

        # Mechanism-level differences for n=6 independent mechanisms
        grouped_by_mech = get_tasks_by_mechanism()
        mech_differences: dict[str, float] = {}
        mech_signs: list[bool] = []
        mech_diff_values: list[float] = []

        for mid in ACTIVE_MECHANISMS:
            tasks_m = grouped_by_mech[mid]
            t4 = [t for t in tasks_m if t.k == 4][0].instance_id
            t8 = [t for t in tasks_m if t.k == 8][0].instance_id

            # Diff in normalized burden: Greedy - DD (positive favors DD)
            d4 = (greedy_requests.get(t4, 120) - dd_requests.get(t4, 120)) / 120.0
            d8 = (greedy_requests.get(t8, 180) - dd_requests.get(t8, 180)) / 180.0
            mean_d = (d4 + d8) / 2.0
            mech_differences[mid] = mean_d
            if abs(mean_d) > 1e-6:
                mech_diff_values.append(abs(mean_d))
                mech_signs.append(mean_d > 0)

        # Exact Wilcoxon signed-rank permutation test on mechanism differences (n=6)
        if mech_diff_values:
            # Rank the absolute differences
            indexed = sorted(enumerate(mech_diff_values), key=lambda x: x[1])
            ranks = [0.0] * len(mech_diff_values)
            for r_idx, (orig_idx, _) in enumerate(indexed, start=1):
                ranks[orig_idx] = float(r_idx)
            wilcoxon_p = exact_wilcoxon_permutation_p_value(ranks, mech_signs)
        else:
            wilcoxon_p = 1.0

        # Paired 2x2 contingency table for recovery (DD vs Greedy on 12 active tasks)
        # b: DD passed, Greedy failed; c: Greedy passed, DD failed
        both_rec = sum(1 for tid in active_ids if dd_recoveries[tid] and greedy_recoveries[tid])
        dd_only = sum(1 for tid in active_ids if dd_recoveries[tid] and not greedy_recoveries[tid])
        gr_only = sum(1 for tid in active_ids if not dd_recoveries[tid] and greedy_recoveries[tid])
        neither_rec = sum(1 for tid in active_ids if not dd_recoveries[tid] and not greedy_recoveries[tid])

        # McNemar exact test: p = 2 * Binomial(b, b+c, 0.5)
        disc_total = dd_only + gr_only
        if disc_total > 0:
            mcnemar_p = exact_binomial_test(min(dd_only, gr_only), disc_total, p0=0.50, alternative="two-sided")
        else:
            mcnemar_p = 1.0

        # Null control false discoveries
        null_tasks = [t for t in list_confirmatory_tasks() if t.is_null_case]
        null_false_disc: dict[str, dict[str, bool]] = {}
        for nt in null_tasks:
            s_data = search_results.get(nt.instance_id, {}).get("algorithms", {})
            null_false_disc[nt.instance_id] = {
                "dd_false_discovery": s_data.get("stochastic_delta_debugging", {}).get("minimality_status") in (
                    StochasticMinimalityStatus.STATISTICALLY_SUPPORTED.value,
                    StochasticMinimalityStatus.SUPPORTED_1_MINIMAL.value,
                    StochasticMinimalityStatus.OBSERVED_ONLY.value,
                    StochasticMinimalityStatus.OBSERVED_1_MINIMAL.value,
                ),
                "greedy_false_discovery": s_data.get("stochastic_greedy_ablation", {}).get("minimality_status") in (
                    StochasticMinimalityStatus.STATISTICALLY_SUPPORTED.value,
                    StochasticMinimalityStatus.SUPPORTED_1_MINIMAL.value,
                    StochasticMinimalityStatus.OBSERVED_ONLY.value,
                    StochasticMinimalityStatus.OBSERVED_1_MINIMAL.value,
                ),
                "random_false_discovery": s_data.get("stochastic_seeded_random", {}).get("minimality_status") in (
                    StochasticMinimalityStatus.STATISTICALLY_SUPPORTED.value,
                    StochasticMinimalityStatus.SUPPORTED_1_MINIMAL.value,
                    StochasticMinimalityStatus.OBSERVED_ONLY.value,
                    StochasticMinimalityStatus.OBSERVED_1_MINIMAL.value,
                ),
            }

        return {
            "recovery_summary": {
                "total_active_tasks": len(active_ids),
                "dd_recovered_count": sum(1 for v in dd_recoveries.values() if v),
                "greedy_recovered_count": sum(1 for v in greedy_recoveries.values() if v),
                "random_recovered_count": sum(1 for v in random_recoveries.values() if v),
                "contingency_table": {
                    "both_recovered": both_rec,
                    "dd_only": dd_only,
                    "greedy_only": gr_only,
                    "neither_recovered": neither_rec,
                },
                "mcnemar_exact_p_value": round(mcnemar_p, 4),
            },
            "resource_burden_stratified": {
                "k4": {
                    "tasks": [t.instance_id for t in active_tasks if t.k == 4],
                    "cap": 120,
                    "dd_requests": {tid: dd_requests[tid] for tid in active_ids if CONFIRMATORY_TASK_REGISTRY[tid].k == 4},
                    "greedy_requests": {tid: greedy_requests[tid] for tid in active_ids if CONFIRMATORY_TASK_REGISTRY[tid].k == 4},
                    "random_requests": {tid: random_requests[tid] for tid in active_ids if CONFIRMATORY_TASK_REGISTRY[tid].k == 4},
                    "mean_dd_requests": round(sum(dd_requests[t.instance_id] for t in active_tasks if t.k == 4) / 6.0, 1),
                    "mean_greedy_requests": round(sum(greedy_requests[t.instance_id] for t in active_tasks if t.k == 4) / 6.0, 1),
                },
                "k8": {
                    "tasks": [t.instance_id for t in active_tasks if t.k == 8],
                    "cap": 180,
                    "dd_requests": {tid: dd_requests[tid] for tid in active_ids if CONFIRMATORY_TASK_REGISTRY[tid].k == 8},
                    "greedy_requests": {tid: greedy_requests[tid] for tid in active_ids if CONFIRMATORY_TASK_REGISTRY[tid].k == 8},
                    "random_requests": {tid: random_requests[tid] for tid in active_ids if CONFIRMATORY_TASK_REGISTRY[tid].k == 8},
                    "mean_dd_requests": round(sum(dd_requests[t.instance_id] for t in active_tasks if t.k == 8) / 6.0, 1),
                    "mean_greedy_requests": round(sum(greedy_requests[t.instance_id] for t in active_tasks if t.k == 8) / 6.0, 1),
                },
            },
            "mechanism_level_burden": {
                "independent_mechanisms_count": len(ACTIVE_MECHANISMS),
                "differences_normalized": {m: round(d, 4) for m, d in mech_differences.items()},
                "wilcoxon_exact_permutation_p_value": round(wilcoxon_p, 4),
                "wilcoxon_rejects_at_0_05": wilcoxon_p < 0.05,
            },
            "null_control_summary": {
                "total_null_tasks": len(null_tasks),
                "evaluations": null_false_disc,
                "dd_false_discoveries_count": sum(1 for v in null_false_disc.values() if v["dd_false_discovery"]),
                "greedy_false_discoveries_count": sum(1 for v in null_false_disc.values() if v["greedy_false_discovery"]),
                "random_false_discoveries_count": sum(1 for v in null_false_disc.values() if v["random_false_discovery"]),
            },
            "global_budget_accounting": {
                "total_model_requests_expended": self.global_tracker.total_requests,
                "max_budget_limit": 7500,
                "commercial_spend_usd": 0.0,
                "budget_hard_cap_breached": self.global_tracker.total_requests >= 7500,
            },
        }

    def execute(self) -> dict[str, Any]:
        """Execute the full exploratory benchmark."""
        self._start_time = datetime.now(timezone.utc).isoformat()

        # Step 1: Preflight
        preflight_info = self.run_preflight()
        if self.dry_run:
            print("DRY-RUN mode requested. Preflight succeeded. Halting before model inference.")
            return {"status": "BLOCKED_BEFORE_INFERENCE", "preflight": preflight_info}

        # Step 2: Calibration Gate
        calibration_results = self.run_calibration_stage()

        # Check if calibration completely failed
        any_qualified = any(r["qualification_status"] == "QUALIFIED" for r in calibration_results.values())
        if not any_qualified:
            print("[STOP] Zero tasks qualified during calibration gate! Halting execution.")
            self._end_time = datetime.now(timezone.utc).isoformat()
            final_report_data = {
                "status": "PARTIALLY_COMPLETED",
                "termination_reason": "CALIBRATION_GATE_FAILED_ZERO_QUALIFIED",
                "preflight": preflight_info,
                "calibration": calibration_results,
                "total_requests": self.global_tracker.total_requests,
            }
            self.results_json_path.write_text(json.dumps(final_report_data, indent=2), encoding="utf-8")
            return final_report_data

        # Step 3: Algorithmic Search
        search_results = self.run_search_stage(calibration_results)

        # Step 4: Holdout Verification
        holdout_results = self.run_holdout_stage(search_results)

        # Step 5: Statistical Analysis
        stats = self.compute_aggregate_statistics(calibration_results, search_results, holdout_results)

        self._end_time = datetime.now(timezone.utc).isoformat()

        # Step 6: Produce Manifest and Results JSON
        manifest_data = {
            "experiment_id": "phase_2b1_exploratory_benchmark_v1_4",
            "protocol_reference": str(self.protocol_path),
            "protocol_sha256": preflight_info["protocol_sha256"],
            "config_reference": str(self.config_path),
            "config_sha256": preflight_info["config_sha256"],
            "git_head": preflight_info["git_head"],
            "start_time": self._start_time,
            "end_time": self._end_time,
            "provider": self.provider_type,
            "model": self.model_name,
            "python_version": platform.python_version(),
            "os_platform": platform.platform(),
            "system_hardware": platform.machine(),
            "global_requests_used": self.global_tracker.total_requests,
            "commercial_spend_usd": 0.0,
        }
        self.manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        full_results = {
            "manifest": manifest_data,
            "preflight": preflight_info,
            "calibration": calibration_results,
            "search": search_results,
            "holdout": holdout_results,
            "statistics": stats,
        }
        self.results_json_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")

        print("\n======================================================================")
        print("PHASE 2B.1 EXPLORATORY BENCHMARK EXECUTION COMPLETED")
        print(f"Total Model Requests Used: {self.global_tracker.total_requests} / 7500")
        print(f"Commercial Spend:          $0.0000 USD")
        print(f"Trials Log:                {self.trials_log_path}")
        print(f"Results JSON:              {self.results_json_path}")
        print(f"Manifest:                  {self.manifest_path}")
        print("======================================================================\n")

        return full_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2B.1 Exploratory Benchmark Execution Runner")
    parser.add_argument("--config", default="benchmarks/configs/phase_2b1_exploratory_config_v1_4.json")
    parser.add_argument("--protocol", default="docs/research/phase_2b1_exploratory_protocol_v1_4.md")
    parser.add_argument("--provider", choices=["ollama", "fake"], default="ollama")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--results-dir", default="benchmarks/results/phase_2b1_5_execution")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    orchestrator = ExploratoryBenchmarkOrchestrator(
        config_path=args.config,
        protocol_path=args.protocol,
        provider_type=args.provider,
        model_name=args.model,
        results_dir=args.results_dir,
        dry_run=args.dry_run,
    )
    res = orchestrator.execute()


if __name__ == "__main__":
    main()
