"""Developer Command-Line Interface (CLI) for AgentXRay.

Exposes developer-facing subcommands ('demo', 'diagnose', 'version') on top of
the public Python SDK (agentxray.diagnose), alongside a restrained monochrome
terminal presentation and strict exit-code contracts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentxray.api import DiagnosticResult, DiagnosticStatus, diagnose
from agentxray.models import InputCase, TargetOutcome, TargetResult
from agentxray.providers.fake import FakeProvider


def get_version() -> str:
    """Return package version dynamically from metadata or fallback to __version__."""
    try:
        from importlib.metadata import version

        return version("agentxray")
    except Exception:
        from agentxray import __version__

        return __version__


class TerminalColors:
    """Restrained monochrome/ANSI styling with automatic NO_COLOR and TTY detection."""

    def __init__(self, disabled: bool = False) -> None:
        if disabled or "NO_COLOR" in os.environ or not sys.stdout.isatty():
            self.bold = ""
            self.dim = ""
            self.green = ""
            self.red = ""
            self.yellow = ""
            self.cyan = ""
            self.reset = ""
        else:
            self.bold = "\033[1m"
            self.dim = "\033[2m"
            self.green = "\033[32m"
            self.red = "\033[31m"
            self.yellow = "\033[33m"
            self.cyan = "\033[36m"
            self.reset = "\033[0m"


class FailureConditionType(str, Enum):
    """Supported declarative failure condition types for offline scenario files."""

    CONTAINS_ANY = "contains_any"
    CONTAINS_ALL = "contains_all"
    EXACT_MATCH = "exact_match"
    ALWAYS_FAIL = "always_fail"
    NEVER_FAIL = "never_fail"


class FailureConditionSpec(BaseModel):
    """Specification for declarative failure condition matching."""

    model_config = ConfigDict(extra="forbid")

    type: FailureConditionType = FailureConditionType.CONTAINS_ANY
    patterns: list[str] = Field(default_factory=list)


class ScenarioFileSpec(BaseModel):
    """Documented schema for AgentXRay JSON scenario files."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_url: str | None = Field(default=None, alias="$schema")
    name: str = "custom_scenario"
    description: str = ""
    candidates: list[str]
    failure_condition: FailureConditionSpec = Field(default_factory=FailureConditionSpec)
    ground_truth: list[str] | None = None


def load_scenario_file(path_str: str) -> ScenarioFileSpec:
    """Load and validate a JSON scenario file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is malformed or violates schema validation rules.
    """
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Scenario file not found: '{path_str}'")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Cannot read scenario file '{path_str}': {exc}") from exc

    try:
        raw_data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in scenario file '{path_str}': {exc.msg} (line {exc.lineno}, col {exc.colno})") from exc

    if not isinstance(raw_data, dict):
        raise ValueError(f"Scenario file root must be a JSON object, got {type(raw_data).__name__}.")

    try:
        spec = ScenarioFileSpec.model_validate(raw_data)
    except ValidationError as exc:
        err_msgs = [f"  - {err['loc']}: {err['msg']}" for err in exc.errors()]
        raise ValueError(f"Schema validation error in scenario file '{path_str}':\n" + "\n".join(err_msgs)) from exc

    # Semantic validation
    if len(spec.candidates) == 0:
        raise ValueError("Candidate pool cannot be empty. Specify at least one candidate directive.")

    seen_candidates: set[str] = set()
    for idx, cand in enumerate(spec.candidates):
        if not cand.strip():
            raise ValueError(f"Candidate element at index {idx} cannot be blank.")
        if cand in seen_candidates:
            raise ValueError(f"Duplicate candidate element found: {cand!r}")
        seen_candidates.add(cand)

    cond_type = spec.failure_condition.type
    if cond_type in (
        FailureConditionType.CONTAINS_ANY,
        FailureConditionType.CONTAINS_ALL,
        FailureConditionType.EXACT_MATCH,
    ):
        if not spec.failure_condition.patterns:
            raise ValueError(
                f"Failure condition type '{cond_type.value}' requires non-empty 'patterns' list."
            )
        for idx, pat in enumerate(spec.failure_condition.patterns):
            if not pat.strip():
                raise ValueError(f"Pattern at index {idx} in failure_condition cannot be blank.")

    if spec.ground_truth is not None:
        for gt in spec.ground_truth:
            if gt not in seen_candidates:
                raise ValueError(
                    f"Ground truth element {gt!r} is not present in the candidate pool."
                )

    return spec


def make_declarative_evaluator(spec: ScenarioFileSpec) -> Callable[[tuple[str, ...]], TargetResult]:
    """Build a safe declarative failure evaluator from a ScenarioFileSpec.

    Operates without eval, exec, or shell execution.
    """
    cond = spec.failure_condition
    cond_type = cond.type
    patterns = cond.patterns

    def evaluate(subset: tuple[str, ...]) -> TargetResult:
        if cond_type == FailureConditionType.ALWAYS_FAIL:
            is_failure = True
        elif cond_type == FailureConditionType.NEVER_FAIL:
            is_failure = False
        elif cond_type == FailureConditionType.EXACT_MATCH:
            is_failure = all(p in subset for p in patterns)
        elif cond_type == FailureConditionType.CONTAINS_ALL:
            is_failure = all(any(p in item for item in subset) for p in patterns)
        else:  # CONTAINS_ANY
            is_failure = any(any(p in item for item in subset) for p in patterns)

        if is_failure:
            return TargetResult(
                outcome=TargetOutcome.FAILURE,
                output_message="Declarative failure condition matched active candidate subset.",
                error_code="CONDITION_MATCHED",
            )
        else:
            return TargetResult(
                outcome=TargetOutcome.SUCCESS,
                output_message="Clean execution (no failure condition matched).",
            )

    return evaluate


def format_terminal_card(
    result: DiagnosticResult,
    *,
    title: str = "AgentXRay",
    scenario_name: str | None = None,
    no_color: bool = False,
) -> str:
    """Format an SDK DiagnosticResult into a restrained monochrome terminal card."""
    c = TerminalColors(disabled=no_color)
    box_width = 76
    line_char = "─"

    lines: list[str] = [
        f"{c.bold}{title}{c.reset}" + (f" {c.dim}({scenario_name}){c.reset}" if scenario_name else ""),
        line_char * box_width,
        "",
        f"{c.bold}DIAGNOSIS{c.reset}",
    ]

    if result.status == DiagnosticStatus.VERIFIED_CULPRIT:
        status_label = f"{c.green}CULPRIT ISOLATED & VERIFIED{c.reset}"
        diag_desc = (
            f"The task fails when {len(result.isolated_candidates)} specific candidate "
            f"directive{' is' if len(result.isolated_candidates) == 1 else 's are'} active."
        )
    elif result.status == DiagnosticStatus.NO_FAILURE:
        status_label = f"{c.cyan}NO FAILURE DETECTED{c.reset}"
        diag_desc = "Baseline execution passed cleanly across all candidates. No failure condition to minimize."
    elif result.status == DiagnosticStatus.INCONCLUSIVE:
        status_label = f"{c.yellow}SEARCH INCONCLUSIVE{c.reset}"
        diag_desc = result.summary
    elif result.status == DiagnosticStatus.BUDGET_EXHAUSTED:
        status_label = f"{c.yellow}BUDGET EXHAUSTED{c.reset}"
        diag_desc = result.summary
    elif result.status == DiagnosticStatus.TIMEOUT:
        status_label = f"{c.red}TIMEOUT EXCEEDED{c.reset}"
        diag_desc = result.summary
    else:  # ERROR
        status_label = f"{c.red}EVALUATOR ERROR{c.reset}"
        diag_desc = result.summary

    lines.append(f"  {diag_desc}")
    lines.append("")

    # Isolated condition section
    if result.status == DiagnosticStatus.VERIFIED_CULPRIT:
        lines.append(f"{c.bold}ISOLATED CONDITION{c.reset}")
        for idx, cand in enumerate(result.isolated_candidates, start=1):
            cand_id = result.isolated_candidate_ids[idx - 1] if idx - 1 < len(result.isolated_candidate_ids) else f"cand_{idx}"
            lines.append(f"  {c.cyan}[{cand_id}]{c.reset} {cand}")
        lines.append("")
        lines.append(f"{c.bold}VERIFICATION{c.reset}")
        lines.append(f"  {c.green}Verified:{c.reset} removing this directive restores clean success (1-minimal).")
        lines.append("")
    elif result.status == DiagnosticStatus.NO_FAILURE:
        lines.append(f"{c.bold}ISOLATED CONDITION{c.reset}")
        lines.append("  (None — clean control execution)")
        lines.append("")
        lines.append(f"{c.bold}VERIFICATION{c.reset}")
        lines.append("  Verified: baseline execution passed without any faults.")
        lines.append("")
    elif result.isolated_candidates:
        lines.append(f"{c.bold}CANDIDATE SUBSET (UNVERIFIED){c.reset}")
        for idx, cand in enumerate(result.isolated_candidates, start=1):
            lines.append(f"  [{idx}] {cand}")
        lines.append("")
        lines.append(f"{c.bold}VERIFICATION{c.reset}")
        lines.append(f"  {c.yellow}Unverified:{c.reset} search stopped before 1-minimality could be confirmed.")
        lines.append("")

    # Search telemetry section
    lines.extend([
        f"{c.bold}SEARCH TELEMETRY{c.reset}",
        f"  Status:          {status_label}",
        f"  Evaluations:     {result.evaluations_used} calls ({result.algorithm})",
        f"  Candidate Pool:  {result.initial_candidates_count} directives",
        f"  Execution Time:  {result.execution_time_seconds:.3f}s",
    ])
    if result.matches_ground_truth is not None:
        gt_str = f"{c.green}True (100% match){c.reset}" if result.matches_ground_truth else f"{c.red}False{c.reset}"
        lines.append(f"  Ground Truth:    {gt_str}")

    lines.extend([
        "",
        f"{c.bold}MODE & BOUNDARIES{c.reset}",
        "  Execution:       Controlled offline execution ($0.00 spend, zero network requests)",
        "  Disclosures:     Operates strictly on observable tool-calling inputs/outputs.",
        "                   Does not inspect internal model weights or proprietary prompt logic.",
        "                   Offline testbeds do not evaluate live third-party stochastic noise.",
        line_char * box_width,
    ])

    return "\n".join(lines)


def resolve_exit_code(result: DiagnosticResult) -> int:
    """Map DiagnosticResult status to the documented CLI process exit code contract.

    Exit codes:
      0: Verified culprit isolated (VERIFIED_CULPRIT) or clean control pass (NO_FAILURE).
      1: Inconclusive search, budget exhaustion, or timeout (INCONCLUSIVE, BUDGET_EXHAUSTED, TIMEOUT).
      2: Argument or schema usage error (handled before execution).
      3: Evaluator exception / runtime failure (ERROR).
    """
    if result.status in (DiagnosticStatus.VERIFIED_CULPRIT, DiagnosticStatus.NO_FAILURE):
        return 0
    elif result.status in (DiagnosticStatus.INCONCLUSIVE, DiagnosticStatus.BUDGET_EXHAUSTED, DiagnosticStatus.TIMEOUT):
        return 1
    elif result.status == DiagnosticStatus.ERROR:
        return 3
    return 1


BUILTIN_SCENARIOS = {
    "currency_ast": "Calibrated Currency Formatting (AST arithmetic baseline)",
    "trace_prompt_01": "Tool Prohibition Directive (prompt steering fault)",
    "trace_schema_01": "Schema Parameter Key Mismatch (interface contract fault)",
    "trace_env_01": "Mock Service 500 Outage (backend environment fault)",
    "trace_null_01": "Clean Execution Control (null case / zero planted faults)",
}


def run_builtin_scenario(
    scenario_id: str,
    *,
    seed: int = 42,
    max_evaluations: int = 50,
    timeout_seconds: float = 60.0,
) -> DiagnosticResult:
    """Execute a predefined benchmark scenario through the public SDK."""
    if scenario_id == "currency_ast":
        from benchmarks.model_pilot.tasks import create_model_pilot_calibrated_currency_task

        target, initial_input, ground_truth = create_model_pilot_calibrated_currency_task()
        return diagnose(
            candidates=initial_input.elements,
            evaluator=lambda subset: target.execute(InputCase(elements=subset)),
            seed=seed,
            max_evaluations=max_evaluations,
            timeout_seconds=timeout_seconds,
            ground_truth=ground_truth.trigger_elements,
        )

    from agentxray.web_demo import _make_trace_mock_agent_handler
    from benchmarks.multi_mechanism.tasks import (
        TraceEvaluatedTaskSpec,
        create_trace_evaluated_clean_null_task,
        create_trace_evaluated_environment_task,
        create_trace_evaluated_prompt_task,
        create_trace_evaluated_schema_task,
    )

    if scenario_id == "trace_prompt_01":
        task: TraceEvaluatedTaskSpec = create_trace_evaluated_prompt_task()
        prompt_tr = task.planted_trigger_elements[0] if task.planted_trigger_elements else ""
        handler = _make_trace_mock_agent_handler(prompt_trigger=prompt_tr)
    elif scenario_id == "trace_schema_01":
        task = create_trace_evaluated_schema_task()
        handler = _make_trace_mock_agent_handler()
    elif scenario_id == "trace_env_01":
        task = create_trace_evaluated_environment_task()
        handler = _make_trace_mock_agent_handler()
    elif scenario_id == "trace_null_01":
        task = create_trace_evaluated_clean_null_task()
        handler = _make_trace_mock_agent_handler()
    else:
        valid_ids = ", ".join(repr(k) for k in BUILTIN_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario ID {scenario_id!r}. Supported IDs: {valid_ids}")

    target = task.create_target(provider=FakeProvider(model_name="fake-agent", custom_handler=handler))
    return diagnose(
        candidates=task.candidate_elements,
        evaluator=lambda subset: target.execute(InputCase(elements=subset)),
        seed=seed,
        max_evaluations=max_evaluations,
        timeout_seconds=timeout_seconds,
        ground_truth=task.planted_trigger_elements,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the unified AgentXRay CLI argument parser."""
    version_str = f"agentxray {get_version()}"
    parser = argparse.ArgumentParser(
        prog="agentxray",
        description="AgentXRay: Developer CLI for Causal AI Agent Failure Minimization.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=version_str,
        help="Show program's version number and exit.",
    )

    subparsers = parser.add_subparsers(
        dest="subcommand",
        title="subcommands",
        description="Valid commands:",
        help="Use 'agentxray <subcommand> --help' for detailed arguments.",
    )

    # 1. 'demo' subcommand
    demo_p = subparsers.add_parser(
        "demo",
        help="Run deterministic offline demonstration using public diagnose() API.",
        description="Run an offline AST calculator demonstration proving 1-minimal culprit isolation in 4 evaluations.",
    )
    demo_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for algorithm sampling (default: 42).",
    )
    demo_p.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable ANSI terminal colors.",
    )
    demo_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON diagnostic result instead of terminal card.",
    )
    demo_p.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save report files to disk in output directory.",
    )
    demo_p.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save report files (default: current directory).",
    )
    demo_p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing report files if present.",
    )

    # 2. 'diagnose' subcommand
    diag_p = subparsers.add_parser(
        "diagnose",
        help="Diagnose failure in a scenario JSON file or built-in testbed scenario.",
        description="Execute causal failure diagnosis using Delta Debugging (ddmin) through the public Python SDK.",
    )
    diag_p.add_argument(
        "scenario_file",
        nargs="?",
        default=None,
        help="Path to JSON scenario specification file (documented schema).",
    )
    diag_p.add_argument(
        "--scenario",
        type=str,
        choices=list(BUILTIN_SCENARIOS.keys()),
        default=None,
        help=f"Execute one of the built-in catalog testbed scenarios: {', '.join(BUILTIN_SCENARIOS.keys())}.",
    )
    diag_p.add_argument(
        "--max-evaluations",
        type=int,
        default=50,
        help="Hard cap on target evaluations budget (default: 50).",
    )
    diag_p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Wall-clock timeout in seconds (default: 60.0).",
    )
    diag_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for algorithm sampling (default: 42).",
    )
    diag_p.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable ANSI terminal colors.",
    )
    diag_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON diagnostic result instead of terminal card.",
    )
    diag_p.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save report files to disk in output directory.",
    )
    diag_p.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save report files (default: current directory).",
    )
    diag_p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing report files if present.",
    )

    # 3. 'version' subcommand
    subparsers.add_parser(
        "version",
        help="Report installed AgentXRay package version metadata.",
        description="Report installed AgentXRay package version metadata.",
    )

    # Legacy benchmark fallback arguments on the top-level parser
    parser.add_argument(
        "--mode",
        type=str,
        choices=["deterministic", "stochastic", "audit", "model", "all", "demo"],
        default=None,
        help="Benchmark execution mode for research suites.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["fake", "openai", "ollama"],
        default="fake",
        help="Model provider for model-backed pilot experiments (default: fake).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Explicit authorization flag required to execute live model API calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Pre-flight validation: inspect proposed experiment size and estimated costs.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="Pilot task to run (default: 'all').",
    )
    parser.add_argument(
        "--max-spend",
        type=float,
        default=1.00,
        help="Hard cap on estimated spend in USD (default: 1.00).",
    )
    parser.add_argument(
        "--max-model-requests",
        type=int,
        default=150,
        help="Hard cap on total model generation requests (default: 150).",
    )
    parser.add_argument(
        "--enable-independent-verification",
        action="store_true",
        default=False,
        help="Enable post-discovery independent 1-minimality verification audit.",
    )
    parser.add_argument(
        "--max-discovery-requests",
        type=int,
        default=None,
        help="Hard cap on model generation requests for discovery phase.",
    )
    parser.add_argument(
        "--max-verification-requests",
        type=int,
        default=None,
        help="Hard cap on model generation requests for verification phase.",
    )
    parser.add_argument(
        "--verification-trials",
        type=int,
        default=None,
        help="Number of trials per candidate subset in verification phase.",
    )
    parser.add_argument(
        "--verification-tier",
        type=str,
        choices=["exploratory", "confirmatory"],
        default="exploratory",
        help="Verification evidence tier classification.",
    )
    parser.add_argument(
        "--sample-size-rationale",
        type=str,
        default=None,
        help="Sample size / statistical power rationale.",
    )
    parser.add_argument(
        "--audit-seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated random seeds for multi-seed stochastic audit.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum trial budget per algorithm execution.",
    )
    parser.add_argument(
        "--trials-per-candidate",
        type=int,
        default=None,
        help="Number of trials per candidate subset evaluation.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.30,
        help="Failure proportion threshold.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Statistical significance threshold.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for uncertainty intervals.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Wall-clock timeout in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for algorithm sampling (default: 42).",
    )
    parser.add_argument(
        "--target-seed",
        type=int,
        default=42,
        help="Random seed for synthetic stochastic target outcomes (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output reports (default: current directory).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Explicitly save report files to disk in output-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing report files in output-dir if present.",
    )

    return parser


def cli_main(argv: Sequence[str] | None = None) -> int:
    """Main CLI command runner. Returns process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate verification parameters if present
    if getattr(args, "max_discovery_requests", None) is not None and args.max_discovery_requests <= 0:
        parser.error("--max-discovery-requests must be a positive integer.")
    if getattr(args, "max_verification_requests", None) is not None and args.max_verification_requests <= 0:
        parser.error("--max-verification-requests must be a positive integer.")
    if getattr(args, "verification_trials", None) is not None and args.verification_trials <= 0:
        parser.error("--verification-trials must be a positive integer.")
    if getattr(args, "verification_tier", "exploratory").lower() == "confirmatory":
        if getattr(args, "verification_trials", None) is None:
            parser.error(
                "Confirmatory verification requires an explicitly configured sample size (--verification-trials). "
                "Defaulting to exploratory 5 trials is not permitted for confirmatory claims."
            )

    # 1. Version command
    if args.subcommand == "version":
        print(f"agentxray {get_version()}")
        return 0

    # 2. Demo subcommand or legacy --mode demo
    if args.subcommand == "demo" or args.mode == "demo":
        from agentxray.demo import format_demo_output, run_demo, write_demo_reports

        res = run_demo(seed=args.seed)
        demo_output = format_demo_output(res)
        print(demo_output)

        if getattr(args, "save", False):
            try:
                write_demo_reports(res, output_dir=args.output_dir, overwrite=args.overwrite)
                print(f"\n[Demo reports saved to: {args.output_dir}]")
            except FileExistsError as err:
                print(f"\n[Warning: {err}]")
        return 0

    # 3. Diagnose subcommand
    if args.subcommand == "diagnose":
        if args.scenario_file is None and args.scenario is None:
            print(
                "Error: agentxray diagnose requires either a scenario file path or --scenario <id>.",
                file=sys.stderr,
            )
            print(
                f"Available built-in scenarios: {', '.join(BUILTIN_SCENARIOS.keys())}",
                file=sys.stderr,
            )
            return 2

        if args.scenario_file is not None and args.scenario is not None:
            print(
                "Error: Specify either a scenario file OR --scenario <id>, not both.",
                file=sys.stderr,
            )
            return 2

        try:
            if args.scenario is not None:
                scenario_name = f"built-in: {args.scenario}"
                diag_result = run_builtin_scenario(
                    args.scenario,
                    seed=args.seed,
                    max_evaluations=args.max_evaluations,
                    timeout_seconds=args.timeout,
                )
            else:
                scenario_spec = load_scenario_file(args.scenario_file)
                scenario_name = scenario_spec.name
                evaluator = make_declarative_evaluator(scenario_spec)
                diag_result = diagnose(
                    candidates=scenario_spec.candidates,
                    evaluator=evaluator,
                    seed=args.seed,
                    max_evaluations=args.max_evaluations,
                    timeout_seconds=args.timeout,
                    ground_truth=scenario_spec.ground_truth,
                )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"Diagnostic runtime failure: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3

        if args.json:
            print(diag_result.model_dump_json(indent=2))
        else:
            card = format_terminal_card(
                diag_result,
                scenario_name=scenario_name,
                no_color=args.no_color,
            )
            print(card)

        if args.save:
            out_p = Path(args.output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            json_file = out_p / "diagnosis_result.json"
            if json_file.exists() and not args.overwrite:
                print(
                    f"\n[Warning: Refusing to overwrite existing '{json_file}'. Use --overwrite to replace.]",
                    file=sys.stderr,
                )
            else:
                json_file.write_text(diag_result.model_dump_json(indent=2), encoding="utf-8")
                print(f"\n[Diagnosis saved to: {json_file}]")

        return resolve_exit_code(diag_result)

    # 4. Fallback: if no subcommand and no legacy mode specified, print help
    if args.subcommand is None and args.mode is None:
        parser.print_help()
        return 0

    # 5. Legacy benchmark mode routing
    from agentxray.__main__ import _run_legacy_benchmark_mode

    return _run_legacy_benchmark_mode(args)
