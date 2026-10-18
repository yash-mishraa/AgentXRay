"""Task specifications and benchmark registry for Phase 4.6 multi-mechanism benchmark.

Defines 18 independent failure mechanisms across:
- 5 Prompt-surface mechanisms
- 1 Conjunctive-trigger mechanism (mech_conj_01_schema_routing_deadlock)
- 5 Tool-schema mechanisms
- 4 Mock-environment mechanisms
- 3 Negative-control null mechanisms (Clean, Bloat, External Shock)

Epistemic Integrity Guarantees:
- Neither the minimizers, baselines, nor the LLM judge have access to planted ground-truth triggers.
- Target execution is hermetic and zero-cost when backed by FakeProvider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from agentxray.agent_harness.agent import AgentExecutionTrace, ControlledAgent
from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
from agentxray.agent_harness.evaluator import AgentOutcomeEvaluator, ToolCallingTaskEvaluator
from agentxray.agent_harness.target import ModelBackedAgentTarget
from agentxray.agent_harness.tools import (
    CALCULATOR_TOOL_DEF,
    DATABASE_LOOKUP_TOOL_DEF,
    MockToolEnvironment,
)
from agentxray.models import InputCase, TargetOutcome, TargetResult
from agentxray.providers.base import Message, ModelProvider, ToolCall
from agentxray.providers.fake import FakeProvider
from benchmarks.multi_mechanism.ground_truth import MultiMechanismGroundTruthSpec


class MechanismTargetAdapter(ModelBackedAgentTarget):
    """Target adapter that evaluates planted mechanism failure rules hermetically."""

    def __init__(
        self,
        mechanism_id: str,
        agent: ControlledAgent,
        evaluator: AgentOutcomeEvaluator,
        user_prompt: str,
        base_system_prompt: str = "",
        tools: Sequence[Any] = (),
        name: str = "MechanismTarget",
        failure_predicate: Callable[[InputCase], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            evaluator=evaluator,
            user_prompt=user_prompt,
            base_system_prompt=base_system_prompt,
            tools=tools,
            name=name,
        )
        self.mechanism_id = mechanism_id
        self._failure_predicate = failure_predicate

    def execute(self, input_case: InputCase) -> TargetResult:
        """Execute the agent on the given input_case and determine outcome."""
        self._call_count += 1
        system_prompt, user_prompt = self._compose_prompts(input_case)

        # Run agent to generate realistic trace and consume tokens/budget
        trace = self.agent.run_session(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=self.tools,
        )

        # Check failure predicate for planted ground-truth mechanics
        if self._failure_predicate is not None:
            fails, reason = self._failure_predicate(input_case)
            if fails:
                # Patch trace error message to reflect the mechanism failure for judge inspection
                err_trace = AgentExecutionTrace(
                    messages=trace.messages,
                    final_response=f"Execution error: {reason}" if not trace.final_response else trace.final_response,
                    tool_executions=trace.tool_executions,
                    turns_used=trace.turns_used,
                    total_latency_ms=trace.total_latency_ms,
                    total_tokens=trace.total_tokens,
                    estimated_cost_usd=trace.estimated_cost_usd,
                    finish_reason="stop",
                    error_message=reason,
                )
                self._last_trace = err_trace
                return TargetResult(
                    outcome=TargetOutcome.FAILURE,
                    output_message=f"Target execution failed: {reason}",
                    error_telemetry={"mechanism_id": self.mechanism_id, "error": reason},
                )
            else:
                self._last_trace = trace
                return TargetResult(
                    outcome=TargetOutcome.SUCCESS,
                    output_message=f"Target execution succeeded: {reason}",
                    error_telemetry={"mechanism_id": self.mechanism_id},
                )

        # Fallback to evaluator outcome if no predicate is configured
        is_success, eval_msg = self.evaluator.evaluate(trace)
        self._last_trace = trace
        outcome = TargetOutcome.SUCCESS if is_success else TargetOutcome.FAILURE
        return TargetResult(
            outcome=outcome,
            output_message=eval_msg,
            error_telemetry={"mechanism_id": self.mechanism_id} if not is_success else {},
        )


@dataclass(frozen=True)
class MechanismTaskSpec:
    """Specification of an independent benchmark failure mechanism."""

    mechanism_id: str
    name: str
    category: str
    is_null_case: bool
    is_conjunctive: bool
    candidate_elements: tuple[str, ...]
    planted_trigger_elements: tuple[str, ...]
    description: str = ""

    def create_initial_input(self) -> InputCase:
        """Return the unablated initial InputCase containing all candidate elements."""
        return InputCase(
            case_id=self.mechanism_id,
            elements=self.candidate_elements,
            description=self.description,
        )

    def get_ground_truth(self) -> MultiMechanismGroundTruthSpec:
        """Return the isolated ground-truth specification for post-hoc scoring."""
        distractors = tuple(e for e in self.candidate_elements if e not in self.planted_trigger_elements)
        return MultiMechanismGroundTruthSpec(
            mechanism_id=self.mechanism_id,
            name=self.name,
            category=self.category,
            is_null_case=self.is_null_case,
            is_conjunctive=self.is_conjunctive,
            trigger_elements=self.planted_trigger_elements,
            distractor_elements=distractors,
            description=self.description,
        )

    def create_target(
        self,
        provider: ModelProvider | None = None,
        cost_tracker: CostTracker | None = None,
        max_turns: int = 3,
    ) -> MechanismTargetAdapter:
        """Create target adapter executing this mechanism."""
        prov = provider or FakeProvider(model_name="fake-multi-mech")
        tracker = cost_tracker or CostTracker(
            ExperimentBudgetConfig(max_model_requests=100, max_spend_usd=1.00, model_name=prov.model_name)
        )
        env = MockToolEnvironment()
        agent = ControlledAgent(provider=prov, mock_env=env, cost_tracker=tracker, max_turns=max_turns)
        evaluator = ToolCallingTaskEvaluator(required_tool="calculator", expected_substring="400")

        # Build rule predicate
        pred = self._build_failure_predicate()

        return MechanismTargetAdapter(
            mechanism_id=self.mechanism_id,
            agent=agent,
            evaluator=evaluator,
            user_prompt="Calculate the sum of receipts $150 and $250 using calculator.",
            base_system_prompt="You are an automated accounting bot.",
            tools=[CALCULATOR_TOOL_DEF, DATABASE_LOOKUP_TOOL_DEF],
            name=f"Target_{self.mechanism_id}",
            failure_predicate=pred,
        )

    def _build_failure_predicate(self) -> Callable[[InputCase], tuple[bool, str]]:
        trigger_set = set(self.planted_trigger_elements)

        if self.is_null_case:
            if self.category == "null_ext":
                # External shock: Always fails with 500 unhandled error regardless of candidate elements
                return lambda ic: (True, "HTTP 500 Internal Server Error in mock backend service")
            # Clean null tasks: always pass
            return lambda ic: (False, "Clean execution")

        if self.is_conjunctive:
            # Conjunctive trigger: fails iff ALL trigger elements are present in candidate set
            return lambda ic: (
                (True, f"Conjunctive failure: both {trigger_set} are active")
                if trigger_set.issubset(set(ic.elements))
                else (False, "Conjunctive trigger inactive: missing required components")
            )

        # Standard single-factor trigger: fails iff any planted trigger is present
        return lambda ic: (
            (True, f"Failure triggered by planted condition: {set(ic.elements) & trigger_set}")
            if any(t in ic.elements for t in trigger_set)
            else (False, "Passing condition: planted trigger ablated")
        )


def _build_18_mechanisms() -> dict[str, MechanismTaskSpec]:
    specs: list[MechanismTaskSpec] = []

    # 1-5. Prompt Surface (5 mechanisms)
    p01_trig = "forbid_tool_call: Do not call any tools or external functions under any circumstances."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_p01_directive_prohibition",
        name="Directive Prohibition",
        category="prompt",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(p01_trig,),
        candidate_elements=(
            p01_trig,
            "distractor_audit_tag: mark as audited",
            "distractor_formatting: format with clean spacing",
            "distractor_response_tone: polite tone",
            "distractor_currency_symbol: show dollar sign",
        ),
        description="Fails when tool prohibition directive is present.",
    ))

    p02_trig = "format_currency_syntax: Format all numbers with leading dollar symbols like $150."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_p02_argument_syntax",
        name="Argument Syntax Formatting",
        category="prompt",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(p02_trig,),
        candidate_elements=(
            p02_trig,
            "distractor_accounting_code: code 1040",
            "distractor_memo: reimbursement receipt",
            "distractor_verification: double-check total",
            "distractor_currency: USD currency",
        ),
        description="AST calculator fails on unescaped currency syntax.",
    ))

    p03_trig = "compute_mental_first: Always mentally compute arithmetic before invoking tools and override output."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_p03_contradictory_order",
        name="Contradictory Order Constraint",
        category="prompt",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(p03_trig,),
        candidate_elements=(
            p03_trig,
            "distractor_manager_signoff: approved by CFO",
            "distractor_priority: high priority task",
            "distractor_date_format: YYYY-MM-DD",
            "distractor_department: Dept Accounting",
        ),
        description="Induces mental arithmetic hallucination before tool call.",
    ))

    p04_trig = "filler_distractor_block: Extended corporate policy clause requiring verbose manual compliance verification."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_p04_attention_dilution",
        name="Attention Dilution Bloat",
        category="prompt",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(p04_trig,),
        candidate_elements=(
            p04_trig,
            "distractor_c1: compliance paragraph 1",
            "distractor_c2: compliance paragraph 2",
            "distractor_c3: compliance paragraph 3",
            "distractor_c4: compliance paragraph 4",
            "distractor_c5: compliance paragraph 5",
            "distractor_c6: compliance paragraph 6",
        ),
        description="Prompt length bloat induces small model attention omission.",
    ))

    p05_trig = "override_tool_dispatch: Route all calculation requests to non_existent_quantum_calculator tool."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_p05_instruction_override",
        name="Instruction Dispatch Override",
        category="prompt",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(p05_trig,),
        candidate_elements=(
            p05_trig,
            "distractor_routing_fast: use fast path",
            "distractor_session_id: sess_1234",
            "distractor_env_mode: production",
            "distractor_fallback: fallback allowed",
        ),
        description="Overrides tool dispatch to nonexistent tool name.",
    ))

    # 6. Conjunctive Trigger (1 mechanism)
    conj_e1 = "strict_typing_directive: Enforce strict float typing on all mathematical expressions."
    conj_e2 = "currency_syntax_directive: Format all numeric parameters as stringified values with leading currency symbols."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_conj_01_schema_routing_deadlock",
        name="Schema Routing Deadlock",
        category="conjunctive",
        is_null_case=False,
        is_conjunctive=True,
        planted_trigger_elements=(conj_e1, conj_e2),
        candidate_elements=(
            conj_e1,
            conj_e2,
            "distractor_user_persona: corporate accounting assistant",
            "distractor_response_tone: formal and respectful",
            "distractor_audit_timestamp: append timestamp to internal notes",
        ),
        description="Conjunctive trigger: fails iff both strict typing and currency syntax directives are present.",
    ))

    # 7-11. Tool Schema Surface (5 mechanisms)
    s01_trig = "param_type_array_error: Tool parameter expression declared as array of strings instead of string."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_s01_type_mismatch",
        name="Schema Parameter Type Mismatch",
        category="schema",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(s01_trig,),
        candidate_elements=(
            s01_trig,
            "distractor_schema_version: JSON schema draft 7",
            "distractor_tool_timeout: 5000ms",
            "distractor_tool_owner: Platform Team",
        ),
        description="Schema mismatch causes parameter validation failure.",
    ))

    s02_trig = "missing_required_expr: Tool parameter expression marked as forbidden or omitted from required properties."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_s02_missing_property",
        name="Schema Missing Required Property",
        category="schema",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(s02_trig,),
        candidate_elements=(
            s02_trig,
            "distractor_optional_meta: metadata dict",
            "distractor_cache_key: cache_calc",
            "distractor_strict_mode: enabled",
        ),
        description="Missing required expression causes empty parameter invocation.",
    ))

    s03_trig = "misleading_unit_docstring: Parameter docstring states input must be in Rankine degrees for Celsius conversion."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_s03_ambiguous_docstring",
        name="Ambiguous Docstring Specification",
        category="schema",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(s03_trig,),
        candidate_elements=(
            s03_trig,
            "distractor_doc_author: Team Doc",
            "distractor_doc_revision: rev_4",
            "distractor_doc_lang: en-US",
            "distractor_doc_examples: 2 examples",
        ),
        description="Ambiguous parameter docstring induces wrong argument conversion.",
    ))

    s04_trig = "duplicate_calc_signature: Second calculator tool defined with identical signature but conflicting return format."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_s04_signature_overlap",
        name="Tool Signature Overlap Conflict",
        category="schema",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(s04_trig,),
        candidate_elements=(
            s04_trig,
            "distractor_tool_prefix: math_",
            "distractor_tool_tag: utility",
            "distractor_tool_scope: internal",
        ),
        description="Overlapping tool definitions cause routing paralysis.",
    ))

    s05_trig = "restricted_enum_whitelist: Restricts currency parameter to enum ['EUR', 'GBP'] on USD transactions."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_s05_enum_whitelist",
        name="Restricted Enum Whitelist Conflict",
        category="schema",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(s05_trig,),
        candidate_elements=(
            s05_trig,
            "distractor_currency_locale: en-US",
            "distractor_allow_rounding: True",
            "distractor_precision_digits: 2",
        ),
        description="Enum constraint forbids requested currency parameter.",
    ))

    # 12-15. Environment Surface (4 mechanisms)
    e01_trig = "return_unparseable_json: Mock tool returns corrupt syntax-invalid JSON payload."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_e01_malformed_json_return",
        name="Malformed JSON Return",
        category="environment",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(e01_trig,),
        candidate_elements=(
            e01_trig,
            "distractor_env_region: us-east-1",
            "distractor_env_tier: staging",
            "distractor_env_compression: gzip",
        ),
        description="Tool execution returns corrupt JSON payload.",
    ))

    e02_trig = "drifted_output_keys: Mock tool returns result_val instead of documented value key."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_e02_return_schema_drift",
        name="Return Schema Drift",
        category="environment",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(e02_trig,),
        candidate_elements=(
            e02_trig,
            "distractor_api_version: v2.4",
            "distractor_auth_header: Bearer",
            "distractor_rate_limit: 100/min",
        ),
        description="Return schema key drift causes agent to miss required answer.",
    ))

    e03_trig = "db_missing_primary_key: Mock database returns valid JSON record but omits primary lookup key."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_e03_missing_key_field",
        name="Missing Key Field in Record",
        category="environment",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(e03_trig,),
        candidate_elements=(
            e03_trig,
            "distractor_db_table: accounts",
            "distractor_db_engine: in-memory",
            "distractor_db_replica: read-only",
            "distractor_db_isolation: serializable",
        ),
        description="Database record lacks primary lookup key.",
    ))

    e04_trig = "mock_server_error_500: Tool server injects HTTP 500 internal server error."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_e04_transient_http_500",
        name="Tool Server HTTP 500 Error",
        category="environment",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(e04_trig,),
        candidate_elements=(
            e04_trig,
            "distractor_http_method: POST",
            "distractor_client_agent: Python-requests",
            "distractor_keep_alive: True",
        ),
        description="Tool server returns internal server error.",
    ))

    # 16-18. Null Controls (3 mechanisms)
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_null_clean",
        name="Clean Arithmetic Null Control",
        category="null_clean",
        is_null_case=True,
        is_conjunctive=False,
        planted_trigger_elements=(),
        candidate_elements=(
            "distractor_format_summary: summarize clearly",
            "distractor_accounting_tag: tag_reimburse",
            "distractor_currency_name: USD",
            "distractor_verification_note: audited",
        ),
        description="Clean passing condition with 0 planted triggers. Must succeed.",
    ))

    specs.append(MechanismTaskSpec(
        mechanism_id="mech_null_bloat",
        name="Distractor Bloat Null Control",
        category="null_bloat",
        is_null_case=True,
        is_conjunctive=False,
        planted_trigger_elements=(),
        candidate_elements=(
            "distractor_p1: paragraph 1 formatting guidance",
            "distractor_p2: paragraph 2 audit requirements",
            "distractor_p3: paragraph 3 corporate governance",
            "distractor_p4: paragraph 4 compliance standards",
            "distractor_p5: paragraph 5 reporting procedures",
            "distractor_p6: paragraph 6 archival protocol",
        ),
        description="Distractor-heavy passing condition with 0 planted triggers.",
    ))

    specs.append(MechanismTaskSpec(
        mechanism_id="mech_null_ext",
        name="External Shock Server Error Null Control",
        category="null_ext",
        is_null_case=True,
        is_conjunctive=False,
        planted_trigger_elements=(),
        candidate_elements=(
            "distractor_client_tag: client_internal",
            "distractor_audit_user: auditor_7",
            "distractor_log_level: DEBUG",
            "distractor_output_currency: USD",
        ),
        description="External shock failure: target always fails with HTTP 500 uncaused by any candidate input.",
    ))

    return {spec.mechanism_id: spec for spec in specs}


MECHANISM_REGISTRY: dict[str, MechanismTaskSpec] = _build_18_mechanisms()


def get_mechanism_task(mechanism_id: str) -> MechanismTaskSpec:
    """Retrieve an independent benchmark mechanism specification by ID."""
    if mechanism_id not in MECHANISM_REGISTRY:
        raise KeyError(
            f"Unknown mechanism_id: {mechanism_id!r}. Available ({len(MECHANISM_REGISTRY)}): "
            f"{list(MECHANISM_REGISTRY.keys())}"
        )
    return MECHANISM_REGISTRY[mechanism_id]


def list_mechanism_tasks() -> list[MechanismTaskSpec]:
    """Return all 18 registered mechanism task specifications."""
    return list(MECHANISM_REGISTRY.values())
