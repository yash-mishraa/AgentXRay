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

import random
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

        # Check failure predicate for observable mechanism symptoms
        if self._failure_predicate is not None:
            fails, reason = self._failure_predicate(input_case)
            if fails:
                # Patch trace error message to reflect the observable symptom for judge inspection
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
                    metadata={"error": reason},
                )
            else:
                self._last_trace = trace
                return TargetResult(
                    outcome=TargetOutcome.SUCCESS,
                    output_message=f"Target execution succeeded: {reason}",
                    metadata={},
                )

        # Fallback to evaluator outcome if no predicate is configured
        is_success, eval_msg = self.evaluator.evaluate(trace)
        self._last_trace = trace
        outcome = TargetOutcome.SUCCESS if is_success else TargetOutcome.FAILURE
        return TargetResult(
            outcome=outcome,
            output_message=eval_msg,
            metadata={"error": eval_msg} if not is_success else {},
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
    observable_failure_symptom: str = ""
    description: str = ""

    def create_initial_input(self, seed: int | None = None) -> InputCase:
        """Return the unablated initial InputCase with optional seeded candidate permutation."""
        if seed is None:
            elements = self.candidate_elements
        else:
            rng = random.Random(seed)
            elements = tuple(rng.sample(self.candidate_elements, len(self.candidate_elements)))
        return InputCase(
            case_id=self.mechanism_id,
            elements=elements,
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
                return lambda ic: (
                    True,
                    self.observable_failure_symptom or "ServiceUnavailableError: HTTP 500 Internal Server Error in mock backend service",
                )
            # Clean null tasks: always pass
            return lambda ic: (False, "Clean execution")

        if self.is_conjunctive:
            # Conjunctive trigger: fails iff ALL trigger elements are present in candidate set
            return lambda ic: (
                (
                    True,
                    self.observable_failure_symptom
                    or "SchemaRoutingDeadlock: Parameter expression failed simultaneous typing and currency format constraints",
                )
                if trigger_set.issubset(set(ic.elements))
                else (False, "Passing condition: required conjunctive components not all present")
            )

        # Standard single-factor trigger: fails iff any planted trigger is present
        return lambda ic: (
            (
                True,
                self.observable_failure_symptom
                or "Task execution failed: observable failure criteria met",
            )
            if any(t in ic.elements for t in trigger_set)
            else (False, "Passing condition: failure triggers absent")
        )


def _build_18_mechanisms() -> dict[str, MechanismTaskSpec]:
    specs: list[MechanismTaskSpec] = []

    # 1-5. Prompt Surface (5 mechanisms)
    p01_trig = "tool_dispatch_policy: Do not call any tools or external functions under any circumstances."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_p01_directive_prohibition",
        name="Directive Prohibition",
        category="prompt",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(p01_trig,),
        candidate_elements=(
            p01_trig,
            "audit_logging_policy: Mark all transaction records as verified.",
            "output_format_layout: Format response with clean spacing.",
            "assistant_tone_style: Maintain a polite and professional tone.",
            "currency_symbol_mode: Display monetary values with leading dollar symbols.",
        ),
        observable_failure_symptom="TaskExecutionError: Required tool 'calculator' was not invoked during agent execution.",
        description="Fails when tool prohibition directive is present.",
    ))

    p02_trig = "currency_syntax_rule: Format all numbers with leading dollar symbols like $150."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_p02_argument_syntax",
        name="Argument Syntax Formatting",
        category="prompt",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(p02_trig,),
        candidate_elements=(
            p02_trig,
            "accounting_code_tag: Assign general ledger accounting code 1040.",
            "transaction_memo_text: Attach reimbursement receipt note.",
            "calculation_verification: Double check computed transaction total.",
            "settlement_currency_unit: Use United States Dollar currency standard.",
        ),
        observable_failure_symptom="ToolExecutionError: SyntaxError in calculator: invalid character '$' in numeric expression.",
        description="AST calculator fails on unescaped currency syntax.",
    ))

    p03_trig = "mental_arithmetic_rule: Always mentally compute arithmetic before invoking tools and override output."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_p03_contradictory_order",
        name="Contradictory Order Constraint",
        category="prompt",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(p03_trig,),
        candidate_elements=(
            p03_trig,
            "manager_approval_tag: Record CFO authorization approval in metadata.",
            "priority_queue_level: Process task with high priority routing.",
            "date_format_convention: Represent dates using ISO standard YYYY-MM-DD.",
            "cost_center_department: Route accounting allocation to Finance department.",
        ),
        observable_failure_symptom="TaskEvaluationError: Expected calculation result '400' not found in final response.",
        description="Induces mental arithmetic hallucination before tool call.",
    ))

    p04_trig = "compliance_policy_clause: Extended corporate policy clause requiring verbose manual compliance verification."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_p04_attention_dilution",
        name="Attention Dilution Bloat",
        category="prompt",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(p04_trig,),
        candidate_elements=(
            p04_trig,
            "regulatory_standard_p1: Corporate compliance and ethics guidelines paragraph 1.",
            "regulatory_standard_p2: Treasury oversight and fiduciary duty paragraph 2.",
            "regulatory_standard_p3: Enterprise risk management protocol paragraph 3.",
            "regulatory_standard_p4: Internal accounting internal controls paragraph 4.",
            "regulatory_standard_p5: Financial reporting disclosure checklist paragraph 5.",
            "regulatory_standard_p6: Document retention and archival policy paragraph 6.",
        ),
        observable_failure_symptom="TaskExecutionError: Agent session exceeded turn limit without producing final calculation.",
        description="Prompt length bloat induces small model attention omission.",
    ))

    p05_trig = "tool_routing_override: Route all calculation requests to non_existent_quantum_calculator tool."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_p05_instruction_override",
        name="Instruction Dispatch Override",
        category="prompt",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(p05_trig,),
        candidate_elements=(
            p05_trig,
            "execution_path_setting: Utilize fast path dispatch pipeline.",
            "session_identifier_tag: Track interaction under session tag sess_1234.",
            "environment_tier_flag: Execute under production runtime constraints.",
            "fallback_routing_mode: Permit standard fallback if primary fails.",
        ),
        observable_failure_symptom="ToolDispatchError: Tool 'non_existent_quantum_calculator' is not available in environment.",
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
            "assistant_persona_role: Act as corporate accounting assistant.",
            "communication_style_guide: Maintain formal and respectful tone.",
            "audit_trail_timestamp: Append timestamp metadata to internal audit notes.",
        ),
        observable_failure_symptom="SchemaRoutingDeadlock: Parameter expression failed simultaneous typing and currency format constraints.",
        description="Conjunctive trigger: fails iff both strict typing and currency syntax directives are present.",
    ))

    # 7-11. Tool Schema Surface (5 mechanisms)
    s01_trig = "schema_param_type_spec: Tool parameter expression declared as array of strings instead of string."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_s01_type_mismatch",
        name="Schema Parameter Type Mismatch",
        category="schema",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(s01_trig,),
        candidate_elements=(
            s01_trig,
            "json_schema_version_spec: Adhere to JSON schema draft 7 specifications.",
            "tool_timeout_limit_ms: Enforce execution timeout limit of 5000ms.",
            "service_owner_team_tag: Assign tool maintenance ownership to Platform Team.",
        ),
        observable_failure_symptom="ValidationError: Tool 'calculator' parameter 'expression' expected string, got array.",
        description="Schema mismatch causes parameter validation failure.",
    ))

    s02_trig = "required_property_spec: Tool parameter expression marked as forbidden or omitted from required properties."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_s02_missing_property",
        name="Schema Missing Required Property",
        category="schema",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(s02_trig,),
        candidate_elements=(
            s02_trig,
            "metadata_field_schema: Include optional metadata dictionary definition.",
            "calculation_cache_key: Use cache key cache_calc for memoization.",
            "validation_strictness_mode: Enable strict validation checking.",
        ),
        observable_failure_symptom="ValidationError: Tool 'calculator' missing required parameter 'expression'.",
        description="Missing required expression causes empty parameter invocation.",
    ))

    s03_trig = "parameter_docstring_unit: Parameter docstring states input must be in Rankine degrees for Celsius conversion."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_s03_ambiguous_docstring",
        name="Ambiguous Docstring Specification",
        category="schema",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(s03_trig,),
        candidate_elements=(
            s03_trig,
            "documentation_author_tag: Document authored by Platform Documentation Team.",
            "documentation_revision_id: Document updated under revision tracking rev_4.",
            "documentation_language_code: Document locale set to English US.",
            "documentation_example_count: Provide 2 usage examples in docstring.",
        ),
        observable_failure_symptom="TaskEvaluationError: Calculation result '-459.67' did not match expected value '400'.",
        description="Ambiguous parameter docstring induces wrong argument conversion.",
    ))

    s04_trig = "duplicate_tool_definition: Second calculator tool defined with identical signature but conflicting return format."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_s04_signature_overlap",
        name="Tool Signature Overlap Conflict",
        category="schema",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(s04_trig,),
        candidate_elements=(
            s04_trig,
            "tool_namespace_prefix: Use namespace prefix math_ for utility functions.",
            "tool_registry_category: Categorize tool under financial utility functions.",
            "tool_visibility_scope: Restrict tool accessibility to internal platform.",
        ),
        observable_failure_symptom="ToolRoutingError: Ambiguous tool call resolution between duplicate tool declarations.",
        description="Overlapping tool definitions cause routing paralysis.",
    ))

    s05_trig = "currency_enum_constraint: Restricts currency parameter to enum ['EUR', 'GBP'] on USD transactions."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_s05_enum_whitelist",
        name="Restricted Enum Whitelist Conflict",
        category="schema",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(s05_trig,),
        candidate_elements=(
            s05_trig,
            "locale_format_setting: Format currency according to en-US standard.",
            "rounding_precision_rule: Allow half-up rounding precision.",
            "decimal_places_limit: Format fractional amounts to 2 decimal places.",
        ),
        observable_failure_symptom="ValidationError: Value 'USD' is not a member of allowed currency enum ['EUR', 'GBP'].",
        description="Enum constraint forbids requested currency parameter.",
    ))

    # 12-15. Environment Surface (4 mechanisms)
    e01_trig = "json_serializer_syntax: Mock tool returns corrupt syntax-invalid JSON payload."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_e01_malformed_json_return",
        name="Malformed JSON Return",
        category="environment",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(e01_trig,),
        candidate_elements=(
            e01_trig,
            "backend_region_code: Deploy service in region us-east-1.",
            "deployment_environment_tier: Target staging environment runtime.",
            "payload_compression_type: Compress payload using standard gzip.",
        ),
        observable_failure_symptom="ToolExecutionError: Failed to parse tool response as valid JSON: JSONDecodeError at position 0.",
        description="Tool execution returns corrupt JSON payload.",
    ))

    e02_trig = "output_key_mapping: Mock tool returns result_val instead of documented value key."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_e02_return_schema_drift",
        name="Return Schema Drift",
        category="environment",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(e02_trig,),
        candidate_elements=(
            e02_trig,
            "api_version_specification: Expose endpoint under api version v2.4.",
            "authorization_scheme_header: Authenticate requests using Bearer token.",
            "client_rate_limit_policy: Throttle client calls at 100 per minute.",
        ),
        observable_failure_symptom="KeyError: Missing expected key 'value' in tool execution response payload.",
        description="Return schema key drift causes agent to miss required answer.",
    ))

    e03_trig = "record_primary_key_rule: Mock database returns valid JSON record but omits primary lookup key."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_e03_missing_key_field",
        name="Missing Key Field in Record",
        category="environment",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(e03_trig,),
        candidate_elements=(
            e03_trig,
            "database_table_target: Query records from accounts table.",
            "storage_engine_type: Execute queries using in-memory engine.",
            "database_replica_policy: Route read queries to read-only replica.",
            "transaction_isolation_level: Enforce serializable transaction isolation.",
        ),
        observable_failure_symptom="DatabaseError: Record retrieved from table 'accounts' is missing primary key 'id'.",
        description="Database record lacks primary lookup key.",
    ))

    e04_trig = "server_fault_injection: Tool server injects HTTP 500 internal server error."
    specs.append(MechanismTaskSpec(
        mechanism_id="mech_e04_transient_http_500",
        name="Tool Server HTTP 500 Error",
        category="environment",
        is_null_case=False,
        is_conjunctive=False,
        planted_trigger_elements=(e04_trig,),
        candidate_elements=(
            e04_trig,
            "http_request_method: Issue tool dispatch over HTTP POST method.",
            "client_user_agent_header: Identify client as Python-requests.",
            "connection_keep_alive: Maintain persistent connection keep alive.",
        ),
        observable_failure_symptom="ServiceUnavailableError: HTTP 500 Internal Server Error returned by mock service.",
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
            "summary_reporting_format: Format arithmetic summary clearly.",
            "accounting_entry_tag: Assign tag tag_reimburse to ledger entry.",
            "currency_unit_label: Display amounts with currency symbol USD.",
            "audit_signoff_verification: Record transaction as verified by accounting.",
        ),
        observable_failure_symptom="",
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
            "presentation_layout_p1: Paragraph 1 formatting and layout guidelines.",
            "audit_verification_p2: Paragraph 2 audit inspection requirements.",
            "corporate_governance_p3: Paragraph 3 corporate governance protocols.",
            "compliance_framework_p4: Paragraph 4 financial compliance standards.",
            "reporting_procedure_p5: Paragraph 5 reporting and reconciliation procedures.",
            "archival_protocol_p6: Paragraph 6 document archival and retention rules.",
        ),
        observable_failure_symptom="",
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
            "client_application_tag: Internal client application identifier.",
            "audit_user_account: Record auditing user account auditor_7.",
            "telemetry_log_level: Configure telemetry logging level to DEBUG.",
            "default_settlement_currency: Default settlement currency configured to USD.",
        ),
        observable_failure_symptom="ServiceUnavailableError: HTTP 500 Internal Server Error in mock backend service.",
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
