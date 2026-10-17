"""Confirmatory Benchmark Task Registry for Phase 2B.1 Experiments.

Defines 16 independently specified, calibrated experimental task instances
across 4 distinct task families and two candidate-space sizes (K = 4, K = 8):
- Family 1: Tool-Argument Constraints (TOOL_ARG_SYNTAX)
- Family 2: Instruction Interaction & Routing Overrides (TOOL_ROUTING_CONFLICT)
- Family 3: Tool Parameter Schema Violations (TOOL_PARAM_SCHEMA)
- Family 4: Null Controls with Zero Planted Triggers (NULL_CONTROL)

Epistemic Integrity Guarantees:
- Every instance defines observable boundary criteria for evaluation.
- Neither the evaluator nor the minimizer inspects true trigger definitions during search.
- All distractors are designed to be orthogonal and non-conflicting (avoiding length/format truncation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from agentxray.agent_harness.agent import ControlledAgent
from agentxray.agent_harness.cost_tracker import CostTracker, ExperimentBudgetConfig
from agentxray.agent_harness.evaluator import ToolCallingTaskEvaluator
from agentxray.agent_harness.target import ModelBackedAgentTarget
from agentxray.agent_harness.tools import (
    CALCULATOR_TOOL_DEF,
    DATABASE_LOOKUP_TOOL_DEF,
    UNIT_CONVERTER_TOOL_DEF,
    MockToolEnvironment,
)
from agentxray.models import InputCase
from agentxray.providers.base import ModelProvider, ToolDefinition


class TaskFamily(str, Enum):
    TOOL_ARG_SYNTAX = "TOOL_ARG_SYNTAX"
    TOOL_ROUTING_CONFLICT = "TOOL_ROUTING_CONFLICT"
    TOOL_PARAM_SCHEMA = "TOOL_PARAM_SCHEMA"
    NULL_CONTROL = "NULL_CONTROL"


@dataclass(frozen=True)
class ConfirmatoryTaskSpec:
    """Specification of an independent confirmatory task instance."""

    instance_id: str
    family: TaskFamily
    k: int
    name: str
    description: str
    is_null_case: bool
    candidate_elements: tuple[str, ...]
    planted_trigger_elements: tuple[str, ...]
    base_system_prompt: str
    user_prompt: str
    tool_names: tuple[str, ...]
    required_tool: str | None
    expected_substring: str | None
    expected_failure_classification: str
    leakage_protection: str = "Evaluator and minimizers have zero access to planted_trigger_elements."

    def create_initial_input(self) -> InputCase:
        """Create the unablated initial InputCase containing all K candidate elements."""
        return InputCase(
            case_id=self.instance_id,
            elements=self.candidate_elements,
            description=self.description,
        )

    def create_target(
        self,
        provider: ModelProvider,
        cost_tracker: CostTracker,
        max_turns: int = 3,
    ) -> ModelBackedAgentTarget:
        """Instantiate the black-box target adapter for this task instance."""
        env = MockToolEnvironment()
        agent = ControlledAgent(
            provider=provider,
            mock_env=env,
            cost_tracker=cost_tracker,
            max_turns=max_turns,
        )
        evaluator = ToolCallingTaskEvaluator(
            required_tool=self.required_tool,
            expected_substring=self.expected_substring,
            target_failure_class=self.expected_failure_classification if not self.is_null_case else None,
        )
        tools = env.get_definitions(self.tool_names)
        return ModelBackedAgentTarget(
            agent=agent,
            evaluator=evaluator,
            base_system_prompt=self.base_system_prompt,
            user_prompt=self.user_prompt,
            tools=tools,
            name=f"Target_{self.instance_id}",
        )


# ==============================================================================
# Orthogonal Distractor Pools
# ==============================================================================
_DISTRACTORS_POOL: list[str] = [
    "expense_category: specify expense category as Travel & Subsistence",
    "compliance_tag: include audit reference code REF-2026",
    "audit_memo: include verification timestamp in final audit notes",
    "currency_label: mention USD currency denomination in final remarks",
    "department_code: assign to department code DEPT-FIN-01",
    "reviewer_initials: append compliance reviewer initials CR-ADM",
    "urgency_level: designate processing priority as standard batch",
    "archival_flag: flag session for standard 7-year records retention",
]


# ==============================================================================
# 16 Confirmatory Task Instance Definitions
# ==============================================================================

def _build_task_registry() -> dict[str, ConfirmatoryTaskSpec]:
    tasks: dict[str, ConfirmatoryTaskSpec] = {}

    # --------------------------------------------------------------------------
    # Family 1: Tool-Argument Constraints (TOOL_ARG_SYNTAX)
    # --------------------------------------------------------------------------

    # Instance 1: Currency syntax rejection (K=4) [Validated in Phase 2B.0.2]
    t1_trig = 'strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. "$150" or "$250")'
    t1_cands = (t1_trig, _DISTRACTORS_POOL[0], _DISTRACTORS_POOL[1], _DISTRACTORS_POOL[2])
    tasks["inst_01_syntax_currency_k4"] = ConfirmatoryTaskSpec(
        instance_id="inst_01_syntax_currency_k4",
        family=TaskFamily.TOOL_ARG_SYNTAX,
        k=4,
        name="Currency Formatting Syntax Rejection (K=4)",
        description="Currency symbols in arithmetic expressions cause calculator AST syntax failure.",
        is_null_case=False,
        candidate_elements=t1_cands,
        planted_trigger_elements=(t1_trig,),
        base_system_prompt="You are an automated accounting assistant. Always use the calculator tool for calculations.",
        user_prompt="Calculate total reimbursement for expenses of 150 and 250.",
        tool_names=("calculator",),
        required_tool="calculator",
        expected_substring="400",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # Instance 2: Currency syntax rejection (K=8)
    t2_cands = (t1_trig, *_DISTRACTORS_POOL[:7])
    tasks["inst_02_syntax_currency_k8"] = ConfirmatoryTaskSpec(
        instance_id="inst_02_syntax_currency_k8",
        family=TaskFamily.TOOL_ARG_SYNTAX,
        k=8,
        name="Currency Formatting Syntax Rejection (K=8)",
        description="Currency symbols in arithmetic expressions cause calculator AST syntax failure in a larger distractor space.",
        is_null_case=False,
        candidate_elements=t2_cands,
        planted_trigger_elements=(t1_trig,),
        base_system_prompt="You are an automated accounting assistant. Always use the calculator tool for calculations.",
        user_prompt="Calculate total reimbursement for expenses of 150 and 250.",
        tool_names=("calculator",),
        required_tool="calculator",
        expected_substring="400",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # Instance 3: Comma thousands separator syntax rejection (K=4)
    t3_trig = 'thousands_separator: format all numbers inside arithmetic expressions with comma thousands separators (e.g. "1,500 + 2,500")'
    t3_cands = (t3_trig, _DISTRACTORS_POOL[0], _DISTRACTORS_POOL[3], _DISTRACTORS_POOL[4])
    tasks["inst_03_syntax_comma_k4"] = ConfirmatoryTaskSpec(
        instance_id="inst_03_syntax_comma_k4",
        family=TaskFamily.TOOL_ARG_SYNTAX,
        k=4,
        name="Comma Thousands Separator Syntax Rejection (K=4)",
        description="Comma separators in arithmetic expressions cause calculator AST syntax failure.",
        is_null_case=False,
        candidate_elements=t3_cands,
        planted_trigger_elements=(t3_trig,),
        base_system_prompt="You are a financial operations agent. Always compute arithmetic totals using the calculator tool.",
        user_prompt="Calculate the total budget allocation for line items of 1500 and 2500.",
        tool_names=("calculator",),
        required_tool="calculator",
        expected_substring="4000",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # Instance 4: Comma thousands separator syntax rejection (K=8)
    t4_cands = (t3_trig, *_DISTRACTORS_POOL[:7])
    tasks["inst_04_syntax_comma_k8"] = ConfirmatoryTaskSpec(
        instance_id="inst_04_syntax_comma_k8",
        family=TaskFamily.TOOL_ARG_SYNTAX,
        k=8,
        name="Comma Thousands Separator Syntax Rejection (K=8)",
        description="Comma separators in arithmetic expressions cause calculator AST syntax failure under K=8 distractors.",
        is_null_case=False,
        candidate_elements=t4_cands,
        planted_trigger_elements=(t3_trig,),
        base_system_prompt="You are a financial operations agent. Always compute arithmetic totals using the calculator tool.",
        user_prompt="Calculate the total budget allocation for line items of 1500 and 2500.",
        tool_names=("calculator",),
        required_tool="calculator",
        expected_substring="4000",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # --------------------------------------------------------------------------
    # Family 2: Instruction Interaction & Routing Overrides (TOOL_ROUTING_CONFLICT)
    # --------------------------------------------------------------------------

    # Instance 5: Field override causing missing database field error (K=4)
    t5_trig = "override_field_name: set query field parameter to non_existent_secret_key"
    t5_cands = (t5_trig, _DISTRACTORS_POOL[1], _DISTRACTORS_POOL[4], _DISTRACTORS_POOL[5])
    tasks["inst_05_routing_override_k4"] = ConfirmatoryTaskSpec(
        instance_id="inst_05_routing_override_k4",
        family=TaskFamily.TOOL_ROUTING_CONFLICT,
        k=4,
        name="Database Field Name Override Error (K=4)",
        description="Field override directive forces lookup of non-existent field, triggering database error.",
        is_null_case=False,
        candidate_elements=t5_cands,
        planted_trigger_elements=(t5_trig,),
        base_system_prompt="You are an audit database assistant. Query user records using the database_lookup tool.",
        user_prompt="Look up the department for user_42 in the database.",
        tool_names=("database_lookup",),
        required_tool="database_lookup",
        expected_substring="finance",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # Instance 6: Field override causing missing database field error (K=8)
    t6_cands = (t5_trig, *_DISTRACTORS_POOL[:7])
    tasks["inst_06_routing_override_k8"] = ConfirmatoryTaskSpec(
        instance_id="inst_06_routing_override_k8",
        family=TaskFamily.TOOL_ROUTING_CONFLICT,
        k=8,
        name="Database Field Name Override Error (K=8)",
        description="Field override directive forces lookup of non-existent field under K=8 distractors.",
        is_null_case=False,
        candidate_elements=t6_cands,
        planted_trigger_elements=(t5_trig,),
        base_system_prompt="You are an audit database assistant. Query user records using the database_lookup tool.",
        user_prompt="Look up the department for user_42 in the database.",
        tool_names=("database_lookup",),
        required_tool="database_lookup",
        expected_substring="finance",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # Instance 7: Prohibit external tool execution constraint (K=4)
    t7_trig = "prohibit_tools: never invoke external tools; always compute results mentally without calling functions"
    t7_cands = (t7_trig, _DISTRACTORS_POOL[0], _DISTRACTORS_POOL[2], _DISTRACTORS_POOL[6])
    tasks["inst_07_routing_prohibition_k4"] = ConfirmatoryTaskSpec(
        instance_id="inst_07_routing_prohibition_k4",
        family=TaskFamily.TOOL_ROUTING_CONFLICT,
        k=4,
        name="Tool Prohibition Constraint (K=4)",
        description="Constraint explicitly forbidding tool calls causes failure to invoke required calculator.",
        is_null_case=False,
        candidate_elements=t7_cands,
        planted_trigger_elements=(t7_trig,),
        base_system_prompt="You are an accounting verification agent. Mandatory policy requires calling the calculator tool.",
        user_prompt="Calculate the total reimbursement for 150 and 250 using the calculator tool.",
        tool_names=("calculator",),
        required_tool="calculator",
        expected_substring="400",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # Instance 8: Prohibit external tool execution constraint (K=8)
    t8_cands = (t7_trig, *_DISTRACTORS_POOL[:7])
    tasks["inst_08_routing_prohibition_k8"] = ConfirmatoryTaskSpec(
        instance_id="inst_08_routing_prohibition_k8",
        family=TaskFamily.TOOL_ROUTING_CONFLICT,
        k=8,
        name="Tool Prohibition Constraint (K=8)",
        description="Constraint forbidding tool calls causes failure to invoke required calculator in K=8 candidate space.",
        is_null_case=False,
        candidate_elements=t8_cands,
        planted_trigger_elements=(t7_trig,),
        base_system_prompt="You are an accounting verification agent. Mandatory policy requires calling the calculator tool.",
        user_prompt="Calculate the total reimbursement for 150 and 250 using the calculator tool.",
        tool_names=("calculator",),
        required_tool="calculator",
        expected_substring="400",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # --------------------------------------------------------------------------
    # Family 3: Tool Parameter Schema Violations (TOOL_PARAM_SCHEMA)
    # --------------------------------------------------------------------------

    # Instance 9: Unit converter incompatible target unit (K=4)
    t9_trig = "incompatible_unit: specify target unit to_unit as light_years"
    t9_cands = (t9_trig, _DISTRACTORS_POOL[0], _DISTRACTORS_POOL[1], _DISTRACTORS_POOL[7])
    tasks["inst_09_schema_unit_k4"] = ConfirmatoryTaskSpec(
        instance_id="inst_09_schema_unit_k4",
        family=TaskFamily.TOOL_PARAM_SCHEMA,
        k=4,
        name="Unsupported Unit Conversion Parameter (K=4)",
        description="Directive forcing an unsupported unit causes unit_converter to return an error.",
        is_null_case=False,
        candidate_elements=t9_cands,
        planted_trigger_elements=(t9_trig,),
        base_system_prompt="You are a logistics measurement assistant. Always convert units using the unit_converter tool.",
        user_prompt="Convert 5 kilometers to miles using the unit_converter tool.",
        tool_names=("unit_converter",),
        required_tool="unit_converter",
        expected_substring="3.1069",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # Instance 10: Unit converter incompatible target unit (K=8)
    t10_cands = (t9_trig, *_DISTRACTORS_POOL[:7])
    tasks["inst_10_schema_unit_k8"] = ConfirmatoryTaskSpec(
        instance_id="inst_10_schema_unit_k8",
        family=TaskFamily.TOOL_PARAM_SCHEMA,
        k=8,
        name="Unsupported Unit Conversion Parameter (K=8)",
        description="Directive forcing an unsupported unit causes unit_converter to return an error under K=8 distractors.",
        is_null_case=False,
        candidate_elements=t10_cands,
        planted_trigger_elements=(t9_trig,),
        base_system_prompt="You are a logistics measurement assistant. Always convert units using the unit_converter tool.",
        user_prompt="Convert 5 kilometers to miles using the unit_converter tool.",
        tool_names=("unit_converter",),
        required_tool="unit_converter",
        expected_substring="3.1069",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # Instance 11: Incompatible unit category (mass to length) (K=4)
    t11_trig = "cross_category_unit: convert mass value into length unit meters"
    t11_cands = (t11_trig, _DISTRACTORS_POOL[2], _DISTRACTORS_POOL[4], _DISTRACTORS_POOL[6])
    tasks["inst_11_schema_cross_unit_k4"] = ConfirmatoryTaskSpec(
        instance_id="inst_11_schema_cross_unit_k4",
        family=TaskFamily.TOOL_PARAM_SCHEMA,
        k=4,
        name="Cross-Category Unit Conversion Parameter (K=4)",
        description="Directive attempting mass to length conversion triggers incompatible unit error.",
        is_null_case=False,
        candidate_elements=t11_cands,
        planted_trigger_elements=(t11_trig,),
        base_system_prompt="You are a cargo compliance bot. Convert weight using unit_converter.",
        user_prompt="Convert 10 kilograms to pounds using unit_converter.",
        tool_names=("unit_converter",),
        required_tool="unit_converter",
        expected_substring="22.0462",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # Instance 12: Cross-category unit conversion (K=8)
    t12_cands = (t11_trig, *_DISTRACTORS_POOL[:7])
    tasks["inst_12_schema_cross_unit_k8"] = ConfirmatoryTaskSpec(
        instance_id="inst_12_schema_cross_unit_k8",
        family=TaskFamily.TOOL_PARAM_SCHEMA,
        k=8,
        name="Cross-Category Unit Conversion Parameter (K=8)",
        description="Cross-category conversion trigger under K=8 candidate distractors.",
        is_null_case=False,
        candidate_elements=t12_cands,
        planted_trigger_elements=(t11_trig,),
        base_system_prompt="You are a cargo compliance bot. Convert weight using unit_converter.",
        user_prompt="Convert 10 kilograms to pounds using unit_converter.",
        tool_names=("unit_converter",),
        required_tool="unit_converter",
        expected_substring="22.0462",
        expected_failure_classification="FAIL_TOOL_EXEC",
    )

    # --------------------------------------------------------------------------
    # Family 4: Null Control Tasks (NULL_CONTROL)
    # --------------------------------------------------------------------------

    # Instance 13: Null Calculator Task (K=4)
    tasks["inst_13_null_calc_k4"] = ConfirmatoryTaskSpec(
        instance_id="inst_13_null_calc_k4",
        family=TaskFamily.NULL_CONTROL,
        k=4,
        name="Null Control Arithmetic (K=4)",
        description="Negative control with 4 benign distractors and zero planted triggers.",
        is_null_case=True,
        candidate_elements=tuple(_DISTRACTORS_POOL[:4]),
        planted_trigger_elements=(),
        base_system_prompt="You are an automated accounting assistant. Always use the calculator tool for calculations.",
        user_prompt="Calculate total reimbursement for expenses of 150 and 250.",
        tool_names=("calculator",),
        required_tool="calculator",
        expected_substring="400",
        expected_failure_classification="CLEAN_PASS",
    )

    # Instance 14: Null Calculator Task (K=8)
    tasks["inst_14_null_calc_k8"] = ConfirmatoryTaskSpec(
        instance_id="inst_14_null_calc_k8",
        family=TaskFamily.NULL_CONTROL,
        k=8,
        name="Null Control Arithmetic (K=8)",
        description="Negative control with 8 benign distractors and zero planted triggers.",
        is_null_case=True,
        candidate_elements=tuple(_DISTRACTORS_POOL[:8]),
        planted_trigger_elements=(),
        base_system_prompt="You are an automated accounting assistant. Always use the calculator tool for calculations.",
        user_prompt="Calculate total reimbursement for expenses of 150 and 250.",
        tool_names=("calculator",),
        required_tool="calculator",
        expected_substring="400",
        expected_failure_classification="CLEAN_PASS",
    )

    # Instance 15: Null Database Task (K=4)
    tasks["inst_15_null_db_k4"] = ConfirmatoryTaskSpec(
        instance_id="inst_15_null_db_k4",
        family=TaskFamily.NULL_CONTROL,
        k=4,
        name="Null Control Database Lookup (K=4)",
        description="Negative control with 4 benign distractors and zero planted triggers for database queries.",
        is_null_case=True,
        candidate_elements=tuple(_DISTRACTORS_POOL[4:8]),
        planted_trigger_elements=(),
        base_system_prompt="You are an audit database assistant. Query user records using database_lookup.",
        user_prompt="Look up the department for user_42 in the database.",
        tool_names=("database_lookup",),
        required_tool="database_lookup",
        expected_substring="finance",
        expected_failure_classification="CLEAN_PASS",
    )

    # Instance 16: Null Database Task (K=8)
    tasks["inst_16_null_db_k8"] = ConfirmatoryTaskSpec(
        instance_id="inst_16_null_db_k8",
        family=TaskFamily.NULL_CONTROL,
        k=8,
        name="Null Control Database Lookup (K=8)",
        description="Negative control with 8 benign distractors and zero planted triggers for database queries.",
        is_null_case=True,
        candidate_elements=tuple(_DISTRACTORS_POOL[:8]),
        planted_trigger_elements=(),
        base_system_prompt="You are an audit database assistant. Query user records using database_lookup.",
        user_prompt="Look up the department for user_42 in the database.",
        tool_names=("database_lookup",),
        required_tool="database_lookup",
        expected_substring="finance",
        expected_failure_classification="CLEAN_PASS",
    )

    return tasks


CONFIRMATORY_TASK_REGISTRY: dict[str, ConfirmatoryTaskSpec] = _build_task_registry()


def get_confirmatory_task(instance_id: str) -> ConfirmatoryTaskSpec:
    """Retrieve an independent confirmatory task specification by instance_id."""
    if instance_id not in CONFIRMATORY_TASK_REGISTRY:
        raise KeyError(
            f"Unknown confirmatory task instance_id: {instance_id!r}. "
            f"Available instances ({len(CONFIRMATORY_TASK_REGISTRY)}): {list(CONFIRMATORY_TASK_REGISTRY.keys())}"
        )
    return CONFIRMATORY_TASK_REGISTRY[instance_id]


def list_confirmatory_tasks() -> list[ConfirmatoryTaskSpec]:
    """Return all 16 registered confirmatory task specifications."""
    return list(CONFIRMATORY_TASK_REGISTRY.values())
