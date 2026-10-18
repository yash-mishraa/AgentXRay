"""Passive LLM-as-a-judge baseline for AgentXRay.

Evaluates multi-turn agent execution traces in a single shot without interactive probing,
adhering strictly to the Phase 4.6 protocol.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from agentxray.agent_harness.agent import AgentExecutionTrace
from agentxray.agent_harness.cost_tracker import CostBudgetExceededError, CostTracker
from agentxray.models import DiagnosticOutcome, InputCase, LLMJudgeComparison, LLMJudgeResult
from agentxray.providers.base import Message, ModelProvider, ModelResponse, UsageMetadata


def _extract_json_substring(text: str) -> str:
    """Extract a candidate JSON substring from raw model output text."""
    text_clean = text.strip()

    # 1. Check for markdown fenced blocks ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_clean, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Check for outermost balanced curly braces { ... }
    first_brace = text_clean.find("{")
    last_brace = text_clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text_clean[first_brace : last_brace + 1].strip()

    return text_clean


def _build_candidate_id_maps(elements: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional candidate ID and alias mappings."""
    id_to_full: dict[str, str] = {}
    display_list: dict[str, str] = {}

    for idx, elem in enumerate(elements):
        auto_id = f"elem_{idx}"
        id_to_full[auto_id] = elem
        id_to_full[elem] = elem

        # If element has a prefix like "forbid_tool_call: ...", map the prefix
        if ":" in elem:
            prefix = elem.split(":", 1)[0].strip()
            if prefix:
                id_to_full[prefix] = elem
                display_id = prefix
            else:
                display_id = auto_id
        else:
            display_id = auto_id

        id_to_full[display_id] = elem
        display_list[display_id] = elem

    return id_to_full, display_list


def _format_trace_for_prompt(trace: AgentExecutionTrace) -> str:
    """Format an agent execution trace into human-readable, structured prompt text."""
    lines: list[str] = [
        f"Terminal Finish Reason: {trace.finish_reason}",
        f"Error Diagnostic: {trace.error_message or 'None'}",
        f"Turns Executed: {trace.turns_used}",
    ]

    if trace.tool_executions:
        lines.append("Executed Tool Invocations:")
        for rec in trace.tool_executions:
            lines.append(f"  - Tool: {rec.tool_name}({rec.arguments}) -> Output: {rec.output}")
    else:
        lines.append("Executed Tool Invocations: None")

    if trace.messages:
        lines.append("Conversation Excerpt:")
        for m in trace.messages:
            lines.append(f"  [{m.role.upper()}]: {m.content}")

    lines.append(f"Final Response: {trace.final_response or 'None'}")
    return "\n".join(lines)


class LLMJudgeBaseline:
    """Passive LLM-as-a-judge baseline for fault localization."""

    def __init__(
        self,
        provider: ModelProvider,
        cost_tracker: CostTracker | None = None,
        max_retries: int = 2,
    ) -> None:
        self.provider = provider
        self.cost_tracker = cost_tracker
        self.max_retries = max_retries  # At most 1 initial + max_retries retries = 3 total attempts

    def diagnose(
        self,
        initial_input: InputCase,
        execution_trace: AgentExecutionTrace,
    ) -> LLMJudgeResult:
        """Diagnose minimal failure-triggering elements from an execution trace in a single-shot prompt."""
        id_to_full, display_map = _build_candidate_id_maps(initial_input.elements)

        # 1. System Prompt
        system_content = (
            "You are an expert AI systems debugger and root-cause fault localization assistant.\n"
            "Your task is to identify which candidate input elements caused an AI agent to fail its task.\n\n"
            "Rules:\n"
            "1. Analyze the observed agent execution trace.\n"
            "2. Identify the MINIMAL subset of candidate element IDs that caused the failure.\n"
            "3. If the agent execution succeeded, or if no candidate element caused the failure (for example, "
            "if the failure was caused by an external unmodeled server error), return an empty list: [].\n"
            "4. Do not blame non-causal formatting or distractor clauses.\n"
            "5. Output MUST be valid JSON adhering strictly to:\n"
            '   {"culprit_element_ids": ["<id>", ...], "rationale": "<explanation>"}'
        )

        # 2. Initial User Prompt
        candidate_lines = [f'- id: "{cid}" | description: "{desc}"' for cid, desc in display_map.items()]
        candidates_text = "\n".join(candidate_lines)
        trace_text = _format_trace_for_prompt(execution_trace)

        initial_user_content = (
            f"Candidate Input Elements:\n{candidates_text}\n\n"
            f"Observed Agent Execution Trace:\n{trace_text}\n\n"
            'Please identify the culprit candidate element IDs. Respond ONLY with the JSON object.'
        )

        messages: list[Message] = [
            Message(role="system", content=system_content),
            Message(role="user", content=initial_user_content),
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        raw_responses: list[str] = []
        max_attempts = 1 + self.max_retries

        for attempt in range(1, max_attempts + 1):
            # Budget guard before request
            if self.cost_tracker is not None:
                try:
                    self.cost_tracker.check_budget_before_request(est_prompt_tokens=250)
                except CostBudgetExceededError:
                    return LLMJudgeResult(
                        predicted_candidate_set=None,
                        is_valid_diagnosis=False,
                        attempts_used=attempt - 1,
                        termination_reason="budget_exhausted",
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        estimated_cost_usd=self.cost_tracker.estimated_spend_usd if self.cost_tracker else 0.0,
                        rationale="Budget exhausted before request",
                        raw_responses=tuple(raw_responses),
                    )

            # Generate response from model provider
            response = self.provider.generate(
                messages=messages,
                tools=(),
                temperature=0.0,
                max_tokens=512,
                max_retries=1,
            )

            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            raw_text = response.content or ""
            raw_responses.append(raw_text)

            if self.cost_tracker is not None:
                try:
                    self.cost_tracker.record_usage(response.usage, success=True, retries=0)
                except CostBudgetExceededError:
                    return LLMJudgeResult(
                        predicted_candidate_set=None,
                        is_valid_diagnosis=False,
                        attempts_used=attempt,
                        termination_reason="budget_exhausted",
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        estimated_cost_usd=self.cost_tracker.estimated_spend_usd,
                        rationale="Budget exhausted during response processing",
                        raw_responses=tuple(raw_responses),
                    )

            # Extract JSON
            json_text = _extract_json_substring(raw_text)
            try:
                parsed = json.loads(json_text)
            except Exception as json_err:
                if attempt < max_attempts:
                    retry_msg = (
                        f"Error: Could not parse response as valid JSON ({json_err}). "
                        'Please respond with ONLY valid JSON adhering strictly to: '
                        '{"culprit_element_ids": ["<id>", ...], "rationale": "<explanation>"}'
                    )
                    messages.append(Message(role="assistant", content=raw_text))
                    messages.append(Message(role="user", content=retry_msg))
                    continue
                else:
                    return LLMJudgeResult(
                        predicted_candidate_set=None,
                        is_valid_diagnosis=False,
                        attempts_used=attempt,
                        termination_reason="malformed_output",
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        estimated_cost_usd=round((total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000, 6),
                        rationale=f"Failed to parse JSON: {json_err}",
                        raw_responses=tuple(raw_responses),
                    )

            has_culprit = isinstance(parsed, dict) and ("culprit_element_ids" in parsed or "predicted_candidate_set" in parsed)
            if not has_culprit:
                if attempt < max_attempts:
                    retry_msg = (
                        "Your previous response could not be parsed as valid JSON or was missing 'culprit_element_ids'. "
                        'Please respond with ONLY a valid JSON object matching: {"culprit_element_ids": ["<id>", ...], "rationale": "..."}. '
                        f"Valid candidate element IDs to choose from: {list(display_map.keys())}"
                    )
                    messages.append(Message(role="assistant", content=raw_text))
                    messages.append(Message(role="user", content=retry_msg))
                    continue
                else:
                    return LLMJudgeResult(
                        predicted_candidate_set=None,
                        is_valid_diagnosis=False,
                        attempts_used=attempt,
                        termination_reason="malformed_output",
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        estimated_cost_usd=round((total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000, 6),
                        rationale="Missing 'culprit_element_ids' in JSON response",
                        raw_responses=tuple(raw_responses),
                    )

            raw_ids = parsed.get("culprit_element_ids") if "culprit_element_ids" in parsed else parsed.get("predicted_candidate_set")
            if not isinstance(raw_ids, list):
                if attempt < max_attempts:
                    retry_msg = "Error: 'culprit_element_ids' must be a list of string identifiers."
                    messages.append(Message(role="assistant", content=raw_text))
                    messages.append(Message(role="user", content=retry_msg))
                    continue
                else:
                    return LLMJudgeResult(
                        predicted_candidate_set=None,
                        is_valid_diagnosis=False,
                        attempts_used=attempt,
                        termination_reason="malformed_output",
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        estimated_cost_usd=round((total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000, 6),
                        rationale="'culprit_element_ids' was not a list",
                        raw_responses=tuple(raw_responses),
                    )

            # Map candidate IDs back to full candidate strings (with case and whitespace normalization)
            resolved_elements: list[str] = []
            invalid_ids: list[str] = []
            lower_id_map = {k.lower(): v for k, v in id_to_full.items()}
            for item in raw_ids:
                s_id = str(item).strip()
                if s_id in id_to_full:
                    full_elem = id_to_full[s_id]
                    if full_elem not in resolved_elements:
                        resolved_elements.append(full_elem)
                elif s_id.lower() in lower_id_map:
                    full_elem = lower_id_map[s_id.lower()]
                    if full_elem not in resolved_elements:
                        resolved_elements.append(full_elem)
                else:
                    invalid_ids.append(s_id)

            if invalid_ids:
                if attempt < max_attempts:
                    valid_opts = list(display_map.keys())
                    retry_msg = (
                        f"The following elements were not found in the valid candidate list: {', '.join(invalid_ids)}. "
                        f"Available valid candidate options: {valid_opts}. Please return only valid candidate IDs."
                    )
                    messages.append(Message(role="assistant", content=raw_text))
                    messages.append(Message(role="user", content=retry_msg))
                    continue
                else:
                    return LLMJudgeResult(
                        predicted_candidate_set=None,
                        is_valid_diagnosis=False,
                        attempts_used=attempt,
                        termination_reason="unrecognized_candidate",
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        estimated_cost_usd=round((total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000, 6),
                        rationale=f"Unrecognized candidate IDs: {invalid_ids}",
                        raw_responses=tuple(raw_responses),
                    )

            # Success! Valid diagnosis produced
            rationale_text = str(parsed.get("rationale", "")).strip()
            cost_usd = (
                self.cost_tracker.estimated_spend_usd
                if self.cost_tracker
                else round((total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000, 6)
            )

            return LLMJudgeResult(
                predicted_candidate_set=tuple(resolved_elements),
                is_valid_diagnosis=True,
                attempts_used=attempt,
                termination_reason="normal_completion",
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                estimated_cost_usd=cost_usd,
                rationale=rationale_text,
                raw_responses=tuple(raw_responses),
            )

        # Retries exhausted fallback
        return LLMJudgeResult(
            predicted_candidate_set=None,
            is_valid_diagnosis=False,
            attempts_used=max_attempts,
            termination_reason="malformed_output",
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            estimated_cost_usd=round((total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000, 6),
            rationale="Max retries exhausted without valid JSON",
            raw_responses=tuple(raw_responses),
        )


def run_llm_judge_baseline(
    initial_input: InputCase,
    execution_trace: AgentExecutionTrace,
    provider: ModelProvider,
    cost_tracker: CostTracker | None = None,
    max_retries: int = 2,
) -> LLMJudgeComparison:
    """Run the passive LLM judge baseline and package result into standard baseline comparison schema."""
    judge = LLMJudgeBaseline(provider=provider, cost_tracker=cost_tracker, max_retries=max_retries)
    res = judge.diagnose(initial_input, execution_trace)

    return LLMJudgeComparison(
        baseline_name="llm_judge",
        discovered_elements=res.predicted_candidate_set or (),
        predicted_candidate_set=res.predicted_candidate_set,
        is_valid_diagnosis=res.is_valid_diagnosis,
        false_attribution_indicator=None,  # Populated post-hoc when scored against ground truth
        exact_recovery_indicator=None,
        attempts_used=res.attempts_used,
        total_model_requests=res.attempts_used,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        total_tokens=res.total_tokens,
        estimated_cost_usd=res.estimated_cost_usd,
        termination_reason=res.termination_reason,
        rationale=res.rationale,
        raw_output=res.raw_responses[-1] if res.raw_responses else "",
    )
