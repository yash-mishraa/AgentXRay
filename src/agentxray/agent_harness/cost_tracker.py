"""Experiment cost tracking, pre-flight estimation, and hard budget enforcement."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from agentxray.providers.base import UsageMetadata
from agentxray.providers.openai_adapter import MODEL_PRICING_PER_1M


class CostBudgetExceededError(Exception):
    """Raised when request counts or estimated expenditure exceed configured limits."""


class ExperimentBudgetConfig(BaseModel):
    """Budget limits and safety guardrails for model-backed experiments."""

    model_config = ConfigDict(frozen=True)

    max_model_requests: int = Field(default=50, gt=0, description="Hard cap on model calls.")
    max_spend_usd: float = Field(default=0.50, gt=0.0, description="Hard cap on estimated spend in USD.")
    max_retries_per_request: int = Field(default=2, ge=0, description="Max retries for transient errors.")
    request_timeout_seconds: float = Field(default=20.0, gt=0.0, description="Per-request HTTP timeout.")
    model_name: str = Field(default="gpt-4o-mini", description="Target model identifier.")


class CostTracker:
    """Tracks token consumption, records model requests, and halts when budgets are breached."""

    def __init__(self, config: ExperimentBudgetConfig) -> None:
        self.config = config
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.estimated_spend_usd = 0.0

        # Retrieve pricing for model or fallback to gpt-4o-mini standard pricing
        self._pricing = MODEL_PRICING_PER_1M.get(config.model_name, (0.15, 0.60))

    def record_usage(
        self,
        usage: UsageMetadata,
        success: bool = True,
        retries: int = 0,
    ) -> None:
        """Record model request outcomes and verify safety budgets.

        Raises
        ------
        CostBudgetExceededError
            If total requests or estimated spend breach configured limits.
        """
        self.total_requests += 1
        self.retries += retries

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens

        # Accumulate spend
        call_cost = (
            usage.prompt_tokens * self._pricing[0] + usage.completion_tokens * self._pricing[1]
        ) / 1_000_000.0
        self.estimated_spend_usd = round(self.estimated_spend_usd + call_cost, 7)

        # Enforce request cap
        if self.total_requests >= self.config.max_model_requests:
            raise CostBudgetExceededError(
                f"Model request budget exhausted: executed {self.total_requests} requests "
                f"(limit: {self.config.max_model_requests})."
            )

        # Enforce spend cap
        if self.estimated_spend_usd >= self.config.max_spend_usd:
            raise CostBudgetExceededError(
                f"Estimated spend budget exhausted: reached ${self.estimated_spend_usd:.4f} USD "
                f"(limit: ${self.config.max_spend_usd:.4f} USD)."
            )

    def estimate_preflight_cost(
        self,
        planned_trials: int,
        turns_per_trial: int = 2,
        est_prompt_tokens_per_turn: int = 250,
        est_completion_tokens_per_turn: int = 60,
    ) -> dict[str, Any]:
        """Compute an upfront worst-case estimation before running live experiments."""
        total_requests = planned_trials * turns_per_trial
        total_prompt = total_requests * est_prompt_tokens_per_turn
        total_comp = total_requests * est_completion_tokens_per_turn
        est_spend = (total_prompt * self._pricing[0] + total_comp * self._pricing[1]) / 1_000_000.0

        return {
            "model_name": self.config.model_name,
            "planned_trials": planned_trials,
            "estimated_requests": total_requests,
            "max_allowed_requests": self.config.max_model_requests,
            "estimated_prompt_tokens": total_prompt,
            "estimated_completion_tokens": total_comp,
            "estimated_total_tokens": total_prompt + total_comp,
            "pricing_per_1m_input": f"${self._pricing[0]:.2f}",
            "pricing_per_1m_output": f"${self._pricing[1]:.2f}",
            "estimated_spend_usd": round(est_spend, 5),
            "max_allowed_spend_usd": self.config.max_spend_usd,
            "within_budget": (total_requests <= self.config.max_model_requests and est_spend <= self.config.max_spend_usd),
        }

    def get_summary(self) -> dict[str, Any]:
        """Return structured summary of consumed resources."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "retries": self.retries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_spend_usd": round(self.estimated_spend_usd, 6),
            "max_spend_usd": self.config.max_spend_usd,
            "max_requests": self.config.max_model_requests,
        }
