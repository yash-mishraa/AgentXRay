"""Experiment cost tracking, pre-flight estimation, and hard budget enforcement."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from agentxray.providers.base import UsageMetadata
from agentxray.providers.openai_adapter import MODEL_PRICING_PER_1M
from agentxray.runner import BudgetExhaustedError


class CostBudgetExceededError(BudgetExhaustedError):
    """Raised when request counts or estimated expenditure exceed configured limits."""


class ExperimentBudgetConfig(BaseModel):
    """Budget limits and safety guardrails for model-backed experiments."""

    model_config = ConfigDict(frozen=True)

    max_model_requests: int = Field(default=50, gt=0, description="Hard cap on model calls.")
    max_spend_usd: float = Field(default=0.50, gt=0.0, description="Hard cap on estimated spend in USD.")
    max_retries_per_request: int = Field(default=2, ge=0, description="Max retries for transient errors.")
    request_timeout_seconds: float = Field(default=20.0, gt=0.0, description="Per-request HTTP timeout.")
    model_name: str = Field(default="gpt-4o-mini", description="Target model identifier.")
    custom_pricing_per_1m: tuple[float, float] | None = Field(
        default=None, description="Optional custom (input_usd, output_usd) per 1M tokens."
    )


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

        # Retrieve pricing for model or fallback to custom/standard pricing
        if config.custom_pricing_per_1m is not None:
            self._pricing = config.custom_pricing_per_1m
        elif config.model_name in MODEL_PRICING_PER_1M:
            self._pricing = MODEL_PRICING_PER_1M[config.model_name]
        elif config.model_name.startswith("fake"):
            self._pricing = (0.15, 0.60)
        elif any(
            config.model_name.lower().startswith(p)
            for p in ("llama", "qwen", "mistral", "phi", "gemma", "ollama", "local", "deepseek", "vicuna")
        ) or any(
            tag in config.model_name.lower()
            for tag in (":3b", ":8b", ":7b", ":1.5b", ":14b", ":70b", ":latest")
        ):
            self._pricing = (0.0, 0.0)
        else:
            raise ValueError(
                f"Unknown pricing for model {config.model_name!r}. Please specify explicit "
                f"custom_pricing_per_1m=(input_usd_per_1m, output_usd_per_1m) or choose from {list(MODEL_PRICING_PER_1M.keys())}."
            )

    def check_budget_before_request(self, est_prompt_tokens: int = 0) -> None:
        """Verify that request count and projected spend are within limits before initiating a call."""
        if self.total_requests >= self.config.max_model_requests:
            raise CostBudgetExceededError(
                f"Model request budget exhausted before request: executed {self.total_requests} requests "
                f"(limit: {self.config.max_model_requests})."
            )

        if est_prompt_tokens > 0:
            est_call_cost = (est_prompt_tokens * self._pricing[0]) / 1_000_000.0
            if (self.estimated_spend_usd + est_call_cost) >= self.config.max_spend_usd:
                raise CostBudgetExceededError(
                    f"Estimated spend budget would be exceeded before request: current ${self.estimated_spend_usd:.4f} USD + "
                    f"projected ${est_call_cost:.4f} USD >= limit ${self.config.max_spend_usd:.4f} USD."
                )
        elif self.estimated_spend_usd >= self.config.max_spend_usd:
            raise CostBudgetExceededError(
                f"Estimated spend budget exhausted before request: reached ${self.estimated_spend_usd:.4f} USD "
                f"(limit: ${self.config.max_spend_usd:.4f} USD)."
            )

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
        # Increment request count including retried requests
        self.total_requests += (1 + retries)
        self.retries += retries

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += (1 + retries)

        # Fallback for omitted/zero token reporting so unknown usage is not treated as zero cost
        p_tok = max(10, usage.prompt_tokens) if (usage.prompt_tokens == 0 and usage.completion_tokens == 0) else usage.prompt_tokens
        c_tok = max(5, usage.completion_tokens) if (usage.prompt_tokens == 0 and usage.completion_tokens == 0) else usage.completion_tokens

        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.total_tokens += (p_tok + c_tok)

        # Accumulate spend
        call_cost = (
            p_tok * self._pricing[0] + c_tok * self._pricing[1]
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

    def record_failed_request(self, retries: int = 0) -> None:
        """Record a request attempt that failed before receiving usage telemetry."""
        self.total_requests += (1 + retries)
        self.retries += retries
        self.failed_requests += (1 + retries)
        if self.total_requests >= self.config.max_model_requests:
            raise CostBudgetExceededError(
                f"Model request budget exhausted: executed {self.total_requests} requests "
                f"(limit: {self.config.max_model_requests})."
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
