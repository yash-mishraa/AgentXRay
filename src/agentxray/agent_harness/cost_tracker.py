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
    max_spend_usd: float = Field(default=0.50, ge=0.0, description="Hard cap on estimated spend in USD.")
    max_retries_per_request: int = Field(default=2, ge=0, description="Max retries for transient errors.")
    request_timeout_seconds: float = Field(default=20.0, gt=0.0, description="Per-request HTTP timeout.")
    model_name: str = Field(default="gpt-4o-mini", description="Target model identifier.")
    custom_pricing_per_1m: tuple[float, float] | None = Field(
        default=None, description="Optional custom (input_usd, output_usd) per 1M tokens."
    )
    max_discovery_requests: int | None = Field(
        default=None, ge=1, description="Explicit cap on model requests during discovery/search phase."
    )
    max_verification_requests: int | None = Field(
        default=None, ge=1, description="Explicit cap on model requests during verification phase."
    )


class CostTracker:
    """Tracks token consumption, records model requests, and halts when budgets are breached."""

    def __init__(self, config: ExperimentBudgetConfig, provider: Any = None) -> None:
        self.config = config
        self._provider = provider
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

    def bind_provider(self, provider: Any) -> None:
        """Bind an active provider instance to this cost tracker."""
        self._provider = provider

    def is_guaranteed_local_provider(self, provider: Any = None) -> bool:
        """Verify that the given or bound provider is an exact approved local OllamaProvider on loopback."""
        prov = provider if provider is not None else self._provider
        if prov is None:
            return False
        from agentxray.providers.ollama_adapter import OllamaProvider
        if type(prov) is not OllamaProvider:
            return False
        base_url = getattr(prov, "base_url", getattr(prov, "_base_url", ""))
        import urllib.parse
        parsed = urllib.parse.urlparse(base_url)
        hostname = (parsed.hostname or "").lower()
        return hostname in ("127.0.0.1", "localhost", "::1")

    def check_budget_before_request(self, est_prompt_tokens: int = 0, provider: Any = None) -> None:
        """Verify that request count and projected spend are within limits before initiating a call."""
        if self.total_requests >= self.config.max_model_requests:
            raise CostBudgetExceededError(
                f"Model request budget exhausted before request: executed {self.total_requests} requests "
                f"(limit: {self.config.max_model_requests})."
            )

        if self.config.max_spend_usd == 0.0:
            if not self.is_guaranteed_local_provider(provider):
                prov_name = type(provider or self._provider).__name__ if (provider or self._provider) else "None"
                raise CostBudgetExceededError(
                    f"Zero-spend budget (max_spend_usd=0.0) is strictly restricted to local execution using an approved "
                    f"OllamaProvider bound to local loopback (127.0.0.1, localhost, or ::1). Got provider: '{prov_name}'."
                )
            if self._pricing != (0.0, 0.0):
                raise CostBudgetExceededError(
                    f"Zero-spend budget (max_spend_usd=0.0) requires zero pricing (0.0, 0.0), but model "
                    f"'{self.config.model_name}' has non-zero pricing {self._pricing}."
                )
            if self.estimated_spend_usd > 0.0:
                raise CostBudgetExceededError(
                    f"Estimated spend budget exhausted before request: reached ${self.estimated_spend_usd:.4f} USD "
                    f"(limit: $0.0000 USD)."
                )
            est_call_cost = (est_prompt_tokens * self._pricing[0]) / 1_000_000.0 if est_prompt_tokens > 0 else 0.0
            if est_call_cost > 0.0:
                raise CostBudgetExceededError(
                    f"Estimated spend budget would be exceeded before request: projected call cost ${est_call_cost:.4f} USD > $0.0000 USD."
                )
        else:
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
        provider: Any = None,
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
        if self.config.max_spend_usd == 0.0:
            if not self.is_guaranteed_local_provider(provider):
                prov_name = type(provider or self._provider).__name__ if (provider or self._provider) else "None"
                raise CostBudgetExceededError(
                    f"Zero-spend budget (max_spend_usd=0.0) is strictly restricted to local execution using an approved "
                    f"OllamaProvider bound to local loopback (127.0.0.1, localhost, or ::1). Got provider: '{prov_name}'."
                )
            if self.estimated_spend_usd > 0.0 or self._pricing != (0.0, 0.0):
                raise CostBudgetExceededError(
                    f"Estimated spend budget exhausted: reached ${self.estimated_spend_usd:.4f} USD "
                    f"(limit: $0.0000 USD)."
                )
        else:
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

        within_spend = (
            est_spend == 0.0
            if self.config.max_spend_usd == 0.0
            else est_spend <= self.config.max_spend_usd
        )

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
            "within_budget": (total_requests <= self.config.max_model_requests and within_spend),
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

    @property
    def is_spend_exceeded(self) -> bool:
        """Return True if estimated spend has reached or exceeded configured budget."""
        if self.config.max_spend_usd == 0.0:
            return self.estimated_spend_usd > 0.0
        return self.estimated_spend_usd >= self.config.max_spend_usd

    @property
    def is_request_cap_exceeded(self) -> bool:
        """Return True if total requests strictly exceeded (>) configured budget."""
        return self.total_requests > self.config.max_model_requests

    @property
    def is_request_cap_reached(self) -> bool:
        """Return True if total requests reached or exceeded (>=) configured budget."""
        return self.total_requests >= self.config.max_model_requests

    @property
    def remaining_request_headroom(self) -> int:
        """Return remaining allowed provider requests before reaching configured cap."""
        return max(0, self.config.max_model_requests - self.total_requests)


class IsolatedAlgorithmCostTracker(CostTracker):
    """Tracks resource consumption for a single algorithm execution while mirroring usage to a global tracker.

    Enforces local algorithm-level request and spend limits to guarantee fair, independent
    budget allowances during multi-algorithm benchmark comparisons, preventing sequential
    budget starvation where earlier algorithms consume later algorithms' quotas.
    """

    def __init__(
        self,
        algorithm_config: ExperimentBudgetConfig,
        global_tracker: CostTracker | None = None,
        enforce_global_spend: bool = True,
        enforce_global_requests: bool = False,
    ) -> None:
        super().__init__(algorithm_config)
        self.global_tracker = global_tracker
        self.enforce_global_spend = enforce_global_spend
        self.enforce_global_requests = enforce_global_requests

    def bind_provider(self, provider: Any) -> None:
        """Bind an active provider instance to this cost tracker and global tracker."""
        super().bind_provider(provider)
        if self.global_tracker is not None and hasattr(self.global_tracker, "bind_provider"):
            self.global_tracker.bind_provider(provider)

    @property
    def is_spend_exceeded(self) -> bool:
        """Return True if local spend or enforced global spend has reached or exceeded budget."""
        if super().is_spend_exceeded:
            return True
        if self.enforce_global_spend and self.global_tracker is not None:
            return self.global_tracker.is_spend_exceeded
        return False

    @property
    def is_request_cap_exceeded(self) -> bool:
        """Return True if local requests or enforced global requests strictly exceeded (>) budget."""
        if super().is_request_cap_exceeded:
            return True
        if self.enforce_global_requests and self.global_tracker is not None:
            return self.global_tracker.is_request_cap_exceeded
        return False

    @property
    def is_request_cap_reached(self) -> bool:
        """Return True if local requests or enforced global requests reached or exceeded (>=) budget."""
        if super().is_request_cap_reached:
            return True
        if self.enforce_global_requests and self.global_tracker is not None:
            return self.global_tracker.is_request_cap_reached
        return False

    @property
    def remaining_request_headroom(self) -> int:
        """Return remaining allowed provider requests considering local and enforced global caps."""
        local_hr = super().remaining_request_headroom
        if self.enforce_global_requests and self.global_tracker is not None:
            return min(local_hr, self.global_tracker.remaining_request_headroom)
        return local_hr

    def check_budget_before_request(self, est_prompt_tokens: int = 0, provider: Any = None) -> None:
        """Verify local algorithm allowance and global safety limits before request."""
        # 1. Enforce local algorithm request and spend allowance
        super().check_budget_before_request(est_prompt_tokens, provider=provider)

        # 2. Enforce global safety limits if global_tracker is attached
        if self.global_tracker is not None:
            if (
                self.enforce_global_requests
                and self.global_tracker.total_requests >= self.global_tracker.config.max_model_requests
            ):
                raise CostBudgetExceededError(
                    f"Global benchmark model request budget exhausted before request: "
                    f"executed {self.global_tracker.total_requests} requests "
                    f"(global limit: {self.global_tracker.config.max_model_requests})."
                )
            if self.enforce_global_spend:
                if self.global_tracker.config.max_spend_usd == 0.0:
                    if not self.global_tracker.is_guaranteed_local_provider(provider):
                        prov_name = type(provider or self.global_tracker._provider).__name__ if (provider or self.global_tracker._provider) else "None"
                        raise CostBudgetExceededError(
                            "Global zero-spend budget (max_spend_usd=0.0) is strictly restricted to local execution "
                            f"using an approved OllamaProvider bound to local loopback. Got provider: '{prov_name}'."
                        )
                    if self.global_tracker._pricing != (0.0, 0.0):
                        raise CostBudgetExceededError(
                            f"Global zero-spend budget requires zero pricing (0.0, 0.0), but model has non-zero pricing {self.global_tracker._pricing}."
                        )
                    if self.global_tracker.estimated_spend_usd > 0.0:
                        raise CostBudgetExceededError(
                            f"Global estimated spend budget exhausted before request: "
                            f"reached ${self.global_tracker.estimated_spend_usd:.4f} USD (limit: $0.0000 USD)."
                        )
                    est_call_cost = (est_prompt_tokens * self.global_tracker._pricing[0]) / 1_000_000.0 if est_prompt_tokens > 0 else 0.0
                    if est_call_cost > 0.0:
                        raise CostBudgetExceededError(
                            f"Global estimated spend budget would be exceeded before request: projected call cost ${est_call_cost:.4f} USD > $0.0000 USD."
                        )
                else:
                    if est_prompt_tokens > 0:
                        est_call_cost = (est_prompt_tokens * self.global_tracker._pricing[0]) / 1_000_000.0
                        if (self.global_tracker.estimated_spend_usd + est_call_cost) >= self.global_tracker.config.max_spend_usd:
                            raise CostBudgetExceededError(
                                f"Global estimated spend budget would be exceeded before request: "
                                f"current ${self.global_tracker.estimated_spend_usd:.4f} USD + "
                                f"projected ${est_call_cost:.4f} USD >= limit ${self.global_tracker.config.max_spend_usd:.4f} USD."
                            )
                    elif self.global_tracker.estimated_spend_usd >= self.global_tracker.config.max_spend_usd:
                        raise CostBudgetExceededError(
                            f"Global estimated spend budget exhausted before request: "
                            f"reached ${self.global_tracker.estimated_spend_usd:.4f} USD "
                            f"(global limit: ${self.global_tracker.config.max_spend_usd:.4f} USD)."
                        )

    def record_usage(
        self,
        usage: UsageMetadata,
        success: bool = True,
        retries: int = 0,
        provider: Any = None,
    ) -> None:
        """Record model request outcomes in both local algorithm tracker and global tracker."""
        local_exc: CostBudgetExceededError | None = None
        try:
            super().record_usage(usage, success=success, retries=retries, provider=provider)
        except CostBudgetExceededError as exc:
            local_exc = exc

        # Always mirror into global tracker so cumulative resource consumption is fully accounted
        if self.global_tracker is not None:
            try:
                self.global_tracker.record_usage(usage, success=success, retries=retries, provider=provider)
            except CostBudgetExceededError as g_exc:
                if (
                    self.global_tracker.is_spend_exceeded
                    and self.enforce_global_spend
                ):
                    if local_exc is None or not super().is_spend_exceeded:
                        local_exc = g_exc
                elif self.enforce_global_requests:
                    if local_exc is None or not super().is_request_cap_exceeded:
                        local_exc = g_exc

        if local_exc is not None:
            raise local_exc

    def record_failed_request(self, retries: int = 0) -> None:
        """Record failed request attempt in both local algorithm tracker and global tracker."""
        local_exc: CostBudgetExceededError | None = None
        try:
            super().record_failed_request(retries=retries)
        except CostBudgetExceededError as exc:
            local_exc = exc

        if self.global_tracker is not None:
            try:
                self.global_tracker.record_failed_request(retries=retries)
            except CostBudgetExceededError as g_exc:
                if self.enforce_global_requests:
                    if local_exc is None or not super().is_request_cap_exceeded:
                        local_exc = g_exc

        if local_exc is not None:
            raise local_exc


class IsolatedPhaseCostTracker(IsolatedAlgorithmCostTracker):
    """Tracks resource consumption for a distinct execution phase (discovery or verification).

    Enforces phase-specific model-request quotas while mirroring usage to an overarching
    global safety tracker. Prevents discovery from consuming the verification allowance
    and vice versa.
    """

    def __init__(
        self,
        algorithm_config: ExperimentBudgetConfig,
        phase: str = "discovery",
        global_tracker: CostTracker | None = None,
        enforce_global_spend: bool = True,
        enforce_global_requests: bool = False,
    ) -> None:
        super().__init__(
            algorithm_config=algorithm_config,
            global_tracker=global_tracker,
            enforce_global_spend=enforce_global_spend,
            enforce_global_requests=enforce_global_requests,
        )
        self.phase = phase.strip().lower()
        if self.phase in ("discovery", "search"):
            self.max_phase_requests = (
                algorithm_config.max_discovery_requests
                if algorithm_config.max_discovery_requests is not None
                else algorithm_config.max_model_requests
            )
        elif self.phase in ("verification", "verify", "audit"):
            self.max_phase_requests = (
                algorithm_config.max_verification_requests
                if algorithm_config.max_verification_requests is not None
                else algorithm_config.max_model_requests
            )
        else:
            self.max_phase_requests = algorithm_config.max_model_requests

    @property
    def is_request_cap_exceeded(self) -> bool:
        """Return True if local phase requests or enforced global requests strictly exceeded (>) budget."""
        if self.total_requests > self.max_phase_requests:
            return True
        if self.enforce_global_requests and self.global_tracker is not None:
            return self.global_tracker.is_request_cap_exceeded
        return False

    @property
    def is_request_cap_reached(self) -> bool:
        """Return True if local phase requests or enforced global requests reached or exceeded (>=) budget."""
        if self.total_requests >= self.max_phase_requests:
            return True
        if self.enforce_global_requests and self.global_tracker is not None:
            return self.global_tracker.is_request_cap_reached
        return False

    @property
    def remaining_request_headroom(self) -> int:
        """Return remaining allowed provider requests considering phase, algorithm, and enforced global caps."""
        phase_hr = max(0, self.max_phase_requests - self.total_requests)
        algo_hr = max(0, self.config.max_model_requests - self.total_requests)
        local_hr = min(phase_hr, algo_hr)
        if self.enforce_global_requests and self.global_tracker is not None:
            return min(local_hr, self.global_tracker.remaining_request_headroom)
        return local_hr

    def check_budget_before_request(self, est_prompt_tokens: int = 0, provider: Any = None) -> None:
        """Verify local phase request allowance and global safety limits before request."""
        # Check local spend limit
        if self.config.max_spend_usd == 0.0:
            if not self.is_guaranteed_local_provider(provider):
                prov_name = type(provider or self._provider).__name__ if (provider or self._provider) else "None"
                raise CostBudgetExceededError(
                    "Phase zero-spend budget (max_spend_usd=0.0) is strictly restricted to local execution "
                    f"using an approved OllamaProvider bound to local loopback. Got provider: '{prov_name}'."
                )
            if self._pricing != (0.0, 0.0):
                raise CostBudgetExceededError(
                    f"Phase zero-spend budget requires zero pricing (0.0, 0.0), but model has non-zero pricing {self._pricing}."
                )
            if self.estimated_spend_usd > 0.0:
                raise CostBudgetExceededError(
                    f"Estimated spend budget exhausted before request: "
                    f"reached ${self.estimated_spend_usd:.4f} USD (limit: $0.0000 USD)."
                )
            est_call_cost = (est_prompt_tokens * self._pricing[0]) / 1_000_000.0 if est_prompt_tokens > 0 else 0.0
            if est_call_cost > 0.0:
                raise CostBudgetExceededError(
                    f"Estimated spend budget would be exceeded before request: projected call cost ${est_call_cost:.4f} USD > $0.0000 USD."
                )
        elif self.estimated_spend_usd >= self.config.max_spend_usd:
            raise CostBudgetExceededError(
                f"Estimated spend budget exhausted before request: "
                f"reached ${self.estimated_spend_usd:.4f} USD (limit: ${self.config.max_spend_usd:.4f} USD)."
            )

        # Check local phase request allowance
        if self.total_requests >= self.max_phase_requests:
            raise CostBudgetExceededError(
                f"{self.phase.capitalize()} model request budget exhausted before request: "
                f"executed {self.total_requests} requests (phase limit: {self.max_phase_requests})."
            )

        # Check global safety limits if global_tracker is attached
        if self.global_tracker is not None:
            if (
                self.enforce_global_requests
                and self.global_tracker.total_requests >= self.global_tracker.config.max_model_requests
            ):
                raise CostBudgetExceededError(
                    f"Global benchmark model request budget exhausted before request: "
                    f"executed {self.global_tracker.total_requests} requests "
                    f"(global limit: {self.global_tracker.config.max_model_requests})."
                )
            if self.enforce_global_spend:
                if self.global_tracker.config.max_spend_usd == 0.0:
                    if not self.global_tracker.is_guaranteed_local_provider(provider):
                        prov_name = type(provider or self.global_tracker._provider).__name__ if (provider or self.global_tracker._provider) else "None"
                        raise CostBudgetExceededError(
                            "Global zero-spend budget (max_spend_usd=0.0) is strictly restricted to local execution "
                            f"using an approved OllamaProvider bound to local loopback. Got provider: '{prov_name}'."
                        )
                    if self.global_tracker._pricing != (0.0, 0.0):
                        raise CostBudgetExceededError(
                            f"Global zero-spend budget requires zero pricing (0.0, 0.0), but model has non-zero pricing {self.global_tracker._pricing}."
                        )
                    if self.global_tracker.estimated_spend_usd > 0.0:
                        raise CostBudgetExceededError(
                            f"Global estimated spend budget exhausted before request: "
                            f"reached ${self.global_tracker.estimated_spend_usd:.4f} USD (limit: $0.0000 USD)."
                        )
                    est_call_cost = (est_prompt_tokens * self.global_tracker._pricing[0]) / 1_000_000.0 if est_prompt_tokens > 0 else 0.0
                    if est_call_cost > 0.0:
                        raise CostBudgetExceededError(
                            f"Global estimated spend budget would be exceeded before request: projected call cost ${est_call_cost:.4f} USD > $0.0000 USD."
                        )
                else:
                    if est_prompt_tokens > 0:
                        est_call_cost = (est_prompt_tokens * self.global_tracker._pricing[0]) / 1_000_000.0
                        if (self.global_tracker.estimated_spend_usd + est_call_cost) >= self.global_tracker.config.max_spend_usd:
                            raise CostBudgetExceededError(
                                f"Global estimated spend budget would be exceeded before request: "
                                f"current ${self.global_tracker.estimated_spend_usd:.4f} USD + "
                                f"projected ${est_call_cost:.4f} USD >= limit ${self.global_tracker.config.max_spend_usd:.4f} USD."
                            )
                    elif self.global_tracker.estimated_spend_usd >= self.global_tracker.config.max_spend_usd:
                        raise CostBudgetExceededError(
                            f"Global estimated spend budget exhausted before request: "
                            f"reached ${self.global_tracker.estimated_spend_usd:.4f} USD "
                            f"(global limit: ${self.global_tracker.config.max_spend_usd:.4f} USD)."
                        )

    def record_usage(
        self,
        usage: UsageMetadata,
        success: bool = True,
        retries: int = 0,
        provider: Any = None,
    ) -> None:
        """Record model request outcomes in phase tracker and mirror to global tracker."""
        self.total_requests += (1 + retries)
        self.retries += retries
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += (1 + retries)

        p_tok = max(10, usage.prompt_tokens) if (usage.prompt_tokens == 0 and usage.completion_tokens == 0) else usage.prompt_tokens
        c_tok = max(5, usage.completion_tokens) if (usage.prompt_tokens == 0 and usage.completion_tokens == 0) else usage.completion_tokens
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.total_tokens += (p_tok + c_tok)
        call_cost = (p_tok * self._pricing[0] + c_tok * self._pricing[1]) / 1_000_000.0
        self.estimated_spend_usd = round(self.estimated_spend_usd + call_cost, 7)

        local_exc: CostBudgetExceededError | None = None
        if self.config.max_spend_usd == 0.0:
            if not self.is_guaranteed_local_provider(provider):
                prov_name = type(provider or self._provider).__name__ if (provider or self._provider) else "None"
                local_exc = CostBudgetExceededError(
                    f"Phase zero-spend budget (max_spend_usd=0.0) is strictly restricted to local execution using an approved "
                    f"OllamaProvider bound to local loopback (127.0.0.1, localhost, or ::1). Got provider: '{prov_name}'."
                )
            elif self.estimated_spend_usd > 0.0 or self._pricing != (0.0, 0.0):
                local_exc = CostBudgetExceededError(
                    f"Phase estimated spend budget exhausted: reached ${self.estimated_spend_usd:.4f} USD "
                    f"(limit: $0.0000 USD)."
                )
        elif self.estimated_spend_usd >= self.config.max_spend_usd:
            local_exc = CostBudgetExceededError(
                f"Estimated spend budget exhausted: reached ${self.estimated_spend_usd:.4f} USD "
                f"(limit: ${self.config.max_spend_usd:.4f} USD)."
            )
        elif self.total_requests >= self.max_phase_requests:
            local_exc = CostBudgetExceededError(
                f"{self.phase.capitalize()} model request budget exhausted: executed {self.total_requests} requests "
                f"(phase limit: {self.max_phase_requests})."
            )

        if self.global_tracker is not None:
            try:
                self.global_tracker.record_usage(usage, success=success, retries=retries, provider=provider)
            except CostBudgetExceededError as g_exc:
                if (
                    self.global_tracker.is_spend_exceeded
                    and self.enforce_global_spend
                ):
                    if local_exc is None or not self.is_spend_exceeded:
                        local_exc = g_exc
                elif self.enforce_global_requests:
                    if local_exc is None or not (self.total_requests > self.max_phase_requests):
                        local_exc = g_exc

        if local_exc is not None:
            raise local_exc

    def record_failed_request(self, retries: int = 0) -> None:
        """Record failed request attempt in phase tracker and mirror to global tracker."""
        self.total_requests += (1 + retries)
        self.failed_requests += (1 + retries)
        self.retries += retries

        local_exc: CostBudgetExceededError | None = None
        if self.total_requests > self.max_phase_requests:
            local_exc = CostBudgetExceededError(
                f"{self.phase.capitalize()} model request budget exceeded on failed request: "
                f"executed {self.total_requests} requests (phase limit: {self.max_phase_requests})."
            )

        if self.global_tracker is not None:
            try:
                self.global_tracker.record_failed_request(retries=retries)
            except CostBudgetExceededError as g_exc:
                if self.enforce_global_requests:
                    if local_exc is None or not (self.total_requests > self.max_phase_requests):
                        local_exc = g_exc

        if local_exc is not None:
            raise local_exc
