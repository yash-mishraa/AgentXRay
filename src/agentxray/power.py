"""Exact Fisher power calculations and sample-size solvers for AgentXRay.

Standard-library-first implementation of finite-domain exact statistical power analysis
for two independent balanced binomial arms under one-sided Fisher exact tests (H1: p1 > p2)
and Intersection-Union Tests (IUT) for 1-minimality verification.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PowerSolverStatus(str, Enum):
    """Validation outcome of prospective power and sample size determination."""

    VALIDATED = "validated"
    RANGE_EXHAUSTED = "range_exhausted"
    INVALID_INPUTS = "invalid_inputs"


class FisherPowerResult(BaseModel):
    """Result of exact Fisher power calculation across a finite sample size domain."""

    model_config = ConfigDict(frozen=True)

    status: PowerSolverStatus = Field(..., description="Solver outcome status.")
    n_required: int | None = Field(
        default=None,
        description="Minimum balanced sample size satisfying power target over [n*, n_max], or None if exhausted.",
    )
    target_power: float = Field(..., ge=0.0, le=1.0, description="Target statistical power 1 - beta.")
    power_achieved: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Power achieved at n_required (if validated) or at n_max (if range exhausted).",
    )
    p1: float = Field(..., ge=0.0, le=1.0, description="Prespecified candidate failure probability.")
    p2: float = Field(..., ge=0.0, le=1.0, description="Prespecified ablation baseline failure probability.")
    alpha: float = Field(..., gt=0.0, lt=1.0, description="Prespecified one-sided significance level.")
    n_min: int = Field(default=1, ge=1, description="Lower bound of evaluation domain.")
    n_max: int = Field(default=120, ge=1, description="Upper bound of evaluation domain.")
    powers_by_n: dict[int, float] = Field(
        default_factory=dict,
        description="Exact power values computed for evaluated sample sizes.",
    )
    diagnostic_message: str = Field(default="", description="Human-readable explanation of solver outcome.")


class JointPowerResult(BaseModel):
    """Result of joint sample size determination for k-component 1-minimality verification."""

    model_config = ConfigDict(frozen=True)

    status: PowerSolverStatus = Field(..., description="Overall joint solver status.")
    n_required: int | None = Field(
        default=None,
        description="Overall required sample size per arm max(n_j*), or None if any range is exhausted.",
    )
    k_components: int = Field(..., ge=1, description="Number of single-component ablation baselines.")
    joint_target_power: float = Field(..., ge=0.0, le=1.0, description="Target joint power 1 - beta_target.")
    joint_beta_target: float = Field(..., ge=0.0, le=1.0, description="Target joint Type II error beta_target.")
    component_target_power: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Per-component target power 1 - beta_target/k derived via Boole's inequality.",
    )
    component_beta_target: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Per-component Type II error allocation beta_target/k.",
    )
    p1: float = Field(..., ge=0.0, le=1.0, description="Prespecified candidate failure probability.")
    p2_values: tuple[float, ...] = Field(..., description="Prespecified ablation failure probabilities.")
    component_results: dict[int, FisherPowerResult] = Field(
        default_factory=dict,
        description="Per-component sample size solver results indexed by component index.",
    )
    diagnostic_message: str = Field(default="", description="Human-readable summary of joint power allocation.")


def exact_fisher_p_value_unrounded(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    alternative: str = "greater",
) -> float:
    """Compute exact unrounded hypergeometric p-value for a 2x2 contingency table.

    Zero external dependencies, standard library math.comb.
    Crucially, does not round p-values to 5 decimal places before returning.

    Contingency table:
                Group 1       Group 2       Total
    Success:      k1            k2            K = k1 + k2
    Failure:    n1 - k1       n2 - k2       N - K
    Total:        n1            n2            N = n1 + n2
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"Sample sizes must be positive, got n1={n1}, n2={n2}")
    if k1 < 0 or k1 > n1 or k2 < 0 or k2 > n2:
        raise ValueError(f"Success counts invalid: k1={k1}/n1={n1}, k2={k2}/n2={n2}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less', got {alternative!r}")

    K = k1 + k2
    N = n1 + n2
    min_x = max(0, K - n2)
    max_x = min(n1, K)

    denom = math.comb(N, K)

    if alternative == "greater":
        # Upper tail: sum_{x = k1}^{max_x} comb(n1, x) * comb(n2, K - x) / denom
        if k1 > max_x:
            return 0.0
        num = sum(math.comb(n1, x) * math.comb(n2, K - x) for x in range(max(min_x, k1), max_x + 1))
        return min(1.0, max(0.0, num / denom))
    elif alternative == "less":
        # Lower tail: sum_{x = min_x}^{min(k1, max_x)} comb(n1, x) * comb(n2, K - x) / denom
        if k1 < min_x:
            return 0.0
        num = sum(math.comb(n1, x) * math.comb(n2, K - x) for x in range(min_x, min(max_x, k1) + 1))
        return min(1.0, max(0.0, num / denom))
    else:  # two-sided
        probs = {
            x: math.comb(n1, x) * math.comb(n2, K - x) / denom
            for x in range(min_x, max_x + 1)
        }
        obs_prob = probs.get(k1, 0.0)
        p_val = sum(p for p in probs.values() if p <= obs_prob * (1.0 + 1e-12))
        return min(1.0, max(0.0, p_val))


def exact_fisher_power(
    n: int,
    p1: float,
    p2: float,
    alpha: float = 0.05,
) -> float:
    """Calculate exact statistical power of one-sided Fisher exact test for balanced sample size n.

    Evaluates the probability that the unrounded Fisher exact p-value is strictly less than alpha
    under product binomial sampling X1 ~ Bin(n, p1) and X2 ~ Bin(n, p2) with alternative H1: p1 > p2.

    Uses an exact marginal-total decomposition: for each marginal sum K = x1 + x2 in [0, 2n],
    the upper-tail hypergeometric p-value is strictly decreasing in x1. The rejection boundary
    x_crit(K) is determined by accumulating terms from min(n, K) downward, achieving exact O(n^2)
    evaluation in standard library arithmetic with zero approximations.

    Parameters
    ----------
    n : int
        Balanced sample size per arm (n >= 1).
    p1 : float
        Candidate failure probability under H1 (0.0 <= p1 <= 1.0).
    p2 : float
        Ablation baseline failure probability under H1 (0.0 <= p2 <= 1.0).
    alpha : float
        One-sided significance threshold (0.0 < alpha < 1.0).

    Returns
    -------
    float
        Exact statistical power in [0.0, 1.0].
    """
    if n <= 0:
        raise ValueError(f"Sample size n must be positive integer, got {n}")
    if not (0.0 <= p1 <= 1.0) or not (0.0 <= p2 <= 1.0):
        raise ValueError(f"Probabilities p1, p2 must be in [0, 1], got p1={p1}, p2={p2}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")

    # Boundary cases
    if p1 == 0.0:
        # If candidate failure rate is 0, it can never exceed p2 >= 0
        return 0.0
    if p1 <= p2:
        # Under H0 or reverse superiority, power against H1: p1 > p2 is bounded by alpha
        pass

    # Precompute binomial pmfs for arm 1 and arm 2
    b1 = [math.comb(n, x1) * (p1 ** x1) * ((1.0 - p1) ** (n - x1)) for x1 in range(n + 1)]
    b2 = [math.comb(n, x2) * (p2 ** x2) * ((1.0 - p2) ** (n - x2)) for x2 in range(n + 1)]

    prob_rej = 0.0
    two_n = 2 * n

    for K in range(two_n + 1):
        min_x = max(0, K - n)
        max_x = min(n, K)
        denom = math.comb(two_n, K)

        # Find critical x_crit such that for all x >= x_crit, tail_num / denom < alpha
        tail_num = 0
        crit_x = max_x + 1
        for x in range(max_x, min_x - 1, -1):
            term = math.comb(n, x) * math.comb(n, K - x)
            tail_num += term
            # Strict inequality p < alpha
            if tail_num / denom < alpha:
                crit_x = x
            else:
                break

        # If any x in [min_x, max_x] rejects, accumulate exact binomial joint probabilities
        if crit_x <= max_x:
            for x1 in range(crit_x, max_x + 1):
                prob_rej += b1[x1] * b2[K - x1]

    return min(1.0, max(0.0, prob_rej))


def solve_fisher_sample_size(
    p1: float,
    p2: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
    n_min: int = 1,
    n_max: int = 120,
) -> FisherPowerResult:
    """Find the minimum balanced sample size n* in [n_min, n_max] satisfying target power.

    Employs a finite-domain monotonic ceiling search to compensate for discrete sawtooth dips:
    n* is the smallest integer in [n_min, n_max] such that for all m in [n*, n_max],
    Power(m; p1, p2, alpha) >= target_power.

    Guarantees:
    - Finite-Domain Ceiling: within [n_min, n_max], any planned sample size N in [n*, n_max]
      has exact power >= target_power.
    - Specificity: power is calculated specifically for the prespecified parameter pair (p1, p2).
      If actual operational failure rates differ from (p1, p2), realized power will vary.
    - Explicit Scope: does not extrapolate or guarantee monotonicity beyond n_max.
    - Range Exhaustion: returns status='range_exhausted' if no qualifying n* exists in [n_min, n_max].
    """
    if not (0.0 <= p1 <= 1.0) or not (0.0 <= p2 <= 1.0):
        raise ValueError(f"Probabilities p1, p2 must be in [0, 1], got p1={p1}, p2={p2}")
    if p1 <= p2:
        raise ValueError(
            f"Candidate failure rate p1 ({p1}) must be strictly greater than ablation rate p2 ({p2}) "
            "for one-sided superiority testing (H1: p1 > p2)."
        )
    if not (0.0 < target_power < 1.0):
        raise ValueError(f"target_power must be in (0, 1), got {target_power}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_min < 1:
        raise ValueError(f"n_min must be >= 1, got {n_min}")
    if n_max < n_min:
        raise ValueError(f"n_max ({n_max}) must be >= n_min ({n_min})")

    powers: dict[int, float] = {}
    for n in range(n_min, n_max + 1):
        powers[n] = exact_fisher_power(n, p1, p2, alpha=alpha)

    # Monotonic ceiling search: find smallest n* such that for all m in [n*, n_max], powers[m] >= target_power
    qualifying_n: int | None = None
    for n in range(n_min, n_max + 1):
        if all(powers[m] >= target_power for m in range(n, n_max + 1)):
            qualifying_n = n
            break

    if qualifying_n is not None:
        return FisherPowerResult(
            status=PowerSolverStatus.VALIDATED,
            n_required=qualifying_n,
            target_power=target_power,
            power_achieved=round(powers[qualifying_n], 6),
            p1=p1,
            p2=p2,
            alpha=alpha,
            n_min=n_min,
            n_max=n_max,
            powers_by_n=powers,
            diagnostic_message=(
                f"Prospective power validated: required balanced sample size n*={qualifying_n} "
                f"achieves exact power {powers[qualifying_n]:.4f} >= {target_power:.2f} across all "
                f"sample sizes up to n_max={n_max} under prespecified (p1={p1:.2f}, p2={p2:.2f}, alpha={alpha:.3f})."
            ),
        )
    else:
        max_power = max(powers.values()) if powers else 0.0
        power_at_n_max = powers[n_max] if n_max in powers else 0.0
        return FisherPowerResult(
            status=PowerSolverStatus.RANGE_EXHAUSTED,
            n_required=None,
            target_power=target_power,
            power_achieved=round(power_at_n_max, 6),
            p1=p1,
            p2=p2,
            alpha=alpha,
            n_min=n_min,
            n_max=n_max,
            powers_by_n=powers,
            diagnostic_message=(
                f"Search range exhausted: target power {target_power:.2f} cannot be achieved within "
                f"supported domain budget [n_min={n_min}, n_max={n_max}]. Maximum power achieved was "
                f"{max_power:.4f} (power at n_max={n_max} was {power_at_n_max:.4f}) under prespecified "
                f"(p1={p1:.2f}, p2={p2:.2f}, alpha={alpha:.3f})."
            ),
        )


def solve_joint_minimality_power(
    p1: float,
    p2_rates: float | Sequence[float],
    k: int | None = None,
    joint_target_power: float = 0.80,
    alpha: float = 0.05,
    n_min: int = 1,
    n_max: int = 120,
) -> JointPowerResult:
    """Solve balanced sample size per arm for k-component 1-minimality verification.

    Under the Intersection-Union Test (IUT) framework, 1-minimality is rejected if ANY of the k
    component tests fails to reject. By Boole's inequality (union bound):
        P_H1(fail joint rejection) <= sum_{j=1}^k beta_j
    Allocating beta_j <= beta_target / k guarantees joint power >= 1 - beta_target WITHOUT
    assuming positive dependence or independence among the k ablation tests.

    Parameters
    ----------
    p1 : float
        Prespecified candidate failure probability under H1.
    p2_rates : float or Sequence[float]
        Prespecified ablation baseline failure probability. If a float, treated as conservative
        weakest-effect scalar applied to all k components. If a sequence of length k, represents
        heterogeneous per-component ablation rates.
    k : int, optional
        Number of components. Required if p2_rates is a scalar float. If p2_rates is a sequence,
        k must equal len(p2_rates) if provided.
    joint_target_power : float
        Target joint statistical power 1 - beta_target (default 0.80).
    alpha : float
        One-sided significance level per component test (default 0.05).
    n_min, n_max : int
        Search range bounds.

    Returns
    -------
    JointPowerResult
        Overall required sample size max(n_j*) and per-component allocations.
    """
    if isinstance(p2_rates, (int, float)):
        if k is None or k < 1:
            raise ValueError(f"k must be positive integer >= 1 when p2_rates is a scalar, got k={k}")
        p2_tuple = tuple(float(p2_rates) for _ in range(k))
    else:
        p2_tuple = tuple(float(p) for p in p2_rates)
        if len(p2_tuple) == 0:
            raise ValueError("p2_rates sequence cannot be empty")
        if k is not None and k != len(p2_tuple):
            raise ValueError(f"k ({k}) does not match length of p2_rates ({len(p2_tuple)})")
        k = len(p2_tuple)

    if not (0.0 < joint_target_power < 1.0):
        raise ValueError(f"joint_target_power must be in (0, 1), got {joint_target_power}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Boole's inequality allocation: beta_j = beta_joint / k
    joint_beta = 1.0 - joint_target_power
    component_beta = joint_beta / float(k)
    component_target_power = 1.0 - component_beta

    component_results: dict[int, FisherPowerResult] = {}
    any_exhausted = False

    for j, p2 in enumerate(p2_tuple):
        res = solve_fisher_sample_size(
            p1=p1,
            p2=p2,
            target_power=component_target_power,
            alpha=alpha,
            n_min=n_min,
            n_max=n_max,
        )
        component_results[j] = res
        if res.status == PowerSolverStatus.RANGE_EXHAUSTED:
            any_exhausted = True

    if any_exhausted:
        overall_status = PowerSolverStatus.RANGE_EXHAUSTED
        overall_n_req = None
        diag = (
            f"Joint 1-minimality power search exhausted: at least one component ablation test cannot "
            f"achieve component target power {component_target_power:.4f} (allocated via Boole's inequality "
            f"for joint power {joint_target_power:.2f} across k={k} components) within [n_min={n_min}, n_max={n_max}]."
        )
    else:
        overall_status = PowerSolverStatus.VALIDATED
        overall_n_req = max(r.n_required for r in component_results.values() if r.n_required is not None)
        diag = (
            f"Joint 1-minimality power validated: required sample size per arm is n*={overall_n_req} "
            f"guaranteeing joint power >= {joint_target_power:.2f} (joint beta <= {joint_beta:.3f}) across "
            f"k={k} components via conservative union bound allocation (component target power {component_target_power:.4f})."
        )

    return JointPowerResult(
        status=overall_status,
        n_required=overall_n_req,
        k_components=k,
        joint_target_power=joint_target_power,
        joint_beta_target=joint_beta,
        component_target_power=round(component_target_power, 6),
        component_beta_target=round(component_beta, 6),
        p1=p1,
        p2_values=p2_tuple,
        component_results=component_results,
        diagnostic_message=diag,
    )
