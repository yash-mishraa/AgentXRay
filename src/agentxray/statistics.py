"""Statistical evaluation layer for AgentXRay.

Provides rigorous, dependency-free statistical methods for binary outcomes:
- Wilson score confidence intervals for single binomial proportions.
- Newcombe hybrid score confidence intervals for differences of proportions.
- Two-sample proportion comparison tests with effect sizes and p-values.
- Benjamini-Hochberg False Discovery Rate (FDR) adjustment for multiple testing.
- Statistical power estimation for two independent binomial proportions.

All methods are implemented using Python's standard library `math` and `statistics`.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

_STANDARD_NORMAL = NormalDist()


class ProportionComparisonResult(BaseModel):
    """Result of a statistical comparison between two independent binomial samples."""

    model_config = ConfigDict(frozen=True)

    k1: int = Field(..., ge=0, description="Observed successes in group 1 (e.g. candidate condition).")
    n1: int = Field(..., ge=0, description="Total trials in group 1.")
    p1: float = Field(..., ge=0.0, le=1.0, description="Sample proportion k1/n1.")
    p1_ci: tuple[float, float] = Field(..., description="Wilson score confidence interval for p1.")

    k2: int = Field(..., ge=0, description="Observed successes in group 2 (e.g. baseline or ablation).")
    n2: int = Field(..., ge=0, description="Total trials in group 2.")
    p2: float = Field(..., ge=0.0, le=1.0, description="Sample proportion k2/n2.")
    p2_ci: tuple[float, float] = Field(..., description="Wilson score confidence interval for p2.")

    difference: float = Field(..., ge=-1.0, le=1.0, description="Risk difference p1 - p2.")
    difference_ci: tuple[float, float] = Field(..., description="Newcombe score confidence interval for p1 - p2.")
    z_statistic: float = Field(..., description="Z-statistic from two-sample pooled z-test.")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Two-sided p-value testing H0: p1 == p2.")
    p_value_one_sided: float = Field(..., ge=0.0, le=1.0, description="One-sided p-value testing H1: p1 > p2.")
    statistically_significant: bool = Field(..., description="Whether p_value < alpha.")


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of observed events / failures (k >= 0).
    trials : int
        Total number of trials (n >= 0).
    confidence : float
        Confidence level, e.g. 0.95 for a 95% interval.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) strictly bounded in [0.0, 1.0].
    """
    if trials < 0:
        raise ValueError(f"trials must be non-negative, got {trials}")
    if successes < 0:
        raise ValueError(f"successes must be non-negative, got {successes}")
    if successes > trials:
        raise ValueError(f"successes ({successes}) cannot exceed trials ({trials})")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if trials == 0:
        return (0.0, 1.0)

    p_hat = successes / trials
    alpha = 1.0 - confidence
    z = _STANDARD_NORMAL.inv_cdf(1.0 - alpha / 2.0)
    z2 = z * z

    denominator = 1.0 + z2 / trials
    center = (p_hat + z2 / (2.0 * trials)) / denominator
    variance_term = (p_hat * (1.0 - p_hat)) / trials + z2 / (4.0 * trials * trials)
    margin = (z / denominator) * math.sqrt(variance_term)

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


def proportion_difference_interval(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate the Newcombe hybrid score interval for the difference of two proportions (p1 - p2).

    Parameters
    ----------
    k1, n1 : int
        Successes and trials for group 1.
    k2, n2 : int
        Successes and trials for group 2.
    confidence : float
        Confidence level, default 0.95.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) for the risk difference p1 - p2, bounded in [-1.0, 1.0].
    """
    if n1 <= 0 or n2 <= 0:
        return (-1.0, 1.0)

    p1_hat = k1 / n1
    p2_hat = k2 / n2
    diff = p1_hat - p2_hat

    l1, u1 = wilson_score_interval(k1, n1, confidence)
    l2, u2 = wilson_score_interval(k2, n2, confidence)

    lower = diff - math.sqrt((p1_hat - l1) ** 2 + (u2 - p2_hat) ** 2)
    upper = diff + math.sqrt((u1 - p1_hat) ** 2 + (p2_hat - l2) ** 2)

    return (round(max(-1.0, lower), 4), round(min(1.0, upper), 4))


def compare_proportions(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> ProportionComparisonResult:
    """Perform a two-sample proportion comparison between group 1 and group 2.

    Uses a pooled two-sample z-test and Newcombe hybrid score interval.

    Parameters
    ----------
    k1, n1 : int
        Failures and trials for group 1 (candidate).
    k2, n2 : int
        Failures and trials for group 2 (control/baseline/reduced).
    alpha : float
        Significance threshold for hypothesis testing.
    confidence : float
        Confidence level for uncertainty intervals.

    Returns
    -------
    ProportionComparisonResult
        Full statistical comparison metrics.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"Sample sizes must be positive, got n1={n1}, n2={n2}")

    p1_hat = k1 / n1
    p2_hat = k2 / n2
    diff = p1_hat - p2_hat

    p1_ci = wilson_score_interval(k1, n1, confidence)
    p2_ci = wilson_score_interval(k2, n2, confidence)
    diff_ci = proportion_difference_interval(k1, n1, k2, n2, confidence)

    # Pooled z-test under H0: p1 == p2
    pooled_p = (k1 + k2) / (n1 + n2)
    se_pooled = math.sqrt(pooled_p * (1.0 - pooled_p) * (1.0 / n1 + 1.0 / n2))

    if se_pooled == 0.0:
        z_stat = 0.0
        p_val_two = 1.0
        p_val_one = 1.0 if diff <= 0.0 else 0.0
    else:
        z_stat = diff / se_pooled
        p_val_two = math.erfc(abs(z_stat) / math.sqrt(2.0))
        p_val_one = 1.0 - _STANDARD_NORMAL.cdf(z_stat)

    return ProportionComparisonResult(
        k1=k1,
        n1=n1,
        p1=round(p1_hat, 4),
        p1_ci=p1_ci,
        k2=k2,
        n2=n2,
        p2=round(p2_hat, 4),
        p2_ci=p2_ci,
        difference=round(diff, 4),
        difference_ci=diff_ci,
        z_statistic=round(z_stat, 4),
        p_value=round(p_val_two, 5),
        p_value_one_sided=round(p_val_one, 5),
        statistically_significant=(p_val_two < alpha),
    )


def adjust_p_values_bh(p_values: Sequence[float]) -> list[float]:
    """Adjust a sequence of p-values using the Benjamini-Hochberg (BH) FDR procedure.

    Parameters
    ----------
    p_values : Sequence[float]
        Raw p-values to adjust.

    Returns
    -------
    list[float]
        FDR-adjusted p-values (q-values) in original index order.
    """
    m = len(p_values)
    if m == 0:
        return []
    if m == 1:
        return [min(1.0, max(0.0, p_values[0]))]

    # Pair each p-value with its original index
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Compute step-up adjustments from largest to smallest
    adjusted_indexed: list[tuple[int, float]] = [None] * m  # type: ignore
    cum_min = 1.0

    for rank in range(m, 0, -1):
        orig_idx, p = indexed[rank - 1]
        adjusted_val = min(1.0, p * m / rank)
        cum_min = min(cum_min, adjusted_val)
        adjusted_indexed[rank - 1] = (orig_idx, cum_min)

    # Restore original input order
    result = [0.0] * m
    for orig_idx, q in adjusted_indexed:
        result[orig_idx] = round(q, 5)

    return result


def estimate_two_proportion_power(
    p1: float,
    p2: float,
    n_per_group: int,
    alpha: float = 0.05,
) -> float:
    """Estimate theoretical statistical power for detecting a difference between two proportions.

    Uses Cohen's arcsine transformation for two independent binomial samples of size n.

    Parameters
    ----------
    p1, p2 : float
        True outcome probabilities in group 1 and group 2 (in [0, 1]).
    n_per_group : int
        Number of trials per group (n >= 1).
    alpha : float
        Two-sided significance level (e.g. 0.05).

    Returns
    -------
    float
        Estimated statistical power in [0.0, 1.0].
    """
    if n_per_group <= 0:
        return 0.0
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("Probabilities must be in [0, 1]")

    # Cohen's h effect size: 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    phi1 = 2.0 * math.asin(math.sqrt(p1))
    phi2 = 2.0 * math.asin(math.sqrt(p2))
    h = abs(phi1 - phi2)

    if h == 0.0:
        return alpha

    # Variance of difference of arcsine transformed proportions is 1/n1 + 1/n2 = 2/n
    se = math.sqrt(2.0 / n_per_group)
    z_crit = _STANDARD_NORMAL.inv_cdf(1.0 - alpha / 2.0)

    # Non-centrality parameter lambda = h / se
    lam = h / se
    power = _STANDARD_NORMAL.cdf(lam - z_crit) + _STANDARD_NORMAL.cdf(-lam - z_crit)
    return round(min(1.0, max(0.0, power)), 4)
