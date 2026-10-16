"""Statistical evaluation layer for AgentXRay.

Provides rigorous, dependency-free statistical methods for binary outcomes:
- Wilson score confidence intervals for single binomial proportions.
- Newcombe hybrid score confidence intervals for differences of proportions.
- Fisher's exact test for exact finite-sample hypergeometric inference.
- Two-sample proportion comparison tests with pooled z-statistics and p-values.
- Benjamini-Hochberg False Discovery Rate (FDR) adjustment for multiple testing.
- Statistical power estimation for two independent binomial proportions.

All methods are implemented purely using Python's standard library `math` and `statistics`.
Zero external scientific dependencies (NumPy, SciPy) required.
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
    n1: int = Field(..., ge=1, description="Total trials in group 1.")
    p1: float = Field(..., ge=0.0, le=1.0, description="Sample proportion k1/n1.")
    p1_ci: tuple[float, float] = Field(..., description="Wilson score confidence interval for p1.")

    k2: int = Field(..., ge=0, description="Observed successes in group 2 (e.g. baseline or ablation).")
    n2: int = Field(..., ge=1, description="Total trials in group 2.")
    p2: float = Field(..., ge=0.0, le=1.0, description="Sample proportion k2/n2.")
    p2_ci: tuple[float, float] = Field(..., description="Wilson score confidence interval for p2.")

    difference: float = Field(..., ge=-1.0, le=1.0, description="Risk difference p1 - p2.")
    difference_ci: tuple[float, float] = Field(..., description="Newcombe hybrid score confidence interval for p1 - p2.")
    z_statistic: float = Field(..., description="Z-statistic from two-sample pooled score test.")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Two-sided asymptotic p-value testing H0: p1 == p2.")
    p_value_one_sided: float = Field(..., ge=0.0, le=1.0, description="One-sided asymptotic p-value testing H1: p1 > p2.")
    fisher_exact_p_value: float = Field(..., ge=0.0, le=1.0, description="Exact finite-sample p-value from Fisher's exact test.")
    alternative: str = Field(default="two-sided", description="Prespecified alternative hypothesis: 'two-sided', 'greater', or 'less'.")
    statistically_significant: bool = Field(..., description="Whether test statistic rejects H0 at alpha under the prespecified alternative.")


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

    Uses Newcombe's Method 10 (without continuity correction) combining independent Wilson intervals.

    Parameters
    ----------
    k1, n1 : int
        Successes and trials for group 1 (0 <= k1 <= n1, n1 > 0).
    k2, n2 : int
        Successes and trials for group 2 (0 <= k2 <= n2, n2 > 0).
    confidence : float
        Confidence level, default 0.95.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) for the risk difference p1 - p2, bounded in [-1.0, 1.0].
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"Sample sizes must be positive integers, got n1={n1}, n2={n2}")
    if k1 < 0 or k1 > n1:
        raise ValueError(f"k1 must be in [0, n1], got k1={k1}, n1={n1}")
    if k2 < 0 or k2 > n2:
        raise ValueError(f"k2 must be in [0, n2], got k2={k2}, n2={n2}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    p1_hat = k1 / n1
    p2_hat = k2 / n2
    diff = p1_hat - p2_hat

    l1, u1 = wilson_score_interval(k1, n1, confidence)
    l2, u2 = wilson_score_interval(k2, n2, confidence)

    lower = diff - math.sqrt((p1_hat - l1) ** 2 + (u2 - p2_hat) ** 2)
    upper = diff + math.sqrt((u1 - p1_hat) ** 2 + (p2_hat - l2) ** 2)

    return (round(max(-1.0, lower), 4), round(min(1.0, upper), 4))


def fisher_exact_test(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    alternative: str = "two-sided",
) -> float:
    """Perform Fisher's exact test for a 2x2 contingency table using exact hypergeometric distribution.

    Implemented using Python's standard library `math.comb`. Zero external dependencies.

    Contingency table:
                Group 1       Group 2       Total
    Success:      k1            k2            K = k1 + k2
    Failure:    n1 - k1       n2 - k2       N - K
    Total:        n1            n2            N = n1 + n2

    Parameters
    ----------
    k1, n1 : int
        Successes and trials for group 1.
    k2, n2 : int
        Successes and trials for group 2.
    alternative : str
        'two-sided', 'greater' (H1: p1 > p2), or 'less' (H1: p1 < p2).

    Returns
    -------
    float
        Exact p-value in [0.0, 1.0].
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
    probs = {
        x: math.comb(n1, x) * math.comb(n2, K - x) / denom
        for x in range(min_x, max_x + 1)
    }
    obs_prob = probs[k1]

    if alternative == "greater":
        p_val = sum(p for x, p in probs.items() if x >= k1)
    elif alternative == "less":
        p_val = sum(p for x, p in probs.items() if x <= k1)
    else:  # two-sided
        # Standard two-sided Fisher: sum probabilities of tables as or less probable than observed
        p_val = sum(p for p in probs.values() if p <= obs_prob * (1.0 + 1e-9))

    return round(min(1.0, max(0.0, p_val)), 5)


def compare_proportions(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    alpha: float = 0.05,
    confidence: float = 0.95,
    alternative: str = "two-sided",
) -> ProportionComparisonResult:
    """Perform a two-sample proportion comparison between group 1 and group 2.

    Computes:
    - Sample proportions and Wilson score intervals.
    - Risk difference p1 - p2 and Newcombe hybrid score interval.
    - Two-sample pooled z-statistic and asymptotic p-values (two-sided and one-sided).
    - Exact finite-sample p-value via Fisher's exact test.

    Parameters
    ----------
    k1, n1 : int
        Failures and trials for group 1 (candidate condition).
    k2, n2 : int
        Failures and trials for group 2 (control/baseline/reduced).
    alpha : float
        Significance threshold for hypothesis testing (e.g. 0.05).
    confidence : float
        Confidence level for uncertainty intervals (e.g. 0.95).
    alternative : str
        Prespecified alternative: 'two-sided' (p1 != p2), 'greater' (p1 > p2), or 'less' (p1 < p2).

    Returns
    -------
    ProportionComparisonResult
        Full statistical metrics and significance decision.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"Sample sizes must be positive, got n1={n1}, n2={n2}")
    if k1 < 0 or k1 > n1 or k2 < 0 or k2 > n2:
        raise ValueError(f"Success counts invalid: k1={k1}/n1={n1}, k2={k2}/n2={n2}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less', got {alternative!r}")

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

    fisher_p = fisher_exact_test(k1, n1, k2, n2, alternative=alternative)

    # Significance decision aligned with prespecified alternative hypothesis
    if alternative == "two-sided":
        sig = (p_val_two < alpha)
    elif alternative == "greater":
        sig = (p_val_one < alpha and diff > 0.0)
    else:  # less
        sig = ((1.0 - p_val_one) < alpha and diff < 0.0)

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
        fisher_exact_p_value=fisher_p,
        alternative=alternative,
        statistically_significant=sig,
    )


def adjust_p_values_bh(p_values: Sequence[float]) -> list[float]:
    """Adjust a sequence of p-values using the Benjamini-Hochberg (BH) FDR procedure.

    Controls the False Discovery Rate across a family of m hypothesis tests.

    Parameters
    ----------
    p_values : Sequence[float]
        Raw p-values to adjust. Values outside [0.0, 1.0] are clamped to [0.0, 1.0].

    Returns
    -------
    list[float]
        FDR-adjusted p-values (q-values) mapped back to original input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    for i, p in enumerate(p_values):
        if math.isnan(p):
            raise ValueError(f"p-value at index {i} cannot be NaN")

    clamped = [min(1.0, max(0.0, p)) for p in p_values]

    if m == 1:
        return [round(clamped[0], 5)]

    # Pair each p-value with its original index
    indexed = sorted(enumerate(clamped), key=lambda x: x[1])

    adjusted_indexed: list[tuple[int, float]] = [None] * m  # type: ignore
    cum_min = 1.0

    # Step-up adjustment from largest to smallest rank
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
    alternative: str = "two-sided",
) -> float:
    """Estimate theoretical statistical power for detecting a difference between two proportions.

    Uses Cohen's arcsine transformation for two independent binomial samples of equal size n.

    IMPORTANT: This is an asymptotic, approximate power calculation intended strictly
    for pre-experimental study design and sample size planning. It does NOT represent
    observed empirical power on a specific executed dataset.

    Parameters
    ----------
    p1, p2 : float
        Hypothesized true outcome probabilities in group 1 and group 2 (in [0, 1]).
    n_per_group : int
        Number of trials per group (n >= 1).
    alpha : float
        Significance level (e.g. 0.05).
    alternative : str
        'two-sided' (default) or 'greater'.

    Returns
    -------
    float
        Estimated statistical power in [0.0, 1.0].
    """
    if n_per_group <= 0:
        return 0.0
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError(f"Probabilities must be in [0, 1], got p1={p1}, p2={p2}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if alternative not in ("two-sided", "greater"):
        raise ValueError(f"alternative must be 'two-sided' or 'greater', got {alternative!r}")

    # Cohen's h effect size: 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    phi1 = 2.0 * math.asin(math.sqrt(p1))
    phi2 = 2.0 * math.asin(math.sqrt(p2))
    h = abs(phi1 - phi2)

    if h == 0.0:
        return alpha

    se = math.sqrt(2.0 / n_per_group)

    if alternative == "two-sided":
        z_crit = _STANDARD_NORMAL.inv_cdf(1.0 - alpha / 2.0)
        lam = h / se
        power = _STANDARD_NORMAL.cdf(lam - z_crit) + _STANDARD_NORMAL.cdf(-lam - z_crit)
    else:  # greater (one-sided)
        z_crit = _STANDARD_NORMAL.inv_cdf(1.0 - alpha)
        lam = h / se
        power = _STANDARD_NORMAL.cdf(lam - z_crit)

    return round(min(1.0, max(0.0, power)), 4)


class PairedDifferenceSummary(BaseModel):
    """Summary of paired difference analysis between two algorithm configurations."""

    model_config = ConfigDict(frozen=True)

    n_pairs: int = Field(..., ge=1, description="Number of paired evaluations.")
    mean_difference: float = Field(..., description="Mean of paired differences (x - y).")
    std_difference: float = Field(..., ge=0.0, description="Standard deviation of differences.")
    median_difference: float = Field(..., description="Median of paired differences.")
    wilcoxon_w: float = Field(..., description="Wilcoxon signed-rank W+ statistic.")
    wilcoxon_p_value: float = Field(..., ge=0.0, le=1.0, description="Two-sided Wilcoxon p-value.")
    paired_t_statistic: float = Field(..., description="Paired t-statistic.")
    paired_t_p_value: float = Field(..., ge=0.0, le=1.0, description="Two-sided paired t-test p-value.")


def wilcoxon_signed_rank_test(
    x: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test for paired continuous/discrete observations.

    Parameters
    ----------
    x, y : Sequence[float]
        Paired numeric samples of identical length.

    Returns
    -------
    tuple[float, float]
        (W_statistic, two_sided_p_value)
    """
    if len(x) != len(y):
        raise ValueError(f"Sample lengths must match: len(x)={len(x)}, len(y)={len(y)}")
    if len(x) == 0:
        raise ValueError("Samples cannot be empty")

    diffs = [xi - yi for xi, yi in zip(x, y)]
    nonzero_diffs = [d for d in diffs if d != 0.0]
    n_r = len(nonzero_diffs)

    if n_r == 0:
        return (0.0, 1.0)

    # Sort absolute differences
    abs_diffs = sorted((abs(d), d > 0) for d in nonzero_diffs)

    # Assign average ranks for ties
    ranks: list[tuple[float, bool]] = []
    i = 0
    tie_lengths: list[int] = []
    while i < n_r:
        j = i
        while j < n_r and math.isclose(abs_diffs[j][0], abs_diffs[i][0], rel_tol=1e-9, abs_tol=1e-9):
            j += 1
        tie_len = j - i
        tie_lengths.append(tie_len)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append((avg_rank, abs_diffs[k][1]))
        i = j

    w_plus = sum(r for r, is_pos in ranks if is_pos)

    # Expected value and variance under null hypothesis
    mean_w = n_r * (n_r + 1.0) / 4.0
    tie_correction = sum(t ** 3 - t for t in tie_lengths) / 48.0
    var_w = (n_r * (n_r + 1.0) * (2.0 * n_r + 1.0) / 24.0) - tie_correction

    if var_w <= 0.0:
        return (w_plus, 1.0)

    # Continuity correction
    diff_from_mean = w_plus - mean_w
    if diff_from_mean > 0:
        z = (diff_from_mean - 0.5) / math.sqrt(var_w)
    elif diff_from_mean < 0:
        z = (diff_from_mean + 0.5) / math.sqrt(var_w)
    else:
        z = 0.0

    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return (round(w_plus, 4), round(min(1.0, max(0.0, p_value)), 5))


def compare_paired_differences(
    x: Sequence[float],
    y: Sequence[float],
) -> PairedDifferenceSummary:
    """Compute comprehensive paired difference statistics between samples x and y."""
    if len(x) != len(y) or len(x) == 0:
        raise ValueError(f"Inputs must be non-empty and equal length, got {len(x)} and {len(y)}")

    n = len(x)
    diffs = [float(xi - yi) for xi, yi in zip(x, y)]
    mean_d = sum(diffs) / n

    if n > 1:
        variance_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
        std_d = math.sqrt(variance_d)
    else:
        std_d = 0.0

    sorted_d = sorted(diffs)
    if n % 2 == 1:
        med_d = sorted_d[n // 2]
    else:
        med_d = (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2.0

    # Paired t-statistic
    se_mean = (std_d / math.sqrt(n)) if n > 0 else 0.0
    if se_mean > 0.0:
        t_stat = mean_d / se_mean
        t_p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    else:
        t_stat = 0.0
        t_p_val = 1.0 if mean_d == 0.0 else 0.0

    w_stat, w_p_val = wilcoxon_signed_rank_test(x, y)

    return PairedDifferenceSummary(
        n_pairs=n,
        mean_difference=round(mean_d, 4),
        std_difference=round(std_d, 4),
        median_difference=round(med_d, 4),
        wilcoxon_w=w_stat,
        wilcoxon_p_value=w_p_val,
        paired_t_statistic=round(t_stat, 4),
        paired_t_p_value=round(t_p_val, 5),
    )

