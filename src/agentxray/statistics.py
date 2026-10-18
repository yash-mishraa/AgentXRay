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


def exact_binomial_test(
    k: int,
    n: int,
    p0: float = 0.50,
    alternative: str = "less",
) -> float:
    """Perform exact binomial test for a single proportion using standard library math.comb.

    Zero external dependencies.

    Parameters
    ----------
    k : int
        Observed number of successes/failures (0 <= k <= n).
    n : int
        Total number of trials (n > 0).
    p0 : float
        Hypothesized proportion under H0 (0.0 < p0 < 1.0), default 0.50.
    alternative : str
        'less' (H1: p < p0), 'greater' (H1: p > p0), or 'two-sided'.

    Returns
    -------
    float
        Exact p-value in [0.0, 1.0].
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    if not (0.0 < p0 < 1.0):
        raise ValueError(f"p0 must be in (0, 1), got {p0}")
    if alternative not in ("less", "greater", "two-sided"):
        raise ValueError(f"alternative must be 'less', 'greater', or 'two-sided', got {alternative!r}")

    if alternative == "less":
        p_val = sum(math.comb(n, x) * (p0 ** x) * ((1.0 - p0) ** (n - x)) for x in range(k + 1))
    elif alternative == "greater":
        p_val = sum(math.comb(n, x) * (p0 ** x) * ((1.0 - p0) ** (n - x)) for x in range(k, n + 1))
    else:  # two-sided
        obs_prob = math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
        all_probs = [math.comb(n, x) * (p0 ** x) * ((1.0 - p0) ** (n - x)) for x in range(n + 1)]
        p_val = sum(p for p in all_probs if p <= obs_prob * (1.0 + 1e-9))

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
    """Adjust a sequence of p-values using the Benjamini-Hochberg (BH) FDR step-up procedure.

    Controls the False Discovery Rate (FDR = E[V / max(R, 1)]) across a family of m hypothesis tests.

    IMPORTANT STATISTICAL DESIGNATION:
    The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR) under independent
    or positive regression dependent (PRDS) test statistics. It does NOT control the Family-Wise
    Error Rate (FWER = P(V >= 1)). FWER is controlled by Bonferroni or Holm-Bonferroni procedures.
    A rejection at FDR threshold alpha guarantees that the expected proportion of false discoveries
    among all rejected hypotheses does not exceed alpha.

    Parameters
    ----------
    p_values : Sequence[float]
        Raw p-values to adjust across the family of m hypotheses.
        Values outside [0.0, 1.0] are clamped to [0.0, 1.0].

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
        q0 = clamped[0]
        return [round(q0, 5) if q0 >= 1e-5 else q0]

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
        result[orig_idx] = round(q, 5) if q >= 1e-5 else q

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
    wilcoxon_p_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Two-sided Wilcoxon asymptotic p-value (normal approximation with continuity and tie corrections).",
    )
    wilcoxon_p_value_is_asymptotic: bool = Field(
        default=True,
        description="Explicit indicator that the Wilcoxon p-value is computed via asymptotic normal approximation, not exact combinatorial distribution.",
    )
    paired_t_statistic: float = Field(..., description="Paired t-statistic.")
    paired_t_p_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Two-sided paired t-test exact Student's t distribution p-value (via regularized incomplete beta).",
    )


def regularized_incomplete_beta(
    a: float,
    b: float,
    x: float,
    max_iter: int = 200,
    tol: float = 1e-15,
) -> float:
    """Compute the regularized incomplete beta function I_x(a, b).

    Uses Lentz's continued fraction expansion with symmetry transformation.
    Implemented purely with Python standard library `math`.

    Parameters
    ----------
    a, b : float
        Positive shape parameters.
    x : float
        Integration upper bound in [0.0, 1.0].
    max_iter : int
        Maximum iterations for continued fraction convergence.
    tol : float
        Relative convergence tolerance.

    Returns
    -------
    float
        Value of I_x(a, b) in [0.0, 1.0].
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"Parameters a and b must be positive, got a={a}, b={b}")
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Symmetry relation for faster/stable convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x, max_iter, tol)

    # Front factor: exp(a*ln(x) + b*ln(1-x) - ln(Beta(a, b))) / a
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a

    tiny = 1e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step: 2m
        num_even = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        d = 1.0 + num_even * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num_even / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        f *= (c * d)

        # Odd step: 2m + 1
        num_odd = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + num_odd * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num_odd / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < tol:
            break

    val = front * f
    return min(1.0, max(0.0, val))


def student_t_distribution_p_value(
    t_stat: float,
    df: int,
    alternative: str = "two-sided",
) -> float:
    """Calculate exact p-value for Student's t distribution.

    Uses regularized incomplete beta function:
    Two-sided p-value = I_{df / (df + t^2)}(df / 2, 1 / 2).

    Parameters
    ----------
    t_stat : float
        Observed t-statistic.
    df : int
        Degrees of freedom (must be >= 1).
    alternative : str
        'two-sided', 'greater' (t > 0), or 'less' (t < 0).

    Returns
    -------
    float
        Exact p-value in [0.0, 1.0].
    """
    if df < 1:
        raise ValueError(f"Degrees of freedom must be >= 1, got {df}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less', got {alternative!r}")

    if t_stat == 0.0:
        return 1.0 if alternative == "two-sided" else 0.5

    x = df / (df + t_stat * t_stat)
    p_two_sided = regularized_incomplete_beta(df / 2.0, 0.5, x)

    if alternative == "two-sided":
        return p_two_sided
    elif alternative == "greater":
        return p_two_sided / 2.0 if t_stat > 0 else 1.0 - p_two_sided / 2.0
    else:  # less
        return p_two_sided / 2.0 if t_stat < 0 else 1.0 - p_two_sided / 2.0


def wilcoxon_exact_p_value_small_sample(
    w_plus: float,
    n_r: int,
) -> float:
    """Compute exact two-sided combinatorial p-value for Wilcoxon signed-rank test without ties.

    Uses dynamic programming over subset sums of {1, ..., n_r} under the null hypothesis
    that all 2^{n_r} sign configurations are equally likely.

    Applicable strictly when there are no ties and n_r <= 25.

    Parameters
    ----------
    w_plus : float
        Observed Wilcoxon W+ statistic (sum of positive ranks).
    n_r : int
        Number of non-zero paired differences (1 <= n_r <= 25).

    Returns
    -------
    float
        Exact two-sided p-value in [0.0, 1.0].
    """
    if n_r < 1:
        raise ValueError(f"n_r must be positive integer, got {n_r}")
    if n_r > 25:
        raise ValueError(f"Exact combinatorial calculation restricted to n_r <= 25, got {n_r}")

    max_w = n_r * (n_r + 1) // 2
    w_int = int(round(w_plus))
    if w_int < 0 or w_int > max_w:
        raise ValueError(f"w_plus must be in [0, {max_w}], got {w_plus}")

    dp = [0] * (max_w + 1)
    dp[0] = 1
    for i in range(1, n_r + 1):
        for s in range(max_w, i - 1, -1):
            dp[s] += dp[s - i]

    w_min = min(w_int, max_w - w_int)
    tail_sum = sum(dp[: w_min + 1])
    total_permutations = 1 << n_r
    return min(1.0, max(0.0, 2.0 * tail_sum / total_permutations))


def wilcoxon_signed_rank_test(
    x: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test for paired continuous/discrete observations.

    IMPORTANT STATISTICAL DESIGNATION:
    The p-value returned is an ASYMPTOTIC NORMAL APPROXIMATION with Wilcoxon (1945) zero-discard,
    average-rank tie correction, and Yates-style continuity correction (+-0.5).
    It is NOT an exact combinatorial distribution result.
    For large paired samples (such as N=400, n_r=340), exact combinatorial enumeration over
    2^340 combinations is computationally intractable, making the continuity-corrected
    asymptotic normal approximation the standard analytical procedure.

    Treatment of edge cases:
    - Zero differences (d_i == 0): discarded, reducing sample size to n_r.
    - Tied absolute differences: assigned average ranks, and null variance is adjusted
      by subtracting sum(t_i^3 - t_i) / 48.
    - Continuity correction: numerator shifted +-0.5 toward the null mean n_r(n_r+1)/4.

    Parameters
    ----------
    x, y : Sequence[float]
        Paired numeric samples of identical length.

    Returns
    -------
    tuple[float, float]
        (W_plus_statistic, two_sided_asymptotic_p_value)
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
    p_clamped = min(1.0, max(0.0, p_value))
    # Preserve full floating point precision for extreme p-values (< 1e-4) while rounding ordinary p-values
    p_ret = round(p_clamped, 5) if p_clamped >= 1e-4 else p_clamped
    return (round(w_plus, 4), p_ret)


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

    # Exact paired t-test using Student's t distribution with df = n - 1
    se_mean = (std_d / math.sqrt(n)) if n > 0 else 0.0
    if se_mean > 0.0 and n > 1:
        t_stat = mean_d / se_mean
        t_p_val = student_t_distribution_p_value(t_stat, df=n - 1, alternative="two-sided")
    else:
        t_stat = 0.0
        t_p_val = 1.0 if mean_d == 0.0 else 0.0

    w_stat, w_p_val = wilcoxon_signed_rank_test(x, y)

    t_p_ret = round(t_p_val, 6) if t_p_val >= 1e-6 else t_p_val

    return PairedDifferenceSummary(
        n_pairs=n,
        mean_difference=round(mean_d, 4),
        std_difference=round(std_d, 4),
        median_difference=round(med_d, 4),
        wilcoxon_w=w_stat,
        wilcoxon_p_value=w_p_val,
        paired_t_statistic=round(t_stat, 4),
        paired_t_p_value=t_p_ret,
    )


class McNemarTestResult(BaseModel):
    """Result of exact McNemar test on paired binary observations."""

    model_config = ConfigDict(frozen=True)

    b: int = Field(..., ge=0, description="Discordant pairs favoring method 1 (e.g. Judge error, ActiveDD correct).")
    c: int = Field(..., ge=0, description="Discordant pairs favoring method 2 (e.g. Judge correct, ActiveDD error).")
    n_discordant: int = Field(..., ge=0, description="Total discordant pairs b + c.")
    odds_ratio: float | None = Field(default=None, description="Discordant odds ratio b / c (or None/inf).")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Exact binomial p-value.")
    alternative: str = Field(default="greater", description="Alternative hypothesis: 'greater' (b > c), 'two-sided', or 'less'.")
    statistically_significant: bool = Field(..., description="Whether p_value <= alpha.")


def mcnemar_exact_test(
    b: int,
    c: int,
    alternative: str = "greater",
    alpha: float = 0.05,
) -> McNemarTestResult:
    """Compute exact McNemar test for paired binary proportions.

    Under H0: P(b) == P(c), the conditional distribution of b given n_d = b + c
    follows Binomial(n_d, 0.5).

    Parameters
    ----------
    b : int
        Count of discordant pairs where method 1 fails and method 2 succeeds (e.g. Y_judge = 1, Y_activedd = 0).
    c : int
        Count of discordant pairs where method 1 succeeds and method 2 fails (e.g. Y_judge = 0, Y_activedd = 1).
    alternative : str
        Direction of H1:
        - "greater": H1 is b > c (favors ActiveDD over Judge).
        - "two-sided": H1 is b != c.
        - "less": H1 is b < c.
    alpha : float
        Nominal significance level.
    """
    if b < 0 or c < 0:
        raise ValueError(f"b and c must be non-negative integers, got b={b}, c={c}")
    if alternative not in ("greater", "two-sided", "less"):
        raise ValueError(f"alternative must be 'greater', 'two-sided', or 'less', got {alternative!r}")

    n_d = b + c
    if n_d == 0:
        return McNemarTestResult(
            b=b,
            c=c,
            n_discordant=0,
            odds_ratio=None,
            p_value=1.0,
            alternative=alternative,
            statistically_significant=False,
        )

    # Odds ratio
    if c == 0:
        odds_ratio = float("inf") if b > 0 else None
    else:
        odds_ratio = round(b / c, 4)

    # Compute exact binomial probability: b ~ Binomial(n_d, 0.5)
    if alternative == "greater":
        # P(K >= b)
        prob = sum(math.comb(n_d, k) for k in range(b, n_d + 1)) * (0.5 ** n_d)
        p_val = min(1.0, max(0.0, prob))
    elif alternative == "less":
        # P(K <= b)
        prob = sum(math.comb(n_d, k) for k in range(0, b + 1)) * (0.5 ** n_d)
        p_val = min(1.0, max(0.0, prob))
    else:  # two-sided
        k_extreme = max(b, c)
        prob = 2.0 * sum(math.comb(n_d, k) for k in range(k_extreme, n_d + 1)) * (0.5 ** n_d)
        p_val = min(1.0, max(0.0, prob))

    sig = bool((p_val <= alpha) and (b > c if alternative == "greater" else (b < c if alternative == "less" else b != c)))
    return McNemarTestResult(
        b=b,
        c=c,
        n_discordant=n_d,
        odds_ratio=odds_ratio,
        p_value=p_val,
        alternative=alternative,
        statistically_significant=sig,
    )


class ContrastSensitivityResult(BaseModel):
    """Non-parametric Manski-Horowitz sensitivity bounds for directional contrast Delta_Risk."""

    model_config = ConfigDict(frozen=True)

    n_evaluable: int = Field(..., ge=1, description="Number of mechanisms in retained analysis cohort.")
    n_complete: int = Field(..., ge=0, description="Number of complete pairs with valid diagnoses for both methods.")
    n_miss_judge: int = Field(..., ge=0, description="Count of missing diagnoses for Judge.")
    n_miss_activedd: int = Field(..., ge=0, description="Count of missing diagnoses for ActiveDD.")
    complete_case_risk_difference: float | None = Field(default=None, description="Observed risk difference on complete cases.")
    upper_bound: float = Field(..., description="Upper bound on Delta_Risk (imputing missing Judge=1, ActiveDD=0; wholly missing pairs contribute +1).")
    lower_bound: float = Field(..., description="Lower bound on Delta_Risk (imputing missing Judge=0, ActiveDD=1; wholly missing pairs contribute -1).")
    bounds_disclaimer: str = Field(
        default="Bounds describe the retained benchmark cohort under extreme missingness; they do not automatically bound target-population risk if infrastructure attrition is non-random.",
        description="Methodological disclaimer on population generalization.",
    )


def compute_contrast_sensitivity_bounds(
    y_judge: Sequence[int | None],
    y_activedd: Sequence[int | None],
) -> ContrastSensitivityResult:
    r"""Compute Manski-Horowitz extreme-value bounds for Delta_Risk = P(Y_judge=1) - P(Y_activedd=1).

    The estimand is the mean per-mechanism false-attribution risk difference across
    the retained evaluable cohort of size n = N_evaluable:
        Delta_Risk = (1/n) * sum_{i=1}^n (Y_judge,i - Y_activedd,i)

    Observed indicators are held fixed. For missing outcomes:
    - Upper bound (maximizing Delta_Risk):
      Imputes missing Y_judge = 1 and missing Y_activedd = 0.
      Numerator: sum_{obs} Y_judge + miss_judge - sum_{obs} Y_activedd.
      Wholly missing pairs (both missing) contribute 1 - 0 = +1 to the upper bound.
    - Lower bound (minimizing Delta_Risk):
      Imputes missing Y_judge = 0 and missing Y_activedd = 1.
      Numerator: sum_{obs} Y_judge - (sum_{obs} Y_activedd + miss_activedd).
      Wholly missing pairs (both missing) contribute 0 - 1 = -1 to the lower bound.

    Consequently, wholly missing pairs do NOT cancel in either extreme-value bound;
    each wholly missing pair widens the sensitivity interval by 2 / n.

    Parameters
    ----------
    y_judge : Sequence[int | None]
        Observed false-attribution indicators for Judge (1, 0, or None if missing).
    y_activedd : Sequence[int | None]
        Observed false-attribution indicators for ActiveDD (1, 0, or None if missing).

    Returns
    -------
    ContrastSensitivityResult
        Complete-case estimate, extreme imputation bounds [lower_bound, upper_bound],
        cohort counts, and methodological disclaimers.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or if the cohort is empty (n == 0).
    """
    if len(y_judge) != len(y_activedd):
        raise ValueError(f"Sequences must have equal length, got {len(y_judge)} and {len(y_activedd)}")
    n = len(y_judge)
    if n == 0:
        raise ValueError("Cohort cannot be empty")

    complete_pairs = [(yj, ya) for yj, ya in zip(y_judge, y_activedd) if yj is not None and ya is not None]
    n_complete = len(complete_pairs)
    if n_complete > 0:
        cc_diff = sum((yj - ya) for yj, ya in complete_pairs) / n_complete
        cc_diff_val = round(cc_diff, 4)
    else:
        cc_diff_val = None

    obs_judge_sum = sum(yj for yj in y_judge if yj is not None)
    obs_activedd_sum = sum(ya for ya in y_activedd if ya is not None)
    miss_judge = sum(1 for yj in y_judge if yj is None)
    miss_activedd = sum(1 for ya in y_activedd if ya is None)

    # Upper bound: Judge worst case (1), ActiveDD best case (0)
    upper = (obs_judge_sum + miss_judge - (obs_activedd_sum + 0)) / n

    # Lower bound: Judge best case (0), ActiveDD worst case (1)
    lower = (obs_judge_sum + 0 - (obs_activedd_sum + miss_activedd)) / n

    return ContrastSensitivityResult(
        n_evaluable=n,
        n_complete=n_complete,
        n_miss_judge=miss_judge,
        n_miss_activedd=miss_activedd,
        complete_case_risk_difference=cc_diff_val,
        upper_bound=round(upper, 4),
        lower_bound=round(lower, 4),
    )


def compute_discovery_level_fdr(
    predictions: Sequence[Sequence[str] | None],
    ground_truths: Sequence[Sequence[str]],
) -> float:
    r"""Compute conventional discovery-level FDR over pooled declared elements.

    FDR_pooled = sum(V_i) / max(1, sum(R_i))
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Lengths of predictions and ground truths must match")
    total_declared = 0
    total_spurious = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred is not None and len(pred) > 0:
            total_declared += len(pred)
            true_set = set(gt)
            spurious = sum(1 for e in pred if e not in true_set)
            total_spurious += spurious
    if total_declared == 0:
        return 0.0
    return round(total_spurious / total_declared, 4)


def compute_mean_mechanism_fdp(
    predictions: Sequence[Sequence[str] | None],
    ground_truths: Sequence[Sequence[str]],
) -> float:
    r"""Compute mean per-mechanism False Discovery Proportion (FDP).

    FDP_i = |S_pred,i \ S_true,i| / |S_pred,i| if |S_pred,i| > 0 else 0.0.
    """
    if len(predictions) != len(ground_truths) or len(predictions) == 0:
        raise ValueError("Inputs must be non-empty and of equal length")
    fdps: list[float] = []
    for pred, gt in zip(predictions, ground_truths):
        if pred is None or len(pred) == 0:
            fdps.append(0.0)
        else:
            true_set = set(gt)
            spurious = sum(1 for e in pred if e not in true_set)
            fdps.append(spurious / len(pred))
    return round(sum(fdps) / len(fdps), 4)


