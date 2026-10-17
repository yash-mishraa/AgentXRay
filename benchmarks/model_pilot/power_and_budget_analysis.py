"""Reproducible Power, Null-Control, and Request Budget Analysis for Phase 2B.1 Protocol.

Implements pure-Python statistical calculations:
- Exact permutation distribution and dynamic programming p-values for Wilcoxon signed-rank test.
- Asymptotic Wilcoxon signed-rank test with Pratt tie-correction and continuity correction.
- Monte Carlo power estimation with explicit Monte Carlo standard errors (MCSE).
- Clopper-Pearson exact and Wilson score confidence intervals for null-control false-positive rates.
- Low, expected, high-demand, and theoretical maximum request budget scenarios.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054 if confidence == 0.95 else 1.6448536269514722
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)) / denom
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def clopper_pearson_upper_bound_zero_success(n: int, alpha: float = 0.05) -> float:
    """Exact Clopper-Pearson 1-alpha upper bound when k = 0."""
    if n <= 0:
        return 1.0
    return round(1.0 - (alpha ** (1.0 / n)), 4)


def solve_minimum_null_tasks_for_bound(target_upper_bound: float = 0.05, alpha: float = 0.05) -> int:
    """Calculate minimum null tasks needed so Clopper-Pearson 1-alpha upper bound is <= target when k=0."""
    return math.ceil(math.log(alpha) / math.log(1.0 - target_upper_bound))


def wilcoxon_signed_rank_exact_critical_value(n: int, alpha: float = 0.05) -> tuple[int, float]:
    """Compute exact two-sided critical value W_crit for Wilcoxon signed-rank test under H0.

    Assumes continuous data without ties and non-zero differences.
    """
    total = n * (n + 1) // 2
    dp = [0] * (total + 1)
    dp[0] = 1
    for r in range(1, n + 1):
        for s in range(total, r - 1, -1):
            dp[s] += dp[s - r]

    tot = 2 ** n
    cum = 0
    half_alpha = alpha / 2.0
    crit = 0
    exact_p = 0.0

    for w in range(len(dp)):
        cum += dp[w]
        if cum / tot > half_alpha:
            crit = max(0, w - 1)
            exact_p = 2.0 * (cum - dp[w]) / tot
            break

    return crit, round(exact_p, 4)


def exact_wilcoxon_permutation_p_value(ranks: Sequence[float], signs: Sequence[bool]) -> float:
    """Compute exact two-sided permutation p-value for signed mid-ranks using subset-sum DP."""
    m = len(ranks)
    if m == 0:
        return 1.0

    # Scale mid-ranks by 2 so all values are integers
    int_ranks = [int(round(r * 2)) for r in ranks]
    total_s = sum(int_ranks)
    w_plus = sum(s for s, sign in zip(int_ranks, signs) if sign)
    w_min = min(w_plus, total_s - w_plus)

    dp = [0] * (total_s + 1)
    dp[0] = 1
    for s in int_ranks:
        for val in range(total_s, s - 1, -1):
            dp[val] += dp[val - s]

    tot_perms = 1 << m
    count = 2 * sum(dp[: w_min + 1])
    return min(1.0, count / tot_perms)


def asymptotic_wilcoxon_p_value_with_ties(
    ranks: Sequence[float], signs: Sequence[bool], tie_counts: Sequence[int]
) -> float:
    """Compute asymptotic normal approximation p-value with Pratt tie-correction and continuity correction."""
    m = len(ranks)
    if m == 0:
        return 1.0

    w_plus = sum(r for r, sign in zip(ranks, signs) if sign)
    w_minus = sum(r for r, sign in zip(ranks, signs) if not sign)
    w = min(w_plus, w_minus)

    mean_w = m * (m + 1) / 4.0
    tie_term = sum(t * (t - 1) * (2 * t + 1) for t in tie_counts) / 48.0
    var_w = (m * (m + 1) * (2 * m + 1) / 24.0) - tie_term
    std_w = math.sqrt(max(1e-9, var_w))

    z = (w - mean_w + 0.5) / std_w
    return min(1.0, 2.0 * normal_cdf(z))


def simulate_wilcoxon_power(
    n: int = 12,
    mu: float = 15.0,
    sigma: float = 12.0,
    n_sims: int = 2000,
    dist: str = "normal",
    p_tie: float = 0.05,
    discrete_step: float | None = None,
    use_exact_permutation: bool = True,
    seed: int = 42,
) -> float:
    """Monte Carlo power simulation of Wilcoxon signed-rank test.

    Supports exact subset-sum DP permutation p-values (default for n <= 14)
    and tie-corrected asymptotic normal approximation.
    """
    rng = random.Random(seed)
    rejections = 0

    for _ in range(n_sims):
        diffs: list[float] = []
        for _ in range(n):
            if rng.random() < p_tie:
                diffs.append(0.0)
            elif dist == "normal":
                diffs.append(rng.gauss(mu, sigma))
            elif dist == "skewed":
                diffs.append(rng.expovariate(1.0 / sigma) - sigma + mu)
            elif dist == "bimodal":
                base_m = 8.0 if rng.random() < 0.5 else 22.0
                diffs.append(rng.gauss(base_m, sigma * 0.7))

        if discrete_step is not None and discrete_step > 0.0:
            diffs = [round(d / discrete_step) * discrete_step for d in diffs]

        # Pratt filtering of zero differences
        non_zeros = [d for d in diffs if d != 0.0]
        m = len(non_zeros)
        if m < 4:
            continue

        abs_pairs = [(abs(d), d > 0.0) for d in non_zeros]
        abs_pairs.sort(key=lambda x: x[0])

        ranks = [0.0] * m
        tie_counts: list[int] = []
        i = 0
        while i < m:
            j = i
            while j < m and abs(abs_pairs[j][0] - abs_pairs[i][0]) < 1e-6:
                j += 1
            tie_len = j - i
            if tie_len > 1:
                tie_counts.append(tie_len)
            avg_r = (i + 1 + j) / 2.0
            for idx in range(i, j):
                ranks[idx] = avg_r
            i = j

        signs = [pair[1] for pair in abs_pairs]
        w_plus = sum(ranks[k] for k in range(m) if signs[k])
        w_minus = sum(ranks[k] for k in range(m) if not signs[k])

        if use_exact_permutation and m <= 14:
            p = exact_wilcoxon_permutation_p_value(ranks, signs)
        else:
            p = asymptotic_wilcoxon_p_value_with_ties(ranks, signs, tie_counts)

        # Directional rejection in favor of DD (mu_delta > 0 => w_plus > w_minus)
        if p < 0.05 and w_plus > w_minus:
            rejections += 1

    return round(rejections / n_sims, 4)


def simulate_wilcoxon_power_detailed(
    n: int = 12,
    mu: float = 15.0,
    sigma: float = 12.0,
    n_sims: int = 2000,
    dist: str = "normal",
    p_tie: float = 0.05,
    discrete_step: float | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Return detailed Monte Carlo power estimates with exact permutation, asymptotic, and MCSE."""
    pw_exact = simulate_wilcoxon_power(
        n=n, mu=mu, sigma=sigma, n_sims=n_sims, dist=dist,
        p_tie=p_tie, discrete_step=discrete_step, use_exact_permutation=True, seed=seed
    )
    pw_asymp = simulate_wilcoxon_power(
        n=n, mu=mu, sigma=sigma, n_sims=n_sims, dist=dist,
        p_tie=p_tie, discrete_step=discrete_step, use_exact_permutation=False, seed=seed
    )
    mcse_exact = math.sqrt(pw_exact * (1.0 - pw_exact) / n_sims)
    mcse_asymp = math.sqrt(pw_asymp * (1.0 - pw_asymp) / n_sims)

    return {
        "n": n,
        "cohen_d": round(mu / sigma, 2),
        "mu": mu,
        "sigma": sigma,
        "n_sims": n_sims,
        "power_exact": pw_exact,
        "mcse_exact": round(mcse_exact, 4),
        "ci_95_exact": (round(max(0.0, pw_exact - 1.96 * mcse_exact), 4), round(min(1.0, pw_exact + 1.96 * mcse_exact), 4)),
        "power_asymptotic": pw_asymp,
        "mcse_asymptotic": round(mcse_asymp, 4),
        "ci_95_asymptotic": (round(max(0.0, pw_asymp - 1.96 * mcse_asymp), 4), round(min(1.0, pw_asymp + 1.96 * mcse_asymp), 4)),
    }


def generate_power_sensitivity_table() -> list[dict[str, Any]]:
    """Generate sensitivity table across plausible Cohen d effect sizes with MCSE."""
    effects = [
        (0.40, 4.8, 12.0, "Small-to-moderate request reduction (~10%)"),
        (0.50, 6.0, 12.0, "Moderate request reduction (~15%)"),
        (0.60, 7.2, 12.0, "Moderate-large request reduction"),
        (0.70, 8.4, 12.0, "Substantial request reduction (~20%)"),
        (0.80, 9.6, 12.0, "Standard Cohen large effect size"),
        (1.00, 12.0, 12.0, "Very large request reduction (~25%)"),
        (1.25, 15.0, 12.0, "Empirical pilot single-task observed effect (Phase 2B.0.2)"),
    ]

    rows = []
    for d, mu, sigma, desc in effects:
        det = simulate_wilcoxon_power_detailed(n=12, mu=mu, sigma=sigma, n_sims=2000, dist="normal", p_tie=0.05, seed=100 + int(d*100))
        pw_skew = simulate_wilcoxon_power(n=12, mu=mu, sigma=sigma, n_sims=2000, dist="skewed", p_tie=0.05, seed=200 + int(d*100))
        pw_ties = simulate_wilcoxon_power(n=12, mu=mu, sigma=sigma, n_sims=2000, dist="normal", p_tie=0.15, discrete_step=5.0, seed=300 + int(d*100))
        pw_n6 = simulate_wilcoxon_power(n=6, mu=mu, sigma=sigma, n_sims=2000, dist="normal", p_tie=0.05, seed=400 + int(d*100))

        rows.append({
            "cohen_d": d,
            "mean_diff_requests": mu,
            "description": desc,
            "power_exact": det["power_exact"],
            "mcse_exact": det["mcse_exact"],
            "power_asymp": det["power_asymptotic"],
            "power_skewed": pw_skew,
            "power_ties_discrete": pw_ties,
            "power_n6_stratified": pw_n6,
        })
    return rows


def generate_budget_scenarios() -> dict[str, dict[str, Any]]:
    """Calculate low, expected, high-demand, and theoretical maximum request-budget scenarios."""
    scenarios = {
        "low_demand": {
            "description": "Fast convergence on all tasks; no budget exhaustion; clean singletons.",
            "calibration_k4": 6 * 16 * 5 * 2,  # 960
            "calibration_k8": 6 * 18 * 5 * 2,  # 1080
            "calibration_null": 4 * 2 * 5 * 2,  # 80
            "search_dd": 12 * 4 * 5 * 2,       # 480
            "search_greedy": 12 * 6 * 5 * 2,   # 720
            "search_random": 12 * 5 * 5 * 2,   # 600
            "search_null": 4 * 3 * 1 * 5 * 2,  # 120 (early exit at full set)
            "holdout": 12 * 3 * 5 * 2,         # 360
            "retries_headroom": 100,
        },
        "expected": {
            "description": "Empirical pilot-consistent convergence; typical distractor counts; minimal retries.",
            "calibration_k4": 6 * 16 * 5 * 2,  # 960
            "calibration_k8": 6 * 18 * 5 * 2,  # 1080
            "calibration_null": 4 * 2 * 5 * 2,  # 80
            "search_dd": (6 * 4 + 6 * 7) * 5 * 2,      # 660
            "search_greedy": (6 * 6 + 6 * 15) * 5 * 2, # 1260
            "search_random": (6 * 5 + 6 * 8) * 5 * 2,  # 780
            "search_null": 4 * 3 * 1 * 5 * 2,          # 120
            "holdout": 12 * 3 * 5 * 2,                 # 360
            "retries_headroom": 250,
        },
        "high_demand_worst_case": {
            "description": "Greedy reaches cap on K=8; multiple ablation rounds; 5% infrastructure retries.",
            "calibration_k4": 6 * 16 * 5 * 2,  # 960
            "calibration_k8": 6 * 18 * 5 * 2,  # 1080
            "calibration_null": 4 * 4 * 5 * 2,  # 160 (tested 4 subsets)
            "search_dd": 6 * 60 + 6 * 120,     # 1080
            "search_greedy": 6 * 120 + 6 * 180, # 1800 (hits caps)
            "search_random": 6 * 80 + 6 * 120, # 1200
            "search_null": 4 * 3 * 2 * 5 * 2,  # 240
            "holdout": 12 * 3 * 5 * 2,         # 360
            "retries_headroom": 450,
        },
        "theoretical_maximum_cap": {
            "description": "All 3 algorithms exhaust caps (120 on K=4, 180 on K=8) across all 12 tasks.",
            "calibration_k4": 6 * 16 * 5 * 2,   # 960
            "calibration_k8": 6 * 18 * 5 * 2,   # 1080
            "calibration_null": 4 * 4 * 5 * 2,   # 160
            "search_dd": 6 * 120 + 6 * 180,     # 1800
            "search_greedy": 6 * 120 + 6 * 180, # 1800
            "search_random": 6 * 120 + 6 * 180, # 1800
            "search_null": 4 * 3 * 2 * 5 * 2,   # 240
            "holdout": 12 * 3 * 5 * 2,          # 360
            "retries_headroom": 0,
        },
    }

    for name, s in scenarios.items():
        total = sum(v for k, v in s.items() if k != "description")
        s["total_model_requests"] = total
        s["estimated_minutes_at_0_45s"] = round(total * 0.45 / 60.0, 1)

    return scenarios


def main() -> None:
    print("=" * 80)
    print("AgentXRay Phase 2B.1 — Rigorous Power and Request Budget Analysis")
    print("=" * 80)

    # 1. Null control analysis
    w_low, w_high = wilson_interval(0, 4, 0.95)
    cp_high = clopper_pearson_upper_bound_zero_success(4, 0.05)
    print("\n[1] Null Control False-Positive Rate Identifiability (N = 4 null tasks):")
    print(f"  Observed False Positives: 0 / 4 (0.0%)")
    print(f"  Wilson 95% Confidence Interval: [{w_low:.4f}, {w_high:.4f}] -> Upper bound = {w_high*100:.1f}%")
    print(f"  Clopper-Pearson Exact 95% Upper Bound: {cp_high*100:.1f}%")
    print("  Statistical Implication: 0/4 null discoveries CANNOT establish FDR <= 0.05.")
    print("  Required null tasks to establish 95% upper bound <= 0.05: N >= 59 tasks.")

    # 2. Exact Wilcoxon Critical Values
    crit, ex_alpha = wilcoxon_signed_rank_exact_critical_value(12, 0.05)
    print(f"\n[2] Exact Wilcoxon Signed-Rank Test Critical Boundary for N = 12:")
    print(f"  Total Permutations: 2^12 = 4,096")
    print(f"  Exact Null Critical W: W <= {crit} (two-sided alpha = {ex_alpha:.4f})")
    print("  Note: Exact W_crit assumes continuous data without ties or zeros.")

    # 3. Power sensitivity table
    print("\n[3] Power Sensitivity Analysis across Plausible Effect Sizes (N = 12 active tasks):")
    print("  Methodology: Exact subset-sum DP permutation p-values vs Pratt-tie-corrected asymptotic.")
    rows = generate_power_sensitivity_table()
    header = f"{'Cohen d':>7} | {'Diff':>5} | {'Exact Pw':>9} | {'MCSE':>6} | {'Asymp Pw':>9} | {'Skewed':>8} | {'Disc/Ties':>9} | {'N=6 Strat':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['cohen_d']:7.2f} | {r['mean_diff_requests']:5.1f} | "
            f"{r['power_exact']*100:8.1f}% | {r['mcse_exact']*100:5.2f}% | "
            f"{r['power_asymp']*100:8.1f}% | {r['power_skewed']*100:7.1f}% | "
            f"{r['power_ties_discrete']*100:8.1f}% | {r['power_n6_stratified']*100:8.1f}%"
        )

    # 4. Budget Scenarios
    print("\n[4] Request Budget Scenarios & Safety Hard-Stop:")
    scenarios = generate_budget_scenarios()
    for name, s in scenarios.items():
        print(f"\n  Scenario: {name.upper()}")
        print(f"    Description: {s['description']}")
        print(f"    Total Model Requests: {s['total_model_requests']} requests")
        print(f"    Estimated Local Wall-Clock Time: {s['estimated_minutes_at_0_45s']} minutes")

    print("\n  Enforceable Safety Guardrail:")
    print("    Configuration Hard Cap: 7,500 requests")
    print("    Safety Margin over High Demand: +170 requests headroom")
    print("    Safety Firewall: Halts gracefully via HARD_BUDGET_BREACH_ABORT before theoretical max (8,200).")


if __name__ == "__main__":
    main()
