"""Tests for Phase 2B.1 protocol audit, mathematical proofs, and Protocol v1.1 repair."""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest

from benchmarks.model_pilot.confirmatory_tasks import (
    CONFIRMATORY_TASK_REGISTRY,
    TaskFamily,
    get_confirmatory_task,
    list_confirmatory_tasks,
)
from benchmarks.model_pilot.power_and_budget_analysis import (
    wilson_interval,
    clopper_pearson_upper_bound_zero_success,
    solve_minimum_null_tasks_for_bound,
    wilcoxon_signed_rank_exact_critical_value,
    simulate_wilcoxon_power,
    generate_budget_scenarios,
)
from benchmarks.model_pilot.validate_confirmatory_protocol import ConfirmatoryProtocolValidator
from agentxray.statistics import adjust_p_values_bh


def test_null_control_identifiability_math() -> None:
    """Verify the mathematical proof that 0/4 null tasks cannot establish FDR <= 0.05."""
    # 0 successes out of 4 trials
    lower, upper = wilson_interval(k=0, n=4, confidence=0.95)
    assert lower == 0.0
    assert 0.48 < upper < 0.50, f"Expected Wilson upper ~0.4899, got {upper}"

    cp_upper = clopper_pearson_upper_bound_zero_success(n=4, alpha=0.05)
    assert 0.52 < cp_upper < 0.54, f"Expected Clopper-Pearson upper ~0.5271, got {cp_upper}"

    # Verify that N >= 59 null tasks are required to bound FDR <= 0.05 with 95% confidence
    min_tasks = solve_minimum_null_tasks_for_bound(target_upper_bound=0.05, alpha=0.05)
    assert min_tasks == 59, f"Expected 59 tasks, got {min_tasks}"

    # Confirm that (1 - 0.05)^59 <= 0.05
    assert (1.0 - 0.05) ** 59 <= 0.05
    assert (1.0 - 0.05) ** 58 > 0.05


def test_wilcoxon_exact_critical_boundary_n12() -> None:
    """Verify the exact Wilcoxon signed-rank critical value for n=12 active instances."""
    crit_w, alpha_exact = wilcoxon_signed_rank_exact_critical_value(n=12, alpha=0.05)
    assert crit_w == 13
    assert alpha_exact < 0.05
    assert 0.040 < alpha_exact < 0.045


def test_power_sensitivity_bounds() -> None:
    """Verify that power with n=12 active tasks is inadequate for moderate effect sizes."""
    # Run targeted simulations with fixed seed for determinism
    # Moderate effect d = 0.50 (mu = 6.0, sigma = 12.0)
    power_d50 = simulate_wilcoxon_power(
        n=12,
        mu=6.0,
        sigma=12.0,
        n_sims=500,
        p_tie=0.05,
        dist="normal",
        seed=101,
    )
    assert power_d50 < 0.35, f"Expected power at d=0.50 to be < 35%, got {power_d50:.1%}"

    # Large pilot-assumed effect d = 1.25 (mu = 15.0, sigma = 12.0)
    power_d125 = simulate_wilcoxon_power(
        n=12,
        mu=15.0,
        sigma=12.0,
        n_sims=500,
        p_tie=0.05,
        dist="normal",
        seed=102,
    )
    assert power_d125 > 0.85, f"Expected power at d=1.25 to be > 85%, got {power_d125:.1%}"

    # Stratified cohort n=6
    power_strat_d125 = simulate_wilcoxon_power(
        n=6,
        mu=15.0,
        sigma=12.0,
        n_sims=500,
        p_tie=0.05,
        dist="normal",
        seed=103,
    )
    assert power_strat_d125 < 0.50, f"Expected stratified n=6 power at d=1.25 to be < 50%, got {power_strat_d125:.1%}"


def test_instance_registry_pseudoreplication_audit() -> None:
    """Audit the task registry to prove the 12 active tasks are 6 paired mechanisms across K=4 and K=8."""
    tasks = list_confirmatory_tasks()
    active_tasks = [t for t in tasks if not t.is_null_case]
    assert len(active_tasks) == 12

    # Group tasks by prompt and required tool
    base_mechanisms: dict[tuple[str, str], list[str]] = {}
    for t in active_tasks:
        key = (t.user_prompt, t.required_tool)
        base_mechanisms.setdefault(key, []).append(t.instance_id)

    # Exactly 6 unique mechanisms
    assert len(base_mechanisms) == 6, f"Expected 6 unique base mechanisms, found {len(base_mechanisms)}"

    # Each mechanism must have exactly 2 instances: one K=4 and one K=8
    for (prompt, tool), inst_list in base_mechanisms.items():
        assert len(inst_list) == 2, f"Expected 2 instances for {tool}: {prompt[:30]}, got {inst_list}"
        t1 = get_confirmatory_task(inst_list[0])
        t2 = get_confirmatory_task(inst_list[1])
        ks = {t1.k, t2.k}
        assert ks == {4, 8}, f"Expected K=4 and K=8 pair, got {ks}"
        # Trigger elements must match exactly
        assert t1.planted_trigger_elements == t2.planted_trigger_elements


def test_benjamini_hochberg_implementation() -> None:
    """Verify that stochastic_minimizer implements Benjamini-Hochberg FDR control."""
    raw_p = [0.01, 0.04, 0.03, 0.20]
    adj_p = adjust_p_values_bh(raw_p)

    assert len(adj_p) == len(raw_p)
    # Check bounds
    for p in adj_p:
        assert 0.0 <= p <= 1.0

    # For m=4:
    # Sorted p: 0.01 (idx 0, rank 1), 0.03 (idx 2, rank 2), 0.04 (idx 1, rank 3), 0.20 (idx 3, rank 4)
    # Step-up BH values:
    # rank 4 (idx 3): 0.20 * 4/4 = 0.20 -> cum_min = 0.20
    # rank 3 (idx 1): 0.04 * 4/3 = 0.05333 -> cum_min = 0.05333
    # rank 2 (idx 2): 0.03 * 4/2 = 0.06 -> min(0.05333, 0.06) = 0.05333
    # rank 1 (idx 0): 0.01 * 4/1 = 0.04 -> min(0.05333, 0.04) = 0.04
    assert pytest.approx(adj_p[0], abs=1e-4) == 0.04      # idx 0 (p=0.01)
    assert pytest.approx(adj_p[1], abs=1e-4) == 0.05333   # idx 1 (p=0.04)
    assert pytest.approx(adj_p[2], abs=1e-4) == 0.05333   # idx 2 (p=0.03)
    assert pytest.approx(adj_p[3], abs=1e-4) == 0.20      # idx 3 (p=0.20)


def test_budget_scenarios_and_headroom() -> None:
    """Verify that worst-case budget exceeds 6000 and is safely encompassed by 7500."""
    scenarios = generate_budget_scenarios()
    low = scenarios["low_demand"]["total_model_requests"]
    expected = scenarios["expected"]["total_model_requests"]
    worst = scenarios["high_demand_worst_case"]["total_model_requests"]

    assert low < expected < worst
    assert worst > 6000, f"Worst-case requests {worst} must demonstrate risk to 6000 cap"
    assert worst < 7500, f"Worst-case requests {worst} must be safely below 7500 cap"
    assert (7500 - worst) >= 150, "Headroom must be at least 150 requests"


def test_protocol_v1_1_config_validity() -> None:
    """Verify that Protocol v1.1 configuration and document pass the validator cleanly."""
    cfg_path = Path("benchmarks/configs/phase_2b1_confirmatory_config_v1_1.json")
    proto_path = Path("docs/research/phase_2b1_confirmatory_protocol_v1_1.md")

    assert cfg_path.exists()
    assert proto_path.exists()

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["protocol_version"] == "Phase_2B.1_v1.1"
    assert cfg["status"] == "PREREGISTERED_FROZEN"
    assert cfg["study_designation"] == "EXPLORATORY_FEASIBILITY_BENCHMARK"
    assert cfg["commercial_spend_cap_usd"] == 0.0
    assert cfg["budget_rules"]["max_total_request_budget"] == 7500
    assert cfg["statistical_analysis_plan"]["primary_analysis_framework"] == "two_part_mcnemar_and_signed_rank"
    assert cfg["statistical_analysis_plan"]["minimality_multiplicity_correction"] == "benjamini_hochberg"

    validator = ConfirmatoryProtocolValidator(config_path=cfg_path, protocol_path=proto_path)
    summary = validator.validate()

    assert summary["validation_status"] == "VALIDATION_PASSED"
    assert summary["all_checks_passed"] is True
    assert summary["total_discrepancies"] == 0
