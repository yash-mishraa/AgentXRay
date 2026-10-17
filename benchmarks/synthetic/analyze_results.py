"""Statistical analysis script for Phase 2A/2A.1 scaling benchmark results."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from agentxray.statistics import adjust_p_values_bh, compare_paired_differences


def analyze_suite(results_path: Path):
    with open(results_path, "r") as f:
        data = json.load(f)

    instances = data["instances"]
    total = len(instances)
    print(f"\n=======================================================")
    print(f"ANALYZING: {results_path} (N={total} instances)")
    print(f"=======================================================")

    algos = ["stochastic_delta_debugging", "stochastic_greedy_ablation", "stochastic_seeded_random"]

    # 1. Overall Metrics by Algorithm
    overall = {}
    for a in algos:
        overall[a] = {
            "target_calls": [],
            "search_calls": [],
            "verif_calls": [],
            "subsets": [],
            "matched_gt_all": 0,
            "matched_gt_non_null": 0,
            "verified_minimal_all": 0,
            "verified_minimal_non_null": 0,
            "budget_exhausted": 0,
            "precisions_all": [],
            "recalls_all": [],
            "precisions_non_null": [],
            "recalls_non_null": [],
            "null_false_positives": 0,
            "null_correct": 0,
            "null_count": 0,
            "non_null_count": 0,
        }

    for inst in instances:
        is_null = inst["is_null_case"]
        for a_key, res_field in [
            ("stochastic_delta_debugging", "dd_result"),
            ("stochastic_greedy_ablation", "greedy_result"),
            ("stochastic_seeded_random", "random_result"),
        ]:
            r = inst[res_field]
            st = overall[a_key]
            st["target_calls"].append(r["target_calls"])
            st["search_calls"].append(r["search_target_calls"])
            st["verif_calls"].append(r["verification_target_calls"])
            st["subsets"].append(r["subsets_examined"])
            st["precisions_all"].append(r["precision"])
            st["recalls_all"].append(r["recall"])

            if r["matches_ground_truth"]:
                st["matched_gt_all"] += 1
            if r["is_verified_one_minimal"]:
                st["verified_minimal_all"] += 1
            if r["budget_exhausted"]:
                st["budget_exhausted"] += 1

            if is_null:
                st["null_count"] += 1
                is_fp = r["failure_reproduced"] or len(r["discovered_elements"]) > 0
                if is_fp:
                    st["null_false_positives"] += 1
                else:
                    st["null_correct"] += 1
            else:
                st["non_null_count"] += 1
                st["precisions_non_null"].append(r["precision"])
                st["recalls_non_null"].append(r["recall"])
                if r["matches_ground_truth"]:
                    st["matched_gt_non_null"] += 1
                if r["is_verified_one_minimal"]:
                    st["verified_minimal_non_null"] += 1

    print("\n=== OVERALL SUMMARY (Separated Null vs Non-Null Evaluation) ===")
    for a in algos:
        st = overall[a]
        mean_calls = sum(st["target_calls"]) / total
        med_calls = sorted(st["target_calls"])[total // 2]
        mean_search = sum(st["search_calls"]) / total
        mean_verif = sum(st["verif_calls"]) / total
        mean_sub = sum(st["subsets"]) / total

        # Corrected separated metrics
        null_n = st["null_count"]
        non_null_n = st["non_null_count"]

        null_detection_rate = st["null_correct"] / null_n if null_n > 0 else 0.0
        null_fdr = st["null_false_positives"] / null_n if null_n > 0 else 0.0
        non_null_gt_recovery = st["matched_gt_non_null"] / non_null_n if non_null_n > 0 else 0.0
        non_null_1_minimality = st["verified_minimal_non_null"] / non_null_n if non_null_n > 0 else 0.0
        mean_prec_non_null = sum(st["precisions_non_null"]) / non_null_n if non_null_n > 0 else 0.0
        mean_rec_non_null = sum(st["recalls_non_null"]) / non_null_n if non_null_n > 0 else 0.0
        exh_rate = st["budget_exhausted"] / total

        print(f"\nAlgorithm: {a}")
        print(f"  Target Calls (mean ± std): {mean_calls:.1f} (median: {med_calls:.1f})")
        print(f"    - Search Calls (mean): {mean_search:.1f}")
        print(f"    - Verification Calls (mean): {mean_verif:.1f}")
        print(f"  Subsets Examined (mean): {mean_sub:.1f}")
        print(f"  [Non-Null Targets N={non_null_n}]:")
        print(f"    - Ground Truth Trigger Recovery Rate: {non_null_gt_recovery*100:.1f}% ({st['matched_gt_non_null']}/{non_null_n})")
        print(f"    - 1-Minimality Verification Rate:     {non_null_1_minimality*100:.1f}% ({st['verified_minimal_non_null']}/{non_null_n})")
        print(f"    - Mean Precision: {mean_prec_non_null:.3f}, Mean Recall: {mean_rec_non_null:.3f}")
        print(f"  [Null Targets N={null_n}]:")
        print(f"    - Null Detection Rate (True Negatives): {null_detection_rate*100:.1f}% ({st['null_correct']}/{null_n})")
        print(f"    - False Discovery Rate on Null Targets: {null_fdr*100:.1f}% ({st['null_false_positives']}/{null_n})")
        print(f"  [Resource Boundaries N={total}]:")
        print(f"    - Budget Exhaustion Rate: {exh_rate*100:.1f}% ({st['budget_exhausted']}/{total})")

    # 2. Scaling by Candidate Space Size K in {4, 8, 12, 16, 20}
    print("\n=== SCALING BY CANDIDATE SIZE K (Mean Target Calls / Subsets) ===")
    for k in [4, 8, 12, 16, 20]:
        k_insts = [inst for inst in instances if inst["k"] == k]
        n_k = len(k_insts)
        n_k_non_null = len([inst for inst in k_insts if not inst["is_null_case"]])
        print(f"\n--- K = {k} (N={n_k} total, N_non_null={n_k_non_null}) ---")
        for a_key, res_field in [
            ("Delta Debugging", "dd_result"),
            ("Greedy Ablation", "greedy_result"),
            ("Random Search", "random_result"),
        ]:
            calls = [inst[res_field]["target_calls"] for inst in k_insts]
            subs = [inst[res_field]["subsets_examined"] for inst in k_insts]
            gt_non_null = [inst[res_field]["matches_ground_truth"] for inst in k_insts if not inst["is_null_case"]]
            mean_c = sum(calls) / n_k
            mean_s = sum(subs) / n_k
            gt_pct = sum(gt_non_null) / n_k_non_null * 100.0 if n_k_non_null > 0 else 0.0
            print(f"  {a_key:18s}: Calls={mean_c:6.1f}, Subsets={mean_s:4.1f}, Non-Null GT Recovery={gt_pct:5.1f}%")

    # 3. Scaling by Target Family
    print("\n=== PERFORMANCE BY TARGET FAMILY (Mean Target Calls / Match Rate) ===")
    families = sorted(list({inst["family"] for inst in instances}))
    for fam in families:
        fam_insts = [inst for inst in instances if inst["family"] == fam]
        n_f = len(fam_insts)
        is_fam_null = fam_insts[0]["is_null_case"]
        fam_label = f"{fam} [NULL]" if is_fam_null else fam
        print(f"\n--- Family: {fam_label} (N={n_f} instances) ---")
        for a_key, res_field in [
            ("Delta Debugging", "dd_result"),
            ("Greedy Ablation", "greedy_result"),
            ("Random Search", "random_result"),
        ]:
            calls = [inst[res_field]["target_calls"] for inst in fam_insts]
            subs = [inst[res_field]["subsets_examined"] for inst in fam_insts]
            gt = [inst[res_field]["matches_ground_truth"] for inst in fam_insts]
            ver = [inst[res_field]["is_verified_one_minimal"] for inst in fam_insts]
            mean_c = sum(calls) / n_f
            mean_s = sum(subs) / n_f
            gt_pct = sum(gt) / n_f * 100.0
            ver_pct = sum(ver) / n_f * 100.0
            metric_label = "Null Detection" if is_fam_null else "GT Recovery"
            print(f"  {a_key:18s}: Calls={mean_c:6.1f}, Subsets={mean_s:4.1f}, {metric_label}={gt_pct:5.1f}%, Verified={ver_pct:5.1f}%")

    # 4. Paired Comparisons: Delta Debugging vs Greedy Ablation across all instances
    print("\n=== PAIRED STATISTICAL COMPARISONS ===")
    dd_calls = [inst["dd_result"]["target_calls"] for inst in instances]
    greedy_calls = [inst["greedy_result"]["target_calls"] for inst in instances]
    random_calls = [inst["random_result"]["target_calls"] for inst in instances]

    dd_vs_greedy = compare_paired_differences(dd_calls, greedy_calls)
    dd_vs_random = compare_paired_differences(dd_calls, random_calls)
    greedy_vs_random = compare_paired_differences(greedy_calls, random_calls)

    print("\nOverall Paired Comparison (Target Calls):")
    print(f"  DD vs Greedy: mean diff = {dd_vs_greedy.mean_difference:+.2f} ± {dd_vs_greedy.std_difference:.2f}, median diff = {dd_vs_greedy.median_difference:+.1f}, Wilcoxon W={dd_vs_greedy.wilcoxon_w}, p={dd_vs_greedy.wilcoxon_p_value:.5f}, t={dd_vs_greedy.paired_t_statistic:.2f}, p={dd_vs_greedy.paired_t_p_value:.5f}")
    print(f"  DD vs Random: mean diff = {dd_vs_random.mean_difference:+.2f} ± {dd_vs_random.std_difference:.2f}, median diff = {dd_vs_random.median_difference:+.1f}, Wilcoxon W={dd_vs_random.wilcoxon_w}, p={dd_vs_random.wilcoxon_p_value:.5f}, t={dd_vs_random.paired_t_statistic:.2f}, p={dd_vs_random.paired_t_p_value:.5f}")
    print(f"  Greedy vs Random: mean diff = {greedy_vs_random.mean_difference:+.2f} ± {greedy_vs_random.std_difference:.2f}, median diff = {greedy_vs_random.median_difference:+.1f}, Wilcoxon W={greedy_vs_random.wilcoxon_w}, p={greedy_vs_random.wilcoxon_p_value:.5f}, t={greedy_vs_random.paired_t_statistic:.2f}, p={greedy_vs_random.paired_t_p_value:.5f}")

    # Paired comparisons by family
    print("\nPaired Comparisons by Family (DD calls minus Greedy calls):")
    family_p_vals = []
    family_diffs = []
    for fam in families:
        f_insts = [inst for inst in instances if inst["family"] == fam]
        d_c = [inst["dd_result"]["target_calls"] for inst in f_insts]
        g_c = [inst["greedy_result"]["target_calls"] for inst in f_insts]
        comp = compare_paired_differences(d_c, g_c)
        family_diffs.append((fam, comp))
        family_p_vals.append(comp.wilcoxon_p_value)

    adjusted_p = adjust_p_values_bh(family_p_vals)
    for (fam, comp), q in zip(family_diffs, adjusted_p):
        print(f"  {fam:24s}: mean diff={comp.mean_difference:+7.1f}, median={comp.median_difference:+6.1f}, W={comp.wilcoxon_w:6.1f}, raw p={comp.wilcoxon_p_value:.5f}, BH q={q:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze scaling benchmark results")
    parser.add_argument(
        "--results-file",
        type=str,
        default="benchmarks/results/phase_2a1_scaling_results.json",
        help="Path to results JSON file",
    )
    args = parser.parse_args()
    p = Path(args.results_file)
    if not p.exists():
        # Fallback to phase_2a if phase_2a1 not generated yet
        fallback = Path("benchmarks/results/phase_2a_scaling_results.json")
        if fallback.exists():
            p = fallback
        else:
            raise FileNotFoundError(f"Neither {p} nor {fallback} exists")

    analyze_suite(p)


if __name__ == "__main__":
    main()
