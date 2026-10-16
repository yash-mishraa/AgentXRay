"""Statistical analysis script for Phase 2A scaling benchmark results."""

import json
from collections import defaultdict
import math
from agentxray.statistics import compare_paired_differences, adjust_p_values_bh

def main():
    with open("benchmarks/results/phase_2a_scaling_results.json", "r") as f:
        data = json.load(f)

    instances = data["instances"]
    total = len(instances)
    print(f"Total instances analyzed: {total}")

    algos = ["stochastic_delta_debugging", "stochastic_greedy_ablation", "stochastic_seeded_random"]

    # 1. Overall Metrics by Algorithm
    overall = {}
    for a in algos:
        overall[a] = {
            "target_calls": [],
            "search_calls": [],
            "verif_calls": [],
            "subsets": [],
            "matched_gt": 0,
            "verified_minimal": 0,
            "budget_exhausted": 0,
            "precisions": [],
            "recalls": [],
            "null_false_positives": 0,
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
            st["precisions"].append(r["precision"])
            st["recalls"].append(r["recall"])
            if r["matches_ground_truth"]:
                st["matched_gt"] += 1
            if r["is_verified_one_minimal"]:
                st["verified_minimal"] += 1
            if r["budget_exhausted"]:
                st["budget_exhausted"] += 1
            if is_null:
                st["null_count"] += 1
                if r["failure_reproduced"] or len(r["discovered_elements"]) > 0:
                    st["null_false_positives"] += 1
            else:
                st["non_null_count"] += 1

    print("\n=== OVERALL SUMMARY (N=400 instances, 1200 runs) ===")
    for a in algos:
        st = overall[a]
        mean_calls = sum(st["target_calls"]) / total
        med_calls = sorted(st["target_calls"])[total // 2]
        mean_search = sum(st["search_calls"]) / total
        mean_verif = sum(st["verif_calls"]) / total
        mean_sub = sum(st["subsets"]) / total
        gt_rate = st["matched_gt"] / total
        verif_rate = st["verified_minimal"] / total
        exh_rate = st["budget_exhausted"] / total
        null_fdr = st["null_false_positives"] / st["null_count"] if st["null_count"] > 0 else 0.0
        mean_prec = sum(st["precisions"]) / total
        mean_rec = sum(st["recalls"]) / total

        print(f"\nAlgorithm: {a}")
        print(f"  Target Calls (mean ± std): {mean_calls:.1f} (median: {med_calls:.1f})")
        print(f"    - Search Calls (mean): {mean_search:.1f}")
        print(f"    - Verification Calls (mean): {mean_verif:.1f}")
        print(f"  Subsets Examined (mean): {mean_sub:.1f}")
        print(f"  Ground Truth Recovery Rate: {gt_rate*100:.1f}%")
        print(f"  1-Minimality Verification Rate: {verif_rate*100:.1f}%")
        print(f"  Mean Precision: {mean_prec:.3f}, Mean Recall: {mean_rec:.3f}")
        print(f"  Null Target False Discovery Rate: {null_fdr*100:.1f}% ({st['null_false_positives']}/{st['null_count']})")
        print(f"  Budget Exhaustion Rate: {exh_rate*100:.1f}% ({st['budget_exhausted']}/{total})")

    # 2. Scaling by Candidate Space Size K in {4, 8, 12, 16, 20}
    print("\n=== SCALING BY CANDIDATE SIZE K (Mean Target Calls / Subsets) ===")
    for k in [4, 8, 12, 16, 20]:
        k_insts = [inst for inst in instances if inst["k"] == k]
        n_k = len(k_insts)
        print(f"\n--- K = {k} (N={n_k} instances) ---")
        for a_key, res_field in [
            ("Delta Debugging", "dd_result"),
            ("Greedy Ablation", "greedy_result"),
            ("Random Search", "random_result"),
        ]:
            calls = [inst[res_field]["target_calls"] for inst in k_insts]
            subs = [inst[res_field]["subsets_examined"] for inst in k_insts]
            gt = [inst[res_field]["matches_ground_truth"] for inst in k_insts]
            mean_c = sum(calls) / n_k
            mean_s = sum(subs) / n_k
            gt_pct = sum(gt) / n_k * 100.0
            print(f"  {a_key:18s}: Calls={mean_c:6.1f}, Subsets={mean_s:4.1f}, GT Match={gt_pct:5.1f}%")

    # 3. Scaling by Target Family
    print("\n=== PERFORMANCE BY TARGET FAMILY (Mean Target Calls / Match Rate) ===")
    families = sorted(list({inst["family"] for inst in instances}))
    for fam in families:
        fam_insts = [inst for inst in instances if inst["family"] == fam]
        n_f = len(fam_insts)
        print(f"\n--- Family: {fam} (N={n_f} instances) ---")
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
            print(f"  {a_key:18s}: Calls={mean_c:6.1f}, Subsets={mean_s:4.1f}, GT={gt_pct:5.1f}%, Verified={ver_pct:5.1f}%")

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

if __name__ == "__main__":
    main()
