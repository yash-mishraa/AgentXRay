# AgentXRay — Phase 3.1: Concise Research Summary & Portfolio Synthesis

**Document Reference:** `docs/research/phase_3_1/phase_3_1_research_summary.md`  
**Date:** 2026-10-17  
**Version:** Phase 3.1 Corrected Successor Document  
**Author:** AI Research & Systems Engineering Team (Antigravity)  
**Target Repository:** `/Users/yashmishra14/Personal projects/AgentXRay`  
**Git Checkpoint:** HEAD at commit `62579fb` on branch `main`  
**Scope:** High-Integrity Research Synthesis for Technical Portfolio and Academic Review

---

## 1. The Research Problem

Autonomous tool-using language model agents frequently fail due to complex interactions between system instructions, user prompts, and structured tool definitions. When a multi-surface agent breaks, developers face an intractable fault-localization challenge: identifying which specific prompt clauses or tool parameters triggered the failure.

Classical fault minimization—most notably Andreas Zeller’s Delta Debugging ($ddmin$)—guarantees 1-minimal failure-inducing inputs in software systems under deterministic, monotonic assumptions. However, applying delta debugging to AI agents introduces three major unsolved challenges:
1. **Black-Box Accessibility:** Enterprise and proprietary agents rarely expose internal execution states, memory snapshots, or trajectory checkpoints required by white-box causal replay systems (e.g., CAR, DoVer).
2. **Stochastic Non-Monotonicity:** Language model outputs are probabilistic ($T > 0$). Adding irrelevant distractor text can unpredictably flip a failure to a pass, violating classical monotonicity.
3. **Multi-Surface Scope:** Prior LLM delta debugging adaptations (e.g., Amazon DD, DDOR) are confined to static, single-turn prompt strings, ignoring multi-turn tool calling, interactive tool arguments, and environment return schemas.

**Central Question:** Can systematic, black-box delta debugging isolate reproducible, 1-minimal failure-inducing conditions across multi-surface agent boundaries with lower model request burdens than standard baseline ablations?

---

## 2. The Implemented Framework

Project **AgentXRay** introduces a local, reproducible research testbed designed to evaluate black-box minimization algorithms under formal statistical controls:

- **Multi-Surface Candidate Modeling:** Represents inputs as sets of discrete candidate interventions spanning prompt directives, tool parameter schemas, and mock environment returns.
- **Stochastic Delta Debugging:** Extends $ddmin$ to binomial response variables, evaluating subsets across repeated probe trials ($m=5$) with Wilson score confidence intervals and Newcombe hybrid difference intervals.
- **Preregistered Statistical Minimality:** Formulates candidate 1-minimality under the Intersection–Union Principle (Berger 1982): a candidate is supported as 1-minimal if and only if every single-element deletion exhibits a statistically significant reduction in failure rate via Fisher's exact test ($\alpha = 0.05$) while satisfying an empirical non-failure guard ($\hat{p} < 0.50$).
- **Rigorous Experimental Controls:** Features a two-stage calibration gate (trigger potency vs. negative control passing), negative null tasks, independent holdout verification, and hard dual-cost budget ceilings (task-level and global request caps).

---

## 3. Strongest Defensible Empirical Findings

In an authorized exploratory benchmark evaluating an open-weights ReAct agent backed by local `llama3.2:3b` across 16 preregistered task configurations ($K \in \{4, 8\}$):

1. **Substantial Request Burden Reduction on Qualified Syntax Mechanisms:**
   On tasks with robust, active causal failure triggers (AST math syntax errors in tool calling):
   - Stochastic Delta Debugging required a mean of **45.0 model requests** (40 at $K=4$, 50 at $K=8$).
   - Stochastic Greedy Ablation required a mean of **80.0 model requests** (60 at $K=4$, 100 at $K=8$).
   - Delta Debugging achieved a **descriptive 43.8% request reduction** while maintaining identical 100% ground-truth trigger recovery.
2. **Superior Scaling Under Candidate Growth ($K=4 \to K=8$):**
   When candidate space doubled from $K=4$ to $K=8$:
   - Delta Debugging request consumption grew by only **+25.0%** (40 to 50 requests).
   - Greedy Ablation request consumption grew by **+66.7%** (60 to 100 requests).
   - Seeded Random Search completely collapsed, exhausting its 180-request cap on both tasks without finding the trigger.
3. **Perfect Holdout Reproducibility:**
   All 12 candidate subsets discovered across resolved active tasks reproduced failure at a 100% rate (5/5 failures, 95% Wilson CI $[0.57, 1.00]$) in independent holdout verification.

---

## 4. Main Negative Findings & Scientific Caveats

1. **Failure to Reject the Primary Null Hypothesis ($p = 0.5000$):**
   In a mechanism-level paired Wilcoxon signed-rank test across all $n=6$ preregistered mechanisms under an Intention-to-Treat design, the result was $W = 1.0, p = 0.5000$. While two mechanisms strongly favored DD, three mechanisms tied at the cap due to calibration disqualification, rendering statistical significance unachievable at $n=6$. This result represents a **failure to reject the null hypothesis, not proof of algorithmic equivalence**.
2. **Operational Minimality $\neq$ Causal Ground Truth:**
   On a schema-unit task (`inst_09`), both DD and Greedy discovered a 2-element interacting subset (`incompatible_unit + distractor`) that was statistically 1-minimal under IUT (removing either element dropped failure to 0/5), but failed to isolate the singleton planted trigger in isolation.
3. **False Discoveries on Database Null Controls:**
   Across 4 null task configurations ($N=4$, spanning 2 mechanisms) and 12 algorithm-task runs, prompt bloat caused `llama3.2:3b` to fail tool calling on 2 database configurations (`inst_15`, `inst_16`), leading algorithms to isolate `department_code` (5 algorithm-task occurrences across 1 unique element). Four null tasks are insufficient to estimate an asymptotic population False Discovery Rate. Small language models exhibit severe vulnerability to prompt length degradation, which black-box algorithms faithfully isolate.

---

## 5. Provenance and Experimental Boundaries

- **Zero Commercial Financial Spend:** Total external API spend was strictly **$0.0000 USD**; all 2,176 requests were served locally by Ollama `llama3.2:3b`.
- **Errata Acknowledged:** 
  - An edge case in budget-cap enforcement omitted four aborted Turn 1 requests from `trials.jsonl`, creating a 4-request delta with `manifest.json` ($2,176$ vs $2,172$).
  - The execution script hash on disk (`69b03d52...`) reflected a critical pre-execution bugfix, while the execution report cited a pre-patch draft hash (`03ff8262...`). Both are fully documented in the official errata record.

---

## 6. What AgentXRay Genuinely Contributes

1. **A Tested Open-Source Architecture:** A complete, modular framework for black-box intervention, stochastic minimization, and statistical validation on tool-calling agents.
2. **Empirical Characterization of LLM Minimization:** A controlled empirical study documenting how stochastic delta debugging performs on an open-weights ReAct agent (`llama3.2:3b`), highlighting the scaling benefits on syntax triggers alongside the operational risks of distractor entrapment and prompt degradation.
3. **A Methodological Benchmark for Research Integrity:** Demonstration of preregistration, intention-to-treat analysis, cryptographic hashing, negative controls, and post-execution forensic auditing in agentic AI evaluation.

---

## 7. Requirements for a Peer-Reviewed Publication Claim

To transition this exploratory study into a confirmatory research paper suitable for top-tier software engineering or AI conferences (e.g., ICSE, FSE, NeurIPS, ICLR), the following additional evidence is required:

1. **Broader Foundation Model Cohort:** Evaluate across larger open-weights models (`llama3.1:8b`, `qwen2.5:7b`, `mistral-nemo`) and commercial frontier APIs (Claude 3.5 Sonnet, GPT-4o) to determine whether larger parameter counts eliminate calibration dropouts.
2. **Expanded Mechanism Sample Size ($n \ge 15$):** Increase the number of independent failure mechanisms to achieve adequate statistical power ($\ge 80\%$) under non-parametric signed-rank tests.
3. **Adaptive Stopping Rules:** Implement sequential testing (e.g., Wald's Sequential Probability Ratio Test) to dynamically allocate model requests and reduce sample burdens on unambiguous subsets.
4. **Real-World Agent Trajectories:** Validate the minimizer on real-world benchmark datasets (e.g., SWE-bench, ToolBench, GAIA) beyond synthetic mock environments.
