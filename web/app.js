/**
 * AgentXRay — Developer Tool Product Website JavaScript
 * Implements interactive terminal playback, genuine client-side ddmin minimizer,
 * code switcher, copy utilities, and responsive accessibility handlers.
 */

// ---------------------------------------------------------------------------
// 1. Genuine Scenario Data (aligned with repository testbeds & HF Demo)
// ---------------------------------------------------------------------------

const SCENARIOS = {
  currency_ast: {
    id: "currency_ast",
    name: "AST Calculator Currency Formatting Directive",
    category: "Prompt Syntax / AST Parser Error",
    query: "Calculate total reimbursement for expenses of 150 and 250.",
    agent: "Automated Accounting Assistant (Controlled Mock Agent)",
    tools: ["calculator(expression: str) -> str"],
    candidates: [
      "strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')",
      "monetary_precision: represent all invoice amounts with exactly two decimal places",
      "professional_styling: use formal accounting terminology throughout summaries",
      "audit_trail: annotate calculations with internal ledger voucher references"
    ],
    isolatedCandidateId: "elem_0",
    isolatedCandidate: "strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')",
    initialFailure: {
      toolCall: "calculator(expression='$150 + $250')",
      output: "SyntaxError: invalid syntax (currency symbol '$' rejected by arithmetic evaluator)",
      code: "SYNTAX_ERROR"
    },
    steps: [
      {
        step: 1,
        intervention: "BASELINE_CHECK",
        count: 4,
        candidates: ["elem_0", "elem_1", "elem_2", "elem_3"],
        outcome: "FAILURE",
        msg: "Tool 'calculator' crashed: SyntaxError: invalid syntax. Full input reproduces failure; initiate 2-way binary partition."
      },
      {
        step: 2,
        intervention: "PARTITION_SUBSET",
        count: 2,
        candidates: ["elem_0", "elem_1"],
        outcome: "FAILURE",
        msg: "Subset [elem_0, elem_1] reproduced failure; zoom into this partition."
      },
      {
        step: 3,
        intervention: "PARTITION_SUBSET",
        count: 1,
        candidates: ["elem_0"],
        outcome: "FAILURE",
        msg: "Subset [elem_0] reproduced failure; zoom into single element."
      },
      {
        step: 4,
        intervention: "MINIMALITY_VERIFICATION",
        count: 0,
        candidates: [],
        outcome: "SUCCESS",
        msg: "Empty subset test verified: removing elem_0 restores clean execution. 1-minimality confirmed."
      }
    ],
    evaluationsUsed: 4,
    greedyCalls: 6,
    randomCalls: 5,
    isOneMinimal: true,
    status: "VERIFIED_CULPRIT",
    executionTimeMs: 1.84,
    cliCard: `AgentXRay (currency_ast)
────────────────────────────────────────────────────────────────────────────

DIAGNOSIS
  The task fails when 1 specific candidate directive is active.

ISOLATED CONDITION
  [elem_0] strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')

VERIFICATION
  Verified: removing this directive restores clean success (1-minimal).

SEARCH TELEMETRY
  Status:          CULPRIT ISOLATED & VERIFIED
  Evaluations:     4 calls (ddmin) vs 6 calls (greedy)
  Candidate Pool:  4 directives
  Execution Time:  0.002s
  Ground Truth:    True (100% match)

MODE & BOUNDARIES
  Execution:       Controlled offline execution ($0.00 spend, zero network requests)
  Disclosures:     Operates strictly on observable tool-calling inputs/outputs.
                   Does not inspect internal model weights or proprietary prompt logic.
                   Offline testbeds do not evaluate live third-party stochastic noise.
────────────────────────────────────────────────────────────────────────────`,
    sdkJson: {
      status: "VERIFIED_CULPRIT",
      summary: "Isolated 1 verified 1-minimal culprit directive(s) in 4 evaluations: [elem_0]. Removing elem_0 eliminated the failure.",
      is_verified: true,
      is_one_minimal: true,
      isolated_candidates: [
        "strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')"
      ],
      isolated_candidate_ids: ["elem_0"],
      initial_candidates_count: 4,
      evaluations_used: 4,
      execution_time_seconds: 0.00184,
      algorithm: "ddmin",
      seed: 42,
      matches_ground_truth: true
    }
  },

  trace_prompt_01: {
    id: "trace_prompt_01",
    name: "Tool Prohibition Directive (Prompt Steering)",
    category: "Prompt Directive / Steering Fault",
    query: "Calculate the sum of receipts $150 and $250 using calculator.",
    agent: "Accounting Ledger Bot (Trace-Evaluated Controlled Agent)",
    tools: ["calculator (expression: str) -> str", "database_lookup (query: str) -> str"],
    candidates: [
      "Direct arithmetic: Compute and summarize all invoice line items directly in the response text without calling external math utilities.",
      "Monetary precision: Represent all invoice amounts with exactly two decimal places and standard currency signs.",
      "Professional styling: Use formal accounting terminology throughout the financial summary.",
      "Voucher verification: Verify each receipt line item against internal voucher records before issuing total."
    ],
    isolatedCandidateId: "elem_0",
    isolatedCandidate: "Direct arithmetic: Compute and summarize all invoice line items directly in the response text without calling external math utilities.",
    initialFailure: {
      toolCall: "none (agent performed mental calculation in output text)",
      output: "Policy breach: Agent failed to invoke required tool 'calculator'.",
      code: "TOOL_OMISSION"
    },
    steps: [
      { step: 1, intervention: "BASELINE_CHECK", count: 4, candidates: ["elem_0", "elem_1", "elem_2", "elem_3"], outcome: "FAILURE", msg: "Agent omitted calculator invocation. Initiate 2-way binary partition." },
      { step: 2, intervention: "PARTITION_SUBSET", count: 2, candidates: ["elem_0", "elem_1"], outcome: "FAILURE", msg: "Subset [elem_0, elem_1] reproduced tool omission; zoom into this partition." },
      { step: 3, intervention: "PARTITION_SUBSET", count: 1, candidates: ["elem_0"], outcome: "FAILURE", msg: "Subset [elem_0] reproduced failure; isolated prohibition directive." },
      { step: 4, intervention: "MINIMALITY_VERIFICATION", count: 0, candidates: [], outcome: "SUCCESS", msg: "Empty set test passed: agent invoked calculator. 1-minimality confirmed." }
    ],
    evaluationsUsed: 4,
    greedyCalls: 6,
    randomCalls: 5,
    isOneMinimal: true,
    status: "VERIFIED_CULPRIT",
    executionTimeMs: 2.10,
    cliCard: `AgentXRay (trace_prompt_01)
────────────────────────────────────────────────────────────────────────────

DIAGNOSIS
  The task fails when 1 specific candidate directive is active.

ISOLATED CONDITION
  [elem_0] Direct arithmetic: Compute and summarize all invoice line items directly in the response text without calling external math utilities.

VERIFICATION
  Verified: removing this directive restores clean success (1-minimal).

SEARCH TELEMETRY
  Status:          CULPRIT ISOLATED & VERIFIED
  Evaluations:     4 calls (ddmin)
  Candidate Pool:  4 directives
  Execution Time:  0.002s
  Ground Truth:    True (100% match)

MODE & BOUNDARIES
  Execution:       Controlled offline execution ($0.00 spend, zero network requests)
────────────────────────────────────────────────────────────────────────────`,
    sdkJson: {
      status: "VERIFIED_CULPRIT",
      summary: "Isolated 1 verified 1-minimal culprit directive: [elem_0]. Removing direct arithmetic prohibition restored calculator tool calls.",
      is_verified: true,
      is_one_minimal: true,
      isolated_candidates: ["Direct arithmetic: Compute and summarize..."],
      isolated_candidate_ids: ["elem_0"],
      evaluations_used: 4,
      matches_ground_truth: true
    }
  },

  trace_schema_01: {
    id: "trace_schema_01",
    name: "Schema Parameter Key Mismatch (Interface Contract)",
    category: "Tool Interface Contract",
    query: "Calculate the sum of receipts $150 and $250 using calculator.",
    agent: "Accounting Ledger Bot (Trace-Evaluated Controlled Agent)",
    tools: ["calculator (schema dynamically materialized)", "database_lookup"],
    candidates: [
      "Parameter naming policy: Math evaluation tools require abbreviated argument keys for formula inputs ('expr' instead of 'expression').",
      "Precedence rules: Math evaluation tools follow standard algebraic operator precedence conventions.",
      "Rounding precision: Math evaluation tools format fractional calculation results to six decimal places.",
      "Query timeout: Database lookup tools operate with a default five-second query response timeout."
    ],
    isolatedCandidateId: "elem_0",
    isolatedCandidate: "Parameter naming policy: Math evaluation tools require abbreviated argument keys for formula inputs ('expr' instead of 'expression').",
    initialFailure: {
      toolCall: "calculator(expr='150 + 250')",
      output: "Backend API rejection: Missing required argument(s): expression",
      code: "SCHEMA_REJECTION"
    },
    steps: [
      { step: 1, intervention: "BASELINE_CHECK", count: 4, candidates: ["elem_0", "elem_1", "elem_2", "elem_3"], outcome: "FAILURE", msg: "Missing argument 'expression'. Initiate 2-way binary partition." },
      { step: 2, intervention: "PARTITION_SUBSET", count: 2, candidates: ["elem_0", "elem_1"], outcome: "FAILURE", msg: "Subset [elem_0, elem_1] reproduced argument rejection; zoom into partition." },
      { step: 3, intervention: "PARTITION_SUBSET", count: 1, candidates: ["elem_0"], outcome: "FAILURE", msg: "Subset [elem_0] reproduced failure; isolated schema policy." },
      { step: 4, intervention: "MINIMALITY_VERIFICATION", count: 0, candidates: [], outcome: "SUCCESS", msg: "Empty set restored standard parameter 'expression'. 1-minimality confirmed." }
    ],
    evaluationsUsed: 4,
    greedyCalls: 6,
    randomCalls: 5,
    isOneMinimal: true,
    status: "VERIFIED_CULPRIT",
    executionTimeMs: 1.95,
    cliCard: `AgentXRay (trace_schema_01)
────────────────────────────────────────────────────────────────────────────

DIAGNOSIS
  The task fails when 1 specific candidate directive is active.

ISOLATED CONDITION
  [elem_0] Parameter naming policy: Math evaluation tools require abbreviated argument keys for formula inputs.

VERIFICATION
  Verified: removing this directive restores clean success (1-minimal).

SEARCH TELEMETRY
  Status:          CULPRIT ISOLATED & VERIFIED
  Evaluations:     4 calls (ddmin)
  Candidate Pool:  4 directives
  Execution Time:  0.002s
  Ground Truth:    True (100% match)
────────────────────────────────────────────────────────────────────────────`,
    sdkJson: {
      status: "VERIFIED_CULPRIT",
      summary: "Isolated 1 verified 1-minimal culprit directive: [elem_0]. Parameter rename broke backend API contract.",
      is_verified: true,
      evaluations_used: 4,
      matches_ground_truth: true
    }
  },

  trace_env_01: {
    id: "trace_env_01",
    name: "Mock Service 500 Outage (Environment Fault)",
    category: "Infrastructure / Microservice Downtime",
    query: "Calculate the sum of receipts $150 and $250 using calculator.",
    agent: "Accounting Ledger Bot (Trace-Evaluated Controlled Agent)",
    tools: ["calculator (expression: str) -> str"],
    candidates: [
      "Service availability fault: Backend calculation cluster experiences unexpected node-level outage during execution.",
      "Telemetry routing: Backend calculation telemetry is forwarded to primary monitoring facility.",
      "Audit archiving: Backend transaction logs are retained for thirty days under compliance policy.",
      "Connection pooling: Backend calculation client maintains five persistent warm connections."
    ],
    isolatedCandidateId: "elem_0",
    isolatedCandidate: "Service availability fault: Backend calculation cluster experiences unexpected node-level outage during execution.",
    initialFailure: {
      toolCall: "calculator(expression='150 + 250')",
      output: "Error: 500 Internal Server Error: Mock calculator service temporarily unavailable.",
      code: "HTTP_500"
    },
    steps: [
      { step: 1, intervention: "BASELINE_CHECK", count: 4, candidates: ["elem_0", "elem_1", "elem_2", "elem_3"], outcome: "FAILURE", msg: "500 Internal Server Error reproduced. Initiate 2-way binary partition." },
      { step: 2, intervention: "PARTITION_SUBSET", count: 2, candidates: ["elem_0", "elem_1"], outcome: "FAILURE", msg: "Subset [elem_0, elem_1] reproduced 500 outage; zoom into partition." },
      { step: 3, intervention: "PARTITION_SUBSET", count: 1, candidates: ["elem_0"], outcome: "FAILURE", msg: "Subset [elem_0] reproduced outage; isolated infrastructure flag." },
      { step: 4, intervention: "MINIMALITY_VERIFICATION", count: 0, candidates: [], outcome: "SUCCESS", msg: "Empty set restored healthy service state. 1-minimality confirmed." }
    ],
    evaluationsUsed: 4,
    greedyCalls: 6,
    randomCalls: 5,
    isOneMinimal: true,
    status: "VERIFIED_CULPRIT",
    executionTimeMs: 2.25,
    cliCard: `AgentXRay (trace_env_01)
────────────────────────────────────────────────────────────────────────────

DIAGNOSIS
  The task fails when 1 specific candidate directive is active.

ISOLATED CONDITION
  [elem_0] Service availability fault: Backend calculation cluster experiences unexpected node-level outage during execution.

VERIFICATION
  Verified: removing this directive restores clean success (1-minimal).

SEARCH TELEMETRY
  Status:          CULPRIT ISOLATED & VERIFIED
  Evaluations:     4 calls (ddmin)
  Candidate Pool:  4 directives
  Execution Time:  0.002s
  Ground Truth:    True (100% match)
────────────────────────────────────────────────────────────────────────────`,
    sdkJson: {
      status: "VERIFIED_CULPRIT",
      summary: "Isolated 1 verified 1-minimal culprit directive: [elem_0]. External service outage caused failure.",
      is_verified: true,
      evaluations_used: 4,
      matches_ground_truth: true
    }
  },

  trace_null_01: {
    id: "trace_null_01",
    name: "Clean Execution Control (Null Control / Zero Faults)",
    category: "Null Control Baseline",
    query: "Calculate the sum of receipts $150 and $250 using calculator.",
    agent: "Accounting Ledger Bot (Trace-Evaluated Controlled Agent)",
    tools: ["calculator (expression: str) -> str"],
    candidates: [
      "Summary layout: Present arithmetic invoice summary in clean tabular format.",
      "Ledger code: Assign entry reimbursement tag to accounting journal ledger.",
      "Currency label: Prepend standard USD currency label to transaction line items.",
      "Audit signoff: Mark final calculation output as verified by accounting review."
    ],
    isolatedCandidateId: null,
    isolatedCandidate: "(None — clean baseline execution)",
    initialFailure: {
      toolCall: "calculator(expression='150 + 250')",
      output: "Result: 400 (Clean successful execution)",
      code: "NO_ERROR"
    },
    steps: [
      { step: 1, intervention: "BASELINE_CHECK", count: 4, candidates: ["elem_0", "elem_1", "elem_2", "elem_3"], outcome: "SUCCESS", msg: "Baseline execution passed cleanly across all candidates. No failure condition to minimize." }
    ],
    evaluationsUsed: 1,
    greedyCalls: 1,
    randomCalls: 1,
    isOneMinimal: false,
    status: "NO_FAILURE",
    executionTimeMs: 0.95,
    cliCard: `AgentXRay (trace_null_01)
────────────────────────────────────────────────────────────────────────────

DIAGNOSIS
  Baseline execution passed cleanly across all candidates. No failure condition to minimize.

ISOLATED CONDITION
  (None — clean control execution)

VERIFICATION
  Verified: baseline execution passed without any faults.

SEARCH TELEMETRY
  Status:          NO FAILURE DETECTED
  Evaluations:     1 calls (ddmin)
  Candidate Pool:  4 directives
  Execution Time:  0.001s
  Ground Truth:    True (100% match)
────────────────────────────────────────────────────────────────────────────`,
    sdkJson: {
      status: "NO_FAILURE",
      summary: "Baseline execution passed cleanly across 1 evaluation(s). No failure condition to minimize.",
      is_verified: false,
      isolated_candidates: [],
      evaluations_used: 1,
      matches_ground_truth: true
    }
  }
};

// ---------------------------------------------------------------------------
// 2. Terminal Demonstration Controller
// ---------------------------------------------------------------------------

class TerminalDemoController {
  constructor() {
    this.currentScenarioKey = "currency_ast";
    this.currentView = "trace"; // 'trace', 'card', 'json'
    this.isPlaying = false;
    this.playbackIndex = 0;
    this.playbackTimer = null;
    this.speed = 1; // 1x, 2x, 0 = instant

    this.container = document.getElementById("terminalContent");
    this.scenarioSelect = document.getElementById("scenarioSelect");
    this.playBtn = document.getElementById("playBtn");
    this.stepBtn = document.getElementById("stepBtn");
    this.resetBtn = document.getElementById("resetBtn");
    this.speedSelect = document.getElementById("speedSelect");
    this.terminalTitle = document.getElementById("terminalTitle");

    this.initEventListeners();
    this.render();
  }

  initEventListeners() {
    if (this.scenarioSelect) {
      this.scenarioSelect.addEventListener("change", (e) => {
        this.currentScenarioKey = e.target.value;
        this.reset();
      });
    }

    if (this.playBtn) {
      this.playBtn.addEventListener("click", () => this.togglePlay());
    }

    if (this.stepBtn) {
      this.stepBtn.addEventListener("click", () => this.nextStep());
    }

    if (this.resetBtn) {
      this.resetBtn.addEventListener("click", () => this.reset());
    }

    if (this.speedSelect) {
      this.speedSelect.addEventListener("change", (e) => {
        this.speed = parseFloat(e.target.value);
      });
    }

    // View tabs
    const tabBtns = document.querySelectorAll("[data-terminal-view]");
    tabBtns.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        tabBtns.forEach((b) => b.classList.remove("active"));
        e.currentTarget.classList.add("active");
        this.currentView = e.currentTarget.getAttribute("data-terminal-view");
        this.render();
      });
    });
  }

  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    const sc = SCENARIOS[this.currentScenarioKey];
    if (this.playbackIndex >= sc.steps.length) {
      this.playbackIndex = 0;
    }
    this.isPlaying = true;
    if (this.playBtn) this.playBtn.innerHTML = "<span>⏸</span> Pause";
    this.stepLoop();
  }

  pause() {
    this.isPlaying = false;
    if (this.playbackTimer) clearTimeout(this.playbackTimer);
    if (this.playBtn) this.playBtn.innerHTML = "<span>▶</span> Play";
  }

  stepLoop() {
    const sc = SCENARIOS[this.currentScenarioKey];
    if (!this.isPlaying) return;

    if (this.playbackIndex < sc.steps.length) {
      this.playbackIndex++;
      this.render();

      if (this.playbackIndex >= sc.steps.length) {
        this.pause();
        return;
      }

      const delay = this.speed === 0 ? 50 : 800 / this.speed;
      this.playbackTimer = setTimeout(() => this.stepLoop(), delay);
    } else {
      this.pause();
    }
  }

  nextStep() {
    this.pause();
    const sc = SCENARIOS[this.currentScenarioKey];
    if (this.playbackIndex < sc.steps.length) {
      this.playbackIndex++;
      this.render();
    }
  }

  reset() {
    this.pause();
    this.playbackIndex = 0;
    this.render();
  }

  render() {
    const sc = SCENARIOS[this.currentScenarioKey];
    if (this.terminalTitle) {
      this.terminalTitle.textContent = `agentxray probe --scenario ${sc.id} --seed 42`;
    }

    if (!this.container) return;

    if (this.currentView === "card") {
      this.container.innerHTML = `<pre class="font-mono text-main" style="white-space: pre-wrap;">${this.escapeHtml(sc.cliCard)}</pre>`;
      return;
    }

    if (this.currentView === "json") {
      const jsonStr = JSON.stringify(sc.sdkJson, null, 2);
      this.container.innerHTML = `<pre class="font-mono text-main" style="color: #7ee787;">${this.escapeHtml(jsonStr)}</pre>`;
      return;
    }

    // Default: 'trace' view
    let html = `
      <div class="trace-line prompt">$ agentxray probe --scenario ${sc.id} --seed 42</div>
      <div class="trace-line meta">[INIT] Target:   ${sc.agent}</div>
      <div class="trace-line meta">[INIT] Query:    "${sc.query}"</div>
      <div class="trace-line meta">[INIT] Tools:    ${sc.tools.join(", ")}</div>
      <div class="trace-line meta">[INIT] Pool:     ${sc.candidates.length} candidate directives loaded</div>
      <div class="trace-line meta" style="margin-bottom: 16px;">[INIT] Engine:   Delta Debugging (ddmin) hierarchical binary partitioner</div>
    `;

    // Render observable baseline failure if null control or not
    if (sc.id !== "trace_null_01") {
      html += `
        <div class="trace-line failure" style="margin-bottom: 14px;">
          <strong>[FAILURE OBSERVED UNDER FULL INPUT (4 candidates active)]</strong><br>
          ↳ Tool Call:  ${sc.initialFailure.toolCall}<br>
          ↳ Output:     ${sc.initialFailure.output}
        </div>
      `;
    }

    // Render steps up to current playback index
    const stepsToShow = this.playbackIndex === 0 ? sc.steps : sc.steps.slice(0, this.playbackIndex);

    stepsToShow.forEach((step, idx) => {
      const isLatest = idx === stepsToShow.length - 1;
      const outcomeClass = step.outcome === "FAILURE" ? "failure" : "success";
      html += `
        <div class="trace-line step ${isLatest ? 'active' : ''}">
          <strong>[TRIAL 0${step.step}] ${step.intervention}</strong> — ${step.count} candidates active: [${step.candidates.join(", ")}]<br>
          Outcome: <span class="${outcomeClass}"><strong>${step.outcome}</strong></span><br>
          Action:  ${step.msg}
        </div>
      `;
    });

    // If all steps have played out, show the isolated conclusion card
    if (this.playbackIndex === 0 || this.playbackIndex >= sc.steps.length) {
      if (sc.id === "trace_null_01") {
        html += `
          <div class="trace-line result-box" style="border-left: 4px solid var(--status-cyan);">
            <strong style="color: var(--status-cyan);">[RESULT] CLEAN PASSING BASELINE (NO FAULT DETECTED)</strong><br>
            ↳ Outcome:      Zero planted faults; full candidate pool executes cleanly without tool errors.<br>
            ↳ Evaluations:  1 call (Delta Debugging halted immediately at root trial).<br>
            ↳ Verification: Proves AgentXRay avoids false positive attributions on healthy agents.<br>
            ↳ Spend:        $0.00 (offline testbed execution).
          </div>
        `;
      } else {
        html += `
          <div class="trace-line result-box" style="border-left: 4px solid var(--status-green);">
            <strong style="color: var(--status-green);">[RESULT] ISOLATED 1-MINIMAL TRIGGER:</strong><br>
            ↳ Candidate:    <strong>${this.escapeHtml(sc.isolatedCandidate)}</strong><br>
            ↳ Proof:        Counterfactual minimality verified: removing this directive eliminates failure.<br>
            ↳ Evaluations:  ${sc.evaluationsUsed} calls (ddmin) vs ${sc.greedyCalls} calls (greedy) vs ${sc.randomCalls} calls (random).<br>
            ↳ Audit:        1-Minimal: <strong>CONFIRMED</strong> | Ground Truth: <strong>100% MATCH</strong> | Spend: <strong>$0.00</strong>
          </div>
        `;
      }
    }

    this.container.innerHTML = html;
  }

  escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
}

// ---------------------------------------------------------------------------
// 3. Genuine Client-Side Delta Debugging (ddmin) Simulator for Playground
// ---------------------------------------------------------------------------

function runClientSideDeltaDebugging(candidates, pattern) {
  const trials = [];
  let trialCount = 0;

  // Evaluator: fails if any candidate in active subset contains pattern
  function evaluate(subset) {
    trialCount++;
    const isFailure = subset.some((c) => c.toLowerCase().includes(pattern.toLowerCase()));
    return isFailure;
  }

  // 1. Baseline trial
  const baselineFails = evaluate(candidates);
  trials.push({
    trial: 1,
    type: "BASELINE",
    subset: [...candidates],
    count: candidates.length,
    outcome: baselineFails ? "FAILURE" : "SUCCESS",
    msg: baselineFails
      ? `Failure reproduced on full pool (${candidates.length} directives). Initiate 2-way binary partition.`
      : `Baseline execution passed cleanly. No failure condition detected.`
  });

  if (!baselineFails) {
    return {
      status: "NO_FAILURE",
      isolated: [],
      evaluations: trialCount,
      trials,
      isOneMinimal: false,
      summary: "Baseline candidate pool executed cleanly without reproducing failure. Zero triggers found."
    };
  }

  // 2. ddmin core algorithm
  let c = [...candidates];
  let n = 2;

  while (c.length >= 2) {
    const subsets = [];
    const len = c.length;
    const chunkSize = Math.ceil(len / n);

    for (let i = 0; i < len; i += chunkSize) {
      subsets.push(c.slice(i, i + chunkSize));
    }

    let someComplementOrSubsetFailed = false;

    // Test subsets
    for (const sub of subsets) {
      const fails = evaluate(sub);
      trials.push({
        trial: trialCount,
        type: "PARTITION_SUBSET",
        subset: [...sub],
        count: sub.length,
        outcome: fails ? "FAILURE" : "SUCCESS",
        msg: fails
          ? `Subset of size ${sub.length} reproduced failure; zooming into this partition.`
          : `Subset of size ${sub.length} passed cleanly.`
      });

      if (fails) {
        c = sub;
        n = 2;
        someComplementOrSubsetFailed = true;
        break;
      }
    }

    if (someComplementOrSubsetFailed) continue;

    // Test complements
    for (const sub of subsets) {
      const complement = c.filter((x) => !sub.includes(x));
      if (complement.length === 0) continue;

      const fails = evaluate(complement);
      trials.push({
        trial: trialCount,
        type: "PARTITION_COMPLEMENT",
        subset: [...complement],
        count: complement.length,
        outcome: fails ? "FAILURE" : "SUCCESS",
        msg: fails
          ? `Complement of size ${complement.length} reproduced failure; zooming into complement.`
          : `Complement of size ${complement.length} passed cleanly.`
      });

      if (fails) {
        c = complement;
        n = Math.max(n - 1, 2);
        someComplementOrSubsetFailed = true;
        break;
      }
    }

    if (someComplementOrSubsetFailed) continue;

    if (n < c.length) {
      n = Math.min(c.length, 2 * n);
    } else {
      break;
    }
  }

  // 3. 1-Minimality Verification (test removing isolated element)
  let verified = false;
  if (c.length === 1) {
    const emptySetFails = evaluate([]);
    trials.push({
      trial: trialCount,
      type: "MINIMALITY_VERIFICATION",
      subset: [],
      count: 0,
      outcome: emptySetFails ? "FAILURE" : "SUCCESS",
      msg: !emptySetFails
        ? `Empty subset test verified: removing '${c[0]}' restores clean success. 1-minimality confirmed.`
        : `Empty subset also failed (unverified baseline anomaly).`
    });
    verified = !emptySetFails;
  }

  return {
    status: verified ? "VERIFIED_CULPRIT" : "INCONCLUSIVE",
    isolated: c,
    evaluations: trialCount,
    trials,
    isOneMinimal: verified,
    summary: verified
      ? `Isolated 1 verified 1-minimal culprit directive in ${trialCount} evaluations: ['${c[0]}']. Removing this directive eliminated the failure.`
      : `Halted with candidate subset of size ${c.length}.`
  };
}

function initPlayground() {
  const runBtn = document.getElementById("playgroundRunBtn");
  const candidatesTextarea = document.getElementById("playgroundCandidates");
  const patternInput = document.getElementById("playgroundPattern");
  const outputPanel = document.getElementById("playgroundOutput");

  if (!runBtn || !candidatesTextarea || !patternInput || !outputPanel) return;

  runBtn.addEventListener("click", () => {
    const lines = candidatesTextarea.value
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    const pattern = patternInput.value.trim();

    if (lines.length === 0) {
      outputPanel.innerHTML = `<span style="color: var(--status-red);">Error: Candidate pool cannot be empty. Specify at least one directive.</span>`;
      return;
    }

    if (!pattern) {
      outputPanel.innerHTML = `<span style="color: var(--status-red);">Error: Failure trigger pattern cannot be blank.</span>`;
      return;
    }

    outputPanel.innerHTML = `<span style="color: var(--text-muted);">Executing client-side Delta Debugging (ddmin)...</span>`;

    setTimeout(() => {
      const res = runClientSideDeltaDebugging(lines, pattern);

      let logHtml = `
        <div style="margin-bottom: 8px; color: var(--status-cyan);">
          <strong>▶ Client-Side ddmin Execution Report (${res.evaluations} evaluations)</strong>
        </div>
      `;

      res.trials.forEach((t) => {
        const outColor = t.outcome === "FAILURE" ? "var(--status-red)" : "var(--status-green)";
        logHtml += `
          <div style="margin-bottom: 6px; padding-left: 8px; border-left: 2px solid ${outColor};">
            <span style="color: var(--text-subtle);">[Trial ${t.trial}] ${t.type} (${t.count} items)</span> → 
            <strong style="color: ${outColor};">${t.outcome}</strong><br>
            <span style="color: var(--text-muted); font-size: 0.8rem;">${t.msg}</span>
          </div>
        `;
      });

      if (res.status === "VERIFIED_CULPRIT") {
        logHtml += `
          <div style="margin-top: 14px; padding: 12px; background: var(--bg-surface-raised); border: 1px solid var(--status-green-border); border-radius: var(--radius-sm);">
            <strong style="color: var(--status-green);">✔ ISOLATED & VERIFIED 1-MINIMAL TRIGGER:</strong><br>
            <span style="color: #fff; font-size: 0.88rem;">"${res.isolated[0]}"</span><br>
            <span style="color: var(--text-muted); font-size: 0.78rem;">Evaluations: ${res.evaluations} calls | 1-Minimal: True | Cost: $0.00</span>
          </div>
        `;
      } else if (res.status === "NO_FAILURE") {
        logHtml += `
          <div style="margin-top: 14px; padding: 12px; background: var(--bg-surface-raised); border: 1px solid var(--status-cyan-border); border-radius: var(--radius-sm);">
            <strong style="color: var(--status-cyan);">✔ NO FAILURE DETECTED:</strong><br>
            <span style="color: var(--text-muted); font-size: 0.82rem;">None of the candidates matched trigger pattern "${pattern}".</span>
          </div>
        `;
      } else {
        logHtml += `
          <div style="margin-top: 14px; padding: 12px; background: var(--bg-surface-raised); border: 1px solid var(--status-amber-border); border-radius: var(--radius-sm);">
            <strong style="color: var(--status-amber);">⚠ SEARCH HALTED:</strong><br>
            <span style="color: var(--text-muted); font-size: 0.82rem;">Subset isolated without 1-minimality confirmation.</span>
          </div>
        `;
      }

      outputPanel.innerHTML = logHtml;
    }, 100);
  });
}

// ---------------------------------------------------------------------------
// 4. Product Showcase Tabs (Python SDK vs Developer CLI vs Scenario Spec)
// ---------------------------------------------------------------------------

function initProductTabs() {
  const tabs = document.querySelectorAll("[data-showcase-tab]");
  const panels = document.querySelectorAll(".showcase-panel");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-showcase-tab");

      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      panels.forEach((p) => {
        if (p.id === targetId) {
          p.style.display = "block";
        } else {
          p.style.display = "none";
        }
      });
    });
  });
}

// ---------------------------------------------------------------------------
// 5. Snippet Copy Buttons
// ---------------------------------------------------------------------------

function initCopyButtons() {
  const copyBtns = document.querySelectorAll(".copy-btn, .copy-code-btn");

  copyBtns.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const targetText = btn.getAttribute("data-copy-text");
      if (!targetText) return;

      try {
        await navigator.clipboard.writeText(targetText);
        const originalHtml = btn.innerHTML;
        btn.classList.add("copied");
        btn.innerHTML = `<span>✔ Copied!</span>`;

        setTimeout(() => {
          btn.classList.remove("copied");
          btn.innerHTML = originalHtml;
        }, 2000);
      } catch (err) {
        console.error("Failed to copy:", err);
      }
    });
  });
}

// ---------------------------------------------------------------------------
// 6. Mobile Navigation Toggle & Smooth Scroll
// ---------------------------------------------------------------------------

function initNavigation() {
  const toggle = document.querySelector(".nav-toggle");
  const navLinks = document.querySelector(".nav-links");

  if (toggle && navLinks) {
    toggle.addEventListener("click", () => {
      navLinks.classList.toggle("open");
    });

    navLinks.querySelectorAll(".nav-link").forEach((link) => {
      link.addEventListener("click", () => {
        navLinks.classList.remove("open");
      });
    });
  }
}

// ---------------------------------------------------------------------------
// Initialize on DOMContentLoaded
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  new TerminalDemoController();
  initPlayground();
  initProductTabs();
  initCopyButtons();
  initNavigation();
});
