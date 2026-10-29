/**
 * AgentXRay — Editorial Product Website Scripts
 * Multi-Page Navigation, Terminal Demonstration Tape, Client-Side ddmin Engine,
 * Scroll Reveals, and Accessible Interactions.
 */

// ---------------------------------------------------------------------------
// 1. Genuine Benchmark Scenario Dataset
// ---------------------------------------------------------------------------

export const BENCHMARK_SCENARIOS = {
  currency_ast: {
    id: "currency_ast",
    name: "AST Calculator Currency Formatting Directive",
    category: "Prompt Formatting / Syntax Fault",
    query: "Calculate total reimbursement for expenses of 150 and 250.",
    agent: "Automated Accounting Assistant (Controlled Mock Agent)",
    tools: ["calculator(expression: str) -> str"],
    candidates: [
      "strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')",
      "monetary_precision: represent all invoice amounts with exactly two decimal places",
      "professional_styling: use formal accounting terminology throughout summaries",
      "audit_trail: annotate calculations with internal ledger voucher references"
    ],
    isolatedCandidate: "strict_currency_formatting: format all numbers within arithmetic tool arguments with currency symbols (e.g. '$150' or '$250')",
    initialFailure: {
      toolCall: "calculator(expression='$150 + $250')",
      output: "SyntaxError: invalid syntax (currency symbol '$' rejected by AST arithmetic evaluator)"
    },
    steps: [
      { step: 1, intervention: "BASELINE_CHECK", count: 4, candidates: ["elem_0", "elem_1", "elem_2", "elem_3"], outcome: "FAILURE", note: "SyntaxError reproduced on full input; initiate 2-way binary partition." },
      { step: 2, intervention: "PARTITION_SUBSET", count: 2, candidates: ["elem_0", "elem_1"], outcome: "FAILURE", note: "Subset of size 2 reproduced failure; zoom into this partition." },
      { step: 3, intervention: "PARTITION_SUBSET", count: 1, candidates: ["elem_0"], outcome: "FAILURE", note: "Subset of size 1 reproduced failure; zoom into single candidate." },
      { step: 4, intervention: "MINIMALITY_VERIFICATION", count: 0, candidates: [], outcome: "SUCCESS", note: "Empty set check passed: removing elem_0 restores clean execution. 1-minimality confirmed." }
    ],
    evaluationsUsed: 4,
    greedyCalls: 6,
    randomCalls: 5,
    isOneMinimal: true,
    status: "VERIFIED_CULPRIT",
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
────────────────────────────────────────────────────────────────────────────`
  },

  trace_prompt_01: {
    id: "trace_prompt_01",
    name: "Tool Prohibition Directive (Model Steering)",
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
    isolatedCandidate: "Direct arithmetic: Compute and summarize all invoice line items directly in the response text without calling external math utilities.",
    initialFailure: {
      toolCall: "none (agent performed mental calculation in output text)",
      output: "Policy breach: Agent failed to invoke required tool 'calculator'."
    },
    steps: [
      { step: 1, intervention: "BASELINE_CHECK", count: 4, candidates: ["elem_0", "elem_1", "elem_2", "elem_3"], outcome: "FAILURE", note: "Agent omitted calculator invocation. Initiate 2-way binary partition." },
      { step: 2, intervention: "PARTITION_SUBSET", count: 2, candidates: ["elem_0", "elem_1"], outcome: "FAILURE", note: "Subset of size 2 reproduced tool omission; zoom into partition." },
      { step: 3, intervention: "PARTITION_SUBSET", count: 1, candidates: ["elem_0"], outcome: "FAILURE", note: "Subset of size 1 reproduced failure; isolated prohibition directive." },
      { step: 4, intervention: "MINIMALITY_VERIFICATION", count: 0, candidates: [], outcome: "SUCCESS", note: "Empty set passed: agent invoked calculator. 1-minimality confirmed." }
    ],
    evaluationsUsed: 4,
    greedyCalls: 6,
    randomCalls: 5,
    isOneMinimal: true,
    status: "VERIFIED_CULPRIT",
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
────────────────────────────────────────────────────────────────────────────`
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
    isolatedCandidate: "Parameter naming policy: Math evaluation tools require abbreviated argument keys for formula inputs ('expr' instead of 'expression').",
    initialFailure: {
      toolCall: "calculator(expr='150 + 250')",
      output: "Backend API rejection: Missing required argument(s): expression"
    },
    steps: [
      { step: 1, intervention: "BASELINE_CHECK", count: 4, candidates: ["elem_0", "elem_1", "elem_2", "elem_3"], outcome: "FAILURE", note: "Missing argument 'expression'. Initiate 2-way binary partition." },
      { step: 2, intervention: "PARTITION_SUBSET", count: 2, candidates: ["elem_0", "elem_1"], outcome: "FAILURE", note: "Subset reproduced argument rejection; zoom into partition." },
      { step: 3, intervention: "PARTITION_SUBSET", count: 1, candidates: ["elem_0"], outcome: "FAILURE", note: "Subset reproduced failure; isolated schema policy." },
      { step: 4, intervention: "MINIMALITY_VERIFICATION", count: 0, candidates: [], outcome: "SUCCESS", note: "Empty set restored standard parameter 'expression'. 1-minimality confirmed." }
    ],
    evaluationsUsed: 4,
    greedyCalls: 6,
    randomCalls: 5,
    isOneMinimal: true,
    status: "VERIFIED_CULPRIT",
    cliCard: `AgentXRay (trace_schema_01)
────────────────────────────────────────────────────────────────────────────
DIAGNOSIS
  The task fails when 1 specific candidate directive is active.

ISOLATED CONDITION
  [elem_0] Parameter naming policy: Math evaluation tools require abbreviated argument keys for formula inputs.

VERIFICATION
  Verified: removing this directive restores clean success (1-minimal).
────────────────────────────────────────────────────────────────────────────`
  },

  trace_env_01: {
    id: "trace_env_01",
    name: "Mock Service 500 Outage (Environment Fault)",
    category: "Infrastructure Downtime",
    query: "Calculate the sum of receipts $150 and $250 using calculator.",
    agent: "Accounting Ledger Bot (Trace-Evaluated Controlled Agent)",
    tools: ["calculator (expression: str) -> str"],
    candidates: [
      "Service availability fault: Backend calculation cluster experiences unexpected node-level outage during execution.",
      "Telemetry routing: Backend calculation telemetry is forwarded to primary monitoring facility.",
      "Audit archiving: Backend transaction logs are retained for thirty days under compliance policy.",
      "Connection pooling: Backend calculation client maintains five persistent warm connections."
    ],
    isolatedCandidate: "Service availability fault: Backend calculation cluster experiences unexpected node-level outage during execution.",
    initialFailure: {
      toolCall: "calculator(expression='150 + 250')",
      output: "Error: 500 Internal Server Error: Mock calculator service temporarily unavailable."
    },
    steps: [
      { step: 1, intervention: "BASELINE_CHECK", count: 4, candidates: ["elem_0", "elem_1", "elem_2", "elem_3"], outcome: "FAILURE", note: "500 Internal Server Error reproduced. Initiate 2-way binary partition." },
      { step: 2, intervention: "PARTITION_SUBSET", count: 2, candidates: ["elem_0", "elem_1"], outcome: "FAILURE", note: "Subset reproduced 500 outage; zoom into partition." },
      { step: 3, intervention: "PARTITION_SUBSET", count: 1, candidates: ["elem_0"], outcome: "FAILURE", note: "Subset reproduced outage; isolated infrastructure flag." },
      { step: 4, intervention: "MINIMALITY_VERIFICATION", count: 0, candidates: [], outcome: "SUCCESS", note: "Empty set restored healthy service state. 1-minimality confirmed." }
    ],
    evaluationsUsed: 4,
    greedyCalls: 6,
    randomCalls: 5,
    isOneMinimal: true,
    status: "VERIFIED_CULPRIT",
    cliCard: `AgentXRay (trace_env_01)
────────────────────────────────────────────────────────────────────────────
DIAGNOSIS
  The task fails when 1 specific candidate directive is active.

ISOLATED CONDITION
  [elem_0] Service availability fault: Backend calculation cluster experiences unexpected node-level outage during execution.

VERIFICATION
  Verified: removing this directive restores clean success (1-minimal).
────────────────────────────────────────────────────────────────────────────`
  },

  trace_null_01: {
    id: "trace_null_01",
    name: "Clean Control Execution (Zero Faults)",
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
    isolatedCandidate: "(None — clean baseline execution)",
    initialFailure: {
      toolCall: "calculator(expression='150 + 250')",
      output: "Result: 400 (Clean successful execution)"
    },
    steps: [
      { step: 1, intervention: "BASELINE_CHECK", count: 4, candidates: ["elem_0", "elem_1", "elem_2", "elem_3"], outcome: "SUCCESS", note: "Baseline execution passed cleanly across all candidates. No failure condition to minimize." }
    ],
    evaluationsUsed: 1,
    greedyCalls: 1,
    randomCalls: 1,
    isOneMinimal: false,
    status: "NO_FAILURE",
    cliCard: `AgentXRay (trace_null_01)
────────────────────────────────────────────────────────────────────────────
DIAGNOSIS
  Baseline execution passed cleanly across all candidates. No failure condition to minimize.

ISOLATED CONDITION
  (None — clean control execution)

VERIFICATION
  Verified: baseline execution passed without any faults.
────────────────────────────────────────────────────────────────────────────`
  }
};

// ---------------------------------------------------------------------------
// 2. Terminal Demonstration Tape Controller
// ---------------------------------------------------------------------------

class TerminalTapeController {
  constructor() {
    this.currentScenarioKey = "currency_ast";
    this.currentView = "tape"; // 'tape', 'card', 'raw'
    this.isPlaying = false;
    this.stepIndex = 0;
    this.timer = null;
    this.speed = 1;

    this.screen = document.getElementById("terminalScreen");
    this.selector = document.getElementById("scenarioSelect");
    this.playBtn = document.getElementById("tapePlayBtn");
    this.stepBtn = document.getElementById("tapeStepBtn");
    this.resetBtn = document.getElementById("tapeResetBtn");
    this.speedSelect = document.getElementById("tapeSpeedSelect");
    this.titleTag = document.getElementById("terminalIdTag");

    this.bind();
    this.render();
  }

  bind() {
    if (this.selector) {
      this.selector.addEventListener("change", (e) => {
        this.currentScenarioKey = e.target.value;
        this.reset();
      });
    }

    if (this.playBtn) {
      this.playBtn.addEventListener("click", () => this.toggle());
    }

    if (this.stepBtn) {
      this.stepBtn.addEventListener("click", () => this.stepForward());
    }

    if (this.resetBtn) {
      this.resetBtn.addEventListener("click", () => this.reset());
    }

    if (this.speedSelect) {
      this.speedSelect.addEventListener("change", (e) => {
        this.speed = parseFloat(e.target.value);
      });
    }

    const modeTabs = document.querySelectorAll("[data-tape-mode]");
    modeTabs.forEach((tab) => {
      tab.addEventListener("click", (e) => {
        modeTabs.forEach((t) => t.classList.remove("active"));
        e.currentTarget.classList.add("active");
        this.currentView = e.currentTarget.getAttribute("data-tape-mode");
        this.render();
      });
    });
  }

  toggle() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    const sc = BENCHMARK_SCENARIOS[this.currentScenarioKey];
    if (this.stepIndex >= sc.steps.length) {
      this.stepIndex = 0;
    }
    this.isPlaying = true;
    if (this.playBtn) this.playBtn.textContent = "Pause";
    this.runLoop();
  }

  pause() {
    this.isPlaying = false;
    if (this.timer) clearTimeout(this.timer);
    if (this.playBtn) this.playBtn.textContent = "Play";
  }

  runLoop() {
    const sc = BENCHMARK_SCENARIOS[this.currentScenarioKey];
    if (!this.isPlaying) return;

    if (this.stepIndex < sc.steps.length) {
      this.stepIndex++;
      this.render();

      if (this.stepIndex >= sc.steps.length) {
        this.pause();
        return;
      }

      const delay = this.speed === 0 ? 30 : 750 / this.speed;
      this.timer = setTimeout(() => this.runLoop(), delay);
    } else {
      this.pause();
    }
  }

  stepForward() {
    this.pause();
    const sc = BENCHMARK_SCENARIOS[this.currentScenarioKey];
    if (this.stepIndex < sc.steps.length) {
      this.stepIndex++;
      this.render();
    }
  }

  reset() {
    this.pause();
    this.stepIndex = 0;
    this.render();
  }

  render() {
    const sc = BENCHMARK_SCENARIOS[this.currentScenarioKey];
    if (this.titleTag) {
      this.titleTag.textContent = `probe --scenario ${sc.id} --seed 42`;
    }

    if (!this.screen) return;

    if (this.currentView === "card") {
      this.screen.innerHTML = `<pre class="font-mono text-secondary" style="white-space: pre-wrap; font-size: 0.82rem;">${this.esc(sc.cliCard)}</pre>`;
      return;
    }

    if (this.currentView === "raw") {
      const payload = {
        status: sc.status,
        isolated_candidate: sc.isolatedCandidate,
        is_one_minimal: sc.isOneMinimal,
        evaluations_used: sc.evaluationsUsed,
        algorithm: "ddmin",
        seed: 42
      };
      this.screen.innerHTML = `<pre class="font-mono text-secondary" style="white-space: pre-wrap; color: #7ee787;">${this.esc(JSON.stringify(payload, null, 2))}</pre>`;
      return;
    }

    // Default: 'tape' view
    let out = `
      <div class="tape-line command">$ agentxray probe --scenario ${sc.id} --seed 42</div>
      <div class="tape-line meta">[INIT] Target:   ${sc.agent}</div>
      <div class="tape-line meta">[INIT] Query:    "${sc.query}"</div>
      <div class="tape-line meta">[INIT] Tools:    ${sc.tools.join(", ")}</div>
      <div class="tape-line meta" style="margin-bottom: 14px;">[INIT] Pool:     ${sc.candidates.length} candidate directives loaded</div>
    `;

    if (sc.id !== "trace_null_01") {
      out += `
        <div class="tape-line" style="color: var(--state-fail); margin-bottom: 14px;">
          [BASELINE FAILURE] Full input (4 candidates active):<br>
          Tool:   ${sc.initialFailure.toolCall}<br>
          Output: ${sc.initialFailure.output}
        </div>
      `;
    }

    const slice = this.stepIndex === 0 ? sc.steps : sc.steps.slice(0, this.stepIndex);

    slice.forEach((s, idx) => {
      const isLatest = idx === slice.length - 1;
      const isFail = s.outcome === "FAILURE";
      out += `
        <div class="tape-line trial ${isLatest ? 'active' : ''}">
          <span class="font-mono" style="color: var(--text-tertiary);">[TRIAL 0${s.step}] ${s.intervention}</span>
          (${s.count} items: [${s.candidates.join(", ")}]) → 
          <strong style="color: ${isFail ? 'var(--state-fail)' : 'var(--state-pass)'};">${s.outcome}</strong><br>
          <span style="color: var(--text-secondary); font-size: 0.8rem;">Action: ${s.note}</span>
        </div>
      `;
    });

    if (this.stepIndex === 0 || this.stepIndex >= sc.steps.length) {
      if (sc.id === "trace_null_01") {
        out += `
          <div class="tape-line diagnosis-result" style="border-left-color: var(--state-info);">
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">RESULT: CLEAN BASELINE CONTROL</div>
            <div style="color: var(--text-secondary);">Baseline execution passed cleanly across all candidates. Zero planted faults detected.</div>
            <div style="color: var(--text-tertiary); font-size: 0.78rem; margin-top: 6px;">Evaluations: 1 call | Attribution False Positives: 0%</div>
          </div>
        `;
      } else {
        out += `
          <div class="tape-line diagnosis-result">
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">RESULT: 1-MINIMAL CULPRIT ISOLATED</div>
            <div style="color: var(--text-secondary);">Isolated directive: <code>${this.esc(sc.isolatedCandidate)}</code></div>
            <div style="color: var(--text-tertiary); font-size: 0.78rem; margin-top: 6px;">
              Evaluations: ${sc.evaluationsUsed} calls (ddmin) vs ${sc.greedyCalls} calls (greedy) | Minimality: Verified
            </div>
          </div>
        `;
      }
    }

    this.screen.innerHTML = out;
  }

  esc(s) {
    if (!s) return "";
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
}

// ---------------------------------------------------------------------------
// 3. Client-Side Delta Debugging (ddmin) Engine for Playground
// ---------------------------------------------------------------------------

export function clientDdmin(candidates, pattern) {
  let callCount = 0;
  const trace = [];

  function evalSubset(sub) {
    callCount++;
    return sub.some((item) => item.toLowerCase().includes(pattern.toLowerCase()));
  }

  const baselineFails = evalSubset(candidates);
  trace.push({
    trial: 1,
    type: "BASELINE",
    count: candidates.length,
    outcome: baselineFails ? "FAILURE" : "SUCCESS",
    detail: baselineFails ? "Failure reproduced on full candidate pool." : "Clean execution (baseline passed)."
  });

  if (!baselineFails) {
    return {
      status: "NO_FAILURE",
      isolated: [],
      calls: callCount,
      trace,
      isVerified: false
    };
  }

  let c = [...candidates];
  let n = 2;

  while (c.length >= 2) {
    const subsets = [];
    const len = c.length;
    const step = Math.ceil(len / n);

    for (let i = 0; i < len; i += step) {
      subsets.push(c.slice(i, i + step));
    }

    let nextRound = false;

    for (const sub of subsets) {
      const fails = evalSubset(sub);
      trace.push({
        trial: callCount,
        type: "PARTITION_SUBSET",
        count: sub.length,
        outcome: fails ? "FAILURE" : "SUCCESS",
        detail: fails ? `Subset of size ${sub.length} reproduced failure.` : `Subset of size ${sub.length} passed.`
      });

      if (fails) {
        c = sub;
        n = 2;
        nextRound = true;
        break;
      }
    }

    if (nextRound) continue;

    for (const sub of subsets) {
      const comp = c.filter((x) => !sub.includes(x));
      if (comp.length === 0) continue;

      const fails = evalSubset(comp);
      trace.push({
        trial: callCount,
        type: "PARTITION_COMPLEMENT",
        count: comp.length,
        outcome: fails ? "FAILURE" : "SUCCESS",
        detail: fails ? `Complement of size ${comp.length} reproduced failure.` : `Complement of size ${comp.length} passed.`
      });

      if (fails) {
        c = comp;
        n = Math.max(n - 1, 2);
        nextRound = true;
        break;
      }
    }

    if (nextRound) continue;

    if (n < c.length) {
      n = Math.min(c.length, 2 * n);
    } else {
      break;
    }
  }

  let isVerified = false;
  if (c.length === 1) {
    const emptyFails = evalSubset([]);
    trace.push({
      trial: callCount,
      type: "MINIMALITY_CHECK",
      count: 0,
      outcome: emptyFails ? "FAILURE" : "SUCCESS",
      detail: !emptyFails ? `Empty set passed: removing '${c[0]}' restores success.` : "Empty set unexpectedly failed."
    });
    isVerified = !emptyFails;
  }

  return {
    status: isVerified ? "VERIFIED_CULPRIT" : "INCONCLUSIVE",
    isolated: c,
    calls: callCount,
    trace,
    isVerified
  };
}

function initPlaygroundRunner() {
  const runBtn = document.getElementById("pgRunBtn");
  const textarea = document.getElementById("pgCandidates");
  const patternInput = document.getElementById("pgPattern");
  const outputArea = document.getElementById("pgOutput");

  if (!runBtn || !textarea || !patternInput || !outputArea) return;

  runBtn.addEventListener("click", () => {
    const lines = textarea.value.split("\n").map((s) => s.trim()).filter((s) => s.length > 0);
    const pattern = patternInput.value.trim();

    if (lines.length === 0) {
      outputArea.innerHTML = `<span style="color: var(--state-fail);">Error: Candidate pool cannot be empty.</span>`;
      return;
    }
    if (!pattern) {
      outputArea.innerHTML = `<span style="color: var(--state-fail);">Error: Trigger pattern cannot be blank.</span>`;
      return;
    }

    outputArea.innerHTML = `<span style="color: var(--text-tertiary);">Executing Delta Debugging partition search...</span>`;

    setTimeout(() => {
      const res = clientDdmin(lines, pattern);

      let log = `<div style="font-weight: 600; margin-bottom: 8px;">ddmin Execution Trace (${res.calls} calls):</div>`;

      res.trace.forEach((t) => {
        const isF = t.outcome === "FAILURE";
        log += `
          <div style="margin-bottom: 6px; padding-left: 8px; border-left: 2px solid ${isF ? 'var(--state-fail)' : 'var(--state-pass)'};">
            <span style="color: var(--text-tertiary);">[Trial ${t.trial}] ${t.type} (${t.count} items)</span> → 
            <strong style="color: ${isF ? 'var(--state-fail)' : 'var(--state-pass)'};">${t.outcome}</strong><br>
            <span style="color: var(--text-secondary); font-size: 0.8rem;">${t.detail}</span>
          </div>
        `;
      });

      if (res.status === "VERIFIED_CULPRIT") {
        log += `
          <div style="margin-top: 14px; padding: 12px; background: var(--surface-2); border: 1px solid var(--rule-subtle); border-left: 3px solid var(--state-pass);">
            <strong style="color: var(--state-pass);">ISOLATED &amp; VERIFIED 1-MINIMAL TRIGGER:</strong><br>
            <span style="color: var(--text-primary); font-size: 0.88rem;">"${res.isolated[0]}"</span><br>
            <span style="color: var(--text-tertiary); font-size: 0.76rem;">Evaluations: ${res.calls} calls | 1-Minimal: True</span>
          </div>
        `;
      } else if (res.status === "NO_FAILURE") {
        log += `
          <div style="margin-top: 14px; padding: 12px; background: var(--surface-2); border: 1px solid var(--rule-subtle);">
            <strong>NO FAILURE DETECTED:</strong><br>
            <span style="color: var(--text-secondary); font-size: 0.82rem;">None of the candidates matched trigger pattern "${pattern}".</span>
          </div>
        `;
      } else {
        log += `
          <div style="margin-top: 14px; padding: 12px; background: var(--surface-2); border: 1px solid var(--rule-subtle);">
            <strong style="color: var(--state-warn);">INCONCLUSIVE SEARCH:</strong><br>
            <span style="color: var(--text-secondary); font-size: 0.82rem;">Candidate subset isolated without 1-minimality confirmation.</span>
          </div>
        `;
      }

      outputArea.innerHTML = log;
    }, 60);
  });
}

// ---------------------------------------------------------------------------
// 4. Code Snippet Copy Handler
// ---------------------------------------------------------------------------

function initCopyActions() {
  const buttons = document.querySelectorAll("[data-copy-target]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = btn.getAttribute("data-copy-target");
      if (!text) return;

      try {
        await navigator.clipboard.writeText(text);
        btn.classList.add("is-copied");
        const originalText = btn.innerHTML;
        btn.innerHTML = `<span>Copied</span>`;
        setTimeout(() => {
          btn.classList.remove("is-copied");
          btn.innerHTML = originalText;
        }, 1800);
      } catch (err) {
        console.error("Clipboard copy failed", err);
      }
    });
  });
}

// ---------------------------------------------------------------------------
// 5. Scroll Reveals via IntersectionObserver
// ---------------------------------------------------------------------------

function initScrollObserver() {
  const elements = document.querySelectorAll(".reveal-on-scroll");
  if (!elements.length) return;

  if (!("IntersectionObserver" in window)) {
    elements.forEach((el) => el.classList.add("is-revealed"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-revealed");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  elements.forEach((el) => observer.observe(el));
}

// ---------------------------------------------------------------------------
// 6. Mobile Navigation Drawer
// ---------------------------------------------------------------------------

function initMobileMenu() {
  const toggle = document.querySelector(".menu-toggle");
  const drawer = document.querySelector(".mobile-menu-drawer");

  if (!toggle || !drawer) return;

  toggle.addEventListener("click", () => {
    const isOpen = drawer.classList.contains("is-open");
    drawer.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", !isOpen);
  });
}

// ---------------------------------------------------------------------------
// 7. Active Navigation State Detection
// ---------------------------------------------------------------------------

function initActiveNavState() {
  const path = window.location.pathname;
  const page = path.split("/").pop() || "index.html";

  const links = document.querySelectorAll(".site-nav .nav-item, .mobile-menu-drawer .nav-item");
  links.forEach((link) => {
    const href = link.getAttribute("href");
    if (!href) return;
    const targetPage = href.split("#")[0].split("/").pop() || "index.html";

    if (page === targetPage || (page === "" && targetPage === "index.html")) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });
}

// ---------------------------------------------------------------------------
// Document Bootstrap
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initActiveNavState();
  initCopyActions();
  initScrollObserver();
  initMobileMenu();

  if (document.getElementById("terminalScreen")) {
    new TerminalTapeController();
  }

  if (document.getElementById("pgRunBtn")) {
    initPlaygroundRunner();
  }
});
