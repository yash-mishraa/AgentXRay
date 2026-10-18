"""Lexical-overlap audit of observable failure symptoms and candidate descriptions.

Phase 4.6J offline research utility:
Measures token-normalized Jaccard overlap and Longest Common Substrings (LCS)
between observable mechanism failure symptoms and candidate element descriptions
presented to passive LLM judges.

CRITICAL EPISTEMIC INTEGRITY GUARANTEES:
1. Descriptive Flags Only: Lexical overlap metrics serve as heuristic sensitivity flags.
   High overlap indicates potential surface-level keyword leakage that a passive model
   might exploit, but does NOT mathematically prove or disprove genuine causal reasoning.
2. Read-Only Reflection: Audits the exact definitions from `benchmarks.multi_mechanism.tasks`.
   Does NOT alter benchmark prompts, candidate definitions, mechanism specifications, or scoring.
3. Complete Coverage: Audits all 18 registered benchmark mechanisms (active, conjunctive, and null).
"""

from __future__ import annotations

import argparse
import re
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.multi_mechanism.tasks import MechanismTaskSpec, list_mechanism_tasks

# Explicit default stop words excluded during tokenization to prevent non-content inflation
DEFAULT_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with",
})


def tokenize_normalized(text: str, filter_stop_words: bool = True) -> set[str]:
    """Tokenize and normalize text according to explicit lexical rules.

    Rules:
    1. Lowercase entire text.
    2. Extract contiguous alphanumeric and underscore character sequences: `[a-z0-9_]+`.
    3. Filter common English stop words if `filter_stop_words` is True.
    4. Return unique token set.
    """
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if filter_stop_words:
        return set(t for t in tokens if t not in DEFAULT_STOP_WORDS)
    return set(tokens)


def compute_jaccard_overlap(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute standard Jaccard similarity coefficient |A n B| / |A u B|."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return round(intersection / union, 4)


def compute_longest_common_substring(s1: str, s2: str, case_sensitive: bool = False) -> tuple[int, str]:
    """Compute longest common contiguous substring between two strings.

    Returns
    -------
    tuple[int, str]
        (length_of_lcs, longest_substring)
    """
    if not s1 or not s2:
        return (0, "")

    text1 = s1 if case_sensitive else s1.lower()
    text2 = s2 if case_sensitive else s2.lower()

    m, n = len(text1), len(text2)
    max_len = 0
    end_idx = 0

    # Dynamic programming table for contiguous substring matching
    prev_row = [0] * (n + 1)
    for i in range(1, m + 1):
        curr_row = [0] * (n + 1)
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                curr_row[j] = prev_row[j - 1] + 1
                if curr_row[j] > max_len:
                    max_len = curr_row[j]
                    end_idx = i
            else:
                curr_row[j] = 0
        prev_row = curr_row

    if max_len == 0:
        return (0, "")

    # Extract substring using original casing if case-sensitive, or lowercased text
    lcs_substr = text1[end_idx - max_len:end_idx]
    return (max_len, lcs_substr)


class MechanismLeakageReport(BaseModel):
    """Detailed lexical overlap audit report for a single benchmark mechanism."""

    model_config = ConfigDict(frozen=True)

    mechanism_id: str = Field(..., description="Unique benchmark mechanism identifier.")
    category: str = Field(..., description="Mechanism category (prompt, schema, environment, conjunctive, null).")
    is_null_case: bool = Field(..., description="Whether this is a negative control null task.")
    is_conjunctive: bool = Field(..., description="Whether this is a multi-trigger conjunctive task.")
    observable_failure_symptom: str = Field(..., description="Observable error message emitted into trace upon failure.")
    candidate_descriptions: tuple[str, ...] = Field(..., description="All candidate element descriptions presented to judge.")
    planted_triggers: tuple[str, ...] = Field(..., description="Planted ground-truth trigger elements.")
    max_jaccard_overlap: float = Field(..., description="Maximum Jaccard overlap between symptom and any candidate element.")
    trigger_jaccard_overlap: float = Field(..., description="Maximum Jaccard overlap between symptom and any true trigger element.")
    max_distractor_jaccard_overlap: float = Field(..., description="Maximum Jaccard overlap between symptom and any distractor element.")
    max_lcs_length: int = Field(..., description="Length of longest common substring between symptom and any candidate.")
    max_lcs_text: str = Field(..., description="Longest common substring between symptom and any candidate.")
    trigger_lcs_length: int = Field(..., description="Length of longest common substring between symptom and any true trigger.")
    trigger_lcs_text: str = Field(..., description="Longest common substring between symptom and any true trigger.")
    is_flagged: bool = Field(..., description="Whether overlap exceeds warning thresholds.")
    warning_message: str | None = Field(default=None, description="Detailed explanation if flagged.")


class BenchmarkLeakageSummary(BaseModel):
    """Aggregated lexical overlap summary across the complete benchmark suite."""

    model_config = ConfigDict(frozen=True)

    total_mechanisms: int = Field(..., description="Total mechanisms audited.")
    flagged_mechanisms: int = Field(..., description="Number of mechanisms exceeding warning thresholds.")
    warning_threshold_jaccard: float = Field(..., description="Configured Jaccard warning threshold.")
    warning_threshold_lcs: int = Field(..., description="Configured LCS length warning threshold.")
    mean_max_jaccard: float = Field(..., description="Mean of maximum candidate Jaccard overlap across active tasks.")
    mean_trigger_jaccard: float = Field(..., description="Mean of trigger Jaccard overlap across active tasks.")
    reports: tuple[MechanismLeakageReport, ...] = Field(..., description="Per-mechanism audit reports.")
    summary_narrative: str = Field(..., description="Interpretive narrative of audit results.")


def audit_mechanism_leakage(
    spec: MechanismTaskSpec,
    warning_threshold_jaccard: float = 0.20,
    warning_threshold_lcs: int = 15,
) -> MechanismLeakageReport:
    """Audit lexical overlap between failure symptom and candidates for a single mechanism."""
    symptom = spec.observable_failure_symptom
    sym_tokens = tokenize_normalized(symptom)

    max_j = 0.0
    trig_j = 0.0
    dist_j = 0.0

    max_lcs_len = 0
    max_lcs_txt = ""
    trig_lcs_len = 0
    trig_lcs_txt = ""

    trigger_set = set(spec.planted_trigger_elements)

    for cand in spec.candidate_elements:
        cand_tokens = tokenize_normalized(cand)
        j = compute_jaccard_overlap(sym_tokens, cand_tokens)
        l_len, l_txt = compute_longest_common_substring(symptom, cand)

        if j > max_j:
            max_j = j
        if l_len > max_lcs_len:
            max_lcs_len = l_len
            max_lcs_txt = l_txt

        if cand in trigger_set:
            if j > trig_j:
                trig_j = j
            if l_len > trig_lcs_len:
                trig_lcs_len = l_len
                trig_lcs_txt = l_txt
        else:
            if j > dist_j:
                dist_j = j

    # Check warning criteria
    is_flagged = False
    reasons: list[str] = []

    # Null clean and null bloat have no failure symptom (symptom == "")
    if symptom:
        if trig_j >= warning_threshold_jaccard:
            is_flagged = True
            reasons.append(f"Trigger Jaccard overlap ({trig_j:.3f}) >= {warning_threshold_jaccard:.2f}")
        if trig_lcs_len >= warning_threshold_lcs:
            is_flagged = True
            reasons.append(f"Trigger LCS length ({trig_lcs_len} chars: '{trig_lcs_txt}') >= {warning_threshold_lcs}")

    warning_msg = "; ".join(reasons) if reasons else None

    return MechanismLeakageReport(
        mechanism_id=spec.mechanism_id,
        category=spec.category,
        is_null_case=spec.is_null_case,
        is_conjunctive=spec.is_conjunctive,
        observable_failure_symptom=symptom,
        candidate_descriptions=spec.candidate_elements,
        planted_triggers=spec.planted_trigger_elements,
        max_jaccard_overlap=max_j,
        trigger_jaccard_overlap=trig_j,
        max_distractor_jaccard_overlap=dist_j,
        max_lcs_length=max_lcs_len,
        max_lcs_text=max_lcs_txt,
        trigger_lcs_length=trig_lcs_len,
        trigger_lcs_text=trig_lcs_txt,
        is_flagged=is_flagged,
        warning_message=warning_msg,
    )


def audit_all_mechanisms(
    mechanisms: Sequence[MechanismTaskSpec] | None = None,
    warning_threshold_jaccard: float = 0.20,
    warning_threshold_lcs: int = 15,
) -> BenchmarkLeakageSummary:
    """Audit all benchmark mechanisms and produce an aggregated summary report."""
    specs = list(mechanisms) if mechanisms is not None else list_mechanism_tasks()
    reports = [
        audit_mechanism_leakage(
            spec=spec,
            warning_threshold_jaccard=warning_threshold_jaccard,
            warning_threshold_lcs=warning_threshold_lcs,
        )
        for spec in specs
    ]

    flagged_count = sum(1 for r in reports if r.is_flagged)
    active_reports = [r for r in reports if not r.is_null_case]

    mean_max_j = (
        sum(r.max_jaccard_overlap for r in active_reports) / max(1, len(active_reports))
        if active_reports
        else 0.0
    )
    mean_trig_j = (
        sum(r.trigger_jaccard_overlap for r in active_reports) / max(1, len(active_reports))
        if active_reports
        else 0.0
    )

    narrative = (
        f"Audited {len(reports)} benchmark mechanisms ({len(active_reports)} active, "
        f"{len(reports) - len(active_reports)} null). "
        f"Detected {flagged_count}/{len(reports)} mechanisms exceeding warning thresholds "
        f"(Jaccard >= {warning_threshold_jaccard:.2f} or LCS >= {warning_threshold_lcs} chars). "
        f"Across active failure tasks, mean trigger Jaccard overlap was {mean_trig_j:.3f} "
        f"(mean max candidate overlap: {mean_max_j:.3f})."
    )

    return BenchmarkLeakageSummary(
        total_mechanisms=len(reports),
        flagged_mechanisms=flagged_count,
        warning_threshold_jaccard=warning_threshold_jaccard,
        warning_threshold_lcs=warning_threshold_lcs,
        mean_max_jaccard=round(mean_max_j, 4),
        mean_trigger_jaccard=round(mean_trig_j, 4),
        reports=tuple(reports),
        summary_narrative=narrative,
    )


def format_leakage_table(summary: BenchmarkLeakageSummary) -> str:
    """Format benchmark leakage summary into an ASCII report table."""
    lines: list[str] = [
        "=" * 105,
        "PHASE 4.6J LEXICAL OVERLAP AUDIT: OBSERVABLE SYMPTOMS VS. CANDIDATE DESCRIPTIONS",
        "=" * 105,
        f"{'Mechanism ID':<38} | {'Cat':<9} | {'Trig J':<6} | {'Max J':<6} | {'Trig LCS':<8} | {'Flagged':<7} | {'LCS Snippet'}",
        "-" * 105,
    ]

    for r in summary.reports:
        flag_str = "YES (!)" if r.is_flagged else "NO"
        snippet = f"'{r.trigger_lcs_text[:22]}'" if r.trigger_lcs_text else "(none)"
        lines.append(
            f"{r.mechanism_id:<38} | {r.category:<9} | {r.trigger_jaccard_overlap:6.3f} | "
            f"{r.max_jaccard_overlap:6.3f} | {r.trigger_lcs_length:8d} | {flag_str:<7} | {snippet}"
        )

    lines.extend([
        "-" * 105,
        f"Summary: {summary.flagged_mechanisms}/{summary.total_mechanisms} mechanisms flagged. "
        f"Mean trigger Jaccard (active): {summary.mean_trigger_jaccard:.3f}. "
        f"Thresholds: Jaccard >= {summary.warning_threshold_jaccard:.2f}, LCS >= {summary.warning_threshold_lcs} chars.",
        "=" * 105,
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit lexical overlap across Phase 4.6 mechanisms.")
    parser.add_argument("--jaccard-threshold", type=float, default=0.20, help="Jaccard warning threshold.")
    parser.add_argument("--lcs-threshold", type=int, default=15, help="LCS character length warning threshold.")
    args = parser.parse_args()

    summary = audit_all_mechanisms(
        warning_threshold_jaccard=args.jaccard_threshold,
        warning_threshold_lcs=args.lcs_threshold,
    )
    print(format_leakage_table(summary))
