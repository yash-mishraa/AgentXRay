"""Tests for lexical-overlap audit utility (Phase 4.6J).

Verifies:
1. Tokenization and normalization rules (lowercasing, punctuation stripping, stop-word filtering).
2. Metric calculations: Jaccard overlap and Longest Common Substring (LCS).
3. Boundary cases: empty strings, disjoint text, exact matches.
4. Complete mechanism coverage across all 18 registered benchmark tasks.
5. Warning threshold triggering and descriptive reporting.
"""

from __future__ import annotations

import pytest

from benchmarks.multi_mechanism.leakage_audit import (
    DEFAULT_STOP_WORDS,
    audit_all_mechanisms,
    audit_mechanism_leakage,
    compute_jaccard_overlap,
    compute_longest_common_substring,
    format_leakage_table,
    tokenize_normalized,
)
from benchmarks.multi_mechanism.tasks import MECHANISM_REGISTRY, get_mechanism_task


class TestLexicalNormalization:
    """Verifies tokenization and normalization rules."""

    def test_tokenize_normalized_lowercasing_and_punctuation(self) -> None:
        raw_text = "ToolExecutionError: SyntaxError in calculator: invalid character '$' in numeric expression."
        tokens = tokenize_normalized(raw_text, filter_stop_words=True)
        # Verify lowercase
        assert all(t == t.lower() for t in tokens)
        # Verify content words retained
        assert "toolexecutionerror" in tokens
        assert "syntaxerror" in tokens
        assert "calculator" in tokens
        assert "invalid" in tokens
        assert "character" in tokens
        assert "numeric" in tokens
        assert "expression" in tokens
        # Stop word 'in' filtered out
        assert "in" not in tokens

    def test_tokenize_without_stop_words_filter(self) -> None:
        raw_text = "in the calculator"
        tokens_filtered = tokenize_normalized(raw_text, filter_stop_words=True)
        assert tokens_filtered == {"calculator"}

        tokens_unfiltered = tokenize_normalized(raw_text, filter_stop_words=False)
        assert tokens_unfiltered == {"in", "the", "calculator"}

    def test_tokenize_empty_and_whitespace_string(self) -> None:
        assert tokenize_normalized("") == set()
        assert tokenize_normalized("   \t\n  ") == set()
        assert tokenize_normalized("!@#$%^&*()") == set()


class TestOverlapMetrics:
    """Verifies Jaccard similarity and LCS computations."""

    def test_jaccard_exact_match(self) -> None:
        toks = {"calculator", "syntax", "error"}
        assert compute_jaccard_overlap(toks, toks) == 1.0

    def test_jaccard_disjoint(self) -> None:
        toks_a = {"alpha", "beta"}
        toks_b = {"gamma", "delta"}
        assert compute_jaccard_overlap(toks_a, toks_b) == 0.0

    def test_jaccard_partial(self) -> None:
        # A = {1, 2}, B = {2, 3} -> intersection=1, union=3 -> 1/3 = 0.3333
        toks_a = {"one", "two"}
        toks_b = {"two", "three"}
        assert compute_jaccard_overlap(toks_a, toks_b) == pytest.approx(0.3333, abs=1e-4)

    def test_jaccard_empty_sets(self) -> None:
        assert compute_jaccard_overlap(set(), {"test"}) == 0.0
        assert compute_jaccard_overlap({"test"}, set()) == 0.0
        assert compute_jaccard_overlap(set(), set()) == 0.0

    def test_lcs_computation(self) -> None:
        s1 = "Tool 'calculator' parameter 'expression' expected string, got array."
        s2 = "schema_param_type_spec: Tool parameter expression declared as array of strings instead of string."
        lcs_len, lcs_str = compute_longest_common_substring(s1, s2)
        assert lcs_len == 11
        assert lcs_str == " parameter "

    def test_lcs_disjoint_and_empty(self) -> None:
        assert compute_longest_common_substring("", "abc") == (0, "")
        assert compute_longest_common_substring("abc", "") == (0, "")
        assert compute_longest_common_substring("abc", "xyz") == (0, "")


class TestBenchmarkLeakageCoverage:
    """Verifies complete coverage across all 18 benchmark mechanisms."""

    def test_all_18_mechanisms_audited(self) -> None:
        summary = audit_all_mechanisms(warning_threshold_jaccard=0.20, warning_threshold_lcs=15)
        assert summary.total_mechanisms == 18
        assert len(summary.reports) == 18

        # Ensure all registered mechanism IDs are represented
        audited_ids = {r.mechanism_id for r in summary.reports}
        assert audited_ids == set(MECHANISM_REGISTRY.keys())

    def test_null_mechanisms_have_zero_trigger_overlap(self) -> None:
        summary = audit_all_mechanisms()
        null_reports = [r for r in summary.reports if r.is_null_case]
        assert len(null_reports) == 3

        for nr in null_reports:
            assert nr.trigger_jaccard_overlap == 0.0
            assert nr.trigger_lcs_length == 0
            assert nr.is_flagged is False

    def test_flagged_mechanisms_meet_threshold_criteria(self) -> None:
        summary = audit_all_mechanisms(warning_threshold_jaccard=0.20, warning_threshold_lcs=15)
        for r in summary.reports:
            if r.is_flagged:
                assert (
                    r.trigger_jaccard_overlap >= 0.20
                    or r.trigger_lcs_length >= 15
                ), f"Mechanism {r.mechanism_id} flagged without meeting threshold!"
                assert r.warning_message is not None
            else:
                assert r.warning_message is None

    def test_format_leakage_table_output(self) -> None:
        summary = audit_all_mechanisms(warning_threshold_jaccard=0.20, warning_threshold_lcs=15)
        table_str = format_leakage_table(summary)
        assert "PHASE 4.6J LEXICAL OVERLAP AUDIT" in table_str
        assert "mech_p05_instruction_override" in table_str
        assert "mech_s01_type_mismatch" in table_str
        assert "mech_null_clean" in table_str
        assert "Summary: 6/18 mechanisms flagged" in table_str
