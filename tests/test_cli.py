"""Unit and integration tests for the AgentXRay Developer CLI (Phase 4.6K.20C).

Verifies entry points, subcommands, exit code contracts, output formatting,
schema validation, --no-color compliance, and SDK integration.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import pytest

from agentxray import __version__
from agentxray.__main__ import main
from agentxray.api import DiagnosticResult, DiagnosticStatus
from agentxray.cli import (
    BUILTIN_SCENARIOS,
    ScenarioFileSpec,
    cli_main,
    format_terminal_card,
    get_version,
    load_scenario_file,
    resolve_exit_code,
)


ANSI_REGEX = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class TestCliBasics:
    """Tests for basic CLI metadata, versioning, and help formatting."""

    def test_version_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray version' reports package version dynamically and exits 0."""
        code = cli_main(["version"])
        assert code == 0
        out = capsys.readouterr().out
        assert f"agentxray {__version__}" in out

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray --version' reports version and raises SystemExit(0)."""
        with pytest.raises(SystemExit) as exc:
            cli_main(["--version"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert f"agentxray {__version__}" in out

    def test_help_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray --help' prints top-level usage with subcommands and exits 0."""
        with pytest.raises(SystemExit) as exc:
            cli_main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "subcommands:" in out
        assert "demo" in out
        assert "diagnose" in out
        assert "version" in out

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray' without args prints help and returns 0."""
        code = cli_main([])
        assert code == 0
        out = capsys.readouterr().out
        assert "subcommands:" in out

    def test_diagnose_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray diagnose --help' prints diagnose options and exits 0."""
        with pytest.raises(SystemExit) as exc:
            cli_main(["diagnose", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--scenario" in out
        assert "--max-evaluations" in out
        assert "--no-color" in out
        assert "--json" in out


class TestCliDemoCommand:
    """Tests for the offline deterministic demo command."""

    def test_demo_execution_and_sdk_integration(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray demo' executes offline, verifies 1-minimal trigger, and exits 0."""
        code = cli_main(["demo"])
        assert code == 0
        out = capsys.readouterr().out

        assert "AgentXRay: Deterministic Offline Demonstration" in out
        assert "[1/5] Scenario Setup & Observable Failure" in out
        assert "[3/5] Isolated Minimal Failure-Inducing Condition" in out
        assert "strict_currency_formatting" in out
        assert "4 evaluations" in out or "Target Calls Used:  4" in out
        assert "$0.00 cost" in out

    def test_demo_no_color(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray demo --no-color' contains no ANSI escape codes."""
        code = cli_main(["demo", "--no-color"])
        assert code == 0
        out = capsys.readouterr().out
        assert not ANSI_REGEX.search(out), "Found ANSI escape codes in --no-color demo output"

    def test_demo_save_reports(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray demo --save --output-dir <tmp>' exports reports."""
        code = cli_main(["demo", "--save", "--output-dir", str(tmp_path)])
        assert code == 0
        assert (tmp_path / "demo_report.json").exists()
        assert (tmp_path / "demo_report.md").exists()


class TestCliDiagnoseCommand:
    """Tests for 'agentxray diagnose' with built-in scenarios and custom JSON files."""

    def test_diagnose_requires_input(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray diagnose' without file or --scenario exits with code 2."""
        code = cli_main(["diagnose"])
        assert code == 2
        err = capsys.readouterr().err
        assert "requires either a scenario file path or --scenario" in err

    def test_diagnose_rejects_both_file_and_scenario(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray diagnose' with both file and --scenario exits with code 2."""
        dummy_file = tmp_path / "scenario.json"
        dummy_file.write_text("{}", encoding="utf-8")
        code = cli_main(["diagnose", str(dummy_file), "--scenario", "trace_env_01"])
        assert code == 2
        err = capsys.readouterr().err
        assert "Specify either a scenario file OR --scenario <id>, not both" in err

    def test_diagnose_builtin_currency_ast(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray diagnose --scenario currency_ast' isolates 1-minimal culprit."""
        code = cli_main(["diagnose", "--scenario", "currency_ast"])
        assert code == 0
        out = capsys.readouterr().out
        assert "AgentXRay (built-in: currency_ast)" in out
        assert "CULPRIT ISOLATED & VERIFIED" in out
        assert "strict_currency_formatting" in out
        assert "Verified: removing this directive restores clean success (1-minimal)" in out

    def test_diagnose_builtin_trace_env_01(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray diagnose --scenario trace_env_01' isolates backend outage trigger."""
        code = cli_main(["diagnose", "--scenario", "trace_env_01"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Service availability fault: Backend calculation cluster experiences unexpected node-level outage" in out
        assert "CULPRIT ISOLATED & VERIFIED" in out

    def test_diagnose_builtin_trace_null_01(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray diagnose --scenario trace_null_01' cleanly passes and reports NO FAILURE."""
        code = cli_main(["diagnose", "--scenario", "trace_null_01"])
        assert code == 0
        out = capsys.readouterr().out
        assert "NO FAILURE DETECTED" in out
        assert "(None — clean control execution)" in out
        assert "1 calls" in out

    def test_diagnose_json_file_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray diagnose examples/scenario_example.json' isolates trigger and exits 0."""
        code = cli_main(["diagnose", "examples/scenario_example.json"])
        assert code == 0
        out = capsys.readouterr().out
        assert "currency_formatting_directive_failure" in out
        assert "strict_currency_formatting" in out
        assert "CULPRIT ISOLATED & VERIFIED" in out

    def test_diagnose_json_file_no_color(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray diagnose <file> --no-color' strips all ANSI escape codes."""
        code = cli_main(["diagnose", "examples/scenario_example.json", "--no-color"])
        assert code == 0
        out = capsys.readouterr().out
        assert not ANSI_REGEX.search(out), "Found ANSI escape codes in --no-color output"

    def test_diagnose_raw_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray diagnose <file> --json' outputs valid machine-readable JSON."""
        code = cli_main(["diagnose", "examples/scenario_example.json", "--json"])
        assert code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["status"] == "VERIFIED_CULPRIT"
        assert data["is_verified"] is True
        assert len(data["isolated_candidates"]) == 1
        assert "strict_currency_formatting" in data["isolated_candidates"][0]
        assert data["evaluations_used"] == 4

    def test_diagnose_save_flag(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """'agentxray diagnose <file> --save --output-dir <tmp>' writes diagnosis_result.json."""
        code = cli_main([
            "diagnose",
            "examples/scenario_example.json",
            "--save",
            "--output-dir",
            str(tmp_path),
        ])
        assert code == 0
        saved_file = tmp_path / "diagnosis_result.json"
        assert saved_file.exists()
        data = json.loads(saved_file.read_text(encoding="utf-8"))
        assert data["status"] == "VERIFIED_CULPRIT"


class TestCliInputValidationAndExitCodes:
    """Tests for file validation errors (exit code 2) and status mappings (0, 1, 2, 3)."""

    def test_missing_file_exit_code_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Non-existent scenario file returns exit code 2."""
        code = cli_main(["diagnose", "non_existent_file.json"])
        assert code == 2
        err = capsys.readouterr().err
        assert "Scenario file not found" in err

    def test_malformed_json_exit_code_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Invalid JSON syntax returns exit code 2."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ unquoted: key, }", encoding="utf-8")
        code = cli_main(["diagnose", str(bad_json)])
        assert code == 2
        err = capsys.readouterr().err
        assert "Invalid JSON in scenario file" in err

    def test_empty_candidates_exit_code_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Empty candidate pool returns exit code 2."""
        empty_cands = tmp_path / "empty_cands.json"
        empty_cands.write_text(
            json.dumps({"name": "empty", "candidates": [], "failure_condition": {"type": "always_fail"}}),
            encoding="utf-8",
        )
        code = cli_main(["diagnose", str(empty_cands)])
        assert code == 2
        err = capsys.readouterr().err
        assert "Candidate pool cannot be empty" in err

    def test_duplicate_candidates_exit_code_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Duplicate candidates return exit code 2."""
        dup_file = tmp_path / "dup.json"
        dup_file.write_text(
            json.dumps({
                "name": "dup",
                "candidates": ["directive_a", "directive_a"],
                "failure_condition": {"type": "always_fail"},
            }),
            encoding="utf-8",
        )
        code = cli_main(["diagnose", str(dup_file)])
        assert code == 2
        err = capsys.readouterr().err
        assert "Duplicate candidate element found" in err

    def test_invalid_ground_truth_exit_code_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Ground truth element missing from candidates returns exit code 2."""
        gt_file = tmp_path / "gt.json"
        gt_file.write_text(
            json.dumps({
                "name": "gt",
                "candidates": ["candidate_1", "candidate_2"],
                "failure_condition": {"type": "contains_any", "patterns": ["candidate_1"]},
                "ground_truth": ["non_existent_trigger"],
            }),
            encoding="utf-8",
        )
        code = cli_main(["diagnose", str(gt_file)])
        assert code == 2
        err = capsys.readouterr().err
        assert "Ground truth element 'non_existent_trigger' is not present in the candidate pool" in err

    def test_resolve_exit_code_contract(self) -> None:
        """Verify the exact exit code mapping contract."""
        dummy_base = {
            "summary": "test",
            "is_one_minimal": False,
            "initial_candidates_count": 2,
            "evaluations_used": 1,
            "execution_time_seconds": 0.01,
        }

        # Code 0: Verified culprit or clean pass
        res_culprit = DiagnosticResult(status=DiagnosticStatus.VERIFIED_CULPRIT, is_verified=True, **dummy_base)
        res_no_fail = DiagnosticResult(status=DiagnosticStatus.NO_FAILURE, is_verified=False, **dummy_base)
        assert resolve_exit_code(res_culprit) == 0
        assert resolve_exit_code(res_no_fail) == 0

        # Code 1: Inconclusive, budget exhausted, or timeout
        res_inconcl = DiagnosticResult(status=DiagnosticStatus.INCONCLUSIVE, is_verified=False, **dummy_base)
        res_budget = DiagnosticResult(status=DiagnosticStatus.BUDGET_EXHAUSTED, is_verified=False, **dummy_base)
        res_timeout = DiagnosticResult(status=DiagnosticStatus.TIMEOUT, is_verified=False, **dummy_base)
        assert resolve_exit_code(res_inconcl) == 1
        assert resolve_exit_code(res_budget) == 1
        assert resolve_exit_code(res_timeout) == 1

        # Code 3: Evaluator error
        res_error = DiagnosticResult(status=DiagnosticStatus.ERROR, is_verified=False, **dummy_base)
        assert resolve_exit_code(res_error) == 3


class TestCliMainWrapper:
    """Tests for the main() function in __main__.py invoking cli_main()."""

    def test_main_runs_demo_via_argv(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """main() executes demo when argv=['agentxray', 'demo']."""
        monkeypatch.setattr("sys.argv", ["agentxray", "demo"])
        main()
        out = capsys.readouterr().out
        assert "AgentXRay: Deterministic Offline Demonstration" in out

    def test_main_exits_with_code_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() calls sys.exit(2) when diagnose receives an invalid file."""
        monkeypatch.setattr("sys.argv", ["agentxray", "diagnose", "non_existent.json"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
