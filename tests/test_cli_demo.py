"""Tests for the offline deterministic demonstration command (agentxray demo)."""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from agentxray.demo import format_demo_output, run_demo, write_demo_reports
from agentxray.__main__ import main


class TestCliDemo:
    """Tests for agentxray demo command, execution speed, output formatting, and report output."""

    def test_run_demo_result_structure(self) -> None:
        """Verify that run_demo executes deterministically and returns complete data."""
        t0 = time.perf_counter()
        res = run_demo(seed=42)
        elapsed = time.perf_counter() - t0

        # Should execute in under 1.0 second offline
        assert elapsed < 1.0

        # Delta Debugging results
        cond = res.discovered_condition
        assert len(cond.subset_elements) == 1
        assert "strict_currency_formatting" in cond.subset_elements[0]
        assert cond.is_one_minimal is True
        assert cond.target_calls_used == 4
        assert res.dd_matches_ground_truth is True

        # Dynamic telemetry derivation
        assert res.initial_failure_tool_call == "calculator(expression='$150 + $250')"
        assert "invalid syntax" in res.initial_failure_output.lower()

        # Trajectory steps (including explicit empty-set 1-minimality check)
        assert len(res.dd_steps) == 4
        assert res.dd_steps[0].intervention_type == "baseline"
        assert res.dd_steps[0].outcome == "FAILURE"
        assert res.dd_steps[1].intervention_type == "partition_subset"
        assert res.dd_steps[1].outcome == "FAILURE"
        assert res.dd_steps[2].intervention_type == "partition_subset"
        assert res.dd_steps[2].outcome == "FAILURE"
        assert res.dd_steps[3].intervention_type == "minimality_verification"
        assert res.dd_steps[3].outcome == "SUCCESS"
        assert res.dd_steps[3].element_count == 0

        # Baseline comparison results
        assert res.greedy_comparison.target_calls == 6
        assert res.greedy_comparison.matches_ground_truth is True
        assert res.greedy_comparison.irrelevant_elements_remaining == 0

        assert res.random_comparison.matches_ground_truth is True
        assert res.random_comparison.irrelevant_elements_remaining == 0

    def test_format_demo_output_contains_required_sections(self) -> None:
        """Verify that the demo output contains all five required sections and disclosures."""
        res = run_demo(seed=42)
        output = format_demo_output(res)

        assert "[1/5] Scenario Setup & Observable Failure" in output
        assert "[2/5] Delta Debugging Execution Trajectory" in output
        assert "[3/5] Isolated Minimal Failure-Inducing Condition" in output
        assert "[4/5] Baseline Method Comparison" in output
        assert "[5/5] Epistemic Boundaries & Scientific Proof" in output

        # Non-interactive terminology verification
        assert "Deterministic Offline Demonstration" in output
        assert "Interactive" not in output

        # Scenario and diagnostic checks
        assert "Automated Accounting Assistant" in output
        assert "calculator" in output
        assert "strict_currency_formatting" in output
        assert "AGENT_FAILURE" in output
        assert "calculator(expression='$150 + $250')" in output

        # Minimality structural explanation and comparisons
        assert "CONFIRMED (1-minimal: explicitly verified that empty set () does not reproduce failure)" in output
        assert "Delta Debugging" in output
        assert "Greedy Ablation" in output
        assert "Seeded Random Search" in output

        # Epistemic limitations
        assert "FakeProvider" in output
        assert "Strict Black-Box Boundary" in output
        assert "$0.00 cost" in output

    def test_demo_reproducibility(self) -> None:
        """Verify that repeated demo runs produce identical output."""
        res1 = run_demo(seed=42)
        res2 = run_demo(seed=42)

        out1 = format_demo_output(res1)
        out2 = format_demo_output(res2)

        assert out1 == out2
        assert res1.discovered_condition.subset_elements == res2.discovered_condition.subset_elements
        assert res1.discovered_condition.target_calls_used == res2.discovered_condition.target_calls_used

    def test_write_demo_reports(self, tmp_path: Path) -> None:
        """Verify that write_demo_reports writes valid JSON/Markdown and protects against accidental overwrites."""
        res = run_demo(seed=42)
        write_demo_reports(res, output_dir=tmp_path)

        json_file = tmp_path / "demo_report.json"
        md_file = tmp_path / "demo_report.md"

        assert json_file.exists()
        assert md_file.exists()

        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert "delta_debugging" in data
        assert data["delta_debugging"]["target_calls"] == 4
        assert data["delta_debugging"]["is_one_minimal"] is True
        assert data["delta_debugging"]["matches_ground_truth"] is True
        assert len(data["delta_debugging"]["steps"]) == 4
        assert data["greedy_ablation"]["matches_ground_truth"] is True
        assert data["seeded_random"]["matches_ground_truth"] is True

        md_content = md_file.read_text(encoding="utf-8")
        assert "AgentXRay: Deterministic Offline Demonstration" in md_content

        # Overwrite protection test: calling again without overwrite=True raises FileExistsError
        with pytest.raises(FileExistsError, match="Refusing to overwrite existing demo report files"):
            write_demo_reports(res, output_dir=tmp_path, overwrite=False)

        # Overwrite=True succeeds
        write_demo_reports(res, output_dir=tmp_path, overwrite=True)
        assert json_file.exists()

    def test_cli_demo_subcommand_stdout(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify that CLI 'agentxray demo' executes via main() and outputs to stdout."""
        monkeypatch.setattr("sys.argv", ["agentxray", "demo"])
        main()

        captured = capsys.readouterr()
        assert "AgentXRay: Deterministic Offline Demonstration" in captured.out
        assert "[1/5] Scenario Setup & Observable Failure" in captured.out
        assert "[3/5] Isolated Minimal Failure-Inducing Condition" in captured.out

    def test_cli_demo_mode_flag_stdout(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify that CLI 'agentxray --mode demo' also triggers the demo."""
        monkeypatch.setattr("sys.argv", ["agentxray", "--mode", "demo"])
        main()

        captured = capsys.readouterr()
        assert "AgentXRay: Deterministic Offline Demonstration" in captured.out

    def test_cli_demo_save_flag_and_overwrite(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Verify that 'agentxray demo --save --output-dir <tmp>' writes report files and respects overwrite flag."""
        # Initial save succeeds
        monkeypatch.setattr("sys.argv", ["agentxray", "demo", "--save", "--output-dir", str(tmp_path)])
        main()

        assert (tmp_path / "demo_report.json").exists()
        assert (tmp_path / "demo_report.md").exists()

        # Second save without --overwrite warns but does not crash CLI
        capsys.readouterr()
        monkeypatch.setattr("sys.argv", ["agentxray", "demo", "--save", "--output-dir", str(tmp_path)])
        main()
        captured = capsys.readouterr()
        assert "Warning: Refusing to overwrite" in captured.out

        # Third save with --overwrite succeeds
        monkeypatch.setattr("sys.argv", ["agentxray", "demo", "--save", "--output-dir", str(tmp_path), "--overwrite"])
        main()
        captured = capsys.readouterr()
        assert "[Demo reports saved to:" in captured.out
