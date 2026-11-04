"""
End-to-end integration tests for evaluator CLI commands (validate, run, report).
"""

import json
from pathlib import Path
from src.evaluator.cli import build_parser, cmd_validate, cmd_run, cmd_report


def test_cli_validate_success():
    """Test CLI validate command on the benchmark dataset."""
    parser = build_parser()
    args = parser.parse_args(["validate", "datasets/ai_agent_evaluator_benchmark.csv"])
    ret = cmd_validate(args)
    assert ret == 0


def test_cli_run_calibration(tmp_path: Path):
    """Test CLI run command outputs valid predictions JSON."""
    out_json = tmp_path / "preds.json"
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--input", "datasets/ai_agent_evaluator_benchmark.csv",
        "--split", "calibration",
        "--output", str(out_json),
        "--engine", "rule",
    ])
    ret = cmd_run(args)
    assert ret == 0
    assert out_json.is_file()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 12
    for record in data:
        assert "case_id" in record
        assert record["verdict"] in {"PASS", "FAIL", "ESCALATE"}
        assert 0 <= record["quality_score"] <= 4
        assert isinstance(record["policy_violation"], bool)
        assert record["reason"]


def test_cli_report_generation(tmp_path: Path):
    """Test CLI report command generates markdown report file."""
    preds_json = tmp_path / "preds.json"
    report_md = tmp_path / "report.md"

    # Step 1: Run predictions on calibration
    parser = build_parser()
    args_run = parser.parse_args([
        "run",
        "--input", "datasets/ai_agent_evaluator_benchmark.csv",
        "--split", "calibration",
        "--output", str(preds_json),
    ])
    assert cmd_run(args_run) == 0

    # Step 2: Generate report
    args_rep = parser.parse_args([
        "report",
        "--input", "datasets/ai_agent_evaluator_benchmark.csv",
        "--predictions", str(preds_json),
        "--output", str(report_md),
    ])
    assert cmd_report(args_rep) == 0
    assert report_md.is_file()

    report_content = report_md.read_text(encoding="utf-8")
    assert "Confusion Matrix" in report_content
    assert "Macro F1 Score" in report_content
