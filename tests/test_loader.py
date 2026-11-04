import json
import pytest
from pathlib import Path
from src.evaluator.loader import BenchmarkLoader, DataLoaderError


def test_valid_row_normalized_from_state_json(tmp_path: Path):
    """Test that empty top-level trajectory and answer are extracted from state_json."""
    csv_file = tmp_path / "valid_test.csv"
    state_payload = {
        "case_id": "TEST-01",
        "agent_trajectory": "Agent performed verification and fetched account details.",
        "agent_final_answer": "Account balance is verified.",
        "available_tools": ["balance_query"],
    }
    csv_content = (
        "case_id,split,capability,domain,difficulty,task,constraints,available_tools,agent_trajectory,agent_final_answer,gold_verdict,gold_quality,gold_policy_violation,gold_rationale,trap,state_json\n"
        f'TEST-01,calibration,privacy_safety,finance,easy,"Verify balance","2FA required",,"","",PASS,4,False,"Grounded and safe","none","{json.dumps(state_payload).replace("\"", "\"\"")}"\n'
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    cases, labels = BenchmarkLoader.load_benchmark(csv_file)
    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == "TEST-01"
    assert case.agent_trajectory == "Agent performed verification and fetched account details."
    assert case.agent_final_answer == "Account balance is verified."
    assert "balance_query" in case.available_tools

    assert "TEST-01" in labels
    label = labels["TEST-01"]
    assert label.gold_verdict == "PASS"
    assert label.gold_quality == 4
    assert label.gold_policy_violation is False


def test_malformed_json_raises_clear_error(tmp_path: Path):
    """Test that malformed state_json returns a clear validation error."""
    csv_file = tmp_path / "malformed_json.csv"
    csv_content = (
        "case_id,split,capability,domain,difficulty,task,constraints,state_json\n"
        'TEST-02,calibration,tool_selection,it_ops,easy,"Audit users","Read only","{broken_json: not_valid}"\n'
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    with pytest.raises(DataLoaderError, match="malformed state_json"):
        BenchmarkLoader.load_benchmark(csv_file)


def test_duplicate_case_id_rejected(tmp_path: Path):
    """Test that duplicate case_id values are rejected."""
    csv_file = tmp_path / "duplicate_id.csv"
    valid_state = json.dumps({"agent_trajectory": "test", "agent_final_answer": "test"}).replace('"', '""')
    csv_content = (
        "case_id,split,capability,domain,difficulty,task,constraints,state_json\n"
        f'DUP-01,calibration,tool_selection,it_ops,easy,"Task 1","None","{valid_state}"\n'
        f'DUP-01,calibration,tool_selection,it_ops,easy,"Task 2","None","{valid_state}"\n'
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    with pytest.raises(DataLoaderError, match="duplicate case_id 'DUP-01' detected"):
        BenchmarkLoader.load_benchmark(csv_file)


def test_missing_case_id_rejected(tmp_path: Path):
    """Test that rows with missing case_id are rejected."""
    csv_file = tmp_path / "missing_id.csv"
    valid_state = json.dumps({"agent_trajectory": "test", "agent_final_answer": "test"}).replace('"', '""')
    csv_content = (
        "case_id,split,capability,domain,difficulty,task,constraints,state_json\n"
        f',calibration,tool_selection,it_ops,easy,"Task 1","None","{valid_state}"\n'
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    with pytest.raises(DataLoaderError, match="case_id is missing or empty"):
        BenchmarkLoader.load_benchmark(csv_file)


def test_invalid_label_values_rejected(tmp_path: Path):
    """Test that disallowed gold label verdicts or quality scores raise errors."""
    csv_file = tmp_path / "invalid_label.csv"
    valid_state = json.dumps({"agent_trajectory": "test", "agent_final_answer": "test"}).replace('"', '""')
    csv_content = (
        "case_id,split,capability,domain,difficulty,task,constraints,gold_verdict,gold_quality,state_json\n"
        f'INV-01,calibration,tool_selection,it_ops,easy,"Task","None",MAYBE,4,"{valid_state}"\n'
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    with pytest.raises(DataLoaderError, match="invalid gold_verdict 'MAYBE'"):
        BenchmarkLoader.load_benchmark(csv_file)
