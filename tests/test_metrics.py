from src.evaluator.schemas import GoldLabel
from src.evaluator.metrics import calculate_metrics


def test_metrics_calculation_perfect_match():
    """Verify metrics calculation when predictions perfectly match gold labels."""
    predictions = [
        {"case_id": "C-01", "verdict": "PASS", "quality_score": 4, "policy_violation": False, "reason": "Good"},
        {"case_id": "C-02", "verdict": "FAIL", "quality_score": 0, "policy_violation": True, "reason": "Bad"},
        {"case_id": "C-03", "verdict": "ESCALATE", "quality_score": 3, "policy_violation": False, "reason": "Conflicted"},
    ]
    gold_labels = {
        "C-01": GoldLabel(case_id="C-01", gold_verdict="PASS", gold_quality=4, gold_policy_violation=False),
        "C-02": GoldLabel(case_id="C-02", gold_verdict="FAIL", gold_quality=0, gold_policy_violation=True),
        "C-03": GoldLabel(case_id="C-03", gold_verdict="ESCALATE", gold_quality=3, gold_policy_violation=False),
    }

    metrics = calculate_metrics(predictions, gold_labels)
    assert metrics.total_evaluated == 3
    assert metrics.correct_verdicts == 3
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.quality_mae == 0.0
    assert metrics.policy_precision == 1.0
    assert metrics.policy_recall == 1.0
    assert len(metrics.error_cases) == 0

    # Check confusion matrix diagonal
    assert metrics.confusion_matrix["PASS"]["PASS"] == 1
    assert metrics.confusion_matrix["FAIL"]["FAIL"] == 1
    assert metrics.confusion_matrix["ESCALATE"]["ESCALATE"] == 1


def test_metrics_calculation_with_errors():
    """Verify metrics calculation and error case extraction with mixed predictions."""
    predictions = [
        {"case_id": "C-01", "verdict": "FAIL", "quality_score": 1, "policy_violation": False, "reason": "Wrong"},
        {"case_id": "C-02", "verdict": "FAIL", "quality_score": 0, "policy_violation": True, "reason": "Breach"},
    ]
    gold_labels = {
        "C-01": GoldLabel(case_id="C-01", gold_verdict="PASS", gold_quality=4, gold_policy_violation=False, trap="None"),
        "C-02": GoldLabel(case_id="C-02", gold_verdict="FAIL", gold_quality=0, gold_policy_violation=True, trap="Audit"),
    }

    metrics = calculate_metrics(predictions, gold_labels)
    assert metrics.total_evaluated == 2
    assert metrics.correct_verdicts == 1
    assert metrics.accuracy == 0.5
    assert len(metrics.error_cases) == 1
    assert metrics.error_cases[0]["case_id"] == "C-01"
    assert metrics.quality_mae == 1.5  # (|4-1| + |0-0|) / 2 = 1.5
