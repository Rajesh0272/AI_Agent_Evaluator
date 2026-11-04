from pathlib import Path
from typing import Optional
from src.evaluator.metrics import ReportMetrics


class EvaluationReporter:
    @staticmethod
    def generate_markdown_report(metrics: ReportMetrics, split_name: str = "test") -> str:
        lines = []
        lines.append(f"# AI Agent Evaluator Benchmark Report ({split_name.upper()} Split)")
        lines.append("")
        lines.append("## 1. Executive Summary")
        lines.append("")
        lines.append(f"- **Total Evaluated Records:** {metrics.total_evaluated}")
        lines.append(f"- **Correct Verdicts:** {metrics.correct_verdicts} / {metrics.total_evaluated}")
        lines.append(f"- **Verdict Accuracy:** {metrics.accuracy * 100:.2f}%")
        lines.append(f"- **Macro F1 Score:** {metrics.macro_f1:.4f}")
        lines.append(f"- **Quality Score MAE:** {metrics.quality_mae:.4f}")
        lines.append(f"- **Policy Violation Precision:** {metrics.policy_precision * 100:.2f}%")
        lines.append(f"- **Policy Violation Recall:** {metrics.policy_recall * 100:.2f}%")
        lines.append("")

        lines.append("## 2. Verdict Distribution")
        lines.append("")
        lines.append("| Verdict | Ground Truth Count | Predicted Count |")
        lines.append("| :--- | :---: | :---: |")
        for cls in ["PASS", "FAIL", "ESCALATE"]:
            gold_c = metrics.gold_distribution.get(cls, 0)
            pred_c = metrics.predicted_distribution.get(cls, 0)
            lines.append(f"| `{cls}` | {gold_c} | {pred_c} |")
        lines.append("")

        lines.append("## 3. Confusion Matrix")
        lines.append("")
        lines.append("Rows represent **Ground Truth (Gold)**; Columns represent **Predicted Verdict**:")
        lines.append("")
        lines.append("| Gold \\ Predicted | Predicted `PASS` | Predicted `FAIL` | Predicted `ESCALATE` |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for gold_cls in ["PASS", "FAIL", "ESCALATE"]:
            row = metrics.confusion_matrix.get(gold_cls, {})
            p_pass = row.get("PASS", 0)
            p_fail = row.get("FAIL", 0)
            p_esc = row.get("ESCALATE", 0)
            lines.append(f"| **Gold `{gold_cls}`** | {p_pass} | {p_fail} | {p_esc} |")
        lines.append("")

        lines.append("## 4. Per-Class Performance")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1-Score | Support |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for cls, scores in metrics.per_class_f1.items():
            lines.append(
                f"| `{cls}` | {scores['precision']:.4f} | {scores['recall']:.4f} | {scores['f1']:.4f} | {scores['support']} |"
            )
        lines.append("")

        lines.append("## 5. Policy Violation Detection")
        lines.append("")
        lines.append(f"- **Precision:** {metrics.policy_precision:.4f}")
        lines.append(f"- **Recall:** {metrics.policy_recall:.4f}")
        lines.append(f"- **F1 Score:** {metrics.policy_f1:.4f}")
        lines.append("")

        lines.append("## 6. Error Analysis & Discrepancies")
        lines.append("")
        if not metrics.error_cases:
            lines.append("No misclassifications observed. The evaluator matched all gold verdicts.")
        else:
            lines.append(f"Found **{len(metrics.error_cases)}** misclassified records out of {metrics.total_evaluated}:")
            lines.append("")
            # Report up to 5 error cases
            for idx, err in enumerate(metrics.error_cases[:5], 1):
                lines.append(f"### Error Case {idx}: `{err['case_id']}`")
                lines.append(f"- **Gold Verdict:** `{err['gold_verdict']}` (Quality: {err['gold_quality']}, Policy Breach: {err['gold_policy_violation']})")
                lines.append(f"- **Predicted Verdict:** `{err['predicted_verdict']}` (Quality: {err['predicted_quality']}, Policy Breach: {err['predicted_policy_violation']})")
                lines.append(f"- **Trap / Failure Mode:** {err.get('trap', 'N/A')}")
                lines.append(f"- **Gold Rationale:** {err.get('gold_rationale', 'N/A')}")
                lines.append(f"- **Evaluator Reason:** {err.get('predicted_reason', 'N/A')}")
                lines.append("")

        return "\n".join(lines)

    @classmethod
    def save_markdown_report(
        cls,
        metrics: ReportMetrics,
        output_path: str | Path,
        split_name: str = "test",
    ) -> None:
        """Write Markdown report to file."""
        report_text = cls.generate_markdown_report(metrics, split_name)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report_text, encoding="utf-8")
