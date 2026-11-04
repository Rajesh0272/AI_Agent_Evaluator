"""
Command Line Interface (CLI) for the AI Agent Response Evaluator.

Supports subcommands:
- validate: Load, verify, and summarize benchmark data integrity
- run: Execute evaluation on a specific split and output predictions
- report: Compare predictions with ground truth labels and generate metrics report
- extension: Run optional claim verification on judge framework datasets
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from src.evaluator.loader import BenchmarkLoader, DataLoaderError
from src.evaluator.engine.factory import create_evaluator
from src.evaluator.metrics import calculate_metrics
from src.evaluator.reporter import EvaluationReporter
from src.evaluator.claim_judge import ClaimJudge


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate CSV data structure, state_json parsing, and label values."""
    csv_path = Path(args.input)
    print(f"Validating dataset: {csv_path}")

    try:
        cases, labels = BenchmarkLoader.load_benchmark(csv_path)
    except DataLoaderError as e:
        print(f"❌ Validation Error: {e}", file=sys.stderr)
        return 1

    split_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}
    capability_counts: Dict[str, int] = {}

    for c in cases:
        split_counts[c.split] = split_counts.get(c.split, 0) + 1
        difficulty_counts[c.difficulty] = difficulty_counts.get(c.difficulty, 0) + 1
        capability_counts[c.capability] = capability_counts.get(c.capability, 0) + 1

    print("\n✅ Dataset validation SUCCESSFUL!")
    print(f"Total valid records: {len(cases)}")
    print(f"Splits: {split_counts}")
    print(f"Difficulties: {difficulty_counts}")
    print(f"Capabilities ({len(capability_counts)}): {list(capability_counts.keys())}")
    print(f"Ground truth labels loaded: {len(labels)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Evaluate cases from benchmark and export predictions."""
    csv_path = Path(args.input)
    split_filter = args.split.lower().strip() if args.split else None
    out_path = Path(args.output)
    engine_mode = getattr(args, "engine", None)

    try:
        cases, _ = BenchmarkLoader.load_benchmark(csv_path)
    except DataLoaderError as e:
        print(f"❌ Loader Error: {e}", file=sys.stderr)
        return 1

    if split_filter:
        cases = [c for c in cases if c.split == split_filter]

    if not cases:
        print(f"❌ No records found for split '{split_filter}'", file=sys.stderr)
        return 1

    print(f"Initializing evaluator engine (mode='{engine_mode or 'default'}')...")
    evaluator = create_evaluator(engine_mode)

    print(f"Running evaluation on {len(cases)} cases (split='{split_filter or 'all'}')...")
    results = evaluator.evaluate_batch(cases)

    # Format predictions to output shape
    output_records = [r.model_dump() for r in results]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=2)

    print(f"✅ Successfully wrote {len(output_records)} predictions to: {out_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Compare predictions with gold labels and generate report."""
    csv_path = Path(args.input)
    pred_path = Path(args.predictions)
    out_path = Path(args.output)

    try:
        cases, gold_labels = BenchmarkLoader.load_benchmark(csv_path)
    except DataLoaderError as e:
        print(f"❌ Loader Error: {e}", file=sys.stderr)
        return 1

    if not pred_path.is_file():
        print(f"❌ Predictions file not found: {pred_path}", file=sys.stderr)
        return 1

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions: List[Dict[str, Any]] = json.load(f)

    # Detect split from evaluated cases
    pred_ids = {p["case_id"] for p in predictions}
    eval_splits = {c.split for c in cases if c.case_id in pred_ids}
    split_name = list(eval_splits)[0] if len(eval_splits) == 1 else "benchmark"

    metrics = calculate_metrics(predictions, gold_labels)
    EvaluationReporter.save_markdown_report(metrics, out_path, split_name=split_name)

    print(f"✅ Generated evaluation report at: {out_path}")
    print(f"Summary: Evaluated={metrics.total_evaluated}, Accuracy={metrics.accuracy*100:.2f}%, Macro F1={metrics.macro_f1:.4f}")
    return 0


def cmd_extension(args: argparse.Namespace) -> int:
    """Evaluate claim verification on judge framework calibration or holdout files."""
    csv_path = Path(args.input)
    if not csv_path.is_file():
        print(f"❌ Input file not found: {csv_path}", file=sys.stderr)
        return 1

    print(f"Evaluating claims from: {csv_path}...")
    res = ClaimJudge.evaluate_file(csv_path)

    print(f"\n✅ Claim Evaluation Complete: {res['correct']}/{res['total']} correct ({res['accuracy']*100:.2f}%)")
    print("\nConfusion Matrix (Rows=Gold, Cols=Pred):")
    labels = ["present", "partial", "missing", "contradicted"]
    print(f"{'Gold \\ Pred':<15} | " + " | ".join(f"{l:<12}" for l in labels))
    print("-" * 75)
    for g in labels:
        row = res["confusion_matrix"].get(g, {})
        counts = [f"{row.get(p, 0):<12}" for p in labels]
        print(f"{g:<15} | " + " | ".join(counts))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved claim evaluation results to: {out}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluator",
        description="AI Agent Response Evaluator CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. validate command
    val_p = subparsers.add_parser("validate", help="Validate benchmark CSV dataset")
    val_p.add_argument("input", help="Path to benchmark CSV file")

    # 2. run command
    run_p = subparsers.add_parser("run", help="Run evaluator on benchmark cases")
    run_p.add_argument("--input", required=True, help="Path to benchmark CSV file")
    run_p.add_argument("--split", choices=["calibration", "test"], help="Dataset split to evaluate")
    run_p.add_argument("--output", required=True, help="Path to write predictions JSON file")
    run_p.add_argument("--engine", choices=["rule", "llm", "hybrid"], default=None, help="Evaluator engine mode")

    # 3. report command
    rep_p = subparsers.add_parser("report", help="Generate benchmark metrics report")
    rep_p.add_argument("--input", required=True, help="Path to benchmark CSV file")
    rep_p.add_argument("--predictions", required=True, help="Path to predictions JSON file")
    rep_p.add_argument("--output", required=True, help="Path to write markdown report")

    # 4. extension command (Optional Claim Verification)
    ext_p = subparsers.add_parser("extension", help="Evaluate claim verification (present, partial, missing, contradicted)")
    ext_p.add_argument("--input", required=True, help="Path to judge framework CSV file")
    ext_p.add_argument("--output", help="Optional path to write evaluation results JSON")

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "report":
        sys.exit(cmd_report(args))
    elif args.command == "extension":
        sys.exit(cmd_extension(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
