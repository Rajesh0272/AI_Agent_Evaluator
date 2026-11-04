import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd

from src.evaluator.schemas import (
    NormalizedCase,
    GoldLabel,
    VerdictType,
    SplitType,
    DifficultyType,
)

VALID_VERDICTS = {"PASS", "FAIL", "ESCALATE"}
VALID_SPLITS = {"calibration", "test"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


class DataLoaderError(Exception):
    """Raised when data loading, schema parsing, or validation fails."""
    pass


class BenchmarkLoader:
    """
    Robust CSV parser and schema normalizer for benchmark evaluation datasets.
    """

    @staticmethod
    def load_benchmark(
        file_path: str | Path,
    ) -> Tuple[List[NormalizedCase], Dict[str, GoldLabel]]:
        """
        Loads the benchmark CSV, validates all constraints, normalizes nested state_json,
        and returns separate lists for evaluator inputs (NormalizedCase) and ground truth (GoldLabel).

        Args:
            file_path: Path to the benchmark CSV file.

        Returns:
            Tuple of (normalized_cases, gold_labels_by_case_id)

        Raises:
            DataLoaderError: If file not found, case_id missing/duplicate,
                            state_json malformed, or label values invalid.
        """
        path = Path(file_path)
        if not path.is_file():
            raise DataLoaderError(f"Benchmark file not found: {path}")

        try:
            # Load CSV using pandas with string preservation
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception as e:
            raise DataLoaderError(f"Failed to read CSV file {path}: {str(e)}") from e

        if df.empty:
            raise DataLoaderError(f"Benchmark CSV file is empty: {path}")

        required_columns = [
            "case_id",
            "split",
            "capability",
            "domain",
            "difficulty",
            "task",
            "constraints",
            "state_json",
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataLoaderError(f"Missing required columns in CSV: {missing_cols}")

        seen_case_ids: set[str] = set()
        normalized_cases: List[NormalizedCase] = []
        gold_labels: Dict[str, GoldLabel] = {}

        for row_idx, row in df.iterrows():
            case_id = str(row.get("case_id", "")).strip()

            # 1. Validate case_id presence
            if not case_id:
                raise DataLoaderError(f"Row {row_idx}: case_id is missing or empty")

            # 2. Validate case_id uniqueness
            if case_id in seen_case_ids:
                raise DataLoaderError(f"Row {row_idx}: duplicate case_id '{case_id}' detected")
            seen_case_ids.add(case_id)

            # 3. Parse state_json safely
            raw_state_json = row.get("state_json", "").strip()
            if not raw_state_json:
                raise DataLoaderError(f"Row {row_idx} ({case_id}): state_json is empty")

            try:
                state_data = json.loads(raw_state_json)
                if not isinstance(state_data, dict):
                    raise ValueError("Parsed JSON root must be an object/dict")
            except Exception as e:
                raise DataLoaderError(
                    f"Row {row_idx} ({case_id}): malformed state_json content: {str(e)}"
                ) from e

            # 4. Extract trajectory and final answer (fall back to state_json if top-level is empty)
            agent_trajectory = str(row.get("agent_trajectory", "")).strip()
            if not agent_trajectory or agent_trajectory.lower() == "nan":
                agent_trajectory = str(state_data.get("agent_trajectory", "")).strip()

            agent_final_answer = str(row.get("agent_final_answer", "")).strip()
            if not agent_final_answer or agent_final_answer.lower() == "nan":
                agent_final_answer = str(state_data.get("agent_final_answer", "")).strip()

            # 5. Extract available tools
            available_tools = str(row.get("available_tools", "")).strip()
            if not available_tools or available_tools.lower() == "nan":
                tools_in_state = state_data.get("available_tools", "")
                if isinstance(tools_in_state, list):
                    available_tools = ", ".join(str(t) for t in tools_in_state)
                else:
                    available_tools = str(tools_in_state or "")

            split = str(row.get("split", "")).strip().lower()
            if split not in VALID_SPLITS:
                raise DataLoaderError(
                    f"Row {row_idx} ({case_id}): invalid split '{split}'. Allowed: {VALID_SPLITS}"
                )

            difficulty = str(row.get("difficulty", "")).strip().lower()
            if difficulty not in VALID_DIFFICULTIES:
                raise DataLoaderError(
                    f"Row {row_idx} ({case_id}): invalid difficulty '{difficulty}'. Allowed: {VALID_DIFFICULTIES}"
                )

            normalized_case = NormalizedCase(
                case_id=case_id,
                split=split,  # type: ignore[arg-type]
                capability=str(row.get("capability", "")).strip(),
                domain=str(row.get("domain", "")).strip(),
                difficulty=difficulty,  # type: ignore[arg-type]
                task=str(row.get("task", "")).strip(),
                constraints=str(row.get("constraints", "")).strip(),
                available_tools=available_tools,
                agent_trajectory=agent_trajectory,
                agent_final_answer=agent_final_answer,
            )
            normalized_cases.append(normalized_case)

            # 6. Parse and validate Gold Labels (if present in the file)
            gold_verdict_raw = str(row.get("gold_verdict", "")).strip().upper()
            if gold_verdict_raw:
                if gold_verdict_raw not in VALID_VERDICTS:
                    raise DataLoaderError(
                        f"Row {row_idx} ({case_id}): invalid gold_verdict '{gold_verdict_raw}'. Allowed: {VALID_VERDICTS}"
                    )

                try:
                    gold_quality = int(float(row.get("gold_quality", 0)))
                    if gold_quality < 0 or gold_quality > 4:
                        raise ValueError(f"Quality score out of range [0, 4]: {gold_quality}")
                except Exception as e:
                    raise DataLoaderError(
                        f"Row {row_idx} ({case_id}): invalid gold_quality: {str(e)}"
                    ) from e

                gold_policy_raw = str(row.get("gold_policy_violation", "")).strip().lower()
                gold_policy_violation = gold_policy_raw in {"true", "1", "yes"}

                gold_labels[case_id] = GoldLabel(
                    case_id=case_id,
                    gold_verdict=gold_verdict_raw,  # type: ignore[arg-type]
                    gold_quality=gold_quality,
                    gold_policy_violation=gold_policy_violation,
                    gold_rationale=str(row.get("gold_rationale", "")).strip(),
                    trap=str(row.get("trap", "")).strip(),
                )

        return normalized_cases, gold_labels
