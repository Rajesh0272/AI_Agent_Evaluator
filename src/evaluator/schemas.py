from typing import Literal, Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# Standardized verdict and capability types
VerdictType = Literal["PASS", "FAIL", "ESCALATE"]
SplitType = Literal["calibration", "test"]
DifficultyType = Literal["easy", "medium", "hard"]


class StateJsonPayload(BaseModel):
    """
    Schema of the nested state_json field inside the benchmark CSV.
    Contains the real agent trajectory and final answer.
    """
    case_id: Optional[str] = None
    task: Optional[str] = None
    constraints: Optional[str] = None
    available_tools: Optional[Union[List[Any], str]] = None
    agent_trajectory: Optional[str] = None
    agent_final_answer: Optional[str] = None


class NormalizedCase(BaseModel):
    """
    Cleaned, normalized input case passed to the evaluator.
    Notice that gold labels are intentionally excluded from this model
    to prevent any data or label leakage into the evaluation logic.
    """
    case_id: str
    split: SplitType
    capability: str
    domain: str
    difficulty: DifficultyType
    task: str
    constraints: str
    available_tools: str = ""
    agent_trajectory: str
    agent_final_answer: str

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("case_id must not be empty")
        return v.strip()


class GoldLabel(BaseModel):
    """
    Ground-truth reference label stored separately for evaluation and scoring.
    """
    case_id: str
    gold_verdict: VerdictType
    gold_quality: int = Field(ge=0, le=4)
    gold_policy_violation: bool
    gold_rationale: Optional[str] = None
    trap: Optional[str] = None


class EvaluationResult(BaseModel):
    """
    Expected output schema for each evaluated case according to the assignment contract.
    """
    case_id: str
    verdict: VerdictType
    quality_score: int = Field(ge=0, le=4)
    policy_violation: bool
    reason: str

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reason explanation must not be empty")
        return v.strip()


class ClaimMetadata(BaseModel):
    """
    Schema for Metadata field in judge_framework extension datasets.
    """
    case_id: Optional[str] = None
    response: str
    context: str
    benchmark_label: Literal["present", "partial", "missing", "contradicted"]
    risk: Optional[str] = None
    reference_notes: Optional[str] = None
