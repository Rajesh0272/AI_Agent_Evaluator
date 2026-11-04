"""
Abstract base class and contract for all evaluation engines.
"""

from abc import ABC, abstractmethod
from typing import List
from src.evaluator.schemas import NormalizedCase, EvaluationResult


class BaseEvaluator(ABC):
    """
    Abstract evaluator interface.
    Every evaluator takes a NormalizedCase (which contains NO gold labels)
    and produces an EvaluationResult conforming to the assignment specification.
    """

    @abstractmethod
    def evaluate(self, case: NormalizedCase) -> EvaluationResult:
        """
        Evaluate a single agent case.

        Args:
            case: Normalized input containing task, constraints, trajectory, and answer.

        Returns:
            EvaluationResult with verdict, quality_score, policy_violation, and reason.
        """
        pass

    def evaluate_batch(self, cases: List[NormalizedCase]) -> List[EvaluationResult]:
        """
        Evaluate a batch of cases sequentially.

        Args:
            cases: List of normalized cases.

        Returns:
            List of EvaluationResults.
        """
        return [self.evaluate(c) for c in cases]
