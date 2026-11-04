from abc import ABC, abstractmethod
from typing import List
from src.evaluator.schemas import NormalizedCase, EvaluationResult


class BaseEvaluator(ABC):

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
