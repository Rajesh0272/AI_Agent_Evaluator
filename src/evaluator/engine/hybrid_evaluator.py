"""
Hybrid Evaluator combining deterministic safety guardrails with semantic evaluation.

Workflow:
1. Fast local deterministic safety gate (catches 100% of hard authority/policy breaches).
2. If safe, delegates to LLM Evaluator when credentials are configured in .env.
3. If LLM credentials are not configured, seamlessly falls back to the local Rule Evaluator.
"""

import logging
from src.evaluator.schemas import NormalizedCase, EvaluationResult
from src.evaluator.engine.base import BaseEvaluator
from src.evaluator.engine.rule_evaluator import RuleEvaluator
from src.evaluator.engine.llm_evaluator import LLMEvaluator
from src.evaluator.config import get_settings

logger = logging.getLogger(__name__)


class HybridEvaluator(BaseEvaluator):
    """
    Evaluator that pairs deterministic rule guardrails with semantic judging.
    """

    def __init__(self):
        self.rule_engine = RuleEvaluator()
        cfg = get_settings()
        self.has_llm_key = bool(cfg.openai_api_key or cfg.external_api_key)
        self.llm_engine = LLMEvaluator() if self.has_llm_key else None

    def evaluate(self, case: NormalizedCase) -> EvaluationResult:
        """
        Evaluate a single case using hybrid policy gates + semantic engine.
        """
        # Step 1: Run deterministic safety and rule check
        rule_result = self.rule_engine.evaluate(case)

        # If a hard policy breach was flagged, uphold it immediately
        if rule_result.policy_violation:
            return rule_result

        # Step 2: If LLM is configured, invoke LLM for nuanced reasoning
        if self.llm_engine:
            try:
                return self.llm_engine.evaluate(case)
            except Exception as e:
                logger.warning(
                    f"LLM evaluation failed for {case.case_id} ({e}); falling back to rule engine."
                )

        # Step 3: Local rule evaluation fallback
        return rule_result
