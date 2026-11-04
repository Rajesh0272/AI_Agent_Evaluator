"""
Factory for creating evaluator instances.
"""

from typing import Optional
from src.evaluator.engine.base import BaseEvaluator
from src.evaluator.engine.rule_evaluator import RuleEvaluator
from src.evaluator.engine.llm_evaluator import LLMEvaluator
from src.evaluator.engine.hybrid_evaluator import HybridEvaluator
from src.evaluator.config import get_settings


def create_evaluator(mode: Optional[str] = None) -> BaseEvaluator:
    """
    Factory function returning the appropriate BaseEvaluator instance.

    Args:
        mode: Engine mode ('rule', 'llm', or 'hybrid'). Defaults to settings.evaluator_engine_mode.

    Returns:
        Instance of BaseEvaluator.
    """
    selected_mode = (mode or get_settings().evaluator_engine_mode).lower().strip()

    if selected_mode == "rule":
        return RuleEvaluator()
    elif selected_mode == "llm":
        return LLMEvaluator()
    elif selected_mode == "hybrid":
        return HybridEvaluator()
    else:
        raise ValueError(f"Unknown evaluator mode '{selected_mode}'. Allowed: rule, llm, hybrid")
