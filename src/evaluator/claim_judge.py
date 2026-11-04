"""
Claim Verification Judge (Optional Extension).

Evaluates whether a candidate agent response preserves, partially preserves,
omits, or contradicts a specific claim from reference context:
- `present`: Fully preserves the claim and qualifiers.
- `partial`: Preserves the core quantity/predicate but omits a scope qualifier.
- `missing`: Fails to mention the required claim or discusses an unrelated fact.
- `contradicted`: Asserts a value, entity, or direction that directly conflicts with context.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd


class ClaimJudge:
    """
    Deterministic claim-verification evaluator for question-answering fidelity.
    """

    QUALIFIER_KEYWORDS = {
        "each",
        "every",
        "only",
        "production",
        "calendar",
        "annual",
        "annually",
        "workspace",
        "business",
        "working",
        "staging",
        "concurrent",
        "maximum",
        "minimum",
        "primary",
    }

    CONTRADICTION_PAIRS = [
        ("east", "north"),
        ("west", "south"),
        ("east", "west"),
        ("north", "south"),
        ("allow", "deny"),
        ("enable", "disable"),
        ("true", "false"),
        ("yes", "no"),
        ("permitted", "forbidden"),
        ("active", "inactive"),
    ]

    @classmethod
    def evaluate_claim(
        cls,
        response: str,
        context: str,
        expected_output: str = "",
    ) -> Tuple[str, str]:
        """
        Evaluate candidate response against supporting context.

        Returns:
            Tuple of (predicted_label, explanation)
            where label is one of: 'present', 'partial', 'missing', 'contradicted'
        """
        resp_lower = response.lower().strip()
        ctx_lower = context.lower().strip()

        resp_words = set(re.findall(r"\b[a-z0-9]{3,}\b", resp_lower))
        ctx_words = set(re.findall(r"\b[a-z0-9]{3,}\b", ctx_lower))

        resp_nums = set(re.findall(r"\b\d+\b", resp_lower))
        ctx_nums = set(re.findall(r"\b\d+\b", ctx_lower))

        # -------------------------------------------------------------
        # STEP 1: Contradiction Detection
        # -------------------------------------------------------------
        # A) Numeric discrepancy: response asserts a number that conflicts with context
        if resp_nums and ctx_nums and resp_nums != ctx_nums:
            return "contradicted", f"Numeric conflict: response cites {resp_nums} while context specifies {ctx_nums}."

        # B) Directional / Geographic / Antonym contradiction
        for word_a, word_b in cls.CONTRADICTION_PAIRS:
            if (word_a in resp_words and word_b in ctx_words) or (word_b in resp_words and word_a in ctx_words):
                return "contradicted", f"Conflicting entity or attribute detected between '{word_a}' and '{word_b}'."

        # -------------------------------------------------------------
        # STEP 2: Missing Claim Detection
        # -------------------------------------------------------------
        overlap = resp_words.intersection(ctx_words)
        shared_nums = resp_nums.intersection(ctx_nums)

        # If there are no shared numbers and lexical overlap is minimal, claim is missing
        if not shared_nums:
            # Check content word overlap (excluding stop words)
            stop_words = {"the", "and", "for", "are", "with", "this", "that", "from", "its"}
            content_overlap = overlap - stop_words
            if len(content_overlap) <= 1:
                return "missing", "The response addresses an unrelated topic and omits the required claim."

        # -------------------------------------------------------------
        # STEP 3: Partial Claim Detection (Omitted Qualifiers)
        # -------------------------------------------------------------
        # Context contains key qualifying restrictions that response omitted
        omitted_qualifiers = (ctx_words.intersection(cls.QUALIFIER_KEYWORDS)) - resp_words
        if omitted_qualifiers:
            omitted_list = ", ".join(f"'{q}'" for q in sorted(omitted_qualifiers))
            return "partial", f"The response preserves the core claim but omits essential scope qualifier(s): {omitted_list}."

        # -------------------------------------------------------------
        # STEP 4: Present (Faithfully preserved)
        # -------------------------------------------------------------
        return "present", "The response faithfully preserves the claim along with all required qualifiers."

    @classmethod
    def evaluate_file(cls, csv_path: str | Path) -> Dict[str, Any]:
        """
        Process a judge_framework CSV file and return accuracy and confusion matrix.
        """
        df = pd.read_csv(csv_path)
        records = []
        correct = 0

        labels = ["present", "partial", "missing", "contradicted"]
        cm = {g: {p: 0 for p in labels} for g in labels}

        for _, row in df.iterrows():
            meta = json.loads(row["Metadata"]) if isinstance(row["Metadata"], str) else row["Metadata"]
            response = meta.get("response", "")
            context = meta.get("context", "")
            gold_label = meta.get("benchmark_label", "").lower().strip()
            case_id = meta.get("case_id", f"Row-{row['Serial No']}")

            pred_label, explanation = cls.evaluate_claim(response, context)
            if pred_label == gold_label:
                correct += 1

            if gold_label in cm and pred_label in cm[gold_label]:
                cm[gold_label][pred_label] += 1

            records.append({
                "case_id": case_id,
                "gold_label": gold_label,
                "predicted_label": pred_label,
                "explanation": explanation,
                "response": response,
                "context": context,
            })

        total = len(records)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "confusion_matrix": cm,
            "records": records,
        }
