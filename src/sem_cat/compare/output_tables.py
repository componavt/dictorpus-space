"""Output table builders for multi-model comparison results."""

from __future__ import annotations

import pandas as pd

from src.sem_cat.compare.data_structures import ComparisonResult, ModelOutput


def build_full_comparison_df(
    results: list[ComparisonResult],
    model_keys: list[str],
) -> pd.DataFrame:
    """Build the full comparison DataFrame with all columns.

    Column order:
    1. Core identifiers
    2. Risk and consensus metrics
    3. Proposal fields
    4. Per-model columns (model_key__field convention)
    """
    rows = []
    for r in results:
        row = {
            "gloss_ru": r.gloss_ru,
            "is_singleword": r.gloss_ru.strip().count(" ") == 0 and len(r.gloss_ru.strip()) > 0,
            "gloss_complexity_score": r.gloss_complexity_score,
            "gloss_complexity_reasons": ";".join(r.gloss_complexity_reasons) if r.gloss_complexity_reasons else "",
            "non_blank_model_count": r.non_blank_model_count,
            "good_model_count": r.good_model_count,
            "unique_output_count": r.unique_output_count,
            "largest_cluster_size": r.largest_cluster_size,
            "consensus_ratio": round(r.consensus_ratio, 3),
            "disagreement_score": round(r.disagreement_score, 3),
            "total_risk": r.total_risk,
            "risk_level": r.risk_level,
            "risk_reasons": ";".join(r.risk_reasons) if r.risk_reasons else "",
            "proposed_gloss_en": r.proposed_gloss_en,
            "preferred_source": r.preferred_source,
            "decision_reason": r.decision_reason,
            "chosen_from_model_key": r.chosen_from_model_key or "",
            "needs_expert_review": r.needs_expert_review,
        }

        # Per-model columns
        for mk in model_keys:
            model_out = next((o for o in r._outputs if o.model_key == mk), None)  # type: ignore[attr-defined]
            prefix = f"{mk}__"
            if model_out:
                row[f"{prefix}gloss_en"] = model_out.gloss_en
                row[f"{prefix}qa_keep"] = model_out.qa_keep
                row[f"{prefix}qa_score"] = model_out.qa_score
                row[f"{prefix}qa_flags"] = ";".join(sorted(model_out.qa_flags)) if model_out.qa_flags else ""
                row[f"{prefix}roundtrip_distance"] = (
                    round(model_out.roundtrip_distance, 3)
                    if model_out.roundtrip_distance is not None
                    else ""
                )
                row[f"{prefix}model_name"] = model_out.model_name
            else:
                row[f"{prefix}gloss_en"] = ""
                row[f"{prefix}qa_keep"] = ""
                row[f"{prefix}qa_score"] = ""
                row[f"{prefix}qa_flags"] = ""
                row[f"{prefix}roundtrip_distance"] = ""
                row[f"{prefix}model_name"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def build_review_queue_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and format the review queue from the full comparison table."""
    review = full_df[full_df["needs_expert_review"] == True].copy()

    review_cols = [
        "gloss_ru",
        "is_singleword",
        "proposed_gloss_en",
        "preferred_source",
        "chosen_from_model_key",
        "decision_reason",
        "total_risk",
        "risk_level",
        "risk_reasons",
        "consensus_ratio",
        "disagreement_score",
        "gloss_complexity_reasons",
    ]
    # Add all per-model columns
    for col in full_df.columns:
        if "__" in col:
            review_cols.append(col)

    available = [c for c in review_cols if c in review.columns]
    return review[available]


def build_gold_template_df(review_df: pd.DataFrame) -> pd.DataFrame:
    """Build the gold template from the review queue."""
    gold = review_df.copy()
    gold["expert_gloss_en"] = ""
    gold["expert_notes"] = ""
    gold["final_decision"] = ""
    gold["include_in_gold"] = ""

    gold_cols = [
        "gloss_ru",
        "is_singleword",
        "proposed_gloss_en",
        "preferred_source",
        "chosen_from_model_key",
        "decision_reason",
        "total_risk",
        "consensus_ratio",
        "disagreement_score",
    ]
    # Add per-model gloss_en columns
    for col in gold.columns:
        if col.endswith("__gloss_en"):
            gold_cols.append(col)

    gold_cols.extend([
        "expert_gloss_en",
        "expert_notes",
        "final_decision",
        "include_in_gold",
    ])

    available = [c for c in gold_cols if c in gold.columns]
    return gold[available]
