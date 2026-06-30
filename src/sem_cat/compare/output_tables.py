"""Output table builders for multi-model comparison results."""

from __future__ import annotations

import pandas as pd

from src.sem_cat.compare.data_structures import ComparisonResult, ModelOutput


def _model_has_roundtrip_data(results: list[ComparisonResult], model_key: str) -> bool:
    """Check if any result has non-null roundtrip_distance for the given model."""
    for r in results:
        for o in getattr(r, "_outputs", []):
            if o.model_key == model_key:
                if o.roundtrip_distance is not None:
                    return True
    return False


def build_full_comparison_df(
    results: list[ComparisonResult],
    model_keys: list[str],
) -> pd.DataFrame:
    """Build the full comparison DataFrame with all columns.

    Column order:
    1. Core identifiers
    2. Task-level metadata (task_key, task_pos if available)
    3. Risk and consensus metrics
    4. Proposal fields
    5. Per-model columns (model_key__field convention)

    Schema policy:
    - Always include: __gloss_en, __qa_keep, __qa_score, __qa_flags
    - Include __roundtrip_distance only if any model has non-null values
    - Never include: __model_name (redundant per-row constant)
    """
    if not model_keys or not results:
        return pd.DataFrame()

    include_roundtrip = {
        mk: _model_has_roundtrip_data(results, mk)
        for mk in model_keys
    }

    has_task_metadata = hasattr(results[0], 'task_key') and results[0].task_key is not None

    rows = []
    for r in results:
        row = {
            "gloss_ru": r.gloss_ru,
        }
        
        if has_task_metadata:
            row["task_key"] = r.task_key
            row["task_pos"] = r.task_pos
            
        row.update({
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
        })

        # Per-model columns
        for mk in model_keys:
            model_out = next((o for o in getattr(r, "_outputs", []) if o.model_key == mk), None)
            prefix = f"{mk}__"
            if model_out:
                row[f"{prefix}gloss_en"] = model_out.gloss_en
                row[f"{prefix}qa_keep"] = model_out.qa_keep
                row[f"{prefix}qa_score"] = model_out.qa_score
                row[f"{prefix}qa_flags"] = ";".join(sorted(model_out.qa_flags)) if model_out.qa_flags else ""
                if include_roundtrip[mk]:
                    row[f"{prefix}roundtrip_distance"] = round(model_out.roundtrip_distance, 3) if model_out.roundtrip_distance is not None else ""
            else:
                row[f"{prefix}gloss_en"] = ""
                row[f"{prefix}qa_keep"] = ""
                row[f"{prefix}qa_score"] = ""
                row[f"{prefix}qa_flags"] = ""
                if include_roundtrip[mk]:
                    row[f"{prefix}roundtrip_distance"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def build_review_queue_df(
    full_df: pd.DataFrame, include_low_risk: bool = False
) -> pd.DataFrame:
    """Extract and format the review queue from the full comparison table.
    
    Args:
        full_df: Full comparison DataFrame
        include_low_risk: If True, include rows with risk_level='low' in review queue
    
    Review queue columns:
    - Core identifiers and risk metrics
    - Per-model: __gloss_en, __qa_keep, __qa_score, __qa_flags, __roundtrip_distance (if data exists)
    """
    if include_low_risk:
        review = full_df[
            (full_df["needs_expert_review"] == True) | 
            (full_df["risk_level"] == "low")
        ].copy()
    else:
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
    # Add per-model columns: always include gloss_en, qa_keep, qa_score, qa_flags
    # roundtrip_distance is only present if model has data
    for col in full_df.columns:
        if "__" in col and (
            col.endswith("__gloss_en") or 
            col.endswith("__qa_keep") or 
            col.endswith("__qa_score") or 
            col.endswith("__qa_flags") or
            col.endswith("__roundtrip_distance")
        ):
            review_cols.append(col)

    available = [c for c in review_cols if c in review.columns]
    return review[available]


def build_gold_template_df(review_df: pd.DataFrame) -> pd.DataFrame:
    """Build the gold template from the review queue.
    
    Gold template columns:
    - Adjudication fields and lightweight comparison metrics
    - Per-model: __gloss_en only (no model_name, no roundtrip)
    """
    gold = review_df.copy()
    gold["expert_gloss_en"] = ""
    gold["expert_notes"] = ""
    gold["final_decision"] = ""
    gold["include_in_gold"] = ""
    gold["accepted_model_key"] = ""
    gold["accepted_raw_output"] = ""
    gold["review_status"] = ""

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
    # Add per-model gloss_en columns only (no model_name, no roundtrip_distance)
    for col in gold.columns:
        if col.endswith("__gloss_en"):
            gold_cols.append(col)

    gold_cols.extend([
        "expert_gloss_en",
        "expert_notes",
        "final_decision",
        "include_in_gold",
        "accepted_model_key",
        "accepted_raw_output",
        "review_status",
    ])

    available = [c for c in gold_cols if c in gold.columns]
    return gold[available]
