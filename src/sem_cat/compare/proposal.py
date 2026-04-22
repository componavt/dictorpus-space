"""Proposed translation selection from multi-model comparison."""

from __future__ import annotations

from src.sem_cat.compare.data_structures import ConsensusCluster, ModelOutput
from src.sem_cat.compare.normalization import output_similarity


def select_proposed_translation(
    clusters: list[ConsensusCluster],
    all_outputs: list[ModelOutput],
    total_risk: float,
    risk_threshold: float = 0.35,
) -> tuple[str, str, str | None, str]:
    """Select the best proposed translation from consensus clusters.

    Args:
        clusters: Consensus clusters sorted by size descending.
        all_outputs: All model outputs (including blank).
        total_risk: Pre-computed total risk score.
        risk_threshold: Threshold above which manual review is preferred.

    Returns:
        (proposed_gloss_en, preferred_source, chosen_from_model_key, decision_reason)
    """
    non_blank = [o for o in all_outputs if o.gloss_en and o.gloss_en.strip()]

    if not non_blank:
        return "", "manual_review", None, "all_blank"

    if not clusters:
        # All outputs are unique and far apart
        return "", "manual_review", None, "no_clear_winner"

    largest = clusters[0]

    # Strategy 1: Strong consensus
    if len(largest.model_keys) >= 3:
        # Pick the output with lowest qa_score from the largest cluster
        best = min(largest.outputs, key=lambda o: o.qa_score)
        return (
            best.gloss_en,
            "consensus",
            best.model_key,
            "strong_consensus",
        )

    # Strategy 2: Near-match consensus (2 models agree closely)
    if len(largest.model_keys) >= 2:
        best = min(largest.outputs, key=lambda o: o.qa_score)
        if total_risk < risk_threshold:
            return (
                best.gloss_en,
                "consensus",
                best.model_key,
                "consensus_near_match",
            )
        # Risk is high even with partial consensus
        return (
            best.gloss_en,
            "low_confidence_consensus",
            best.model_key,
            "low_confidence_consensus",
        )

    # Strategy 3: Only one good output
    good_outputs = [o for o in non_blank if o.qa_keep and o.qa_score < 0.3]
    if len(good_outputs) == 1:
        best = good_outputs[0]
        return (
            best.gloss_en,
            best.model_key,
            best.model_key,
            "only_one_good_output",
        )

    # Strategy 4: Single output, pick the best qa
    if len(non_blank) == 1:
        best = non_blank[0]
        if best.qa_score < 0.3:
            return (
                best.gloss_en,
                best.model_key,
                best.model_key,
                "only_one_good_output",
            )
        else:
            return (
                best.gloss_en,
                "low_confidence",
                best.model_key,
                "low_confidence_single_output",
            )

    # Strategy 5: No clear winner
    return "", "manual_review", None, "no_clear_winner"
