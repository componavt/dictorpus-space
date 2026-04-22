"""Total risk scoring for N-model comparison."""

from __future__ import annotations

from dataclasses import dataclass

from src.sem_cat.compare.data_structures import ModelOutput


@dataclass(frozen=True)
class ComparisonRiskConfig:
    """Penalty weights and thresholds for total risk scoring."""
    # Coverage risk
    all_blank_penalty: float = 0.40
    low_coverage_penalty: float = 0.20
    # QA risk
    high_qa_penalty: float = 0.25
    # Agreement risk
    disagreement_penalty: float = 0.25
    # Sentence-like expansion risk
    sentence_like_penalty: float = 0.20
    # Round-trip risk
    roundtrip_penalty: float = 0.15
    # Gloss complexity risk
    complexity_weight: float = 1.0
    # Strong consensus discount
    strong_consensus_discount: float = 0.20
    # Consensus threshold
    consensus_ratio_threshold: float = 0.60
    min_good_for_consensus: int = 3
    min_cluster_for_consensus: int = 3
    # Risk level thresholds
    high_risk_threshold: float = 0.65
    medium_risk_threshold: float = 0.35


def compute_total_risk(
    outputs: list[ModelOutput],
    total_models: int,
    largest_cluster_size: int,
    good_model_count: int,
    consensus_ratio: float,
    disagreement_score: float,
    complexity_score: float,
    config: ComparisonRiskConfig | None = None,
) -> tuple[float, list[str]]:
    """Compute total risk score in [0.0, 1.0] with reason strings.

    Args:
        outputs: All model outputs for this gloss.
        total_models: Total number of models being compared.
        largest_cluster_size: Size of biggest consensus cluster.
        good_model_count: Number of models with qa_keep=True and non-blank.
        consensus_ratio: largest_cluster_size / non_blank_model_count.
        disagreement_score: Pre-computed disagreement metric.
        complexity_score: From compute_gloss_complexity().
        config: Risk configuration (uses defaults if None).

    Returns:
        (total_risk, risk_reasons)
    """
    if config is None:
        config = ComparisonRiskConfig()

    risk_score = 0.0
    reasons: list[str] = []

    non_blank = [o for o in outputs if o.gloss_en and o.gloss_en.strip()]
    non_blank_count = len(non_blank)

    # 1. Coverage risk
    if non_blank_count == 0:
        risk_score += config.all_blank_penalty
        reasons.append("all_blank")
    elif non_blank_count == 1:
        risk_score += config.low_coverage_penalty
        reasons.append("only_one_output")
    elif non_blank_count < total_models * 0.5:
        risk_score += config.low_coverage_penalty * 0.5
        reasons.append("low_coverage")

    # 2. QA risk
    if non_blank:
        max_qa = max(o.qa_score for o in non_blank)
        avg_qa = sum(o.qa_score for o in non_blank) / len(non_blank)
        if max_qa > 0.5:
            risk_score += config.high_qa_penalty
            reasons.append("high_max_qa_score")
        elif avg_qa > 0.3:
            risk_score += config.high_qa_penalty * 0.5
            reasons.append("elevated_avg_qa_score")

        # Count models with qa_keep=False
        bad_qa = [o for o in non_blank if not o.qa_keep]
        if len(bad_qa) > 0:
            risk_score += config.high_qa_penalty * 0.3
            reasons.append("some_qa_keep_false")

    # 3. Agreement risk
    if non_blank_count >= 2:
        if disagreement_score > 0.5:
            risk_score += config.disagreement_penalty
            reasons.append("severe_disagreement")
        elif disagreement_score > 0.3:
            risk_score += config.disagreement_penalty * 0.5
            reasons.append("moderate_disagreement")

    # 4. Sentence-like expansion risk
    sentence_like_count = sum(
        1 for o in outputs
        if "sentence_like_singleword_expansion" in o.qa_flags
    )
    if sentence_like_count >= 2:
        risk_score += config.sentence_like_penalty
        reasons.append("multiple_sentence_like_outputs")
    elif sentence_like_count == 1:
        risk_score += config.sentence_like_penalty * 0.5
        reasons.append("sentence_like_output")

    # 5. Round-trip risk
    rt_values = [
        o.roundtrip_distance for o in outputs
        if o.roundtrip_distance is not None and o.roundtrip_distance > 0.5
    ]
    if len(rt_values) >= 2:
        risk_score += config.roundtrip_penalty
        reasons.append("multiple_poor_roundtrips")
    elif len(rt_values) == 1:
        risk_score += config.roundtrip_penalty * 0.5
        reasons.append("poor_roundtrip")

    # 6. Complexity risk
    if complexity_score > 0:
        risk_score += complexity_score * config.complexity_weight
        # Reasons added by caller from complexity helper

    # 7. Strong consensus discount
    if (
        largest_cluster_size >= config.min_cluster_for_consensus
        and good_model_count >= config.min_good_for_consensus
        and consensus_ratio >= config.consensus_ratio_threshold
    ):
        risk_score -= config.strong_consensus_discount
        reasons.append("strong_consensus")

    # Clip to [0.0, 1.0]
    risk_score = max(0.0, min(1.0, risk_score))
    return round(risk_score, 2), reasons


def compute_risk_level(risk_score: float, config: ComparisonRiskConfig | None = None) -> str:
    """Convert risk score to a human-readable level."""
    if config is None:
        config = ComparisonRiskConfig()
    if risk_score >= config.high_risk_threshold:
        return "high"
    if risk_score >= config.medium_risk_threshold:
        return "medium"
    return "low"
