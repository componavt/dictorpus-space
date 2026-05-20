"""Data structures for multi-model translation comparison."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelOutput:
    """Normalized output from a single translation model for one gloss."""
    model_key: str
    model_name: str
    gloss_en: str
    qa_keep: bool
    qa_score: float
    qa_flags: set[str] = field(default_factory=set)
    roundtrip_distance: float | None = None
    normalized_gloss_en: str = ""


@dataclass
class ConsensusCluster:
    """A cluster of near-identical outputs from different models."""
    representative: str
    model_keys: list[str] = field(default_factory=list)
    outputs: list[ModelOutput] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Final comparison result for one gloss_ru."""
    gloss_ru: str
    proposed_gloss_en: str
    preferred_source: str
    chosen_from_model_key: str | None
    decision_reason: str
    total_risk: float
    risk_level: str
    risk_reasons: list[str] = field(default_factory=list)
    consensus_ratio: float = 0.0
    disagreement_score: float = 0.0
    non_blank_model_count: int = 0
    good_model_count: int = 0
    unique_output_count: int = 0
    largest_cluster_size: int = 0
    gloss_complexity_score: float = 0.0
    gloss_complexity_reasons: list[str] = field(default_factory=list)
    needs_expert_review: bool = False
