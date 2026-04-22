"""Translation QA analysis for dictionary-style gloss outputs.

Produces a QAResult dataclass with keep/score/flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.sem_cat.utils.text_utils import is_blank, contains_ascii_letters
from src.sem_cat.utils.distance import normalized_edit_distance
from src.sem_cat.qa.translation_flags import (
    detect_repetition,
    detect_sentence_like_expansion,
    detect_length_inflation,
    detect_name_expansion_patterns,
    detect_placeholder_or_garbage,
)


@dataclass(frozen=True)
class TranslationQAConfig:
    """Penalty weights and thresholds for translation QA."""
    no_ascii_penalty: float = 0.30
    too_long_penalty: float = 0.40
    multiword_penalty: float = 0.20
    roundtrip_far_penalty: float = 0.30
    sentence_like_penalty: float = 0.35
    token_inflation_penalty: float = 0.20
    probable_name_overexpansion_penalty: float = 0.20
    roundtrip_far_threshold: float = 0.50


@dataclass
class QAResult:
    """Result of translation quality analysis."""
    qa_keep: bool
    qa_score: float
    qa_flags: list[str] = field(default_factory=list)
    roundtrip_distance: float | None = None


def analyze_translation(
    ru: str,
    en: str,
    roundtrip_text: str | None = None,
    config: TranslationQAConfig | None = None,
) -> QAResult:
    """Analyze a translation and return a QAResult.

    Args:
        ru: Original Russian gloss
        en: Translated English gloss
        roundtrip_text: Optional back-translated Russian text
        config: QA configuration (uses defaults if None)

    Returns:
        QAResult with qa_keep, qa_score, qa_flags, and roundtrip_distance.
    """
    if config is None:
        config = TranslationQAConfig()

    qa_flags: list[str] = []
    qa_score = 0.0

    # Fatal checks -> qa_keep = False
    if is_blank(en):
        return QAResult(qa_keep=False, qa_score=1.0, qa_flags=["empty_translation"])

    if not en.strip():
        return QAResult(qa_keep=False, qa_score=1.0, qa_flags=["empty_translation"])

    # Punctuation-only
    from src.sem_cat.utils.text_utils import is_punctuation_only
    if is_punctuation_only(en):
        return QAResult(qa_keep=False, qa_score=1.0, qa_flags=["punctuation_only"])

    # Repetition loops
    if detect_repetition(en):
        return QAResult(qa_keep=False, qa_score=1.0, qa_flags=["repeated_token_loop"])

    # Placeholder / garbage
    garbage_flags = detect_placeholder_or_garbage(en)
    qa_flags.extend(garbage_flags)

    # No ASCII letters
    if not contains_ascii_letters(en):
        qa_flags.append("no_ascii_letters")
        qa_score += config.no_ascii_penalty

    # Length inflation and multiword checks
    length_flags = detect_length_inflation(ru, en)
    for flag in length_flags:
        if flag == "too_long_for_gloss":
            qa_score += config.too_long_penalty
        elif flag == "multiword_for_singleword":
            qa_score += config.multiword_penalty
        elif flag == "token_inflation":
            qa_score += config.token_inflation_penalty
    qa_flags.extend(length_flags)

    # Sentence-like expansion
    sentence_flags = detect_sentence_like_expansion(ru, en)
    if sentence_flags:
        qa_score += config.sentence_like_penalty
    qa_flags.extend(sentence_flags)

    # Proper-name overexpansion
    name_flags = detect_name_expansion_patterns(ru, en)
    if name_flags:
        qa_score += config.probable_name_overexpansion_penalty
    qa_flags.extend(name_flags)

    # Round-trip distance
    roundtrip_distance: float | None = None
    if roundtrip_text is not None and not is_blank(ru):
        roundtrip_distance = normalized_edit_distance(ru, roundtrip_text)
        if roundtrip_distance > config.roundtrip_far_threshold:
            qa_flags.append("roundtrip_far")
            qa_score += config.roundtrip_far_penalty

    # Cap score at 1.0
    qa_score = min(1.0, qa_score)

    # qa_keep=True for anything that survived fatal checks
    return QAResult(
        qa_keep=True,
        qa_score=round(qa_score, 2),
        qa_flags=qa_flags,
        roundtrip_distance=roundtrip_distance,
    )
