"""Canonical output row builder for translation CSV files."""

from __future__ import annotations

from src.sem_cat.qa.translation_qa import QAResult

CANONICAL_COLUMNS = [
    "pos",
    "meaning_ru",
    "meaning_en",
    "qa_keep",
    "qa_score",
    "qa_flags",
    "meaning_ru_back",
    "roundtrip_distance",
]


def build_translation_row(
    pos: str,
    meaning_ru: str,
    meaning_en: str,
    qa_result: QAResult,
    meaning_ru_back: str | None = None,
    roundtrip_distance: float | None = None,
) -> dict[str, object]:
    """Build a single canonical output row for the translation CSV.

    All columns are always present to ensure stable schema.
    
    Args:
        pos: Part of speech (e.g., NOUN, VERB)
        meaning_ru: Full Russian meaning string (task identity)
        meaning_en: Translated English meaning
        qa_result: Quality analysis results
        meaning_ru_back: Optional reverse translation for QA
        roundtrip_distance: Optional distance metric for round-trip QA
    
    Returns:
        Dictionary with all canonical columns
    """
    return {
        "pos": pos,
        "meaning_ru": meaning_ru,
        "meaning_en": meaning_en,
        "qa_keep": qa_result.qa_keep,
        "qa_score": qa_result.qa_score,
        "qa_flags": ";".join(qa_result.qa_flags) if qa_result.qa_flags else "",
        "meaning_ru_back": meaning_ru_back if meaning_ru_back else "",
        "roundtrip_distance": (
            round(roundtrip_distance, 3)
            if roundtrip_distance is not None
            else ""
        ),
    }
