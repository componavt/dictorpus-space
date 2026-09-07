"""Canonical output row builder for translation CSV files."""

from __future__ import annotations

from src.sem_cat.qa.translation_qa import QAResult
from src.sem_cat.utils.text_utils import is_blank, token_count

QA_VERSION = "v3"

CANONICAL_COLUMNS = [
    "pos",
    "meaning_ru",
    "meaning_en",
    "qa_keep",
    "qa_score",
    "qa_flags",
    "qa_version",
    "model_key",
    "model_name",
    "backend_family",
    "translation_input_mode",
    "input_text_used",
    "meaning_ru_back",
    "roundtrip_distance",
    "is_single_word_ru",
    "input_token_count",
    "output_token_count",
]


def build_translation_row(
    pos: str,
    meaning_ru: str,
    meaning_en: str,
    qa_result: QAResult,
    model_key: str,
    model_name: str,
    backend_family: str,
    translation_input_mode: str,
    input_text_used: str,
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
        model_key: Model identifier
        model_name: Human-readable model name
        backend_family: Backend type (e.g., "marian", "google", "nllb", "hf_causal")
        translation_input_mode: How input was prepared ("raw" or "pos")
        input_text_used: Exact text sent to translator
        meaning_ru_back: Optional reverse translation for QA
        roundtrip_distance: Optional distance metric for round-trip QA
    
    Returns:
        Dictionary with all canonical columns
    """
    ru_tokens = token_count(meaning_ru) if not is_blank(meaning_ru) else 0
    en_tokens = token_count(meaning_en) if not is_blank(meaning_en) else 0

    return {
        "pos": pos,
        "meaning_ru": meaning_ru,
        "meaning_en": meaning_en,
        "qa_keep": qa_result.qa_keep,
        "qa_score": qa_result.qa_score,
        "qa_flags": ";".join(qa_result.qa_flags) if qa_result.qa_flags else "",
        "qa_version": QA_VERSION,
        "model_key": model_key,
        "model_name": model_name,
        "backend_family": backend_family,
        "translation_input_mode": translation_input_mode,
        "input_text_used": input_text_used,
        "meaning_ru_back": meaning_ru_back if meaning_ru_back else "",
        "roundtrip_distance": (
            round(roundtrip_distance, 3)
            if roundtrip_distance is not None
            else ""
        ),
        "is_single_word_ru": ru_tokens == 1,
        "input_token_count": ru_tokens,
        "output_token_count": en_tokens,
    }
