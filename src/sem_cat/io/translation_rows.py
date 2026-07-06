"""Canonical output row builder for translation CSV files."""

from __future__ import annotations

from src.sem_cat.qa.translation_qa import QAResult
from src.sem_cat.utils.text_utils import is_blank, token_count

QA_VERSION = "v2"

CANONICAL_COLUMNS = [
    # Core translation
    "gloss_ru",
    "gloss_en",
    # Task metadata
    "task_key",
    "task_pos",
    # QA
    "qa_keep",
    "qa_score",
    "qa_flags",
    "qa_version",
    "primary_gloss_ru",
    # Model metadata
    "model_key",
    "model_name",
    "backend_family",
    # Input metadata
    "translation_input_mode",
    "input_text_used",
    # Context hints
    "pos_hint",
    "meaning_hint",
    "source_count",
    # Round-trip
    "gloss_ru_back",
    "roundtrip_distance",
    # Token counts
    "is_singleword_ru",
    "input_token_count",
    "output_token_count",
]


def build_translation_row(
    gloss_ru: str,
    gloss_en: str,
    qa_result: QAResult,
    model_key: str,
    model_name: str,
    backend_family: str,
    translation_input_mode: str,
    input_text_used: str,
    pos_hint: str | None = None,
    meaning_hint: str | None = None,
    source_count: int | None = None,
    gloss_ru_back: str | None = None,
    task_key: str | None = None,
    task_pos: str | None = None,
) -> dict[str, object]:
    """Build a single canonical output row for the translation CSV.

    All columns are always present to ensure stable schema.
    """
    ru_tokens = token_count(gloss_ru) if not is_blank(gloss_ru) else 0
    en_tokens = token_count(gloss_en) if not is_blank(gloss_en) else 0

    return {
        "gloss_ru": gloss_ru,
        "gloss_en": gloss_en,
        "task_key": task_key or "",
        "task_pos": task_pos or "",
        "qa_keep": qa_result.qa_keep,
        "qa_score": qa_result.qa_score,
        "qa_flags": ";".join(qa_result.qa_flags) if qa_result.qa_flags else "",
        "qa_version": QA_VERSION,
        "primary_gloss_ru": gloss_ru,
        "model_key": model_key,
        "model_name": model_name,
        "backend_family": backend_family,
        "translation_input_mode": translation_input_mode,
        "input_text_used": input_text_used,
        "pos_hint": pos_hint or "",
        "meaning_hint": meaning_hint or "",
        "source_count": source_count if source_count is not None else 0,
        "gloss_ru_back": gloss_ru_back if gloss_ru_back else "",
        "roundtrip_distance": (
            round(qa_result.roundtrip_distance, 3)
            if qa_result.roundtrip_distance is not None
            else ""
        ),
        "is_singleword_ru": ru_tokens == 1,
        "input_token_count": ru_tokens,
        "output_token_count": en_tokens,
    }
