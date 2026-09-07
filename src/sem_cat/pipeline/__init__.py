"""Pipeline helpers for translation input preparation."""

from .vepkar_translation_selection import (
    canonical_existing_en,
    has_existing_english,
    build_task_key,
    prepare_meanings_for_translation,
    prepare_translation_input_for_task,
    build_translation_tasks_from_pos_meaning_ru,
    TranslationTaskMetadata,
)
from .reuse_analysis import (
    ReuseAnalysisResult,
    analyze_missing_en_reuse,
    write_reuse_outputs,
    print_reuse_summary,
)

__all__ = [
    "canonical_existing_en",
    "has_existing_english",
    "build_task_key",
    "prepare_meanings_for_translation",
    "prepare_translation_input_for_task",
    "build_translation_tasks_from_pos_meaning_ru",
    "TranslationTaskMetadata",
    "ReuseAnalysisResult",
    "analyze_missing_en_reuse",
    "write_reuse_outputs",
    "print_reuse_summary",
]
