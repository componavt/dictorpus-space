"""Pipeline helpers for translation input preparation."""

from .translation_input import (
    GlossMetadata,
    extract_unique_primary_glosses,
    build_gloss_metadata_map,
    prepare_translation_input,
)
from .vepkar_translation_selection import (
    serialize_task_key,
    parse_serialized_task_key,
    canonical_existing_en,
    has_existing_english,
    build_task_key,
    prepare_meanings_for_translation,
    split_by_existing_en_reuse,
    prepare_translation_input_for_task,
    extract_unique_translation_tasks,
    build_task_metadata_map,
    TranslationTaskMetadata,
    compute_suggested_candidate_index,
)
from .reuse_analysis import (
    ReuseAnalysisResult,
    analyze_missing_en_reuse,
    write_reuse_outputs,
    print_reuse_summary,
)

__all__ = [
    "GlossMetadata",
    "extract_unique_primary_glosses",
    "build_gloss_metadata_map",
    "prepare_translation_input",
    "serialize_task_key",
    "parse_serialized_task_key",
    "canonical_existing_en",
    "has_existing_english",
    "build_task_key",
    "prepare_meanings_for_translation",
    "split_by_existing_en_reuse",
    "prepare_translation_input_for_task",
    "extract_unique_translation_tasks",
    "build_task_metadata_map",
    "TranslationTaskMetadata",
    "compute_suggested_candidate_index",
    "ReuseAnalysisResult",
    "analyze_missing_en_reuse",
    "write_reuse_outputs",
    "print_reuse_summary",
]
