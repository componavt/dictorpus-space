"""Gloss metadata extraction and translation input preparation.

This module provides task-based helpers for VepKar-aware translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.sem_cat.utils.gloss_normalizer import primary_gloss

TASK_KEY_SEP = "::"


def serialize_task_key(pos: str, meaning_ru: str) -> str:
    """Serialize task key to a stable string format.
    
    Args:
        pos: Part of speech (e.g., "NOUN", "VERB")
        meaning_ru: Russian meaning string
        
    Returns:
        Serialized task key in format "POS::meaning_ru"
    """
    return f"{pos}{TASK_KEY_SEP}{meaning_ru}"


def parse_serialized_task_key(value: str) -> tuple[str, str] | None:
    """Parse a serialized task key back to (pos, meaning_ru) tuple.
    
    Supports both new :: format and legacy \\t format for backward compatibility.
    
    Args:
        value: Serialized task key string
        
    Returns:
        (pos, meaning_ru) tuple or None if parsing fails
    """
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if TASK_KEY_SEP in value:
        pos, meaning_ru = value.split(TASK_KEY_SEP, 1)
        return pos, meaning_ru
    if "\t" in value:
        pos, meaning_ru = value.split("\t", 1)
        return pos, meaning_ru
    return None





def canonical_existing_en(meaning_en: str | None) -> str:
    """Canonicalize existing English value.
    
    Args:
        meaning_en: Raw English meaning value
        
    Returns:
        Stripped English string or empty string
    """
    if not meaning_en or not str(meaning_en).strip():
        return ""
    return str(meaning_en).strip()


def has_existing_english(meaning_en: str | None) -> bool:
    """Check if meaning has existing English translation.
    
    Args:
        meaning_en: Raw English meaning value
        
    Returns:
        True if non-empty after stripping
    """
    return bool(canonical_existing_en(meaning_en))


def build_task_key(pos: str, meaning_ru: str) -> tuple[str, str]:
    """Build task key as (pos, meaning_ru) tuple.
    
    Args:
        pos: Part of speech
        meaning_ru: Full Russian meaning string
        
    Returns:
        (pos, meaning_ru) tuple
    """
    return (pos, meaning_ru)


@dataclass(frozen=True)
class TranslationTaskMetadata:
    """Metadata for a translation task."""
    task_key: str
    meaning_ru: str
    pos: str


def prepare_translation_input_for_task(
    task: TranslationTaskMetadata,
    mode: Literal["raw", "pos"],
) -> str:
    """Prepare translation input for a task.
    
    Args:
        task: Translation task metadata
        mode: One of "raw", "pos"
        
    Returns:
        Input string to send to translator
    """
    if mode == "raw":
        return task.meaning_ru

    pos_str = task.pos

    if mode == "pos":
        return f"{pos_str} | {task.meaning_ru}"

    return task.meaning_ru


def prepare_meanings_for_translation(df_meanings: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper; use prepare_meanings_for_reuse_and_translation.

    This wrapper exists only for backward compatibility.
    The implementation delegates to the shared neutral function.

    Args:
        df_meanings: Raw meanings DataFrame

    Returns:
        Prepared DataFrame (same as prepare_meanings_for_reuse_and_translation)
    """
    from src.sem_cat.pipeline.meaning_preparation import prepare_meanings_for_reuse_and_translation
    return prepare_meanings_for_reuse_and_translation(df_meanings)


def split_by_existing_en_reuse(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into reusable (unambiguous), reusable (ambiguous), and needs-model.
    
    Args:
        df: Prepared DataFrame from prepare_meanings_for_translation
        
    Returns:
        Tuple of (unambiguous, ambiguous, needs_model) DataFrames
    """
    # Group by task_key to analyze reuse patterns
    reusable_unambiguous_parts = []
    reusable_ambiguous_parts = []
    needs_model_parts = []
    
    for task_key, group in df.groupby("task_key", dropna=False, sort=False):
        # Get all existing English values for this task
        existing_en_values = group["meaning_en"].dropna().apply(canonical_existing_en)
        existing_en_values = existing_en_values[existing_en_values.str.len() > 0]
        unique_existing = set(existing_en_values)
        
        # Rows with existing English reuse the existing translation
        has_en_mask = group["has_existing_en"]
        missing_en_mask = ~has_en_mask
        
        if len(unique_existing) == 0:
            # No existing English values - all missing rows go to needs_model
            needs_model_parts.append(group[missing_en_mask])
        elif len(unique_existing) == 1:
            # Exactly one existing English - all missing become unambiguous reuse
            reused_value = next(iter(unique_existing))
            group_copy = group[missing_en_mask].copy()
            group_copy["reused_existing_en"] = reused_value
            reusable_unambiguous_parts.append(group_copy)
        else:
            # Multiple existing English values - ambiguous reuse
            candidates_str = " || ".join(sorted(unique_existing))
            candidates_count = len(unique_existing)
            group_copy = group[missing_en_mask].copy()
            group_copy["existing_en_candidates"] = candidates_str
            group_copy["existing_en_candidate_count"] = candidates_count
            reusable_ambiguous_parts.append(group_copy)
    
    # Combine results
    reusable_unambiguous_df = pd.DataFrame()
    reusable_ambiguous_df = pd.DataFrame()
    needs_model_df = pd.DataFrame()
    
    if reusable_unambiguous_parts:
        reusable_unambiguous_df = pd.concat(reusable_unambiguous_parts, ignore_index=True)
    if reusable_ambiguous_parts:
        reusable_ambiguous_df = pd.concat(reusable_ambiguous_parts, ignore_index=True)
    if needs_model_parts:
        needs_model_df = pd.concat(needs_model_parts, ignore_index=True)
    
    return reusable_unambiguous_df, reusable_ambiguous_df, needs_model_df


def build_task_metadata_map(df: pd.DataFrame) -> dict[str, TranslationTaskMetadata]:
    """Build a map from task_key to TranslationTaskMetadata.
    
    Args:
        df: DataFrame with task_key column
        
    Returns:
        Dict mapping task_key to TranslationTaskMetadata
    """
    metadata_map: dict[str, TranslationTaskMetadata] = {}
    
    if df.empty:
        return metadata_map
    
    for task_key, group in df.groupby("task_key", dropna=False, sort=False):
        first_row = group.iloc[0]
        
        metadata_map[str(task_key)] = TranslationTaskMetadata(
            task_key=str(task_key),
            meaning_ru=first_row.get("meaning_ru"),
            pos=first_row.get("pos"),
        )
    
    return metadata_map


def compute_suggested_candidate_index(existing_en_candidates: str) -> int | None:
    """Compute recommended candidate index for ambiguous tasks.
    
    Returns a 1-based index of the recommended candidate for UI display.
    The first candidate is always recommended.
    
    Args:
        existing_en_candidates: String of candidates separated by " || "
        
    Returns:
        1-based index of recommended candidate, or None if no candidates
    """
    if not isinstance(existing_en_candidates, str):
        return None
    candidates = [x.strip() for x in existing_en_candidates.split(" || ") if x.strip()]
    return 1 if candidates else None


def extract_unique_translation_tasks(
    df: pd.DataFrame,
) -> list[TranslationTaskMetadata]:
    """Extract unique translation tasks from needs-model DataFrame.
    
    Args:
        df: DataFrame from needs_model output (missing meaning_en)
        
    Returns:
        List of unique TranslationTaskMetadata objects
    """
    tasks: list[TranslationTaskMetadata] = []
    
    if df.empty:
        return tasks

    seen_keys = set()
    for task_key in df["task_key"].dropna().unique():
        if task_key in seen_keys:
            continue
        seen_keys.add(task_key)
        
        first_row = df[df["task_key"] == task_key].iloc[0]
        
        task = TranslationTaskMetadata(
            task_key=str(task_key),
            meaning_ru=first_row.get("meaning_ru"),
            pos=first_row.get("pos"),
        )
        tasks.append(task)
    
    return tasks


def build_translation_tasks_from_pos_meaning_ru(
    df: pd.DataFrame,
) -> list[TranslationTaskMetadata]:
    """Convert validated pos_meaning_ru_reader DataFrame to task metadata.

    One input row becomes one TranslationTaskMetadata.
    Preserves file order and does not deduplicate again.

    Args:
        df: DataFrame from read_pos_meaning_ru_tasks with columns ["pos", "meaning_ru"]

    Returns:
        List of TranslationTaskMetadata objects in file order
    """
    tasks: list[TranslationTaskMetadata] = []

    if df.empty:
        return tasks

    for _, row in df.iterrows():
        pos = row.get("pos", "")
        meaning_ru = row.get("meaning_ru", "")

        metadata = TranslationTaskMetadata(
            task_key=serialize_task_key(str(pos), str(meaning_ru)),
            meaning_ru=str(meaning_ru),
            pos=str(pos),
        )
        tasks.append(metadata)

    return tasks
