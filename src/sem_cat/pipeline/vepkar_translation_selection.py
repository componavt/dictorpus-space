"""Gloss metadata extraction and translation input preparation.

This module provides task-based helpers for Step-02 translation workflow.
The helpers build TranslationTaskMetadata objects from the pos_meanings_ru.csv task file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.sem_cat.utils.gloss_normalizer import primary_gloss



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
    """Return (pos, meaning_ru) tuple as task identity.
    
    Args:
        pos: Part of speech
        meaning_ru: Full Russian meaning string
        
    Returns:
        (pos, meaning_ru) tuple
    """
    return (pos, meaning_ru)


@dataclass(frozen=True)
class TranslationTaskMetadata:
    """Metadata for a translation task.
    
    The semantic identity of a translation task is the ordered pair (pos, meaning_ru).
    """
    pos: str
    meaning_ru: str


def prepare_translation_input_for_task(
    task: TranslationTaskMetadata,
) -> str:
    """Prepare translation input for a task.
    
    Fixed input format: POS | meaning_ru
    
    Args:
        task: Translation task metadata
        
    Returns:
        Input string to send to translator
    """
    return f"{task.pos} | {task.meaning_ru}"


def prepare_meanings_for_translation(df_meanings: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper; use prepare_meanings_for_reuse_and_translation.

    This wrapper exists only for backward compatibility with reuse analysis.
    The implementation delegates to the shared neutral function.

    Args:
        df_meanings: Raw meanings DataFrame

    Returns:
        Prepared DataFrame (same as prepare_meanings_for_reuse_and_translation)
    """
    from src.sem_cat.pipeline.meaning_preparation import prepare_meanings_for_reuse_and_translation
    return prepare_meanings_for_reuse_and_translation(df_meanings)


def build_task_metadata_map(df: pd.DataFrame) -> dict[tuple[str, str], TranslationTaskMetadata]:
    """Build a map from (pos, meaning_ru) tuple to TranslationTaskMetadata.
    
    Args:
        df: DataFrame with pos and meaning_ru columns
        
    Returns:
        Dict mapping (pos, meaning_ru) tuple to TranslationTaskMetadata
    """
    metadata_map: dict[tuple[str, str], TranslationTaskMetadata] = {}
    
    if df.empty:
        return metadata_map
    
    for (pos, meaning_ru), group in df.groupby(["pos", "meaning_ru"], dropna=False, sort=False):
        first_row = group.iloc[0]
        
        metadata_map[(pos, meaning_ru)] = TranslationTaskMetadata(
            pos=first_row.get("pos"),
            meaning_ru=first_row.get("meaning_ru"),
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
            pos=str(pos),
            meaning_ru=str(meaning_ru),
        )
        tasks.append(metadata)

    return tasks
