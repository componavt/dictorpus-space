"""Shared meaning preparation for reuse analysis and translation steps.

This module provides neutral helpers that are used by both Step 01 (reuse analysis)
and Step 02 (translation) to prepare meanings for processing.

Key responsibilities:
- Normalizing primary_gloss_ru and task_pos
- Determining has_existing_en
- Creating normalized English comparison values
- Building task keys when part of the shared contract
"""

from __future__ import annotations

import pandas as pd

from src.sem_cat.pipeline.vepkar_translation_selection import (
    serialize_task_key,
    canonical_existing_en,
)


def normalize_text(value: object) -> str:
    """Normalize text by stripping whitespace and collapsing internal spaces.

    Args:
        value: Value to normalize

    Returns:
        Normalized string, or empty string if None/empty
    """
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def prepare_meanings_for_reuse_and_translation(df_meanings: pd.DataFrame) -> pd.DataFrame:
    """Prepare meanings DataFrame for reuse analysis and translation workflow.

    This is the shared preparation logic used by both Step 01 and Step 02.

    Adds derived columns:
    - primary_gloss_ru: normalized primary gloss from meaning_ru
    - meaning_en: filledna with empty string
    - existing_en_norm: normalized existing English value
    - has_primary_gloss_ru: whether primary_gloss_ru is non-empty
    - has_existing_en: whether meaning_en has content after normalization
    - task_key: serialized task key (pos::gloss)

    Filters out rows with empty primary_gloss_ru.

    Args:
        df_meanings: Raw meanings DataFrame from VepKar

    Returns:
        Prepared DataFrame with derived columns, filtered to non-empty primary_gloss_ru
    """
    from src.sem_cat.utils.gloss_normalizer import primary_gloss

    out = df_meanings.copy()

    out["primary_gloss"] = out["meaning_ru"].apply(
        lambda x: primary_gloss(x) if pd.notna(x) else ""
    )
    out["primary_gloss_ru"] = out["primary_gloss"].map(normalize_text)
    out["meaning_en"] = out["meaning_en"].fillna("")
    out["existing_en_norm"] = out["meaning_en"].map(canonical_existing_en)

    out["has_primary_gloss_ru"] = out["primary_gloss_ru"].ne("")
    out = out.loc[out["has_primary_gloss_ru"]].copy()

    out["has_existing_en"] = out["existing_en_norm"].ne("")
    out["task_key"] = [
        serialize_task_key(pos, gloss_ru)
        for pos, gloss_ru in zip(
            out["pos"], out["primary_gloss_ru"], strict=True
        )
    ]
    return out


def prepare_meanings_for_translation(df_meanings: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper; use prepare_meanings_for_reuse_and_translation.

    This wrapper exists only for backward compatibility with Step 02.
    The implementation delegates to the shared neutral function.

    Args:
        df_meanings: Raw meanings DataFrame

    Returns:
        Prepared DataFrame (same as prepare_meanings_for_reuse_and_translation)
    """
    return prepare_meanings_for_reuse_and_translation(df_meanings)
