"""Gloss metadata extraction and translation input preparation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.sem_cat.utils.gloss_normalizer import primary_gloss


@dataclass
class GlossMetadata:
    """Metadata for a single primary gloss."""
    gloss_ru: str
    dominant_pos: str | None
    meaning_hint: str | None
    source_count: int


def extract_unique_primary_glosses(df_meanings: pd.DataFrame) -> list[str]:
    """Extract unique primary glosses from a meanings DataFrame."""
    unique_glosses_arr = (
        df_meanings["meaning_ru"]
        .dropna()
        .apply(primary_gloss)
        .pipe(lambda s: s[s.str.len() > 0])
        .unique()
    )
    return unique_glosses_arr.tolist()


def build_gloss_metadata_map(df_meanings: pd.DataFrame) -> dict[str, GlossMetadata]:
    """Build a mapping from primary gloss to GlossMetadata."""
    df_work = df_meanings.copy()
    df_work["primary_gloss"] = df_work["meaning_ru"].apply(
        lambda x: primary_gloss(x) if pd.notna(x) else ""
    )
    df_work = df_work[df_work["primary_gloss"].str.len() > 0]

    metadata_map: dict[str, GlossMetadata] = {}
    for gloss, group in df_work.groupby("primary_gloss"):
        pos_counts = group["pos"].dropna().value_counts()
        dominant_pos = pos_counts.index[0] if len(pos_counts) > 0 else None

        meaning_candidates = group["meaning_ru"].dropna()
        meaning_candidates = meaning_candidates[
            meaning_candidates.str.contains(gloss, regex=False)
        ]
        if len(meaning_candidates) > 0:
            meaning_hint = meaning_candidates.loc[
                meaning_candidates.str.len().idxmin()
            ]
        else:
            meaning_hint = (
                group["meaning_ru"].dropna().iloc[0]
                if len(group["meaning_ru"].dropna()) > 0
                else None
            )

        metadata_map[gloss] = GlossMetadata(
            gloss_ru=gloss,
            dominant_pos=dominant_pos,
            meaning_hint=meaning_hint,
            source_count=len(group),
        )

    return metadata_map


def prepare_translation_input(
    gloss_ru: str,
    mode: str,
    metadata: GlossMetadata | None = None,
) -> str:
    """Prepare the input string for translation based on the mode.

    Args:
        gloss_ru: The Russian gloss
        mode: One of 'raw', 'pos', 'pos_meaning'
        metadata: Optional gloss metadata for context

    Returns:
        Input string to send to translator
    """
    if mode == "raw":
        return gloss_ru

    pos = metadata.dominant_pos if metadata else None
    meaning = metadata.meaning_hint if metadata else None

    if mode == "pos":
        pos_str = pos if pos else "UNKNOWN"
        return f"{pos_str} | {gloss_ru}"

    if mode == "pos_meaning":
        pos_str = pos if pos else "UNKNOWN"
        meaning_str = meaning if meaning else ""
        return f"{pos_str} | {gloss_ru} | {meaning_str}"

    return gloss_ru
