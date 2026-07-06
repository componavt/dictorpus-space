"""
Reuse analysis for missing-English rows grouped by (pos, primary_gloss_ru).

This module provides pure helpers for analyzing which missing-English rows
can reuse existing English translations based on exact grouping by
(pos, primary_gloss_ru).

Output files are written to data/sem_cat/2translate/:
- missing_en_reusable_unambiguous_pos_gloss_ru.csv
- missing_en_reusable_ambiguous_pos_gloss_ru.csv
- missing_en_reusable_unambiguous_pos_gloss_ru_summary.csv
- missing_en_reusable_ambiguous_pos_gloss_ru_summary.csv
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

TOP_LEVEL_CANDIDATE_SEP = " || "


@dataclass(frozen=True)
class ReuseAnalysisResult:
    """Result of reuse analysis for missing-English rows."""
    missing_en_reusable_unambiguous: pd.DataFrame
    missing_en_reusable_ambiguous: pd.DataFrame
    missing_en_without_reuse: pd.DataFrame
    unambiguous_summary: pd.DataFrame
    ambiguous_summary: pd.DataFrame
    stats: dict[str, int]


def is_nonblank_text(value: object) -> bool:
    """Check if value is a non-blank string.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is a non-empty string after stripping
    """
    return isinstance(value, str) and value.strip() != ""


def normalize_existing_en(value: object) -> str | None:
    """Normalize an existing English value.
    
    Args:
        value: Raw English value
        
    Returns:
        Normalized string with whitespace collapsed, or None if blank
    """
    if not is_nonblank_text(value):
        return None
    return " ".join(str(value).strip().split())


def build_pos_gloss_ru_key(pos: object, primary_gloss_ru: object) -> str:
    """Build a stable key from (pos, primary_gloss_ru).
    
    Args:
        pos: Part of speech
        primary_gloss_ru: Russian primary gloss
        
    Returns:
        Serialized key in format "pos::gloss"
    """
    pos_s = str(pos or "").strip()
    gloss_s = str(primary_gloss_ru or "").strip()
    return f"{pos_s}::{gloss_s}"


def distinct_existing_en_candidates(group_df: pd.DataFrame) -> list[str]:
    """Extract distinct existing English candidates from a group, sorted alphabetically.
    
    Args:
        group_df: DataFrame group with meaning_en column
        
    Returns:
        List of distinct normalized English values in sorted alphabetical order
    """
    seen: set[str] = set()
    for value in group_df["meaning_en"].tolist():
        norm = normalize_existing_en(value)
        if norm is None:
            continue
        seen.add(norm)
    return sorted(seen)


def analyze_missing_en_reuse(df: pd.DataFrame) -> ReuseAnalysisResult:
    """Analyze missing-English rows for reuse evidence grouped by (pos, primary_gloss_ru).
    
    Args:
        df: Prepared meanings DataFrame with columns:
            - lang, lemma, pos, primary_gloss_ru, meaning_en
            - task_key, task_pos, primary_gloss_ru (from prepare_meanings_for_translation)
            
    Returns:
        ReuseAnalysisResult with all output DataFrames and stats
    """
    work = df.copy()

    work["task_pos"] = work["pos"].fillna("").astype(str).str.strip()
    work["primary_gloss_ru"] = work["primary_gloss_ru"].fillna("").astype(str).str.strip()
    work["has_primary_gloss_ru"] = work["primary_gloss_ru"] != ""
    work["existing_en_norm"] = work["meaning_en"].map(normalize_existing_en)
    work["has_existing_en"] = work["existing_en_norm"].notna()

    work = work[work["has_primary_gloss_ru"]].copy()
    work["pos_gloss_ru_key"] = work.apply(
        lambda row: build_pos_gloss_ru_key(row["task_pos"], row["primary_gloss_ru"]),
        axis=1,
    )

    missing_df = work[~work["has_existing_en"]].copy()
    if missing_df.empty:
        empty_row_cols = [
            "pos_gloss_ru_key",
            "task_pos",
            "primary_gloss_ru",
            "lang",
            "lemma",
            "existing_en_candidates",
            "existing_en_candidate_count",
            "missing_row_count_for_pos_gloss_ru",
            "existing_en_row_count_for_pos_gloss_ru",
        ]
        empty_unamb = pd.DataFrame(columns=empty_row_cols)
        empty_amb = pd.DataFrame(columns=empty_row_cols + ["suggested_candidate_index"])
        return ReuseAnalysisResult(
            missing_en_reusable_unambiguous=empty_unamb,
            missing_en_reusable_ambiguous=empty_amb,
            missing_en_without_reuse=missing_df.copy(),
            unambiguous_summary=build_reuse_summary(empty_unamb, include_suggested_index=True),
            ambiguous_summary=build_reuse_summary(empty_amb, include_suggested_index=True),
            stats={
                "rows_with_primary_gloss_ru": int(len(work)),
                "rows_with_existing_en": int(work["has_existing_en"].sum()),
                "rows_missing_en": 0,
                "rows_reusable_unambiguous": 0,
                "rows_reusable_ambiguous": 0,
                "rows_missing_en_without_reuse": 0,
                "pos_gloss_ru_unambiguous_count": 0,
                "pos_gloss_ru_ambiguous_count": 0,
                "pos_gloss_ru_without_reuse_count": 0,
            },
        )

    groups: list[dict[str, object]] = []
    for (task_pos, primary_gloss_ru), full_group in work.groupby(
        ["task_pos", "primary_gloss_ru"], sort=False, dropna=False
    ):
        missing_group = full_group[~full_group["has_existing_en"]].copy()
        if missing_group.empty:
            continue

        existing_group = full_group[full_group["has_existing_en"]].copy()
        candidates = distinct_existing_en_candidates(existing_group)
        candidate_count = len(candidates)
        if candidate_count == 0:
            groups.append({
                "kind": "no_reuse",
                "rows": missing_group,
                "candidates": [],
                "existing_en_row_count": len(existing_group),
            })
            continue

        base = missing_group.copy()
        base["existing_en_candidates"] = TOP_LEVEL_CANDIDATE_SEP.join(candidates)
        base["existing_en_candidate_count"] = candidate_count
        base["missing_row_count_for_pos_gloss_ru"] = len(missing_group)
        base["existing_en_row_count_for_pos_gloss_ru"] = len(existing_group)

        if candidate_count == 1:
            base["suggested_candidate_index"] = 1
            groups.append({"kind": "unambiguous", "rows": base})
        else:
            base["suggested_candidate_index"] = 1
            groups.append({"kind": "ambiguous", "rows": base})

    unambiguous_parts = [g["rows"] for g in groups if g["kind"] == "unambiguous"]
    ambiguous_parts = [g["rows"] for g in groups if g["kind"] == "ambiguous"]
    no_reuse_parts = [g["rows"] for g in groups if g["kind"] == "no_reuse"]

    unambiguous_df = (
        pd.concat(unambiguous_parts, ignore_index=True)
        if unambiguous_parts
        else pd.DataFrame(columns=[
            "pos_gloss_ru_key",
            "task_pos",
            "primary_gloss_ru",
            "lang",
            "lemma",
            "existing_en_candidates",
            "existing_en_candidate_count",
            "missing_row_count_for_pos_gloss_ru",
            "existing_en_row_count_for_pos_gloss_ru",
            "suggested_candidate_index",
        ])
    )
    ambiguous_df = (
        pd.concat(ambiguous_parts, ignore_index=True)
        if ambiguous_parts
        else pd.DataFrame(columns=[
            "pos_gloss_ru_key",
            "task_pos",
            "primary_gloss_ru",
            "lang",
            "lemma",
            "existing_en_candidates",
            "existing_en_candidate_count",
            "missing_row_count_for_pos_gloss_ru",
            "existing_en_row_count_for_pos_gloss_ru",
            "suggested_candidate_index",
        ])
    )
    no_reuse_df = (
        pd.concat(no_reuse_parts, ignore_index=True)
        if no_reuse_parts
        else pd.DataFrame(columns=missing_df.columns.tolist())
    )

    return ReuseAnalysisResult(
        missing_en_reusable_unambiguous=unambiguous_df,
        missing_en_reusable_ambiguous=ambiguous_df,
        missing_en_without_reuse=no_reuse_df,
        unambiguous_summary=build_reuse_summary(unambiguous_df, include_suggested_index=True),
        ambiguous_summary=build_reuse_summary(ambiguous_df, include_suggested_index=True),
        stats={
            "rows_with_primary_gloss_ru": int(len(work)),
            "rows_with_existing_en": int(work["has_existing_en"].sum()),
            "rows_missing_en": int((~work["has_existing_en"]).sum()),
            "rows_reusable_unambiguous": int(len(unambiguous_df)),
            "rows_reusable_ambiguous": int(len(ambiguous_df)),
            "rows_missing_en_without_reuse": int(len(no_reuse_df)),
            "pos_gloss_ru_unambiguous_count": int(unambiguous_df["pos_gloss_ru_key"].nunique()) if not unambiguous_df.empty else 0,
            "pos_gloss_ru_ambiguous_count": int(ambiguous_df["pos_gloss_ru_key"].nunique()) if not ambiguous_df.empty else 0,
            "pos_gloss_ru_without_reuse_count": int(no_reuse_df["pos_gloss_ru_key"].nunique()) if not no_reuse_df.empty else 0,
        },
    )


def build_reuse_summary(df: pd.DataFrame, *, include_suggested_index: bool) -> pd.DataFrame:
    """Build summary DataFrame with one row per unique (pos, primary_gloss_ru) group.
    
    Args:
        df: Row-level DataFrame from reuse analysis (unambiguous or ambiguous)
        include_suggested_index: Whether to include suggested_candidate_index column
        
    Returns:
        Summary DataFrame with aggregated info per group
    """
    if df.empty:
        cols = [
            "pos_gloss_ru_key",
            "task_pos",
            "primary_gloss_ru",
            "existing_en_candidates",
            "existing_en_candidate_count",
            "missing_row_count",
            "existing_en_row_count",
            "langs",
            "example_lemma",
        ]
        if include_suggested_index:
            cols.append("suggested_candidate_index")
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, object]] = []
    for _, group in df.groupby(
        ["pos_gloss_ru_key", "task_pos", "primary_gloss_ru"], sort=False, dropna=False
    ):
        candidates_text = (
            str(group["existing_en_candidates"].iloc[0])
            if "existing_en_candidates" in group.columns
            else ""
        )
        row = {
            "pos_gloss_ru_key": group["pos_gloss_ru_key"].iloc[0],
            "task_pos": group["task_pos"].iloc[0],
            "primary_gloss_ru": group["primary_gloss_ru"].iloc[0],
            "existing_en_candidates": candidates_text,
            "existing_en_candidate_count": int(group["existing_en_candidate_count"].iloc[0]),
            "missing_row_count": len(group),
            "existing_en_row_count": int(group["existing_en_row_count_for_pos_gloss_ru"].iloc[0]),
            "langs": " || ".join(
                sorted({str(x) for x in group["lang"].dropna().tolist() if str(x).strip()})
            ),
            "example_lemma": str(group["lemma"].iloc[0]) if "lemma" in group.columns else "",
        }
        if include_suggested_index:
            row["suggested_candidate_index"] = (
                1 if candidates_text.strip() else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_reuse_outputs(result: ReuseAnalysisResult, translate_dir: Path) -> None:
    """Write reuse output CSV files.
    
    Writes all four CSV files, including empty files with headers.
    
    Args:
        result: ReuseAnalysisResult from analyze_missing_en_reuse
        translate_dir: Output directory (typically data/sem_cat/2translate)
    """
    translate_dir.mkdir(parents=True, exist_ok=True)

    result.missing_en_reusable_unambiguous.to_csv(
        translate_dir / "missing_en_reusable_unambiguous_pos_gloss_ru.csv",
        index=False,
    )
    result.missing_en_reusable_ambiguous.to_csv(
        translate_dir / "missing_en_reusable_ambiguous_pos_gloss_ru.csv",
        index=False,
    )
    result.unambiguous_summary.to_csv(
        translate_dir / "missing_en_reusable_unambiguous_pos_gloss_ru_summary.csv",
        index=False,
    )
    result.ambiguous_summary.to_csv(
        translate_dir / "missing_en_reusable_ambiguous_pos_gloss_ru_summary.csv",
        index=False,
    )


def print_reuse_summary(stats: dict[str, int]) -> None:
    """Print reuse analysis summary to console.
    
    Args:
        stats: Stats dict from ReuseAnalysisResult.stats
    """
    print("=" * 60)
    print("Missing-English reuse analysis by (pos, primary_gloss_ru)")
    print("=" * 60)
    print()
    print(f"Rows with non-empty primary_gloss_ru:        {stats['rows_with_primary_gloss_ru']}")
    print(f"Rows with existing human English:            {stats['rows_with_existing_en']}")
    print(f"Rows missing English:                        {stats['rows_missing_en']}")
    print()
    print("Among rows missing English:")
    print(f"  Reusable, one EN variant:                  {stats['rows_reusable_unambiguous']}")
    print(f"  Reusable, multiple EN variants:            {stats['rows_reusable_ambiguous']}")
    print(f"  No reusable EN evidence:                   {stats['rows_missing_en_without_reuse']}")
    print()
    print("Unique (pos, primary_gloss_ru) groups among rows missing English:")
    print(f"  Reusable, one EN variant:                  {stats['pos_gloss_ru_unambiguous_count']}")
    print(f"  Reusable, multiple EN variants:            {stats['pos_gloss_ru_ambiguous_count']}")
    print(f"  No reusable EN evidence:                   {stats['pos_gloss_ru_without_reuse_count']}")
