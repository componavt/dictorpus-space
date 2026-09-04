"""
Reuse analysis for missing-English rows grouped by (pos, meaning_ru).

This module provides pure helpers for analyzing which missing-English rows
can reuse existing English translations based on exact grouping by
(pos, meaning_ru).

Output files are written to data/sem_cat/2translate/:
- needs_translation_no_reuse.csv
- pos_meanings_ru.csv
- reusable_english/one_english.csv
- reusable_english/one_english_summary.csv
- reusable_english/several_english.csv
- reusable_english/several_english_summary.csv
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.sem_cat.io import POS_MEANINGS_RU_COLUMNS

TOP_LEVEL_CANDIDATE_SEP = " || "

CONCEPT_CATEGORY_AUDIT_COLUMNS = [
    "id",
    "meaning_id",
    "lemma_id",
    "lemma",
    "lang",
    "pos",
    "meaning_ru",
    "concept_id",
    "category_id",
]

CORE_ROW_LEVEL_COLUMNS = [
    "id",
    "meaning_id",
    "lemma_id",
    "lemma",
    "lang",
    "pos",
    "meaning_ru",
]

NO_REUSE_ROW_LEVEL_COLUMNS = [
    *CORE_ROW_LEVEL_COLUMNS,
]

UNAMBIGUOUS_ROW_LEVEL_COLUMNS = [
    *CORE_ROW_LEVEL_COLUMNS,
    "existing_en_candidates",
    "existing_en_candidate_count",
    "missing_row_count",
    "existing_en_row_count",
]

AMBIGUOUS_ROW_LEVEL_COLUMNS = [
    *UNAMBIGUOUS_ROW_LEVEL_COLUMNS,
    "suggested_candidate_index",
]

UNAMBIGUOUS_SUMMARY_COLUMNS = [
    "pos",
    "meaning_ru",
    "existing_en_candidates",
    "existing_en_candidate_count",
    "missing_row_count",
    "existing_en_row_count",
    "missing_langs",
    "existing_en_langs",
    "example_missing_lemma",
]

AMBIGUOUS_SUMMARY_COLUMNS = [
    *UNAMBIGUOUS_SUMMARY_COLUMNS,
    "suggested_candidate_index",
]


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure DataFrame has all specified columns, adding missing ones as empty strings."""
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            result[col] = ""
    return result[cols].copy()


@dataclass(frozen=True)
class ReuseAnalysisResult:
    """Result of reuse analysis for missing-English rows."""
    missing_en_reusable_unambiguous: pd.DataFrame
    missing_en_reusable_ambiguous: pd.DataFrame
    missing_en_without_reuse: pd.DataFrame
    concept_category_without_english: pd.DataFrame
    invalid_concept_category_pairs: pd.DataFrame
    unambiguous_summary: pd.DataFrame
    ambiguous_summary: pd.DataFrame
    stats: dict[str, int]
    per_lang_stats: dict[str, int] | None = None
    pos_meanings_ru: pd.DataFrame = None  # type: ignore


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


def join_sorted_nonblank(values: pd.Series) -> str:
    items = {
        str(value).strip()
        for value in values.tolist()
        if value is not None and str(value).strip()
    }
    return " || ".join(sorted(items))


def first_nonblank(values: pd.Series) -> str:
    for value in values.tolist():
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return ""


def _is_blank_series(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def count_unique_pos_meaning_groups(df: pd.DataFrame) -> int:
    """Return the number of exact unique (pos, meaning_ru) groups."""
    if df.empty:
        return 0
    return int(
        df[["pos", "meaning_ru"]]
        .drop_duplicates()
        .shape[0]
    )


def analyze_missing_en_reuse(df: pd.DataFrame) -> ReuseAnalysisResult:
    """Analyze missing-English rows for reuse evidence grouped by (pos, meaning_ru).
    
    Args:
        df: Prepared meanings DataFrame with columns:
            - lang, lemma, pos, meaning_ru, meaning_en
            - meaning_ru (from prepare_meanings_for_translation)
            
    Returns:
        ReuseAnalysisResult with all output DataFrames and stats
    """
    work = df.copy()

    work["meaning_ru"] = work["meaning_ru"].fillna("").astype(str).str.strip()
    work["existing_en_norm"] = work["meaning_en"].map(normalize_existing_en)
    work["has_existing_en"] = work["existing_en_norm"].notna()

    missing_df = work[~work["has_existing_en"]].copy()
    if missing_df.empty:
        empty_unamb = pd.DataFrame(columns=UNAMBIGUOUS_ROW_LEVEL_COLUMNS)
        empty_amb = pd.DataFrame(columns=AMBIGUOUS_ROW_LEVEL_COLUMNS)
        empty_unamb_sum = pd.DataFrame(columns=UNAMBIGUOUS_SUMMARY_COLUMNS)
        empty_amb_sum = pd.DataFrame(columns=AMBIGUOUS_SUMMARY_COLUMNS)
        empty_pos_meanings = pd.DataFrame(columns=POS_MEANINGS_RU_COLUMNS)
        empty_audit = pd.DataFrame(columns=CONCEPT_CATEGORY_AUDIT_COLUMNS)
        return ReuseAnalysisResult(
            missing_en_reusable_unambiguous=empty_unamb,
            missing_en_reusable_ambiguous=empty_amb,
            missing_en_without_reuse=empty_unamb,
            concept_category_without_english=empty_audit,
            invalid_concept_category_pairs=empty_audit,
            unambiguous_summary=empty_unamb_sum,
            ambiguous_summary=empty_amb_sum,
            stats={
                "rows_with_meaning_ru": int(len(work)),
                "rows_with_existing_en": int(work["has_existing_en"].sum()),
                "rows_missing_en": 0,
                "rows_reusable_unambiguous": 0,
                "rows_reusable_ambiguous": 0,
                "rows_missing_en_without_reuse": 0,
                "rows_concept_covered_skip": 0,
                "rows_invalid_concept_category_pair": 0,
                "unambiguous_group_count": 0,
                "ambiguous_group_count": 0,
                "no_reuse_group_count": 0,
            },
            per_lang_stats=None,
            pos_meanings_ru=empty_pos_meanings,
        )

    has_concept_id = "concept_id" in missing_df.columns
    has_category_id = "category_id" in missing_df.columns

    if has_concept_id and has_category_id:
        concept_blank = _is_blank_series(missing_df["concept_id"])
        category_blank = _is_blank_series(missing_df["category_id"])

        translatable_mask = concept_blank & category_blank
        concept_covered_mask = ~concept_blank & ~category_blank
        invalid_pair_mask = concept_blank ^ category_blank
    else:
        # If concept_id or category_id columns are missing, treat all rows as translatable
        translatable_mask = pd.Series([True] * len(missing_df), index=missing_df.index)
        concept_covered_mask = pd.Series([False] * len(missing_df), index=missing_df.index)
        invalid_pair_mask = pd.Series([False] * len(missing_df), index=missing_df.index)

    # Add classification flags to missing_df for use in the loop
    missing_df = missing_df.copy()
    missing_df["_is_translatable"] = translatable_mask.values
    missing_df["_is_concept_covered"] = concept_covered_mask.values
    missing_df["_is_invalid_pair"] = invalid_pair_mask.values

    translatable_df = missing_df[missing_df["_is_translatable"]].copy()
    concept_covered_df = missing_df[missing_df["_is_concept_covered"]].copy()
    invalid_pair_df = missing_df[missing_df["_is_invalid_pair"]].copy()

    # Copy flags to work for use in the loop
    work["_is_translatable"] = work.index.isin(translatable_df.index)
    work["_is_concept_covered"] = work.index.isin(concept_covered_df.index)
    work["_is_invalid_pair"] = work.index.isin(invalid_pair_df.index)

    groups: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    pos_meanings_records: list[dict[str, object]] = []
    for (pos, meaning_ru), full_group in work.groupby(
        ["pos", "meaning_ru"], sort=False, dropna=False
    ):
        missing_group = full_group[~full_group["has_existing_en"]].copy()
        if missing_group.empty:
            continue

        # Check if this group has any translatable rows
        has_translatable = missing_group["_is_translatable"].any()
        if not has_translatable:
            # All missing rows in this group are concept-covered or invalid-pair
            continue

        existing_group = full_group[full_group["has_existing_en"]].copy()
        candidates = distinct_existing_en_candidates(existing_group)
        candidate_count = len(candidates)
        
        # Only process translatable rows
        translatable_missing = missing_group[missing_group["_is_translatable"]].copy()
        
        if candidate_count == 0:
            no_reuse_rows = translatable_missing.copy()

            groups.append({
                "kind": "no_reuse",
                "rows": no_reuse_rows,
                "candidates": [],
                "existing_en_row_count": len(existing_group),
            })

            for raw_pos, raw_meaning_ru, meaning_id, lemma_id in zip(
                translatable_missing["pos"],
                translatable_missing["meaning_ru"],
                translatable_missing.get("meaning_id", pd.Series([None] * len(translatable_missing))),
                translatable_missing.get("lemma_id", pd.Series([None] * len(translatable_missing))),
            ):
                if pd.isna(raw_pos) or not str(raw_pos).strip():
                    warnings.warn(
                        (
                            "No-reuse translation input has blank raw pos: "
                            f"meaning_id={meaning_id!r}, "
                            f"lemma_id={lemma_id!r}"
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                pos_meanings_records.append(
                    {
                        "pos": raw_pos if pd.notna(raw_pos) else "",
                        "meaning_ru": raw_meaning_ru if pd.notna(raw_meaning_ru) else "",
                    }
                )
            continue

        base = translatable_missing.copy()
        base["existing_en_candidates"] = TOP_LEVEL_CANDIDATE_SEP.join(candidates)
        base["existing_en_candidate_count"] = candidate_count
        base["missing_row_count"] = len(translatable_missing)
        base["existing_en_row_count"] = len(existing_group)

        missing_langs = join_sorted_nonblank(translatable_missing["lang"])
        existing_en_langs = join_sorted_nonblank(existing_group["lang"])
        example_missing_lemma = first_nonblank(translatable_missing["lemma"])

        summary_record = {
            "pos": pos,
            "meaning_ru": meaning_ru,
            "existing_en_candidates": TOP_LEVEL_CANDIDATE_SEP.join(candidates),
            "existing_en_candidate_count": candidate_count,
            "missing_row_count": len(translatable_missing),
            "existing_en_row_count": len(existing_group),
            "missing_langs": missing_langs,
            "existing_en_langs": existing_en_langs,
            "example_missing_lemma": example_missing_lemma,
        }

        if candidate_count == 1:
            groups.append({"kind": "unambiguous", "rows": base})
            summary_record["suggested_candidate_index"] = None
            summary_records.append(summary_record)
        else:
            base["suggested_candidate_index"] = 1
            groups.append({"kind": "ambiguous", "rows": base})
            summary_record["suggested_candidate_index"] = 1
            summary_records.append(summary_record)

    unambiguous_parts = [g["rows"] for g in groups if g["kind"] == "unambiguous"]
    ambiguous_parts = [g["rows"] for g in groups if g["kind"] == "ambiguous"]
    no_reuse_parts = [g["rows"] for g in groups if g["kind"] == "no_reuse"]

    unambiguous_df = (
        ensure_columns(pd.concat(unambiguous_parts, ignore_index=True), UNAMBIGUOUS_ROW_LEVEL_COLUMNS)[UNAMBIGUOUS_ROW_LEVEL_COLUMNS]
        if unambiguous_parts
        else pd.DataFrame(columns=UNAMBIGUOUS_ROW_LEVEL_COLUMNS)
    )
    ambiguous_df = (
        ensure_columns(pd.concat(ambiguous_parts, ignore_index=True), AMBIGUOUS_ROW_LEVEL_COLUMNS)[AMBIGUOUS_ROW_LEVEL_COLUMNS]
        if ambiguous_parts
        else pd.DataFrame(columns=AMBIGUOUS_ROW_LEVEL_COLUMNS)
    )
    no_reuse_df = (
        ensure_columns(pd.concat(no_reuse_parts, ignore_index=True), NO_REUSE_ROW_LEVEL_COLUMNS)[NO_REUSE_ROW_LEVEL_COLUMNS]
        if no_reuse_parts
        else pd.DataFrame(columns=NO_REUSE_ROW_LEVEL_COLUMNS)
    )

    unamb_summary_records = [r for r in summary_records if r.get("suggested_candidate_index") is None]
    unambiguous_summary = pd.DataFrame(unamb_summary_records)[UNAMBIGUOUS_SUMMARY_COLUMNS] if unamb_summary_records else pd.DataFrame(columns=UNAMBIGUOUS_SUMMARY_COLUMNS)
    amb_summary_records = [r for r in summary_records if r.get("suggested_candidate_index") is not None]
    ambiguous_summary = pd.DataFrame(amb_summary_records)[AMBIGUOUS_SUMMARY_COLUMNS] if amb_summary_records else pd.DataFrame(columns=AMBIGUOUS_SUMMARY_COLUMNS)

    pos_meanings_ru = (
        pd.DataFrame(pos_meanings_records, columns=POS_MEANINGS_RU_COLUMNS)
        .drop_duplicates(subset=["pos", "meaning_ru"], keep="first")
        if pos_meanings_records
        else pd.DataFrame(columns=POS_MEANINGS_RU_COLUMNS)
    )

    concept_category_without_english = ensure_columns(
        concept_covered_df, CONCEPT_CATEGORY_AUDIT_COLUMNS
    )
    invalid_concept_category_pairs = ensure_columns(
        invalid_pair_df, CONCEPT_CATEGORY_AUDIT_COLUMNS
    )

    return ReuseAnalysisResult(
        missing_en_reusable_unambiguous=unambiguous_df,
        missing_en_reusable_ambiguous=ambiguous_df,
        missing_en_without_reuse=no_reuse_df,
        concept_category_without_english=concept_category_without_english,
        invalid_concept_category_pairs=invalid_concept_category_pairs,
        unambiguous_summary=unambiguous_summary,
        ambiguous_summary=ambiguous_summary,
        stats={
            "rows_with_meaning_ru": int(len(work)),
            "rows_with_existing_en": int(work["has_existing_en"].sum()),
            "rows_missing_en": int((~work["has_existing_en"]).sum()),
            "rows_reusable_unambiguous": int(len(unambiguous_df)),
            "rows_reusable_ambiguous": int(len(ambiguous_df)),
            "rows_missing_en_without_reuse": int(len(no_reuse_df)),
            "rows_concept_covered_skip": int(len(concept_covered_df)),
            "rows_invalid_concept_category_pair": int(len(invalid_pair_df)),
            "unambiguous_group_count": count_unique_pos_meaning_groups(unambiguous_df),
            "ambiguous_group_count": count_unique_pos_meaning_groups(ambiguous_df),
            "no_reuse_group_count": count_unique_pos_meaning_groups(no_reuse_df),
        },
        per_lang_stats=None,
        pos_meanings_ru=pos_meanings_ru,
    )


def write_reuse_outputs(result: ReuseAnalysisResult, translate_dir: Path) -> None:
    """Write reuse output CSV files.
    
    Writes all six CSV files, including empty files with headers.
    
    Args:
        result: ReuseAnalysisResult from analyze_missing_en_reuse
        translate_dir: Output directory (typically data/sem_cat/2translate)
    """
    translate_dir.mkdir(parents=True, exist_ok=True)

    reusable_english_dir = translate_dir / "reusable_english"
    reusable_english_dir.mkdir(parents=True, exist_ok=True)

    result.missing_en_reusable_unambiguous.to_csv(
        reusable_english_dir / "one_english.csv",
        index=False,
    )
    result.missing_en_reusable_ambiguous.to_csv(
        reusable_english_dir / "several_english.csv",
        index=False,
    )
    no_reuse_output = ensure_columns(
        result.missing_en_without_reuse,
        NO_REUSE_ROW_LEVEL_COLUMNS,
    )
    no_reuse_output.to_csv(
        translate_dir / "needs_translation_no_reuse.csv",
        index=False,
    )
    result.unambiguous_summary.to_csv(
        reusable_english_dir / "one_english_summary.csv",
        index=False,
    )
    result.ambiguous_summary.to_csv(
        reusable_english_dir / "several_english_summary.csv",
        index=False,
    )
    result.pos_meanings_ru.to_csv(
        translate_dir / "pos_meanings_ru.csv",
        index=False,
    )
    result.concept_category_without_english.to_csv(
        translate_dir / "concept_category_without_english.csv",
        index=False,
    )
    result.invalid_concept_category_pairs.to_csv(
        translate_dir / "invalid_concept_category_pairs.csv",
        index=False,
    )


def print_reuse_summary(stats: dict[str, int], *, per_lang_stats: dict[str, int] | None = None) -> None:
    """Print reuse analysis summary to console.
    
    Args:
        stats: Stats dict from ReuseAnalysisResult.stats
        per_lang_stats: Optional per-language stats dict
    """
    print("=" * 60)
    print("Missing-English reuse analysis by (pos, meaning_ru)")
    print("=" * 60)
    print()
    print(f"Rows with non-empty meaning_ru:              {stats['rows_with_meaning_ru']}")
    print(f"Rows with existing human English:            {stats['rows_with_existing_en']}")
    print(f"Rows missing English:                        {stats['rows_missing_en']}")
    print()
    print("Among rows missing English:")
    print(f"  Reusable, one EN variant:                  {stats['rows_reusable_unambiguous']}")
    print(f"  Reusable, multiple EN variants:            {stats['rows_reusable_ambiguous']}")
    print(f"  No reusable EN evidence:                   {stats['rows_missing_en_without_reuse']}")
    print(
        "Rows with concept-level coverage but no English translation yet\n"
        f"  (see concept_category_without_english.csv): {stats['rows_concept_covered_skip']}"
    )
    print(
        "Rows with inconsistent concept/category pair\n"
        f"  (see invalid_concept_category_pairs.csv): {stats['rows_invalid_concept_category_pair']}"
    )
    print()
    print("Unique (pos, meaning_ru) groups among rows missing English:")
    print(f"  Reusable, one EN variant:                  {stats['unambiguous_group_count']}")
    print(f"  Reusable, multiple EN variants:            {stats['ambiguous_group_count']}")
    print(f"  No reusable EN evidence:                   {stats['no_reuse_group_count']}")
    print()

