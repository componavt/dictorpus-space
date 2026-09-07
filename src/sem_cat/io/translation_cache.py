"""Translation cache loading and validation."""

from __future__ import annotations

import pathlib
import dataclasses
from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class TranslationCacheLoadResult:
    """Result of loading/validating a translation cache file."""
    state: Literal["missing", "valid", "malformed"]
    df: pd.DataFrame
    reason: str | None = None
    columns: tuple[str, ...] = ()
    row_count: int = 0


CANONICAL_COLUMNS = (
    "pos",
    "meaning_ru",
    "meaning_en",
    "qa_keep",
    "qa_score",
    "qa_flags",
    "meaning_ru_back",
    "roundtrip_distance",
)


def _detect_legacy_fields(detected_columns: list[str]) -> list[str]:
    """Check if the cache file uses the legacy gloss-based schema.
    
    Args:
        detected_columns: List of column names from the CSV
        
    Returns:
        List of legacy field names that were detected
    """
    legacy_fields = [
        "task_key", "task_key_str", "task_pos", "primary_gloss_ru",
        "gloss_ru", "gloss_en", "gloss_ru_back", "pos_hint",
        "meaning_hint", "sourcecount"
    ]
    return [f for f in legacy_fields if f in detected_columns]


def load_translation_cache(
    out_path: pathlib.Path,
    expected_model_key: str | None = None,
) -> TranslationCacheLoadResult:
    """Load and validate an existing translation cache file.
    
    The cache uses (pos, meaning_ru) as the composite identity for tasks.
    
    Args:
        out_path: Path to the CSV cache file.
        expected_model_key: If provided, validates that cached rows match (ignored).
        
    Returns:
        TranslationCacheLoadResult with structured state information.
    """
    if not out_path.exists():
        return TranslationCacheLoadResult(
            state="missing",
            df=pd.DataFrame(columns=CANONICAL_COLUMNS),
            reason="file does not exist",
        )

    try:
        df = pd.read_csv(out_path, encoding="utf-8", dtype=str)
    except Exception as e:
        return TranslationCacheLoadResult(
            state="malformed",
            df=pd.DataFrame(columns=CANONICAL_COLUMNS),
            reason=f"csv read failed: {e}",
        )

    detected_columns = df.columns.tolist()
    cols = tuple(detected_columns)
    
    # Check if legacy schema is being used
    legacy_fields = _detect_legacy_fields(detected_columns)
    if legacy_fields:
        return TranslationCacheLoadResult(
            state="malformed",
            df=pd.DataFrame(columns=CANONICAL_COLUMNS),
            reason=(
                f"Cache uses obsolete translation schema with legacy fields: "
                f"{sorted(legacy_fields)}. "
                f"New translation cache requires exact columns: {CANONICAL_COLUMNS}. "
                "Please rename or remove the old cache file and rerun."
            ),
            columns=cols,
            row_count=len(df),
        )

    # Strict schema validation: exact columns in exact order
    if detected_columns != list(CANONICAL_COLUMNS):
        return TranslationCacheLoadResult(
            state="malformed",
            df=pd.DataFrame(columns=CANONICAL_COLUMNS),
            reason=(
                f"Cache columns do not match canonical schema. "
                f"Expected: {list(CANONICAL_COLUMNS)}, "
                f"got: {detected_columns}. "
                "Exact column order and all 8 columns are required."
            ),
            columns=cols,
            row_count=len(df),
        )

    df = df.copy()
    
    df = _deduplicate_cache_by_pos_meaning_ru(df)

    return TranslationCacheLoadResult(
        state="valid",
        df=df,
        columns=cols,
        row_count=len(df),
    )


def _deduplicate_cache_by_pos_meaning_ru(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate cache rows by (pos, meaning_ru), keeping best qa_keep then lowest qa_score.
    
    Duplicate selection rule (in order):
    1. Prefer qa_keep=True
    2. Then prefer lowest numeric qa_score
    3. On a tie, retain the first source-file row
    
    Args:
        df: DataFrame with pos and meaning_ru columns
        
    Returns:
        Deduplicated DataFrame
    """
    if df.empty:
        return df
    
    duplicates = df.duplicated(subset=["pos", "meaning_ru"], keep=False)
    if not duplicates.any():
        return df
    
    dup_count = int(duplicates.sum())
    dup_examples = []
    for (pos, meaning), group in df.groupby(["pos", "meaning_ru"], dropna=False, sort=False):
        if len(group) > 1:
            dup_examples.append((pos, meaning, len(group)))
            if len(dup_examples) >= 10:
                break
    
    print(f"WARNING: cache has {dup_count} duplicate (pos, meaning_ru) rows.")
    print(f"  Examples: {dup_examples[:5]}")
    print(f"  Keeping rows per rule: qa_keep=True preferred, then lowest qa_score, then first-row order.")
    
    df_work = df.copy()
    df_work = df_work.assign(
        _qa_keep_bool=df_work.get("qa_keep", "").apply(
            lambda x: str(x).lower() in ("true", "1", "yes") if pd.notna(x) else False
        ),
        _qa_score_num=pd.to_numeric(df_work.get("qa_score", 0), errors="coerce").fillna(0.0),
    )
    df_work = df_work.sort_values(
        ["_qa_keep_bool", "_qa_score_num"],
        ascending=[False, True],
    )
    df_work = df_work.drop_duplicates(subset=["pos", "meaning_ru"], keep="first")
    df_work = df_work.drop(columns=["_qa_keep_bool", "_qa_score_num"])
    
    return df_work


def build_cached_identity_set(cache_df: pd.DataFrame) -> set[tuple[str, str]]:
    """Build a set of cached task identities (pos, meaning_ru) from a validated cache.
    
    Args:
        cache_df: DataFrame with pos and meaning_ru columns
        
    Returns:
        Set of (pos, meaning_ru) tuples
    """
    if cache_df.empty or "pos" not in cache_df.columns or "meaning_ru" not in cache_df.columns:
        return set()
    
    non_null = cache_df[cache_df["pos"].notna() & cache_df["meaning_ru"].notna()]
    if non_null.empty:
        return set()
    
    return set(zip(non_null["pos"].tolist(), non_null["meaning_ru"].tolist()))


def count_cached_rows(cache_df: pd.DataFrame) -> int:
    """Return the number of unique cached task entries by (pos, meaning_ru).
    
    Args:
        cache_df: DataFrame with pos and meaning_ru columns
        
    Returns:
        Count of unique (pos, meaning_ru) pairs
    """
    return len(build_cached_identity_set(cache_df))
