"""Translation cache loading and validation."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Literal

import pandas as pd


TASK_KEY_SEP = "::"


@dataclass(frozen=True)
class TranslationCacheLoadResult:
    """Result of loading/validating a translation cache file."""
    state: Literal["missing", "valid", "malformed"]
    df: pd.DataFrame
    reason: str | None = None
    columns: tuple[str, ...] = ()
    row_count: int = 0


def normalize_loaded_task_key(value: object) -> str | None:
    """Normalize task key from legacy formats.
    
    Accepts:
    - New format: "NOUN::obida"
    - Legacy tab format: "NOUN\tobida"
    
    Args:
        value: Raw task key value from CSV
        
    Returns:
        Normalized task key in :: format, or None if empty/invalid
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if TASK_KEY_SEP in s:
        pos, gloss = s.split(TASK_KEY_SEP, 1)
        return f"{pos}{TASK_KEY_SEP}{gloss}"
    if "\t" in s:
        pos, gloss = s.split("\t", 1)
        return f"{pos}{TASK_KEY_SEP}{gloss}"
    return s


REQUIRED_CACHE_COLUMNS = [
    "gloss_ru",
    "gloss_en",
    "qa_keep",
    "qa_score",
    "qa_flags",
    "model_key",
]

OPTIONAL_CACHE_COLUMNS = [
    "gloss_en_back",
    "roundtrip_distance",
    "translation_input_mode",
    "pos_hint",
    "meaning_hint",
    "source_count",
    "qa_version",
    "is_singleword_ru",
    "input_token_count",
    "output_token_count",
    "task_key",
    "task_pos",
]

ALL_CACHE_COLUMNS = REQUIRED_CACHE_COLUMNS + OPTIONAL_CACHE_COLUMNS

LEGACY_CACHE_COLUMNS = [
    "gloss_ru",
    "gloss_en",
]


def load_translation_cache(
    out_path: pathlib.Path,
    expected_model_key: str | None = None,
) -> TranslationCacheLoadResult:
    """Load and validate an existing translation cache file.

    Args:
        out_path: Path to the CSV cache file.
        expected_model_key: If provided, validates that cached rows match.

    Returns:
        TranslationCacheLoadResult with structured state information.
    """
    if not out_path.exists():
        return TranslationCacheLoadResult(
            state="missing",
            df=pd.DataFrame(columns=ALL_CACHE_COLUMNS),
            reason="file does not exist",
        )

    try:
        df = pd.read_csv(out_path, encoding="utf-8", dtype=str)
    except Exception as e:
        return TranslationCacheLoadResult(
            state="malformed",
            df=pd.DataFrame(columns=ALL_CACHE_COLUMNS),
            reason=f"csv read failed: {e}",
        )

    cols = tuple(df.columns.tolist())
    if "gloss_ru" not in df.columns:
        return TranslationCacheLoadResult(
            state="malformed",
            df=pd.DataFrame(columns=ALL_CACHE_COLUMNS),
            reason="missing required column 'gloss_ru'",
            columns=cols,
            row_count=len(df),
        )

    # Handle task_key normalization
    if "task_key" in df.columns:
        df = df.copy()
        df["task_key"] = df["task_key"].map(normalize_loaded_task_key)
    elif "task_key_str" in df.columns:
        df = df.copy()
        df["task_key"] = df["task_key_str"].map(normalize_loaded_task_key)

    # Schema upgrade: if model_key is missing, infer it
    if "model_key" not in df.columns:
        if expected_model_key:
            df["model_key"] = expected_model_key
        else:
            df["model_key"] = "unknown"

    # Ensure required columns exist (fill missing with defaults)
    for col in REQUIRED_CACHE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Ensure optional columns exist
    for col in OPTIONAL_CACHE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Handle duplicate gloss_ru: keep row with highest qa_score
    if df["gloss_ru"].duplicated().any():
        dup_count = int(df["gloss_ru"].duplicated().sum())
        dup_examples = df[df["gloss_ru"].duplicated(keep=False)]["gloss_ru"].unique()[:10].tolist()
        print(f"WARNING: cache has {dup_count} duplicate glossru rows.")
        print(f"  Examples: {dup_examples}")
        print(f"  Keeping row with highest qa_score per glossru.")
        df["_qa_score_num"] = pd.to_numeric(df.get("qa_score", 0), errors="coerce").fillna(0.0)
        df = df.sort_values("_qa_score_num", ascending=False)
        df = df.drop_duplicates(subset="gloss_ru", keep="first")
        df = df.drop(columns=["_qa_score_num"])

    return TranslationCacheLoadResult(
        state="valid",
        df=df,
        columns=cols,
        row_count=len(df),
    )


def build_cached_gloss_set(cache_df: pd.DataFrame) -> set[str]:
    """Build a set of already-translated gloss_ru values from a validated cache."""
    if cache_df.empty or "gloss_ru" not in cache_df.columns:
        return set()
    return set(cache_df["gloss_ru"].dropna().tolist())


def build_cached_task_key_set(cache_df: pd.DataFrame) -> set[str]:
    """Build a set of already-translated task_key values from a validated cache.
    
    If task_key column is missing, falls back to gloss_ru for backward compatibility.
    """
    if cache_df.empty or "task_key" not in cache_df.columns:
        return set()
    
    non_null_keys = cache_df["task_key"].dropna()
    if not non_null_keys.empty:
        return set(non_null_keys.astype(str).tolist())
    
    if "gloss_ru" in cache_df.columns:
        non_null_glosses = cache_df["gloss_ru"].dropna()
        if not non_null_glosses.empty:
            return set(non_null_glosses.tolist())
    
    return set()


def count_cached_rows(cache_df: pd.DataFrame) -> int:
    """Return the number of unique cached gloss entries."""
    return len(build_cached_gloss_set(cache_df))


def count_cached_tasks(cache_df: pd.DataFrame) -> int:
    """Return the number of unique cached task keys, or gloss entries for legacy caches."""
    return len(build_cached_task_key_set(cache_df))
