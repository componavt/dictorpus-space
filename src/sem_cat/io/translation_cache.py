"""Translation cache loading and validation."""

from __future__ import annotations

import pathlib

import pandas as pd


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
) -> pd.DataFrame:
    """Load and validate an existing translation cache file.

    Args:
        out_path: Path to the CSV cache file.
        expected_model_key: If provided, validates that cached rows match.

    Returns:
        Validated DataFrame, or empty DataFrame with canonical columns if
        the file doesn't exist or is invalid.
    """
    if not out_path.exists():
        return pd.DataFrame(columns=ALL_CACHE_COLUMNS)

    try:
        df = pd.read_csv(out_path, encoding="utf-8", dtype=str)
    except Exception as e:
        print(f"WARNING: could not read cache file ({e}) — starting fresh.")
        return pd.DataFrame(columns=ALL_CACHE_COLUMNS)

    if "gloss_ru" not in df.columns:
        print("WARNING: cache file exists but has no 'gloss_ru' column — ignoring cache.")
        return pd.DataFrame(columns=ALL_CACHE_COLUMNS)

    # Schema upgrade: if model_key is missing, infer it
    if "model_key" not in df.columns:
        if expected_model_key:
            print(
                f"WARNING: cache predates model_key column. "
                f"Assigning model_key='{expected_model_key}' to all cached rows."
            )
            df["model_key"] = expected_model_key
        else:
            print(
                "WARNING: cache has no model_key and no expected key provided. "
                "Proceeding with caution."
            )
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

    return df


def build_cached_gloss_set(cache_df: pd.DataFrame) -> set[str]:
    """Build a set of already-translated gloss_ru values from a validated cache."""
    if cache_df.empty or "gloss_ru" not in cache_df.columns:
        return set()
    return set(cache_df["gloss_ru"].dropna().tolist())


def build_cached_task_key_set(cache_df: pd.DataFrame) -> set[str]:
    """Build a set of already-translated task_key values from a validated cache.
    
    If task_key column is missing, falls back to gloss_ru for backward compatibility.
    """
    if cache_df.empty:
        return set()
    
    if "task_key" in cache_df.columns and not cache_df["task_key"].isna().all():
        return set(cache_df["task_key"].dropna().astype(str).tolist())
    
    if "gloss_ru" in cache_df.columns:
        return set(cache_df["gloss_ru"].dropna().tolist())
    
    return set()


def count_cached_rows(cache_df: pd.DataFrame) -> int:
    """Return the number of unique cached gloss entries."""
    return len(build_cached_gloss_set(cache_df))


def count_cached_tasks(cache_df: pd.DataFrame) -> int:
    """Return the number of unique cached task keys, or gloss entries for legacy caches."""
    return len(build_cached_task_key_set(cache_df))
