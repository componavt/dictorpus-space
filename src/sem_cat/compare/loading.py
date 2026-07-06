"""Loading and merging multiple translation CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "gloss_ru",
    "gloss_en",
    "qa_keep",
    "qa_score",
    "qa_flags",
    "model_key",
]

OPTIONAL_COLUMNS = [
    "model_name",
    "backend_family",
    "roundtrip_distance",
    "gloss_ru_back",
    "translation_input_mode",
    "pos_hint",
    "meaning_hint",
    "source_count",
    "qa_version",
    "is_singleword_ru",
    "task_key",
    "task_pos",
]


def parse_translation_arg(raw: str) -> tuple[str, Path]:
    """Parse a --translations argument of the form model_key=path.csv.

    Raises:
        ValueError: If format is invalid or model_key is empty.
    """
    if "=" not in raw:
        raise ValueError(
            f"Expected model_key=path.csv, got: {raw!r}"
        )
    model_key, raw_path = raw.split("=", 1)
    model_key = model_key.strip()
    path = Path(raw_path.strip())
    if not model_key:
        raise ValueError("Empty model_key in --translations argument")
    if not path.exists():
        raise FileNotFoundError(f"Translation file not found: {path}")
    return model_key, path


def load_single_model(path: Path, model_key: str) -> pd.DataFrame:
    """Load a single translation CSV and validate minimum schema.

    If the file has a non-empty 'model_key' column with exactly one
    distinct value that differs from the CLI label, a ValueError is
    raised to prevent misleading aliases in comparison summaries.

    Returns a DataFrame with columns prefixed as {model_key}__{field}.
    """
    df = pd.read_csv(path, dtype=str)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"File {path} is missing required column: {col}"
            )

    # Validate that the CLI model_key matches the file's content, if present
    if "model_key" in df.columns:
        file_keys = df["model_key"].dropna().astype(str).str.strip()
        file_keys = file_keys[file_keys != ""]
        if not file_keys.empty:
            unique_keys = set(file_keys)
            if len(unique_keys) == 1:
                file_key = next(iter(unique_keys))
                if file_key != model_key:
                    raise ValueError(
                        f"Mismatched model key: CLI label is {model_key!r}, "
                        f"but the file contains model_key={file_key!r} in "
                        f"every row. "
                        f"Use --translations {file_key}={path} instead."
                    )
            # Mixed or blank model_key values: keep the CLI label as-is.
            # The comparison will still work, but the summary may be
            # misleading. This is documented as a fallback.

    rename_map = {
        "gloss_en": f"{model_key}__gloss_en",
        "qa_keep": f"{model_key}__qa_keep",
        "qa_score": f"{model_key}__qa_score",
        "qa_flags": f"{model_key}__qa_flags",
        "roundtrip_distance": f"{model_key}__roundtrip_distance",
        "gloss_ru_back": f"{model_key}__gloss_ru_back",
        "model_name": f"{model_key}__model_name",
        "backend_family": f"{model_key}__backend_family",
    }

    available = {k: v for k, v in rename_map.items() if k in df.columns}
    keep_cols = ["gloss_ru", "task_key", "task_pos"] + list(available.keys())
    present_cols = [c for c in keep_cols if c in df.columns]
    result = df[present_cols].rename(columns=available)
    result[f"{model_key}__model_key"] = model_key

    return result


def merge_all_models(
    model_dfs: dict[str, pd.DataFrame],
    verbose: bool = False,
) -> pd.DataFrame:
    """Merge all model DataFrames on task_key (or gloss_ru for legacy files).

    Args:
        model_dfs: Mapping of model_key -> prefixed DataFrame.
        verbose: If True, print loading progress.

    Returns:
        Merged DataFrame with one row per unique translation task.
    """
    dfs = list(model_dfs.values())
    if not dfs:
        return pd.DataFrame(columns=["gloss_ru"])

    merged = dfs[0]
    for df in dfs[1:]:
        merge_on = "task_key" if "task_key" in merged.columns and "task_key" in df.columns else "gloss_ru"
        merged = pd.merge(merged, df, on=merge_on, how="outer").reset_index(drop=True)

    if verbose:
        has_task_key = "task_key" in merged.columns
        print(f"Merged {len(model_dfs)} models, {len(merged)} unique {'task_key' if has_task_key else 'gloss_ru'} rows")

    return merged


TASK_KEY_SEP = "::"


def parse_serialized_task_key(value: str) -> tuple[str, str] | None:
    """Parse a serialized task key back to (pos, gloss) tuple.
    
    Supports both new :: format and legacy \\t format for backward compatibility.
    """
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if TASK_KEY_SEP in value:
        pos, gloss = value.split(TASK_KEY_SEP, 1)
        return pos, gloss
    if "\t" in value:
        pos, gloss = value.split("\t", 1)
        return pos, gloss
    return None


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
    result = parse_serialized_task_key(s)
    if result is None:
        return s
    pos, gloss = result
    return f"{pos}{TASK_KEY_SEP}{gloss}"



