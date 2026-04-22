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

    Returns a DataFrame with columns prefixed as {model_key}__{field}.
    """
    df = pd.read_csv(path, dtype=str)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"File {path} is missing required column: {col}"
            )

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
    keep_cols = ["gloss_ru"] + list(available.keys())
    result = df[keep_cols].rename(columns=available)
    result[f"{model_key}__model_key"] = model_key

    return result


def merge_all_models(
    model_dfs: dict[str, pd.DataFrame],
    verbose: bool = False,
) -> pd.DataFrame:
    """Merge all model DataFrames on gloss_ru using outer join.

    Args:
        model_dfs: Mapping of model_key -> prefixed DataFrame.
        verbose: If True, print loading progress.

    Returns:
        Merged DataFrame with one row per gloss_ru.
    """
    dfs = list(model_dfs.values())
    if not dfs:
        return pd.DataFrame(columns=["gloss_ru"])

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="gloss_ru", how="outer")

    if verbose:
        print(f"Merged {len(model_dfs)} models, {len(merged)} unique gloss_ru rows")

    return merged
