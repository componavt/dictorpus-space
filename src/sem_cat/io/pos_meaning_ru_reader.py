"""Strict reader for Step 01 pos_meanings_ru.csv task file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

POS_MEANINGS_RU_COLUMNS = [
    "pos",
    "meaning_ru",
]


def read_pos_meaning_ru_tasks(path: str | Path) -> pd.DataFrame:
    """Read and validate a Step 01 pos_meanings_ru.csv task file.

    Args:
        path: Path to the CSV file (string or Path object)

    Returns:
        DataFrame with exactly two columns: pos, meaning_ru

    Raises:
        ValueError: If file doesn't exist, columns are invalid, or data has issues
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Translation task file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Translation task file is not a regular file: {path}")

    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Failed to read translation task file {path}: {e}")

    required_cols = ["pos", "meaning_ru"]

    if len(df.columns) != 2:
        raise ValueError(
            f"Translation task file has invalid columns. Expected exactly {required_cols}, got {list(df.columns)}."
        )

    if list(df.columns) != required_cols:
        raise ValueError(
            f"Translation task file has invalid columns. Expected exactly {required_cols}, got {list(df.columns)}."
        )

    for idx, val in enumerate(df["pos"], start=2):
        if pd.isna(val) or str(val).strip() == "":
            raise ValueError(f"Translation task file contains blank pos at row {idx}.")

    for idx, val in enumerate(df["meaning_ru"], start=2):
        if pd.isna(val) or str(val).strip() == "":
            raise ValueError(f"Translation task file contains blank meaning_ru at row {idx}.")

    duplicates = df.duplicated(subset=["pos", "meaning_ru"], keep=False)
    if duplicates.any():
        dup_rows = df[duplicates].iloc[0]
        pos_val = dup_rows["pos"]
        meaning_val = dup_rows["meaning_ru"]
        raise ValueError(
            f"Translation task file contains duplicate (pos, meaning_ru) task: pos='{pos_val}', meaning_ru='{meaning_val}'."
        )

    return df[["pos", "meaning_ru"]].copy()
