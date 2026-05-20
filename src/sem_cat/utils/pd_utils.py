"""Pandas utility helpers used across the comparison pipeline."""

from __future__ import annotations

import pandas as pd


def is_blank_pd(value) -> bool:
    """Check if a pandas value is blank (NaN, None, or whitespace-only string)."""
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def safe_float(value, default: float = 0.0) -> float:
    """Parse float from string/number, returning *default* on failure."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value, default: bool = True) -> bool:
    """Parse bool from string/bool, returning *default* on failure."""
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return default


def parse_flags(flags_str) -> set[str]:
    """Parse semicolon-separated flags string into a set of flag names."""
    if is_blank_pd(flags_str):
        return set()
    return set(str(flags_str).split(";"))
