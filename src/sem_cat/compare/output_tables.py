"""Output table builders for multi-model translation comparison."""

from __future__ import annotations

import pandas as pd


def build_comparison_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Return the already-merged DataFrame with pos and meaning_ru guaranteed first.
    
    Args:
        merged_df: DataFrame from merge_all_models() with columns:
            pos, meaning_ru,
            {model_key}_en, {model_key}_keep, {model_key}_score, {model_key}_flags,
            {model_key}_ru, {model_key}_rt,
            ... repeated per model
    
    Returns:
        DataFrame with pos and meaning_ru first, followed by per-model columns in order.
    """
    if merged_df.empty:
        return merged_df
    
    cols = ["pos", "meaning_ru"]
    for col in merged_df.columns:
        if col not in cols:
            cols.append(col)
    
    return merged_df[cols].copy()
