"""Loads the four meanings_*.csv files from data/vepkar/
into a single merged pandas DataFrame.
"""

import pandas as pd
from typing import List

LANGS = ["vep", "olo", "lud", "krl"]
MEANINGS_COLS = ["id", "lemma_id", "meaning_id", "meaning_num",
                 "lemma", "lang", "pos", "meaning_ru"]


def load_meanings(data_dir: str) -> pd.DataFrame:
    """Read meanings_{lang}.csv for each lang in LANGS.
    sep=",", encoding="utf-8-sig", dtype=str.
    Strip whitespace from all string columns.
    Concatenate and reset index.
    Log row counts per language to stdout.
    Return merged DataFrame.
    """
    dfs = []
    
    for lang in LANGS:
        filepath = f"{data_dir}/meanings_{lang}.csv"
        df = pd.read_csv(filepath, sep=",", encoding="utf-8-sig", dtype=str)
        
        # Strip whitespace from all string columns
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip()
        
        # Log row count for this language
        print(f"Loaded {len(df)} rows for language '{lang}'")
        
        dfs.append(df)
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    return merged_df