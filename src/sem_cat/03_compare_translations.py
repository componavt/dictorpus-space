"""Multi-model translation comparison.

Compares translation outputs from N models and produces a single CSV
with raw per-model results. No consensus, risk, or proposal logic.

This script is step 03 in the semantic domain mapping pipeline:
  01 - count meanings and glosses (Jupyter notebook, read-only)
  02 - translate Russian glosses to English
  03 - compare translation outputs [THIS FILE]
  04 - WordNet synset lookup
  05 - assign semantic domains to meanings

This script does NOT use WordNet, NLTK, or any translation model.
It only reads CSV files and writes a merged comparison CSV.

EXAMPLE COMMANDS:
  # Compare all 6 models
  python3 -m src.sem_cat.03_compare_translations \\
      --translations google=data/sem_cat/02_meanings_translated_google.csv \\
      --translations helsinki_opus_mt_ru_en=data/sem_cat/02_meanings_translated_helsinki_opus_mt_ru_en.csv \\
      --translations nllb_3_3b=data/sem_cat/02_meanings_translated_nllb_3_3b.csv \\
      --translations tower_plus_9b=data/sem_cat/02_meanings_translated_tower_plus_9b.csv \\
      --translations hy_mt2_30b_a3b=data/sem_cat/02_meanings_translated_hy_mt2_30b_a3b.csv \\
      --translations alma_7b_r=data/sem_cat/02_meanings_translated_alma_7b_r.csv

  # Compare only 2 models
  python3 -m src.sem_cat.03_compare_translations \\
      --translations google=data/sem_cat/02_meanings_translated_google.csv \\
      --translations nllb_3_3b=data/sem_cat/02_meanings_translated_nllb_3_3b.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.sem_cat.compare.loading import (
    parse_translation_arg,
    load_single_model,
    merge_all_models,
)
from src.sem_cat.compare.output_tables import build_comparison_df


def print_summary(model_keys: list[str], merged_df: pd.DataFrame) -> None:
    """Print console summary statistics."""
    total = len(merged_df)
    if total == 0:
        print("No meanings to compare.")
        return

    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total meanings merged:      {total}")
    print(f"Models compared:            {len(model_keys)}")
    print(f"Model keys:                 {', '.join(model_keys)}")
    print()

    print("Per-model coverage:")
    for mk in model_keys:
        col = f"{mk}_en"
        if col in merged_df.columns:
            non_blank = merged_df[col].notna() & (merged_df[col].astype(str).str.strip() != "")
            count = non_blank.sum()
            coverage = count / total * 100 if total > 0 else 0
            print(f"  {mk:30s}  coverage={coverage:5.1f}%  rows={count}")
        else:
            print(f"  {mk:30s}  coverage=  0.0%  rows=0")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare N-model translation outputs",
    )

    parser.add_argument(
        "--translations",
        action="append",
        default=[],
        metavar="MODEL_KEY=PATH",
        help="Translation file as model_key=path.csv (repeatable)",
    )

    parser.add_argument(
        "--out-file",
        type=str,
        default="data/sem_cat/03_translation_comparison.csv",
        help="Path to output comparison CSV",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    if not args.translations:
        print("ERROR: No translation files provided.")
        print("  Use --translations model_key=path.csv (repeatable)")
        sys.exit(1)

    model_map: dict[str, Path] = {}
    for raw in args.translations:
        try:
            mk, path = parse_translation_arg(raw)
        except (ValueError, FileNotFoundError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        if mk in model_map:
            print(f"ERROR: Duplicate model_key in --translations: {mk}")
            sys.exit(1)
        model_map[mk] = path

    model_keys = sorted(model_map.keys())
    total_models = len(model_keys)

    if args.verbose:
        print(f"Loading {total_models} model files: {', '.join(model_keys)}")

    model_dfs: dict[str, pd.DataFrame] = {}
    for mk, path in model_map.items():
        if args.verbose:
            print(f"  Loading {mk}: {path}")
        try:
            model_dfs[mk] = load_single_model(path, mk)
        except (ValueError, FileNotFoundError) as e:
            print(f"ERROR loading {mk}: {e}")
            sys.exit(1)

    merged = merge_all_models(model_dfs, verbose=args.verbose)

    if merged.empty:
        print("ERROR: No data after merging. Check input files.")
        sys.exit(1)

    out_df = build_comparison_df(merged)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print_summary(model_keys, merged)


if __name__ == "__main__":
    main()
