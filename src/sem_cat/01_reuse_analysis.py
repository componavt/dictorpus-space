"""
Step 01: Missing-English reuse analysis.

Analyze rows that do not have an English translation yet, grouping by
(pos, primary_gloss_ru) to identify reusable existing English evidence.

This step produces three row-level files and two summary files:
- missing_en_reusable_unambiguous_pos_gloss_ru.csv
- missing_en_reusable_ambiguous_pos_gloss_ru.csv
- needs_translation_no_reuse.csv
- missing_en_reusable_unambiguous_pos_gloss_ru_summary.csv
- missing_en_reusable_ambiguous_pos_gloss_ru_summary.csv

Unambiguous reuse: exactly one distinct existing English value for the group
Ambiguous reuse: two or more distinct existing English values for the group
No reuse: missing-English rows with zero existing English candidates

This step does NOT auto-fill translations and does NOT invoke any MT backend.
It only exports reuse evidence for review and later steps.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "vepkar"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "data" / "sem_cat"
_DEFAULT_TRANSLATE_DIR = _DEFAULT_OUT_DIR / "2translate"

from src.sem_cat.utils.vepkar_loader import load_meanings
from src.sem_cat.pipeline.meaning_preparation import prepare_meanings_for_reuse_and_translation
from src.sem_cat.pipeline.reuse_analysis import (
    analyze_missing_en_reuse,
    write_reuse_outputs,
    print_reuse_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze missing-English rows for reusable existing English evidence grouped by (pos, primary_gloss_ru)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_DEFAULT_DATA_DIR),
        help=f"path to data/vepkar/ (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--translate-dir",
        type=str,
        default=str(_DEFAULT_TRANSLATE_DIR),
        help=f"output directory for helper files (default: {_DEFAULT_TRANSLATE_DIR})",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    translate_dir = Path(args.translate_dir)
    translate_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data dir: {data_dir}")
    print(f"Translate dir: {translate_dir}")

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    print("Loading meanings...")
    df_meanings = load_meanings(str(data_dir))

    print("Preparing meanings for analysis...")
    work = prepare_meanings_for_reuse_and_translation(df_meanings)
    print(f"Total rows with non-empty primary gloss: {len(work)}")

    print("Analyzing missing-English reuse by (pos, primary_gloss_ru)...")
    result = analyze_missing_en_reuse(work)

    print("Writing reuse output files...")
    write_reuse_outputs(result, translate_dir)

    print_reuse_summary(result.stats)

    print()
    print(f"Output directory: {translate_dir}")
    print("  - missing_en_reusable_unambiguous_pos_gloss_ru.csv")
    print("  - missing_en_reusable_ambiguous_pos_gloss_ru.csv")
    print("  - needs_translation_no_reuse.csv")
    print("  - missing_en_reusable_unambiguous_pos_gloss_ru_summary.csv")
    print("  - missing_en_reusable_ambiguous_pos_gloss_ru_summary.csv")


if __name__ == "__main__":
    main()
