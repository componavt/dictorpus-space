"""Step 06: Build concept-level WDH table.

Reads concept_categories_wdh.tsv and concepts_with_english_417.csv,
produces data/sem_cat/concepts/concepts_wdh.tsv.

Usage:
    python3 -m src.sem_cat.06_concepts_wdh \
        --cat-wdh data/sem_cat/concept_categories/concept_categories_wdh.tsv \
        --concepts data/sem_cat/concepts/concepts_with_english_417.csv \
        --out-file data/sem_cat/concepts/concepts_wdh.tsv
"""

import argparse
from pathlib import Path

from src.sem_cat.utils.concept_wdh import (
    load_category_wdh,
    load_concepts,
    build_concepts_wdh,
    save_concepts_wdh,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build concept-level WDH table from category inheritance"
    )
    parser.add_argument(
        "--cat-wdh",
        type=str,
        required=True,
        help="Path to concept_categories_wdh.tsv",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        required=True,
        help="Path to concepts_with_english_417.csv (tab-separated)",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Output TSV path (default: data/sem_cat/concepts/concepts_wdh.tsv)",
    )

    args = parser.parse_args()

    # Resolve output path
    if args.out_file is None:
        out_dir = Path(args.concepts).parent
        out_file = str(out_dir / "concepts_wdh.tsv")
    else:
        out_file = args.out_file

    # Load inputs
    print("Loading category WDH...")
    cat_wdh = load_category_wdh(args.cat_wdh)
    print(f"  {len(cat_wdh)} categories loaded")

    print("Loading concepts...")
    concepts = load_concepts(args.concepts)
    print(f"  {len(concepts)} concepts loaded")

    # Build concept-level WDH
    print("Building concept-level WDH...")
    concepts_wdh = build_concepts_wdh(cat_wdh, concepts)

    # Summary
    wdh_assigned = (concepts_wdh["wdh"].notna()) & (concepts_wdh["wdh"] != "")
    wdh_missing = ~wdh_assigned
    print(f"  WDH assigned: {wdh_assigned.sum()}")
    print(f"  WDH missing:  {wdh_missing.sum()}")

    # Source distribution
    src_counts = concepts_wdh["wdh_source"].value_counts()
    for src, cnt in src_counts.items():
        print(f"  {src}: {cnt}")

    # Confidence distribution
    conf_counts = concepts_wdh["wdh_confidence"].value_counts()
    for conf, cnt in conf_counts.items():
        print(f"  confidence={conf}: {cnt}")

    # Save
    save_concepts_wdh(concepts_wdh, out_file)


if __name__ == "__main__":
    main()
