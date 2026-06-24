"""Step 06: Build concept-level WDH table.

By default, paths are loaded from src/sem_cat/sem_cat_paths.toml.

Reads:
- data/sem_cat/concept_categories/concept_categories_wdh.tsv
- data/sem_cat/concepts/concepts_with_english_1445.csv

Produces:
- data/sem_cat/concepts/concepts_wdh.tsv

Usage:
    python3 -m src.sem_cat.06_concepts_wdh

Override example:
    python3 -m src.sem_cat.06_concepts_wdh \
        --concepts /tmp/concepts_experiment.csv \
        --out-file /tmp/concepts_wdh_experiment.tsv
"""

import argparse
from pathlib import Path

from src.sem_cat.paths_config import load_sem_cat_paths, SemCatPaths
from src.sem_cat.utils.concept_wdh import (
    load_category_wdh,
    load_concepts,
    build_concepts_wdh,
    save_concepts_wdh,
)


def resolve_step06_paths(
    args,
    paths_config: SemCatPaths,
) -> tuple[Path, Path, Path]:
    """Resolve step-06 input/output paths from CLI args and config.

    Args:
        args: Parsed argparse arguments namespace.
        paths_config: SemCatPaths instance loaded from config.

    Returns:
        Tuple of (cat_wdh_path, concepts_path, out_file_path) as resolved Paths.
    """
    cat_wdh_path = Path(args.cat_wdh) if args.cat_wdh else paths_config.concept_categories_wdh
    concepts_path = Path(args.concepts) if args.concepts else paths_config.concepts_catalog
    out_file_path = Path(args.out_file) if args.out_file else paths_config.concepts_wdh
    return cat_wdh_path, concepts_path, out_file_path


def main():
    parser = argparse.ArgumentParser(
        description="Build concept-level WDH table from category inheritance"
    )
    parser.add_argument(
        "--paths-config",
        type=str,
        default=None,
        help="Optional TOML config override. Default: src/sem_cat/sem_cat_paths.toml",
    )
    parser.add_argument(
        "--cat-wdh",
        type=str,
        default=None,
        help="Override path to concept_categories_wdh.tsv",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=None,
        help="Override path to concepts_with_english_1445.csv (tab-separated)",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Override output TSV path (default comes from sem_cat_paths.toml)",
    )

    args = parser.parse_args()

    cfg = load_sem_cat_paths(args.paths_config)

    cat_wdh_path, concepts_path, out_file_path = resolve_step06_paths(args, cfg)

    print(f"Category WDH file: {cat_wdh_path}")
    print(f"Concept catalog:   {concepts_path}")
    print(f"Output file:       {out_file_path}")

    cat_wdh = load_category_wdh(str(cat_wdh_path))
    print(f"  {len(cat_wdh)} categories loaded")

    concepts = load_concepts(str(concepts_path))
    print(f"  {len(concepts)} concepts loaded")

    concepts_wdh = build_concepts_wdh(cat_wdh, concepts)

    wdh_assigned = (concepts_wdh["wdh"].notna()) & (concepts_wdh["wdh"] != "")
    wdh_missing = ~wdh_assigned
    print(f"  WDH assigned: {wdh_assigned.sum()}")
    print(f"  WDH missing:  {wdh_missing.sum()}")

    src_counts = concepts_wdh["wdh_source"].value_counts()
    for src, cnt in src_counts.items():
        print(f"  {src}: {cnt}")

    conf_counts = concepts_wdh["wdh_confidence"].value_counts()
    for conf, cnt in conf_counts.items():
        print(f"  confidence={conf}: {cnt}")

    save_concepts_wdh(concepts_wdh, str(out_file_path))


if __name__ == "__main__":
    main()
