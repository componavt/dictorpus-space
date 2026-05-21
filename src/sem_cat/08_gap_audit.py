"""Step 08: Gap audit for VepKar concept coverage.

Produces tabular reports that identify missing or weak concept coverage
in VepKar meanings. Primary grouping axis: meaning_ru.

Outputs (all in out-dir):
    audit_meanings_without_concept.csv
    audit_concept_usage.csv
    audit_category_coverage.csv
    audit_wdh_disagreement.csv  (if enriched meanings provided)
    audit_meaning_ru_clusters.csv

Usage:
    python3 -m src.sem_cat.08_gap_audit \
        --data-dir data/vepkar/ \
        --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
        --out-dir data/sem_cat/results/

    # With enriched meanings for WDH disagreement analysis:
    python3 -m src.sem_cat.08_gap_audit \
        --data-dir data/vepkar/ \
        --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
        --enriched-dir data/sem_cat/results/ \
        --out-dir data/sem_cat/results/
"""

import argparse
from pathlib import Path

import pandas as pd

from src.sem_cat.utils.concept_wdh import load_concepts as load_concepts_catalog
from src.sem_cat.utils.meaning_propagation import load_concepts_wdh
from src.sem_cat.utils.gap_audit import run_gap_audit


def _load_meanings_merged(data_dir: str) -> pd.DataFrame:
    """Load and merge all meanings_*.csv files."""
    langs = ["vep", "olo", "lud", "krl"]
    dfs = []
    for lang in langs:
        filepath = f"{data_dir}/meanings_{lang}.csv"
        path = Path(filepath)
        if not path.exists():
            continue

        df = pd.read_csv(filepath, sep=",", encoding="utf-8-sig", dtype=str)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip()

        if "concept_id" not in df.columns:
            df["concept_id"] = ""
        if "category_id" not in df.columns:
            df["category_id"] = ""
        if "meaning_en" not in df.columns:
            df["meaning_en"] = ""

        print(f"  Loaded {len(df)} rows for {lang}")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No meanings_*.csv files found in {data_dir}")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"  Total merged: {len(merged)} rows")
    return merged


def _load_enriched_meanings(enriched_dir: str) -> pd.DataFrame | None:
    """Load enriched meanings files (meanings_*_concept_wdh.csv)."""
    p = Path(enriched_dir)
    if not p.exists():
        return None

    files = list(p.glob("meanings_*_concept_wdh.csv"))
    if not files:
        return None

    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f, dtype=str)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip()
        print(f"  Loaded enriched: {f.name} ({len(df)} rows)")
        dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Gap audit for VepKar concept coverage"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data/vepkar/ directory with meanings_*.csv",
    )
    parser.add_argument(
        "--concepts-wdh",
        type=str,
        default=None,
        help="Optional: path to concepts_wdh.tsv for concept catalog enrichment",
    )
    parser.add_argument(
        "--enriched-dir",
        type=str,
        default=None,
        help="Optional: directory with meanings_*_concept_wdh.csv for WDH disagreement analysis",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: data/sem_cat/results/)",
    )

    args = parser.parse_args()

    # Resolve output directory
    if args.out_dir is None:
        out_dir = Path("data/sem_cat/results")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load concepts_wdh if provided
    concepts_wdh = None
    if args.concepts_wdh:
        print("Loading concepts_wdh...")
        concepts_wdh = load_concepts_wdh(args.concepts_wdh)
        print(f"  {len(concepts_wdh)} concepts")

    # Load enriched meanings if provided
    enriched_df = None
    if args.enriched_dir:
        enriched_path = Path(args.enriched_dir)
        print("Loading enriched meanings...")
        if enriched_path.exists():
            matching = list(enriched_path.glob("meanings_*_concept_wdh.csv"))
            print(f"  Found {len(matching)} enriched file(s) in {args.enriched_dir}")
            for f in sorted(matching):
                print(f"    {f.name}")
        else:
            matching = []
            print(f"  WARNING: enriched directory not found: {args.enriched_dir}")
        enriched_df = _load_enriched_meanings(args.enriched_dir)
        if enriched_df is not None:
            print(f"  Total enriched: {len(enriched_df)} rows")
        elif not matching:
            print(
                "  WARNING: No enriched meanings_*_concept_wdh.csv files were loaded.\n"
                "  WDH disagreement analysis will be skipped.\n"
                "  To enable it, run step 07 (propagate_wdh) first, then point\n"
                "  --enriched-dir to the directory containing its output files."
            )
        else:
            print("  WARNING: Failed to load enriched files (format error?).")
            print("  WDH disagreement analysis will be skipped.")

    # Load merged meanings
    print("Loading meanings...")
    meanings_df = _load_meanings_merged(args.data_dir)

    # Run gap audit
    print("\nRunning gap audit...\n")
    reports = run_gap_audit(
        meanings_df,
        concepts_wdh_df=concepts_wdh,
        enriched_df=enriched_df,
        out_dir=str(out_dir),
    )

    print(f"\nAudit reports saved to {out_dir}/")
    for name, df in reports.items():
        print(f"  {name}: {len(df)} rows")


if __name__ == "__main__":
    main()
