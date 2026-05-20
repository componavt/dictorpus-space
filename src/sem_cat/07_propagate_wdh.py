"""Step 07: Propagate concept WDH to meanings with conflict detection.

Reads concepts_wdh.tsv and VepKar meanings files, propagates WDH from
concepts to meanings that have a concept_id. Optionally compares with
gloss-based WDH from WordNet domain lookup.

Outputs per-language enriched meanings files and a conflict report.

Usage:
    python3 -m src.sem_cat.07_propagate_wdh \
        --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
        --data-dir data/vepkar/ \
        --out-dir data/sem_cat/results/

    # With gloss-based WDH for conflict detection:
    python3 -m src.sem_cat.07_propagate_wdh \
        --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
        --data-dir data/vepkar/ \
        --domains-file data/sem_cat/04_glosses_wn_domains.csv \
        --out-dir data/sem_cat/results/
"""

import argparse
from pathlib import Path

import pandas as pd

from src.sem_cat.utils.concept_wdh import load_concepts as load_concepts_catalog
from src.sem_cat.utils.meaning_propagation import (
    load_concepts_wdh,
    propagate_wdh_to_meanings,
    save_propagated_meanings,
    save_conflicts,
    print_propagation_summary,
)


def _load_meanings_per_lang(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load meanings_{lang}.csv for each language separately.

    Returns dict of lang -> DataFrame.
    Handles both 8-column (vep/olo/lud) and 10-column (krl) schemas.
    """
    langs = ["vep", "olo", "lud", "krl"]
    result = {}
    for lang in langs:
        filepath = f"{data_dir}/meanings_{lang}.csv"
        path = Path(filepath)
        if not path.exists():
            print(f"  Skipping {filepath} (not found)")
            continue

        df = pd.read_csv(filepath, sep=",", encoding="utf-8-sig", dtype=str)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip()

        # Ensure concept_id column exists (may be missing in older files)
        if "concept_id" not in df.columns:
            df["concept_id"] = ""
        if "category_id" not in df.columns:
            df["category_id"] = ""
        if "meaning_en" not in df.columns:
            df["meaning_en"] = ""

        print(f"  Loaded {len(df)} rows for {lang}")
        result[lang] = df

    return result


def _load_gloss_wdh(filepath: str) -> pd.DataFrame:
    """Load glosses_wn_domains.csv for gloss-based WDH comparison.

    Returns DataFrame with columns: gloss_ru, wn_domain, lookup_status.
    lookup_status is used during WDH resolution to distinguish real
    domain signals (status='found') from lookup failures (factotum fallback).
    """
    df = pd.read_csv(filepath, dtype=str)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    # Keep columns needed for resolution
    needed = {"gloss_ru", "wn_domain", "lookup_status"}
    available = needed & set(df.columns)
    if "gloss_ru" not in available or "wn_domain" not in available:
        raise ValueError(
            f"gloss WDH file missing required columns: gloss_ru or wn_domain. "
            f"Found: {list(df.columns)}"
        )
    keep = [c for c in ["gloss_ru", "wn_domain", "lookup_status"] if c in df.columns]
    return df[keep]


def main():
    parser = argparse.ArgumentParser(
        description="Propagate concept WDH to meanings with conflict detection"
    )
    parser.add_argument(
        "--concepts-wdh",
        type=str,
        required=True,
        help="Path to concepts_wdh.tsv (output of step 06)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data/vepkar/ directory with meanings_*.csv",
    )
    parser.add_argument(
        "--domains-file",
        type=str,
        default=None,
        help="Optional: path to glosses_wn_domains.csv for conflict detection",
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

    # Load concepts_wdh
    print("Loading concepts_wdh...")
    concepts_wdh = load_concepts_wdh(args.concepts_wdh)
    print(f"  {len(concepts_wdh)} concepts with WDH")

    # Load gloss-based WDH if provided
    gloss_wdh = None
    if args.domains_file:
        print("Loading gloss-based WDH...")
        gloss_wdh = _load_gloss_wdh(args.domains_file)
        print(f"  {len(gloss_wdh)} gloss-domain mappings")

    # Load meanings per language
    print("Loading meanings per language...")
    meanings_by_lang = _load_meanings_per_lang(args.data_dir)

    # Propagate WDH for each language
    all_conflicts = []
    for lang, meanings_df in meanings_by_lang.items():
        print(f"\nProcessing {lang}...")
        enriched, conflicts = propagate_wdh_to_meanings(
            meanings_df,
            concepts_wdh,
            gloss_wdh_df=gloss_wdh,
        )

        # Save enriched meanings
        out_file = out_dir / f"meanings_{lang}_concept_wdh.csv"
        save_propagated_meanings(enriched, str(out_file), lang=lang)

        # Collect conflicts
        if len(conflicts) > 0:
            conflicts["lang"] = lang
            all_conflicts.append(conflicts)

        # Print per-language summary
        wdh_assigned = (enriched["wdh"] != "").sum()
        with_concept = (enriched["concept_wdh"] != "").sum()
        print(f"  WDH assigned: {wdh_assigned}/{len(enriched)}")
        print(f"  With concept_id: {with_concept}")

    # Save combined conflict report
    if all_conflicts:
        combined_conflicts = pd.concat(all_conflicts, ignore_index=True)
        conflict_file = out_dir / "wdh_conflicts.csv"
        save_conflicts(combined_conflicts, str(conflict_file))

        # Print conflict summary
        print(f"\nConflict summary:")
        print(f"  Total conflicts: {len(combined_conflicts)}")
        if "wdh_conflict_note" in combined_conflicts.columns:
            # Show top conflict patterns
            notes = combined_conflicts["wdh_conflict_note"].value_counts().head(10)
            for note, cnt in notes.items():
                print(f"  [{cnt}x] {note[:120]}...")
    else:
        print("\nNo WDH conflicts detected.")


if __name__ == "__main__":
    main()
