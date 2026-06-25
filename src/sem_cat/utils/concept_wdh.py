"""Build concept-level WDH (WordNet Domain Hierarchy) table.

Reads concept_categories_wdh.tsv and the current concept catalog
(concepts_with_english_1445.csv by default), produces a per-concept WDH
assignment.

Output schema (flat concept-level lookup):
    category_id, pos, concept_id, concept_ru, concept_en, wdh

Provenance (source/confidence/note) is implicit in the step-06 logic and
in the category mapping input, not duplicated per output row.
"""

import pandas as pd
from pathlib import Path
from collections import Counter


STEP06_OUTPUT_COLUMNS = [
    "category_id",
    "pos",
    "concept_id",
    "concept_ru",
    "concept_en",
    "wdh",
]


def _explode_wdh_labels(wdh_series) -> list[str]:
    """Extract atomic WDH labels from a pandas Series.

    Splits comma-separated values, strips whitespace, ignores empty parts,
    and returns a flat list of atomic labels (duplicates within a row are
    counted only once per row). Case preserved for counting but normalization
    uses lowercase.

    Args:
        wdh_series: Pandas Series with WDH values (may contain None, NaN,
            blank strings, or comma-separated labels).

    Returns:
        List of atomic WDH labels (strings) after splitting and deduplication
        within each row. Labels are returned as-is (case preserved).
    """
    labels = []
    for raw in wdh_series:
        if pd.isna(raw):
            continue
        text = str(raw).strip()
        if not text:
            continue
        parts = [p.strip() for p in text.split(",") if p.strip()]
        labels.extend(sorted(set(parts)))
    return labels


def collect_wdh_label_stats(wdh_series) -> tuple[int, list[tuple[str, int]]]:
    """Collect statistics on atomic WDH labels.

    Splits each WDH value by comma, counts frequencies of individual labels
    across all concepts, and returns summary stats.

    Args:
        wdh_series: Pandas Series with WDH values.

    Returns:
        Tuple of (unique_label_count, top_labels_list).
        - unique_label_count: number of distinct atomic labels
        - top_labels_list: list of (label, count) tuples, sorted by count
          descending, then label ascending for ties
    """
    labels = _explode_wdh_labels(wdh_series)
    counter = Counter(labels)
    top = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return len(counter), top


def load_category_wdh(filepath: str) -> pd.DataFrame:
    """Load concept_categories_wdh.tsv.

    Returns DataFrame with columns: category_id, name_ru, name_en, wdh.
    """
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    return df


def load_concepts(filepath: str) -> pd.DataFrame:
    """Load concepts catalog (tab-separated).

    Returns DataFrame with columns:
        category_id, pos, concept_id, concept_ru, concept_en,
        definition_ru, definition_en
    """
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    return df


def _normalize_wdh(value: str) -> str:
    """Normalize a WDH string: strip, lowercase, sort comma-separated values.

    Deduplicates labels within a single row.
    """
    if pd.isna(value) or not value.strip():
        return ""
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    return ", ".join(sorted(set(parts)))


def build_concepts_wdh(
    cat_wdh_df: pd.DataFrame,
    concepts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-concept WDH table from category inheritance.

    Strategy:
    1. Join concepts with category WDH on category_id.
    2. Inherit wdh from category.

    All concepts should have a category_id that exists in cat_wdh_df.
    If a category is missing from WDH, wdh is set to empty.
    """
    merged = concepts_df.merge(
        cat_wdh_df[["category_id", "wdh"]],
        on="category_id",
        how="left",
    )

    merged["wdh"] = merged["wdh"].apply(_normalize_wdh)

    for col in STEP06_OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""

    return merged[STEP06_OUTPUT_COLUMNS].copy()


def save_concepts_wdh(
    concepts_wdh_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save concepts_wdh DataFrame to TSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df = concepts_wdh_df.copy()
    out_df = out_df[STEP06_OUTPUT_COLUMNS]
    out_df.to_csv(out, sep="\t", index=False, encoding="utf-8")
    print(f"Saved {len(out_df)} concept WDH rows to {out}")


def run_concepts_wdh(
    cat_wdh_path: str,
    concepts_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Full pipeline: load, build, save concepts_wdh."""
    print("Loading category WDH...")
    cat_wdh = load_category_wdh(cat_wdh_path)
    print(f"  {len(cat_wdh)} categories loaded")

    print("Loading concepts...")
    concepts = load_concepts(concepts_path)
    print(f"  {len(concepts)} concepts loaded")

    print("Building concept-level WDH...")
    concepts_wdh = build_concepts_wdh(cat_wdh, concepts)

    wdh_assigned = (concepts_wdh["wdh"].fillna("") != "")
    wdh_missing = ~wdh_assigned
    print(f"  WDH assigned: {wdh_assigned.sum()}")
    print(f"  WDH missing:  {wdh_missing.sum()}")

    unique_count, top_labels = collect_wdh_label_stats(
        concepts_wdh.loc[wdh_assigned, "wdh"]
    )
    print(f"  Unique WDH labels: {unique_count}")

    if top_labels:
        print("  Top WDH labels:")
        for label, count in top_labels[:10]:
            print(f"    {label}: {count}")

    save_concepts_wdh(concepts_wdh, output_path)
    return concepts_wdh
