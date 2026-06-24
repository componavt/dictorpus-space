"""Build concept-level WDH (WordNet Domain Hierarchy) table.

Reads concept_categories_wdh.tsv and the current concept catalog
(concepts_with_english_1445.csv by default), produces a per-concept WDH
assignment with source, confidence, and notes.

Output schema:
    category_id, pos, concept_id, concept_ru, concept_en,
    wdh, wdh_source, wdh_confidence, wdh_note
"""

import pandas as pd
from pathlib import Path


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
    """Normalize a WDH string: strip, lowercase, sort comma-separated values."""
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
    3. Assign source, confidence, and note.

    All concepts should have a category_id that exists in cat_wdh_df.
    If a category is missing from WDH, wdh is set to empty with low confidence.
    """
    # Join concepts with category WDH
    merged = concepts_df.merge(
        cat_wdh_df[["category_id", "wdh"]],
        on="category_id",
        how="left",
    )

    # Normalize WDH values
    merged["wdh"] = merged["wdh"].apply(_normalize_wdh)

    # Determine source, confidence, and note
    def _assign_metadata(row: pd.Series) -> tuple[str, str, str]:
        wdh_val = row.get("wdh", "")
        if wdh_val:
            return (
                "inherited_from_category",
                "medium",
                f"WDH inherited from category {row['category_id']}",
            )
        else:
            return (
                "",
                "low",
                f"No WDH defined for category {row['category_id']}",
            )

    meta = merged.apply(_assign_metadata, axis=1, result_type="expand")
    merged["wdh_source"] = meta[0]
    merged["wdh_confidence"] = meta[1]
    merged["wdh_note"] = meta[2]

    # Select output columns in a stable order
    out_cols = [
        "category_id",
        "pos",
        "concept_id",
        "concept_ru",
        "concept_en",
        "wdh",
        "wdh_source",
        "wdh_confidence",
        "wdh_note",
    ]
    return merged[out_cols].copy()


def save_concepts_wdh(
    concepts_wdh_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save concepts_wdh DataFrame to TSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    concepts_wdh_df.to_csv(out, sep="\t", index=False, encoding="utf-8")
    print(f"Saved {len(concepts_wdh_df)} concept WDH rows to {out}")


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

    print(f"  WDH assigned: {concepts_wdh['wdh'].notna().sum() & (concepts_wdh['wdh'] != '')}")
    print(f"  WDH missing:  {(concepts_wdh['wdh'].isna()) | (concepts_wdh['wdh'] == '')}")

    save_concepts_wdh(concepts_wdh, output_path)
    return concepts_wdh
