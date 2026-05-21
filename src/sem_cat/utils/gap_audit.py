"""Gap audit for VepKar concept coverage.

Produces tabular reports that identify missing or weak concept coverage
in VepKar meanings. Primary grouping axis: meaning_ru.

Reports:
1. meanings_without_concept: meanings that lack concept_id, grouped by
   meaning_ru to find recurring unassigned glosses.
2. concept_usage: how many meanings use each concept_id.
3. category_coverage: per-category statistics on concept assignment.
4. wdh_disagreement: systematic WDH conflicts between concept and gloss sources.
5. meaning_ru_clusters: groups of meanings with the same Russian gloss
   (normalized), optionally enriched with pos, category_id, meaning_en.
"""

import pandas as pd
from pathlib import Path
from collections import Counter


def _normalize_gloss(text: str) -> str:
    """Normalize a Russian gloss for grouping.

    Strips whitespace, lowercases, removes parentheticals.
    """
    if pd.isna(text) or not text.strip():
        return ""
    text = text.strip().lower()
    # Remove parenthetical content
    import re
    text = re.sub(r"\s*\(.*?\)", "", text)
    text = text.strip()
    return text


def analyze_meanings_without_concept(
    meanings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Find meanings that lack concept_id, at both row and gloss levels.

    Returns DataFrame sorted by frequency descending:
        level, meaning_ru, meaning_ru_norm, count, langs, pos_values, sample_ids
    """
    df = meanings_df.copy()
    df["concept_id"] = df["concept_id"].astype(str).replace("nan", "")
    df["meaning_ru_norm"] = df["meaning_ru"].apply(_normalize_gloss)

    without = df[df["concept_id"] == ""].copy()

    # Row-level: count of meaning rows without concept_id across all languages
    row_count = len(without)

    if without.empty:
        result = pd.DataFrame(
            columns=["level", "meaning_ru", "meaning_ru_norm", "count", "langs", "pos_values", "sample_ids"]
        )
        return result

    # Gloss-level: group by unique (meaning_ru, meaning_ru_norm)
    grouped = without.groupby(["meaning_ru", "meaning_ru_norm"]).agg(
        count=("id", "count"),
        langs=("lang", lambda x: ", ".join(sorted(x.unique()))),
        pos_values=("pos", lambda x: ", ".join(sorted(x.unique()))),
        sample_ids=("id", lambda x: ", ".join(x.head(5).astype(str))),
    ).reset_index()

    grouped["level"] = "gloss"
    grouped = grouped.sort_values("count", ascending=False).reset_index(drop=True)

    # Prepend a summary row for row-level coverage
    summary_row = pd.DataFrame([{
        "level": "row",
        "meaning_ru": f"TOTAL ({row_count} rows without concept_id across all languages)",
        "meaning_ru_norm": "",
        "count": row_count,
        "langs": ", ".join(sorted(without["lang"].unique())) if "lang" in without.columns else "",
        "pos_values": "",
        "sample_ids": "",
    }])
    grouped = pd.concat([summary_row, grouped], ignore_index=True)

    return grouped


def analyze_concept_usage(
    meanings_df: pd.DataFrame,
    concepts_wdh_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Analyze how many meanings use each concept_id.

    Returns:
        (in_catalog_usage_df, out_of_catalog_df)
        - in_catalog_usage_df: concept_ids present in concepts_wdh_df
        - out_of_catalog_df: concept_ids found in meanings but absent from catalog

    Schema for in_catalog:
        concept_id, category_id, concept_ru, concept_en, meaning_count,
        langs, wdh, wdh_source, wdh_confidence
    Schema for out_of_catalog:
        concept_id, meaning_count, langs
    """
    df = meanings_df.copy()
    df["concept_id"] = df["concept_id"].astype(str).replace("nan", "")
    with_concept = df[df["concept_id"] != ""].copy()

    usage = with_concept.groupby("concept_id").agg(
        meaning_count=("id", "count"),
        langs=("lang", lambda x: ", ".join(sorted(x.unique()))) if "lang" in with_concept.columns else None,
    ).reset_index()

    if "langs" in usage.columns:
        usage["langs"] = usage["langs"].fillna("")

    if concepts_wdh_df is not None:
        catalog_ids = set(concepts_wdh_df["concept_id"].astype(str).str.strip())
        usage_ids = set(usage["concept_id"].astype(str).str.strip())

        # In-catalog usage
        in_catalog_ids = usage_ids & catalog_ids
        if in_catalog_ids:
            in_catalog = usage[usage["concept_id"].isin(in_catalog_ids)].copy()
            concept_info = concepts_wdh_df[
                ["concept_id", "category_id", "concept_ru", "concept_en",
                 "wdh", "wdh_source", "wdh_confidence"]
            ].copy()
            concept_info["concept_id"] = concept_info["concept_id"].astype(str).str.strip()
            in_catalog["concept_id"] = in_catalog["concept_id"].astype(str).str.strip()
            in_catalog = in_catalog.merge(concept_info, on="concept_id", how="left")
            in_catalog = in_catalog.sort_values("meaning_count", ascending=False).reset_index(drop=True)
        else:
            in_catalog = pd.DataFrame(
                columns=["concept_id", "meaning_count", "langs", "category_id",
                         "concept_ru", "concept_en", "wdh", "wdh_source", "wdh_confidence"]
            )

        # Out-of-catalog observed IDs
        out_ids = usage_ids - catalog_ids
        if out_ids:
            out_of_catalog = usage[usage["concept_id"].isin(out_ids)].copy()
            out_of_catalog = out_of_catalog.sort_values("meaning_count", ascending=False).reset_index(drop=True)
        else:
            out_of_catalog = pd.DataFrame(columns=["concept_id", "meaning_count", "langs"])
    else:
        in_catalog = usage.sort_values("meaning_count", ascending=False).reset_index(drop=True)
        out_of_catalog = None

    return in_catalog, out_of_catalog


def analyze_category_coverage(
    meanings_df: pd.DataFrame,
    concepts_wdh_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-category statistics on concept assignment.

    Returns DataFrame:
        category_id, total_meanings, with_concept, without_concept,
        coverage_pct, unique_concepts, top_concept_id, top_concept_count
    """
    df = meanings_df.copy()
    df["concept_id"] = df["concept_id"].astype(str).replace("nan", "")

    # Ensure category_id column
    if "category_id" not in df.columns:
        df["category_id"] = ""

    # Group by category_id
    stats = df.groupby("category_id").agg(
        total_meanings=("id", "count"),
        with_concept=("concept_id", lambda x: (x != "").sum()),
        unique_concepts=("concept_id", lambda x: x[x != ""].nunique()),
    ).reset_index()

    stats["without_concept"] = stats["total_meanings"] - stats["with_concept"]
    stats["coverage_pct"] = (
        stats["with_concept"] / stats["total_meanings"] * 100
    ).round(1)

    # Find top concept per category
    with_concept = df[df["concept_id"] != ""]
    top_concepts = (
        with_concept.groupby(["category_id", "concept_id"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("category_id", keep="first")
        .rename(columns={"concept_id": "top_concept_id", "cnt": "top_concept_count"})
    )

    stats = stats.merge(top_concepts[["category_id", "top_concept_id", "top_concept_count"]],
                        on="category_id", how="left")

    stats = stats.sort_values("total_meanings", ascending=False).reset_index(drop=True)
    return stats


def analyze_wdh_disagreement(
    enriched_meanings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Find systematic WDH disagreements between concept and gloss sources.

    Returns DataFrame:
        concept_wdh, gloss_wdh, conflict_count, meaning_ru_samples, langs
    """
    df = enriched_meanings_df.copy()
    conflicts = df[df["wdh_conflict"] == "yes"].copy()

    if conflicts.empty:
        return pd.DataFrame(
            columns=["concept_wdh", "gloss_wdh", "conflict_count",
                     "meaning_ru_samples", "langs"]
        )

    grouped = conflicts.groupby(["concept_wdh", "gloss_wdh"]).agg(
        conflict_count=("id", "count"),
        meaning_ru_samples=("meaning_ru", lambda x: "; ".join(x.head(3))),
        langs=("lang", lambda x: ", ".join(sorted(x.unique()))),
    ).reset_index()

    grouped = grouped.sort_values("conflict_count", ascending=False).reset_index(drop=True)
    return grouped


def build_meaning_ru_clusters(
    meanings_df: pd.DataFrame,
    min_count: int = 2,
) -> pd.DataFrame:
    """Group meanings by normalized meaning_ru for clustering analysis.

    Returns DataFrame:
        meaning_ru_norm, count, concept_ids, category_ids, pos_values,
        meaning_en_values, langs, sample_meaning_ru
    """
    df = meanings_df.copy()
    df["concept_id"] = df["concept_id"].fillna("").astype(str).replace("nan", "")
    if "category_id" not in df.columns:
        df["category_id"] = ""
    df["category_id"] = df["category_id"].fillna("").astype(str).replace("nan", "")
    if "meaning_en" not in df.columns:
        df["meaning_en"] = ""
    df["meaning_ru_norm"] = df["meaning_ru"].apply(_normalize_gloss)

    # Filter out empty glosses
    df = df[df["meaning_ru_norm"] != ""]

    grouped = df.groupby("meaning_ru_norm").agg(
        count=("id", "count"),
        concept_ids=("concept_id", lambda x: ", ".join(sorted(
            str(v) for v in x.unique() if str(v).strip() and str(v) != "nan"
        ))),
        category_ids=("category_id", lambda x: ", ".join(sorted(
            str(v) for v in x.unique() if str(v).strip() and str(v) != "nan"
        ))),
        pos_values=("pos", lambda x: ", ".join(sorted(x.unique()))),
        meaning_en_values=("meaning_en", lambda x: ", ".join(sorted(set(
            str(v) for v in x if pd.notna(v) and str(v).strip() and str(v) != "nan"
        )))),
        langs=("lang", lambda x: ", ".join(sorted(x.unique()))),
        sample_meaning_ru=("meaning_ru", lambda x: "; ".join(x.head(3))),
    ).reset_index()

    # Filter by minimum count
    if min_count > 1:
        grouped = grouped[grouped["count"] >= min_count]

    grouped = grouped.sort_values("count", ascending=False).reset_index(drop=True)
    return grouped


def run_gap_audit(
    meanings_df: pd.DataFrame,
    concepts_wdh_df: pd.DataFrame | None = None,
    enriched_df: pd.DataFrame | None = None,
    out_dir: str = "data/sem_cat/results/",
) -> dict[str, pd.DataFrame]:
    """Run all gap audit analyses and save reports.

    Returns dict of report_name -> DataFrame.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reports = {}

    # 1. Meanings without concept
    print("Analyzing meanings without concept_id...")
    no_concept = analyze_meanings_without_concept(meanings_df)
    reports["meanings_without_concept"] = no_concept
    no_concept.to_csv(out / "audit_meanings_without_concept.csv", index=False)
    row_count = no_concept[no_concept["level"] == "row"]["count"].sum()
    gloss_count = len(no_concept[no_concept["level"] == "gloss"])
    print(f"  {int(row_count)} meaning rows without concept_id (row-level, from raw meanings_*.csv)")
    print(f"  {gloss_count} unique glosses without any concept (gloss-level, deduplicated)")
    if gloss_count > 0:
        gloss_rows = no_concept[no_concept["level"] == "gloss"]
        print(f"  Top 5: {gloss_rows.head(5)[['meaning_ru', 'count']].to_dict('records')}")

    # 2. Concept usage (split into in-catalog and out-of-catalog)
    print("Analyzing concept usage...")
    usage, usage_outside_catalog = analyze_concept_usage(meanings_df, concepts_wdh_df)
    reports["concept_usage"] = usage
    usage.to_csv(out / "audit_concept_usage.csv", index=False)
    print(f"  {len(usage)} concepts in use (from catalog)")

    if usage_outside_catalog is not None:
        usage_outside_catalog.to_csv(out / "audit_concept_ids_outside_catalog.csv", index=False)
        reports["concept_ids_outside_catalog"] = usage_outside_catalog
        print(f"  {len(usage_outside_catalog)} concept_ids outside catalog -> audit_concept_ids_outside_catalog.csv")
    else:
        print(f"  0 concept_ids outside catalog")

    if len(usage) > 0:
        unused_in_catalog = 0
        if concepts_wdh_df is not None:
            catalog_ids = set(concepts_wdh_df["concept_id"].astype(str).str.strip())
            used_ids = set(usage["concept_id"].astype(str).str.strip())
            unused_in_catalog = len(catalog_ids - used_ids)
        print(f"  Concepts in catalog but not used: {unused_in_catalog}")

    # 3. Category coverage
    print("Analyzing category coverage...")
    cat_cov = analyze_category_coverage(meanings_df, concepts_wdh_df)
    reports["category_coverage"] = cat_cov
    cat_cov.to_csv(out / "audit_category_coverage.csv", index=False)
    print(f"  {len(cat_cov)} categories found in meanings")

    # 4. WDH disagreement (if enriched data available)
    if enriched_df is not None and "wdh_conflict" in enriched_df.columns:
        print("Analyzing WDH disagreements...")
        disagreement = analyze_wdh_disagreement(enriched_df)
        reports["wdh_disagreement"] = disagreement
        disagreement.to_csv(out / "audit_wdh_disagreement.csv", index=False)
        print(f"  {len(disagreement)} unique WDH conflict patterns")

    # 5. Meaning RU clusters
    print("Building meaning_ru clusters...")
    clusters = build_meaning_ru_clusters(meanings_df, min_count=2)
    reports["meaning_ru_clusters"] = clusters
    clusters.to_csv(out / "audit_meaning_ru_clusters.csv", index=False)
    print(f"  {len(clusters)} clusters (min_count=2)")

    return reports
