"""Propagate concept-level WDH to meanings with conflict detection.

Propagates WDH from concepts_wdh.tsv to meanings rows that have a
concept_id. Where gloss-based WDH (from WordNet domain lookup) is also
available, compares the two sources and records conflicts.

Output schema adds these columns to the original meanings DataFrame:
    concept_wdh, gloss_wdh, wdh, wdh_source, wdh_conflict, wdh_conflict_note
"""

import pandas as pd
from pathlib import Path
from collections import Counter


def load_concepts_wdh(filepath: str) -> pd.DataFrame:
    """Load concepts_wdh.tsv produced by step 06.

    Required columns:
        concept_id, wdh
    Optional context columns:
        category_id, pos, concept_ru, concept_en
    Extra columns (e.g., legacy provenance) are accepted but ignored.

    Returns DataFrame with columns: concept_id, wdh, and optional context columns.
    """
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()

    required = {"concept_id", "wdh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"concepts_wdh file missing required columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    return df


def propagate_wdh_to_meanings(
    meanings_df: pd.DataFrame,
    concepts_wdh_df: pd.DataFrame,
    gloss_wdh_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate concept WDH to meanings and detect conflicts.

    Args:
        meanings_df: VepKar meanings DataFrame (all languages or per-language).
            Must contain: meaning_id, concept_id (nullable), gloss_primary (if
            gloss_wdh_df is provided), and optionally meaning_en.
        concepts_wdh_df: Output from build_concepts_wdh().
        gloss_wdh_df: Optional DataFrame from WordNet domain lookup with
            columns: gloss_ru, wn_domain. If None, no gloss-based WDH
            comparison is performed.

    Returns:
        (enriched_meanings_df, conflicts_df)
        - enriched_meanings_df: original meanings + new WDH columns
        - conflicts_df: rows where concept_wdh and gloss_wdh disagree
    """
    df = meanings_df.copy()

    df["concept_id"] = df["concept_id"].astype(str).replace("nan", "")

    concept_wdh_map = {}
    for _, row in concepts_wdh_df.iterrows():
        cid = str(row["concept_id"]).strip()
        wdh = str(row.get("wdh", "")).strip()
        if cid and wdh:
            concept_wdh_map[cid] = wdh

    df["concept_wdh"] = df["concept_id"].map(concept_wdh_map).fillna("")

    if gloss_wdh_df is not None:
        gloss_map = {}
        gloss_status_map = {}
        has_lookup_status = "lookup_status" in gloss_wdh_df.columns
        for _, row in gloss_wdh_df.iterrows():
            gr = str(row.get("gloss_ru", "")).strip()
            wd = str(row.get("wn_domain", "")).strip()
            if gr and wd:
                gloss_map[gr] = wd
            if has_lookup_status:
                st = str(row.get("lookup_status", "")).strip()
                if gr and st:
                    gloss_status_map[gr] = st

        if "gloss_primary" not in df.columns:
            from src.sem_cat.utils.gloss_normalizer import primary_gloss
            df["gloss_primary"] = df["meaning_ru"].fillna("").apply(primary_gloss)

        df["gloss_wdh"] = df["gloss_primary"].map(gloss_map).fillna("")
        if has_lookup_status:
            df["gloss_lookup_status"] = df["gloss_primary"].map(gloss_status_map).fillna("")
        else:
            df["gloss_lookup_status"] = ""
    else:
        df["gloss_wdh"] = ""
        df["gloss_lookup_status"] = ""

    def _resolve_wdh(row: pd.Series) -> tuple[str, str, str, str]:
        concept_wdh = str(row.get("concept_wdh", "")).strip()
        gloss_wdh = str(row.get("gloss_wdh", "")).strip()
        lookup_status = str(row.get("gloss_lookup_status", "")).strip()

        gloss_is_meaningful = (
            gloss_wdh
            and gloss_wdh.lower() != "factotum"
            and lookup_status == "found"
        )

        if gloss_is_meaningful:
            is_conflict = bool(concept_wdh and concept_wdh != gloss_wdh)
            if is_conflict:
                note = (
                    f"concept_wdh={concept_wdh} vs gloss_wdh={gloss_wdh}; "
                    f"gloss-based evidence (lookup_status=found) takes priority"
                )
                return (gloss_wdh, "gloss_override", "yes", note)
            else:
                return (gloss_wdh, "gloss_based", "no", "")
        elif concept_wdh:
            return (concept_wdh, "concept_based", "no", "")
        elif gloss_wdh:
            return (gloss_wdh, "gloss_based_fallback", "no", "")
        else:
            return ("", "none", "no", "")

    resolved = df.apply(_resolve_wdh, axis=1, result_type="expand")
    df["wdh"] = resolved[0]
    df["wdh_source"] = resolved[1]
    df["wdh_conflict"] = resolved[2]
    df["wdh_conflict_note"] = resolved[3]

    conflicts_df = df[df["wdh_conflict"] == "yes"].copy()

    return df, conflicts_df


def save_propagated_meanings(
    enriched_df: pd.DataFrame,
    output_path: str,
    lang: str | None = None,
) -> None:
    """Save enriched meanings to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(out, index=False, encoding="utf-8")

    label = f" ({lang})" if lang else ""
    print(f"Saved {len(enriched_df)} meanings{label} to {out}")


def save_conflicts(
    conflicts_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save WDH conflict report to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    conflicts_df.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved {len(conflicts_df)} WDH conflicts to {out}")


def print_propagation_summary(enriched_df: pd.DataFrame) -> None:
    """Print summary statistics for the propagation run."""
    total = len(enriched_df)
    with_concept = enriched_df[enriched_df["concept_wdh"] != ""].shape[0]
    with_gloss = enriched_df["gloss_wdh"] != ""
    with_gloss = with_gloss.sum() if hasattr(with_gloss, "sum") else 0
    with_conflict = (enriched_df["wdh_conflict"] == "yes").sum()
    wdh_assigned = (enriched_df["wdh"] != "").sum()

    print(f"\nPropagation summary:")
    print(f"  Total meanings: {total}")
    print(f"  With concept_id: {with_concept}")
    print(f"  With gloss WDH: {with_gloss}")
    print(f"  WDH conflicts: {with_conflict}")
    print(f"  Final WDH assigned: {wdh_assigned}")

    src_counts = enriched_df["wdh_source"].value_counts()
    for src, cnt in src_counts.items():
        print(f"  wdh_source={src}: {cnt}")
