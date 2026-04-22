"""
Multi-model translation comparison and expert review queue builder.

Compares translation outputs from N models, computes total risk scores,
detects consensus vs disagreement, and produces:
  1) Full comparison table
  2) Suspicious review queue (sorted by risk)
  3) Gold template for expert editing

This script is step 03 in the semantic domain mapping pipeline:
  01 - count meanings and glosses (Jupyter notebook, read-only)
  02 - translate Russian glosses to English
  03 - compare translation outputs and build expert review queue [THIS FILE]
  04 - WordNet synset lookup
  05 - assign semantic domains to meanings

This script does NOT use WordNet, NLTK, or any translation model.
It only reads CSV files, computes metrics, and writes CSV files.

EXAMPLE COMMANDS:
  # Compare all 6 models
  python3 -m src.sem_cat.03_compare_translations \\
      --translations google=data/sem_cat/02_glosses_translated_google.csv \\
      --translations helsinki_opus_mt_ru_en=data/sem_cat/02_glosses_translated_helsinki_opus_mt_ru_en.csv \\
      --translations nllb_distilled_1_3b=data/sem_cat/02_glosses_translated_nllb_distilled_1_3b.csv \\
      --translations nllb_1_3b=data/sem_cat/02_glosses_translated_nllb_1_3b.csv \\
      --translations nllb_3_3b=data/sem_cat/02_glosses_translated_nllb_3_3b.csv \\
      --translations wmt19_ru_en=data/sem_cat/02_glosses_translated_wmt19_ru_en.csv

  # Compare only 3 models
  python3 -m src.sem_cat.03_compare_translations \\
      --translations google=data/sem_cat/02_glosses_translated_google.csv \\
      --translations helsinki_opus_mt_ru_en=data/sem_cat/02_glosses_translated_helsinki_opus_mt_ru_en.csv \\
      --translations nllb_distilled_1_3b=data/sem_cat/02_glosses_translated_nllb_distilled_1_3b.csv

  # Export top-k risky rows only
  python3 -m src.sem_cat.03_compare_translations \\
      --translations google=data/sem_cat/02_glosses_translated_google.csv \\
      --translations nllb_distilled_1_3b=data/sem_cat/02_glosses_translated_nllb_distilled_1_3b.csv \\
      --top-k 500 --single-word-first

  # Legacy compatibility (pairwise Marian/Google)
  python3 -m src.sem_cat.03_compare_translations \\
      --marian-file data/sem_cat/02_glosses_translated_marian.csv \\
      --google-file data/sem_cat/02_glosses_translated_google.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

from src.sem_cat.compare.loading import (
    parse_translation_arg,
    load_single_model,
    merge_all_models,
)
from src.sem_cat.compare.normalization import (
    normalize_output_for_comparison,
    output_similarity,
)
from src.sem_cat.compare.consensus import cluster_outputs
from src.sem_cat.compare.complexity import compute_gloss_complexity
from src.sem_cat.compare.risk import (
    compute_total_risk,
    compute_risk_level,
    ComparisonRiskConfig,
)
from src.sem_cat.compare.proposal import select_proposed_translation
from src.sem_cat.compare.output_tables import (
    build_full_comparison_df,
    build_review_queue_df,
    build_gold_template_df,
)
from src.sem_cat.compare.data_structures import (
    ComparisonResult,
    ModelOutput,
)


# ---------------------------------------------------------------------------
# Row-level comparison logic
# ---------------------------------------------------------------------------

def _is_blank_pd(value) -> bool:
    """Check if a pandas value is blank."""
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _safe_float(value, default=0.0) -> float:
    """Parse float from string/number."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_bool(value, default=True) -> bool:
    """Parse bool from string/bool."""
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return default


def _parse_flags(flags_str) -> set[str]:
    """Parse semicolon-separated flags string."""
    if _is_blank_pd(flags_str):
        return set()
    return set(str(flags_str).split(";"))


def _collect_model_outputs(row: pd.Series, model_keys: list[str]) -> list[ModelOutput]:
    """Extract ModelOutput objects from a merged DataFrame row."""
    outputs: list[ModelOutput] = []
    for mk in model_keys:
        gloss_en_col = f"{mk}__gloss_en"
        if gloss_en_col not in row.index:
            continue

        gloss_en = str(row.get(gloss_en_col, "")).strip() if not _is_blank_pd(row.get(gloss_en_col)) else ""
        qa_keep = _safe_bool(row.get(f"{mk}__qa_keep", True))
        qa_score = _safe_float(row.get(f"{mk}__qa_score", 0.0))
        qa_flags = _parse_flags(row.get(f"{mk}__qa_flags", ""))
        rt = row.get(f"{mk}__roundtrip_distance")
        rt_val = _safe_float(rt) if not _is_blank_pd(rt) else None
        model_name = str(row.get(f"{mk}__model_name", mk)) if not _is_blank_pd(row.get(f"{mk}__model_name")) else mk

        norm_en = normalize_output_for_comparison(gloss_en) if gloss_en else ""

        outputs.append(ModelOutput(
            model_key=mk,
            model_name=model_name,
            gloss_en=gloss_en,
            qa_keep=qa_keep,
            qa_score=qa_score,
            qa_flags=qa_flags,
            roundtrip_distance=rt_val,
            normalized_gloss_en=norm_en,
        ))
    return outputs


def _compute_consensus_metrics(
    outputs: list[ModelOutput],
    clusters: list,
) -> dict[str, float | int]:
    """Compute consensus and disagreement metrics."""
    non_blank = [o for o in outputs if o.gloss_en and o.gloss_en.strip()]
    good = [o for o in non_blank if o.qa_keep]
    unique_norm = set(o.normalized_gloss_en for o in non_blank if o.normalized_gloss_en)

    non_blank_count = len(non_blank)
    good_count = len(good)
    unique_count = len(unique_norm)
    largest_cluster = len(clusters[0].model_keys) if clusters else 0

    consensus_ratio = largest_cluster / max(1, non_blank_count) if non_blank_count > 0 else 0.0

    # Disagreement: higher when many unique outputs and low consensus
    if non_blank_count <= 1:
        disagreement = 0.0
    else:
        disagreement = (1.0 - consensus_ratio) * (unique_count / max(1, non_blank_count))

    return {
        "non_blank_model_count": non_blank_count,
        "good_model_count": good_count,
        "unique_output_count": unique_count,
        "largest_cluster_size": largest_cluster,
        "consensus_ratio": round(consensus_ratio, 3),
        "disagreement_score": round(disagreement, 3),
    }


def process_gloss_row(
    row: pd.Series,
    model_keys: list[str],
    total_models: int,
    risk_threshold: float = 0.35,
    risk_config: ComparisonRiskConfig | None = None,
) -> ComparisonResult:
    """Process a single gloss_ru row into a ComparisonResult."""
    gloss_ru = str(row.get("gloss_ru", "")).strip()
    outputs = _collect_model_outputs(row, model_keys)

    # Complexity
    complexity_score, complexity_reasons = compute_gloss_complexity(gloss_ru)

    # Consensus clustering
    non_blank = [o for o in outputs if o.gloss_en and o.gloss_en.strip()]
    clusters = cluster_outputs(non_blank)

    # Consensus metrics
    metrics = _compute_consensus_metrics(outputs, clusters)

    # Total risk
    risk_score, risk_reasons = compute_total_risk(
        outputs=outputs,
        total_models=total_models,
        largest_cluster_size=metrics["largest_cluster_size"],
        good_model_count=metrics["good_model_count"],
        consensus_ratio=metrics["consensus_ratio"],
        disagreement_score=metrics["disagreement_score"],
        complexity_score=complexity_score,
        config=risk_config,
    )

    # Add complexity reasons to risk reasons
    for reason in complexity_reasons:
        if reason not in risk_reasons:
            risk_reasons.append(reason)

    risk_level = compute_risk_level(risk_score, risk_config)

    # Proposed translation
    proposed, preferred_source, chosen_key, decision_reason = select_proposed_translation(
        clusters=clusters,
        all_outputs=outputs,
        total_risk=risk_score,
        risk_threshold=risk_threshold,
    )

    # Override to manual review if risk is very high
    if risk_score >= 0.65 and decision_reason not in ("all_blank", "no_clear_winner"):
        preferred_source = "manual_review"
        decision_reason = "high_risk_manual_review"

    needs_review = (
        risk_score >= risk_threshold
        or decision_reason in ("manual_review", "no_clear_winner", "all_blank", "high_risk_manual_review")
        or (proposed and "sentence_like" in decision_reason)
    )

    # Attach outputs to result for later column building
    result = ComparisonResult(
        gloss_ru=gloss_ru,
        proposed_gloss_en=proposed,
        preferred_source=preferred_source,
        chosen_from_model_key=chosen_key,
        decision_reason=decision_reason,
        total_risk=risk_score,
        risk_level=risk_level,
        risk_reasons=risk_reasons,
        consensus_ratio=metrics["consensus_ratio"],
        disagreement_score=metrics["disagreement_score"],
        non_blank_model_count=metrics["non_blank_model_count"],
        good_model_count=metrics["good_model_count"],
        unique_output_count=metrics["unique_output_count"],
        largest_cluster_size=metrics["largest_cluster_size"],
        gloss_complexity_score=complexity_score,
        gloss_complexity_reasons=complexity_reasons,
        needs_expert_review=needs_review,
    )
    # Attach outputs for output table building (not in dataclass)
    result._outputs = outputs  # type: ignore[attr-defined]

    return result


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_comparison_summary(
    results: list[ComparisonResult],
    model_keys: list[str],
    out_path: Path,
    review_path: Path,
    gold_path: Path,
) -> None:
    """Print console summary statistics."""
    total = len(results)
    if total == 0:
        print("No glosses to compare.")
        return

    high = sum(1 for r in results if r.risk_level == "high")
    medium = sum(1 for r in results if r.risk_level == "medium")
    low = sum(1 for r in results if r.risk_level == "low")

    all_blank = sum(1 for r in results if r.non_blank_model_count == 0)
    strong_consensus = sum(1 for r in results if r.consensus_ratio >= 0.6 and r.good_model_count >= 3)
    needs_review = sum(1 for r in results if r.needs_expert_review)

    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total glosses merged:       {total}")
    print(f"Models compared:            {len(model_keys)}")
    print(f"Model keys:                 {', '.join(model_keys)}")
    print()
    print(f"Coverage:")
    print(f"  All blank outputs:        {all_blank}")
    print(f"  Strong consensus:         {strong_consensus}")
    print()
    print(f"Risk distribution:")
    print(f"  High   (>= 0.65):        {high}")
    print(f"  Medium (0.35-0.65):      {medium}")
    print(f"  Low    (< 0.35):         {low}")
    print()
    print(f"Expert review queue:        {needs_review} rows -> {review_path}")
    print(f"Gold standard template:     {needs_review} rows -> {gold_path}")
    print(f"Full comparison:            {total} rows -> {out_path}")

    # Per-model stats
    print()
    print(f"Per-model coverage:")
    for mk in model_keys:
        non_blank = sum(
            1 for r in results
            for o in getattr(r, '_outputs', [])
            if o.model_key == mk and o.gloss_en and o.gloss_en.strip()
        )
        avg_qa = 0.0
        qa_count = 0
        for r in results:
            for o in getattr(r, '_outputs', []):
                if o.model_key == mk and o.gloss_en and o.gloss_en.strip():
                    avg_qa += o.qa_score
                    qa_count += 1
        avg_qa = avg_qa / qa_count if qa_count > 0 else 0.0
        coverage = non_blank / total * 100 if total > 0 else 0
        print(f"  {mk:30s}  coverage={coverage:5.1f}%  avg_qa={avg_qa:.2f}")


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------

def _build_legacy_translation_map(
    marian_file: str | None,
    google_file: str | None,
) -> dict[str, Path]:
    """Build model_key -> path map from legacy --marian-file / --google-file args."""
    mapping: dict[str, Path] = {}
    if marian_file:
        p = Path(marian_file)
        if p.exists():
            mapping["helsinki_opus_mt_ru_en"] = p
    if google_file:
        p = Path(google_file)
        if p.exists():
            mapping["google"] = p
    return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare N-model translation outputs and build expert review queue",
    )

    # New repeated argument
    parser.add_argument(
        "--translations",
        action="append",
        default=[],
        metavar="MODEL_KEY=PATH",
        help="Translation file as model_key=path.csv (repeatable)",
    )

    # Legacy compatibility
    parser.add_argument(
        "--marian-file",
        type=str,
        default=None,
        help="Legacy: Path to Marian translations CSV",
    )
    parser.add_argument(
        "--google-file",
        type=str,
        default=None,
        help="Legacy: Path to Google translations CSV",
    )

    # Output paths
    parser.add_argument(
        "--out-file",
        type=str,
        default="data/sem_cat/03_translation_comparison_full.csv",
        help="Path to full comparison output CSV",
    )
    parser.add_argument(
        "--review-file",
        type=str,
        default="data/sem_cat/03_translation_review_queue.csv",
        help="Path to expert review queue CSV",
    )
    parser.add_argument(
        "--gold-template-file",
        type=str,
        default="data/sem_cat/03_translation_gold_template.csv",
        help="Path to gold standard template CSV",
    )

    # Options
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.35,
        help="Risk threshold for flagging reviews (default: 0.35)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Include only top-k most risky rows in review file",
    )
    parser.add_argument(
        "--include-low-risk",
        action="store_true",
        help="Include all rows in review file regardless of risk",
    )
    parser.add_argument(
        "--single-word-first",
        action="store_true",
        help="Place single-word Russian glosses earlier in review queue",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    # Resolve translation inputs
    model_map: dict[str, Path] = {}

    # Parse --translations arguments
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

    # Legacy fallback
    if not model_map:
        legacy_map = _build_legacy_translation_map(args.marian_file, args.google_file)
        if legacy_map:
            print("WARNING: Using legacy --marian-file / --google-file arguments.")
            print("         Prefer --translations model_key=path.csv for N-model comparison.")
            model_map = legacy_map
        else:
            print("ERROR: No translation files provided.")
            print("  Use --translations model_key=path.csv (repeatable)")
            print("  Or legacy --marian-file / --google-file")
            sys.exit(1)

    model_keys = sorted(model_map.keys())
    total_models = len(model_keys)

    if args.verbose:
        print(f"Loading {total_models} model files: {', '.join(model_keys)}")

    # Load and merge
    model_dfs = {}
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

    # Process each row
    risk_config = ComparisonRiskConfig()
    results: list[ComparisonResult] = []

    for _, row in merged.iterrows():
        result = process_gloss_row(
            row,
            model_keys=model_keys,
            total_models=total_models,
            risk_threshold=args.risk_threshold,
            risk_config=risk_config,
        )
        results.append(result)

    # Build output DataFrame
    full_df = build_full_comparison_df(results, model_keys)

    # Sort review queue
    if args.single_word_first:
        full_df = full_df.sort_values(
            by=["total_risk", "is_singleword", "gloss_ru"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    else:
        full_df = full_df.sort_values(
            by=["total_risk", "gloss_ru"],
            ascending=[False, True],
        ).reset_index(drop=True)

    # Build review queue
    if args.include_low_risk:
        review_df = build_review_queue_df(full_df)
    else:
        review_df = build_review_queue_df(full_df)

    # Apply top-k
    if args.top_k is not None:
        review_df = review_df.head(args.top_k)

    # Build gold template
    gold_df = build_gold_template_df(review_df)

    # Write output files
    out_path = Path(args.out_file)
    review_path = Path(args.review_file)
    gold_path = Path(args.gold_template_file)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(out_path, index=False)
    review_df.to_csv(review_path, index=False)
    gold_df.to_csv(gold_path, index=False)

    # Summary
    print_comparison_summary(results, model_keys, out_path, review_path, gold_path)


if __name__ == "__main__":
    main()
