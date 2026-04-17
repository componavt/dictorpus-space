"""
Compare translation outputs from Marian and Google backends to find the most
reliable translation per gloss_ru, and produce a sorted expert review queue.

This script is step 03 in the semantic domain mapping pipeline:
  01 – count meanings and glosses (Jupyter notebook, read-only)
  02 – translate Russian glosses to English [02_translate_glosses.py]
  03 – compare two translation backends and build expert review queue [THIS FILE]
  04 – WordNet synset lookup [04_wordnet_lookup.py]
  05 – assign semantic domains to meanings [05_assign_domains.py]

This script does NOT use WordNet, NLTK, or any translation model.
It only reads CSV files, computes metrics, and writes CSV files.

EXAMPLE COMMANDS:
  # Default run
  python3 -m src.sem_cat.03_compare_translations

  # Single-word glosses first, stricter threshold
  python3 -m src.sem_cat.03_compare_translations \\
      --single-word-first --risk-threshold 0.30

  # Top 500 most risky rows only
  python3 -m src.sem_cat.03_compare_translations \\
      --top-k 500 --single-word-first

  # Conservative mode (more rows go to manual review)
  python3 -m src.sem_cat.03_compare_translations \\
      --prefer-backend-strategy conservative

  # Custom input files
  python3 -m src.sem_cat.03_compare_translations \\
      --marian-file data/sem_cat/02_glosses_translated_marian_rt.csv \\
      --google-file data/sem_cat/02_glosses_translated_google_rt.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Set

import pandas as pd


def normalize_text(text) -> str:
    """Lowercase, strip whitespace, remove leading/trailing punctuation.
    Returns "" for NaN/None/empty."""
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text).lower().strip()
    text = text.strip(".,!?;:\"'()-")
    return text


def is_blank(value) -> bool:
    """True if NaN, None, empty string, or whitespace-only after strip."""
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def token_count(text) -> int:
    """Count whitespace-separated tokens. Returns 0 for blank."""
    if is_blank(text):
        return 0
    return len(str(text).split())


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of lowercased token sets. Returns 0.0 if either blank."""
    if is_blank(a) or is_blank(b):
        return 0.0
    tokens_a = set(str(a).lower().split())
    tokens_b = set(str(b).lower().split())
    if not tokens_a and not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard dynamic programming Levenshtein."""
    s1, s2 = str(s1), str(s2)
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[len1][len2]


def normalized_edit_similarity(a: str, b: str) -> float:
    """1 - normalized_edit_distance (case-insensitive).
    Returns 1.0 if both blank, 0.0 if one blank."""
    a, b = str(a).lower(), str(b).lower()
    if is_blank(a) and is_blank(b):
        return 1.0
    if is_blank(a) or is_blank(b):
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    distance = levenshtein_distance(a, b)
    return 1.0 - (distance / max_len)


def parse_qa_flags(flags_str) -> Set[str]:
    """Split semicolon-separated flags. Return empty set for blank."""
    if is_blank(flags_str):
        return set()
    return set(str(flags_str).split(";"))


def safe_float(value, default=None):
    """Parse float from string/number. Return default if unparseable."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value, default=True) -> bool:
    """Parse bool from "True"/"False"/True/False. Return default if unparseable."""
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


def is_singleword_russian(text: str) -> bool:
    """True if gloss_ru has exactly one whitespace-separated token."""
    if is_blank(text):
        return False
    return token_count(text) == 1


def load_and_prepare_data(
    marian_path: Path,
    google_path: Path,
    verbose: bool = False
) -> pd.DataFrame:
    """Load both CSV files and merge on gloss_ru with outer join."""
    if not marian_path.exists():
        print(f"ERROR: Marian file not found: {marian_path}")
        sys.exit(1)
    if not google_path.exists():
        print(f"ERROR: Google file not found: {google_path}")
        sys.exit(1)
    
    if verbose:
        print(f"Loading Marian file: {marian_path}")
    df_marian = pd.read_csv(marian_path, dtype=str)
    
    if verbose:
        print(f"Loading Google file: {google_path}")
    df_google = pd.read_csv(google_path, dtype=str)
    
    # Prefix columns
    marian_cols = {
        "gloss_en": "marian_gloss_en",
        "qa_keep": "marian_qa_keep",
        "qa_score": "marian_qa_score",
        "qa_flags": "marian_qa_flags",
        "gloss_ru_back": "marian_gloss_ru_back",
        "roundtrip_distance": "marian_roundtrip_distance"
    }
    google_cols = {
        "gloss_en": "google_gloss_en",
        "qa_keep": "google_qa_keep",
        "qa_score": "google_qa_score",
        "qa_flags": "google_qa_flags",
        "gloss_ru_back": "google_gloss_ru_back",
        "roundtrip_distance": "google_roundtrip_distance"
    }
    
    df_marian = df_marian.rename(columns=marian_cols)
    df_google = df_google.rename(columns=google_cols)
    
    # Keep only gloss_ru and prefixed columns
    marian_keep = ["gloss_ru"] + list(marian_cols.values())
    google_keep = ["gloss_ru"] + list(google_cols.values())
    
    df_marian = df_marian[[c for c in marian_keep if c in df_marian.columns]]
    df_google = df_google[[c for c in google_keep if c in df_google.columns]]
    
    # Merge on gloss_ru with outer join
    df_merged = pd.merge(df_marian, df_google, on="gloss_ru", how="outer")
    
    return df_merged


def normalize_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and type-convert all merged fields."""
    df = df.copy()
    
    # Normalize gloss_en fields
    for col in ["marian_gloss_en", "google_gloss_en"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "" if is_blank(x) else str(x).strip())
    
    # Normalize qa_keep (default=True when gloss_en is non-blank)
    for prefix in ["marian", "google"]:
        qa_keep_col = f"{prefix}_qa_keep"
        gloss_en_col = f"{prefix}_gloss_en"
        if qa_keep_col in df.columns:
            if gloss_en_col in df.columns:
                df[qa_keep_col] = df.apply(
                    lambda row: safe_bool(row[qa_keep_col], default=not is_blank(row[gloss_en_col])),
                    axis=1
                )
            else:
                df[qa_keep_col] = df[qa_keep_col].apply(lambda x: safe_bool(x, default=True))
        else:
            df[qa_keep_col] = True
    
    # Normalize qa_score (default=0.0)
    for prefix in ["marian", "google"]:
        col = f"{prefix}_qa_score"
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_float(x, default=0.0))
        else:
            df[col] = 0.0
    
    # Normalize roundtrip_distance (default=None)
    for prefix in ["marian", "google"]:
        col = f"{prefix}_roundtrip_distance"
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_float(x, default=None))
        else:
            df[col] = None
    
    # Add is_singleword column
    df["is_singleword"] = df["gloss_ru"].apply(is_singleword_russian)
    
    return df


def compute_risk_score(row: pd.DataFrame) -> Tuple[float, List[str]]:
    """Compute risk_score in [0.0, 1.0] and risk_reasons list."""
    risk_score = 0.0
    risk_reasons = []
    
    marian_en = row.get("marian_gloss_en", "")
    google_en = row.get("google_gloss_en", "")
    marian_blank = is_blank(marian_en)
    google_blank = is_blank(google_en)
    
    marian_qa_keep = row.get("marian_qa_keep", True)
    google_qa_keep = row.get("google_qa_keep", True)
    
    marian_qa_score = row.get("marian_qa_score", 0.0)
    google_qa_score = row.get("google_qa_score", 0.0)
    
    marian_rt = row.get("marian_roundtrip_distance")
    google_rt = row.get("google_roundtrip_distance")
    
    marian_flags = parse_qa_flags(row.get("marian_qa_flags", ""))
    google_flags = parse_qa_flags(row.get("google_qa_flags", ""))
    
    # +0.45 both_blank
    if marian_blank and google_blank:
        risk_score += 0.45
        risk_reasons.append("both_blank")
    # +0.30 one_blank
    elif marian_blank or google_blank:
        risk_score += 0.30
        risk_reasons.append("one_blank")
    
    # +0.20 either_qa_keep_false
    if not marian_qa_keep or not google_qa_keep:
        risk_score += 0.20
        risk_reasons.append("either_qa_keep_false")
    
    # +min(0.20, 0.20 * max(marian_qa_score, google_qa_score)) high_qa_score
    max_qa_score = max(marian_qa_score, google_qa_score)
    qa_penalty = min(0.20, 0.20 * max_qa_score)
    if qa_penalty > 0:
        risk_score += qa_penalty
        risk_reasons.append("high_qa_score")
    
    # +0.15 marian_roundtrip_poor
    if marian_rt is not None and marian_rt > 0.50:
        risk_score += 0.15
        risk_reasons.append("marian_roundtrip_poor")
    
    # +0.15 google_roundtrip_poor
    if google_rt is not None and google_rt > 0.50:
        risk_score += 0.15
        risk_reasons.append("google_roundtrip_poor")
    
    # +0.20 backends_disagree
    if not marian_blank and not google_blank:
        similarity = normalized_edit_similarity(marian_en, google_en)
        if similarity < 0.40:
            risk_score += 0.20
            risk_reasons.append("backends_disagree")
    
    # +0.10 both_flagged
    if marian_flags and google_flags:
        risk_score += 0.10
        risk_reasons.append("both_flagged")
    
    # +0.10 length_mismatch
    if not marian_blank and not google_blank:
        marian_tokens = token_count(marian_en)
        google_tokens = token_count(google_en)
        if abs(marian_tokens - google_tokens) >= 3:
            risk_score += 0.10
            risk_reasons.append("length_mismatch")
    
    # Cap at 1.0
    risk_score = min(1.0, risk_score)
    
    return risk_score, risk_reasons


def determine_preferred_backend(
    row: pd.DataFrame,
    strategy: str = "heuristic",
    risk_score: float = 0.0,
    risk_threshold: float = 0.35
) -> Tuple[str, str, str]:
    """
    Determine preferred_backend, proposed_gloss_en, decision_reason.
    
    Returns:
        (preferred_backend, proposed_gloss_en, decision_reason)
    """
    marian_en = row.get("marian_gloss_en", "")
    google_en = row.get("google_gloss_en", "")
    marian_blank = is_blank(marian_en)
    google_blank = is_blank(google_en)
    
    marian_qa_keep = row.get("marian_qa_keep", True)
    google_qa_keep = row.get("google_qa_keep", True)
    
    marian_qa_score = row.get("marian_qa_score", 0.0)
    google_qa_score = row.get("google_qa_score", 0.0)
    
    marian_flags = parse_qa_flags(row.get("marian_qa_flags", ""))
    google_flags = parse_qa_flags(row.get("google_qa_flags", ""))
    
    marian_rt = row.get("marian_roundtrip_distance")
    google_rt = row.get("google_roundtrip_distance")
    
    is_singleword = row.get("is_singleword", False)
    
    # Adjust near_match threshold for conservative strategy
    near_match_threshold = 0.95 if strategy == "conservative" else 0.85
    
    # 1. NEAR MATCH
    if not marian_blank and not google_blank:
        similarity = normalized_edit_similarity(marian_en, google_en)
        if similarity >= near_match_threshold:
            return "tie", marian_en, "near_match"
    
    # 2. BLANK vs NON-BLANK
    if marian_blank and not google_blank:
        return "google", google_en, "other_blank"
    if google_blank and not marian_blank:
        return "marian", marian_en, "other_blank"
    
    # 3. QA_KEEP MISMATCH
    if marian_qa_keep != google_qa_keep:
        if marian_qa_keep:
            return "marian", marian_en, "qa_keep_wins"
        else:
            return "google", google_en, "qa_keep_wins"
    
    # 4. HEURISTIC SCORE (both non-blank, both qa_keep=True)
    if not marian_blank and not google_blank:
        marian_points = 0
        google_points = 0
        
        # a. lower qa_score
        if marian_qa_score < google_qa_score:
            marian_points += 1
        elif google_qa_score < marian_qa_score:
            google_points += 1
        
        # b. fewer qa_flags
        if len(marian_flags) < len(google_flags):
            marian_points += 1
        elif len(google_flags) < len(marian_flags):
            google_points += 1
        
        # c. lower roundtrip_distance (only if both available)
        if marian_rt is not None and google_rt is not None:
            if marian_rt < google_rt:
                marian_points += 1
            elif google_rt < marian_rt:
                google_points += 1
        
        # d. shorter token_count (only if is_singleword)
        if is_singleword:
            marian_tokens = token_count(marian_en)
            google_tokens = token_count(google_en)
            if marian_tokens < google_tokens:
                marian_points += 1
            elif google_tokens < marian_tokens:
                google_points += 1
        
        # e. higher token_overlap (if overlap >= 0.6, award to neither)
        overlap = token_overlap(marian_en, google_en)
        if overlap < 0.6:
            # No points awarded if overlap is high
            pass
        
        if marian_points > google_points:
            return "marian", marian_en, "heuristic_score"
        elif google_points > marian_points:
            return "google", google_en, "heuristic_score"
        else:
            # Tie
            if strategy == "conservative" and risk_score >= risk_threshold:
                return "manual_review", "", "conservative_review"
            return "tie", marian_en, "heuristic_tie"
    
    # 5. BOTH BLANK OR NO CLEAR WINNER
    if strategy == "conservative" and risk_score >= risk_threshold:
        if is_blank(marian_en) and is_blank(google_en):
            return "manual_review", "", "no_clear_winner"
        # Check if would be tie
        marian_points = 0
        google_points = 0
        if marian_qa_score < google_qa_score:
            marian_points += 1
        elif google_qa_score < marian_qa_score:
            google_points += 1
        if len(marian_flags) < len(google_flags):
            marian_points += 1
        elif len(google_flags) < len(marian_flags):
            google_points += 1
        if marian_rt is not None and google_rt is not None:
            if marian_rt < google_rt:
                marian_points += 1
            elif google_rt < marian_rt:
                google_points += 1
        if marian_points == google_points:
            return "manual_review", "", "conservative_review"
    
    return "manual_review", "", "no_clear_winner"


def compute_risk_level(risk_score: float) -> str:
    """Compute risk_level from risk_score."""
    if risk_score >= 0.65:
        return "high"
    elif risk_score >= 0.35:
        return "medium"
    else:
        return "low"


def main():
    parser = argparse.ArgumentParser(
        description="Compare Marian and Google translation outputs and build expert review queue"
    )
    parser.add_argument(
        "--marian-file", type=str,
        default="data/sem_cat/02_glosses_translated_marian_rt.csv",
        help="Path to Marian translations CSV (default: data/sem_cat/02_glosses_translated_marian_rt.csv)"
    )
    parser.add_argument(
        "--google-file", type=str,
        default="data/sem_cat/02_glosses_translated_google_rt.csv",
        help="Path to Google translations CSV (default: data/sem_cat/02_glosses_translated_google_rt.csv)"
    )
    parser.add_argument(
        "--out-file", type=str,
        default="data/sem_cat/03_translation_comparison_full.csv",
        help="Path to full comparison output CSV (default: data/sem_cat/03_translation_comparison_full.csv)"
    )
    parser.add_argument(
        "--review-file", type=str,
        default="data/sem_cat/03_translation_review_queue.csv",
        help="Path to expert review queue CSV (default: data/sem_cat/03_translation_review_queue.csv)"
    )
    parser.add_argument(
        "--gold-template-file", type=str,
        default="data/sem_cat/03_translation_gold_template.csv",
        help="Path to gold standard template CSV (default: data/sem_cat/03_translation_gold_template.csv)"
    )
    parser.add_argument(
        "--risk-threshold", type=float, default=0.35,
        help="Risk threshold for flagging reviews (default: 0.35)"
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Include only top-k most risky rows in review file (default: None)"
    )
    parser.add_argument(
        "--include-low-risk", action="store_true",
        help="Include all rows in review file regardless of risk"
    )
    parser.add_argument(
        "--prefer-backend-strategy", type=str,
        choices=["heuristic", "conservative"],
        default="heuristic",
        help="Strategy for preferring backend (default: heuristic)"
    )
    parser.add_argument(
        "--single-word-first", action="store_true",
        help="Place single-word Russian glosses earlier in review queue"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Resolve paths with fallbacks
    marian_path = Path(args.marian_file)
    google_path = Path(args.google_file)
    out_path = Path(args.out_file)
    review_path = Path(args.review_file)
    gold_path = Path(args.gold_template_file)
    
    # Fallback paths
    if not marian_path.exists():
        fallback = Path("data/sem_cat/glosses_translated_marian_rt.csv")
        if fallback.exists():
            marian_path = fallback
    if not google_path.exists():
        fallback = Path("data/sem_cat/glosses_translated_google_rt.csv")
        if fallback.exists():
            google_path = fallback
    
    # Create output directories
    out_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load and merge data
    df = load_and_prepare_data(marian_path, google_path, verbose=args.verbose)
    
    # Normalize fields
    df = normalize_fields(df)
    
    # Compute risk scores
    risk_results = df.apply(compute_risk_score, axis=1)
    df["risk_score"] = [r[0] for r in risk_results]
    df["risk_reasons"] = [";".join(r[1]) for r in risk_results]
    
    # Compute risk level
    df["risk_level"] = df["risk_score"].apply(compute_risk_level)
    
    # Determine preferred backend
    backend_results = df.apply(
        lambda row: determine_preferred_backend(
            row,
            strategy=args.prefer_backend_strategy,
            risk_score=row["risk_score"],
            risk_threshold=args.risk_threshold
        ),
        axis=1
    )
    df["preferred_backend"] = [r[0] for r in backend_results]
    df["proposed_gloss_en"] = [r[1] for r in backend_results]
    df["decision_reason"] = [r[2] for r in backend_results]
    
    # Compute needs_expert_review
    def needs_review(row) -> bool:
        if row["preferred_backend"] == "manual_review":
            return True
        if row["risk_score"] >= args.risk_threshold:
            return True
        if is_blank(row.get("marian_gloss_en", "")) and is_blank(row.get("google_gloss_en", "")):
            return True
        return False
    
    df["needs_expert_review"] = df.apply(needs_review, axis=1)
    
    # Prepare full comparison output
    full_cols = [
        "gloss_ru", "is_singleword",
        "proposed_gloss_en", "preferred_backend", "decision_reason",
        "risk_score", "risk_level", "risk_reasons", "needs_expert_review",
        "marian_gloss_en", "marian_qa_keep", "marian_qa_score", "marian_qa_flags",
        "marian_gloss_ru_back", "marian_roundtrip_distance",
        "google_gloss_en", "google_qa_keep", "google_qa_score", "google_qa_flags",
        "google_gloss_ru_back", "google_roundtrip_distance"
    ]
    df_full = df[[c for c in full_cols if c in df.columns]].copy()
    
    # Prepare review queue
    if args.include_low-risk:
        df_review = df_full.copy()
    else:
        df_review = df_full[df_full["needs_expert_review"] == True].copy()
    
    # Sort review queue
    if args.single_word_first:
        df_review = df_review.sort_values(
            by=["risk_score", "is_singleword", "gloss_ru"],
            ascending=[False, False, True]
        )
    else:
        df_review = df_review.sort_values(
            by=["risk_score", "gloss_ru"],
            ascending=[False, True]
        )
    
    # Apply top-k
    if args.top_k is not None:
        df_review = df_review.head(args.top_k)
    
    # Review queue columns
    review_cols = [
        "gloss_ru", "is_singleword",
        "proposed_gloss_en", "preferred_backend", "decision_reason",
        "risk_score", "risk_level", "risk_reasons",
        "marian_gloss_en", "marian_qa_score", "marian_qa_flags", "marian_roundtrip_distance",
        "google_gloss_en", "google_qa_score", "google_qa_flags", "google_roundtrip_distance"
    ]
    df_review = df_review[[c for c in review_cols if c in df_review.columns]]
    
    # Prepare gold template
    df_gold = df_review.copy()
    df_gold["expert_gloss_en"] = ""
    df_gold["expert_notes"] = ""
    
    gold_cols = [
        "gloss_ru", "is_singleword",
        "proposed_gloss_en", "preferred_backend",
        "marian_gloss_en", "google_gloss_en",
        "risk_score", "risk_reasons",
        "expert_gloss_en", "expert_notes"
    ]
    df_gold = df_gold[[c for c in gold_cols if c in df_gold.columns]]
    
    # Write output files
    df_full.to_csv(out_path, index=False)
    df_review.to_csv(review_path, index=False)
    df_gold.to_csv(gold_path, index=False)
    
    # Console summary
    total_merged = len(df)
    only_marian = len(df[df["marian_gloss_en"].notna() & df["google_gloss_en"].isna()])
    only_google = len(df[df["google_gloss_en"].notna() & df["marian_gloss_en"].isna()])
    both_backends = len(df[df["marian_gloss_en"].notna() & df["google_gloss_en"].notna()])
    
    marian_non_blank = df["marian_gloss_en"].apply(lambda x: not is_blank(x))
    google_non_blank = df["google_gloss_en"].apply(lambda x: not is_blank(x))
    
    both_non_blank = (marian_non_blank & google_non_blank).sum()
    only_marian_non_blank = (marian_non_blank & ~google_non_blank).sum()
    only_google_non_blank = (~marian_non_blank & google_non_blank).sum()
    both_blank = (~marian_non_blank & ~google_non_blank).sum()
    
    pref_marian = (df["preferred_backend"] == "marian").sum()
    pref_google = (df["preferred_backend"] == "google").sum()
    pref_tie = (df["preferred_backend"] == "tie").sum()
    pref_manual = (df["preferred_backend"] == "manual_review").sum()
    
    risk_high = (df["risk_level"] == "high").sum()
    risk_medium = (df["risk_level"] == "medium").sum()
    risk_low = (df["risk_level"] == "low").sum()
    
    print(f"\nTotal glosses merged:  {total_merged}")
    print(f"  Only Marian:         {only_marian}")
    print(f"  Only Google:         {only_google}")
    print(f"  Both backends:       {both_backends}")
    print()
    print(f"Translation availability:")
    print(f"  Both non-blank:             {both_non_blank}")
    print(f"  Only Marian non-blank:      {only_marian_non_blank}")
    print(f"  Only Google non-blank:      {only_google_non_blank}")
    print(f"  Both blank:                 {both_blank}")
    print()
    print(f"Preferred backend:")
    print(f"  Marian:             {pref_marian}")
    print(f"  Google:             {pref_google}")
    print(f"  Tie (near-match):   {pref_tie}")
    print(f"  Manual review:      {pref_manual}")
    print()
    print(f"Risk distribution:")
    print(f"  High   (>= 0.65):  {risk_high}")
    print(f"  Medium (0.35–0.65): {risk_medium}")
    print(f"  Low    (< 0.35):   {risk_low}")
    print()
    print(f"Expert review queue: {len(df_review)} rows → {review_path}")
    print(f"Gold standard template: {len(df_gold)} rows → {gold_path}")
    print(f"Full comparison: {len(df_full)} rows → {out_path}")


if __name__ == "__main__":
    main()
