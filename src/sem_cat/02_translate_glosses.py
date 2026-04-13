"""
Translates all unique primary Russian glosses from VepKar meanings files
into English, using a pluggable backend (MarianMT by default).
Results are saved to data/sem_cat/glosses_translated_{backend}.csv.
Already-cached translations are never re-computed (incremental mode).
"""

import sys
import pathlib
import pandas as pd
import argparse
import time
import re
from tqdm import tqdm
from math import ceil
from collections import Counter
import random

# Add project root to sys.path to allow absolute imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Anchor all default paths to the project root (2 levels up from this file).
# This file lives at:  <project_root>/src/sem_cat/02_translate_glosses.py
_THIS_FILE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent   # → <project_root>
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "vepkar"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "data" / "sem_cat"

from src.sem_cat.utils.vepkar_loader import load_meanings
from src.sem_cat.utils.gloss_normalizer import primary_gloss
from src.sem_cat.translators.marian_translator import MarianTranslator
from src.sem_cat.translators.google_translator import GoogleTranslator


def _is_punctuation_only(text: str) -> bool:
    """Return True if text contains only punctuation/whitespace."""
    if not text or not text.strip():
        return True
    return bool(re.fullmatch(r"[\W\s]+", text))


def _detect_repetition(text: str) -> bool:
    """Detect obvious repetition loops like 'No, no, no, no...' or '. . . . .'."""
    if not text or not text.strip():
        return False
    
    text = text.strip()
    
    # Tokenize by whitespace and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    
    if len(tokens) < 4:
        return False
    
    # Check for repeated single token (e.g., "no no no no")
    token_counts = Counter(tokens)
    most_common_token, most_common_count = token_counts.most_common(1)[0]
    if most_common_count >= int(len(tokens) * 0.7) and len(tokens) >= 4:
        return True
    
    # Check for repeated 2-grams
    if len(tokens) >= 6:
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        bigram_counts = Counter(bigrams)
        most_common_bigram, most_common_bigram_count = bigram_counts.most_common(1)[0]
        if most_common_bigram_count >= int(len(bigrams) * 0.6):
            return True
    
    # Check for repeated 3-grams
    if len(tokens) >= 8:
        trigrams = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]
        trigram_counts = Counter(trigrams)
        most_common_trigram, most_common_trigram_count = trigram_counts.most_common(1)[0]
        if most_common_trigram_count >= int(len(trigrams) * 0.5):
            return True
    
    return False


def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute normalized edit distance between two strings (0.0 = identical, 1.0 = completely different)."""
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    
    # Use Levenshtein distance
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    
    # Create distance matrix
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
    
    return dp[len1][len2] / max_len


def analyze_translation(ru: str, en: str, roundtrip_text: str | None = None) -> tuple[bool, list[str], float]:
    """
    Analyzes a translation and returns QA information.
    
    Args:
        ru: Original Russian gloss
        en: Translated English gloss
        roundtrip_text: Optional back-translated Russian text
    
    Returns:
        keep_translation: bool - False only for truly unusable output
        qa_flags: list[str] - List of QA flag identifiers
        qa_score: float - 0.0 (good) to 1.0 (suspicious)
    """
    qa_flags: list[str] = []
    qa_score = 0.0
    
    # Check for empty/whitespace only
    if not en or not en.strip():
        return False, ["empty_translation"], 1.0
    
    # Check for punctuation-only
    if _is_punctuation_only(en):
        return False, ["punctuation_only"], 1.0
    
    # Check for repetition loops
    if _detect_repetition(en):
        return False, ["repeated_token_loop"], 1.0
    
    # Check for no ASCII letters (may indicate transliteration or garbage)
    if not any(c.isalpha() and ord(c) < 128 for c in en):
        qa_flags.append("no_ascii_letters")
        qa_score += 0.3
    
    # Check for suspicious length ratio (English > 5x Russian)
    if ru and len(en) > max(80, len(ru) * 5):
        qa_flags.append("too_long_for_gloss")
        qa_score += 0.4
    
    # Check for multi-word English from single-word Russian
    ru_words = ru.strip().split()
    en_words = en.strip().split()
    if len(ru_words) == 1 and len(en_words) > 2:
        qa_flags.append("multiword_for_singleword")
        qa_score += 0.2
    
    # Check round-trip distance if provided
    if roundtrip_text is not None and ru:
        distance = _normalized_edit_distance(ru, roundtrip_text)
        if distance > 0.5:
            qa_flags.append("roundtrip_far")
            qa_score += 0.3
    
    # Cap qa_score at 1.0
    qa_score = min(1.0, qa_score)
    
    # keep_translation is True unless we already flagged as unusable
    keep_translation = True
    
    return keep_translation, qa_flags, qa_score


def build_gloss_metadata(df_meanings: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-primary-gloss metadata from meanings DataFrame.
    
    Returns DataFrame with columns:
        gloss_ru, dominant_pos, meaning_hint, source_count
    """
    # Extract primary gloss for each row
    df_meanings = df_meanings.copy()
    df_meanings['primary_gloss'] = df_meanings['meaning_ru'].apply(
        lambda x: primary_gloss(x) if pd.notna(x) else ''
    )
    df_meanings = df_meanings[df_meanings['primary_gloss'].str.len() > 0]
    
    # Group by primary gloss
    gloss_metadata = []
    for gloss, group in df_meanings.groupby('primary_gloss'):
        # Most frequent POS
        pos_counts = group['pos'].dropna().value_counts()
        dominant_pos = pos_counts.index[0] if len(pos_counts) > 0 else None
        
        # Shortest meaning_ru containing this gloss (as a hint)
        meaning_candidates = group['meaning_ru'].dropna()
        # Filter to those that contain the gloss
        meaning_candidates = meaning_candidates[meaning_candidates.str.contains(gloss, regex=False)]
        if len(meaning_candidates) > 0:
            # Pick shortest one as hint
            meaning_hint = meaning_candidates.loc[meaning_candidates.str.len().idxmin()]
        else:
            meaning_hint = group['meaning_ru'].dropna().iloc[0] if len(group) > 0 else None
        
        gloss_metadata.append({
            'gloss_ru': gloss,
            'dominant_pos': dominant_pos,
            'meaning_hint': meaning_hint,
            'source_count': len(group)
        })
    
    return pd.DataFrame(gloss_metadata)


def prepare_translation_input(gloss_ru: str, mode: str, pos_hint: str | None = None, meaning_hint: str | None = None) -> str:
    """
    Prepare the input string for translation based on the mode.
    
    Args:
        gloss_ru: The Russian gloss
        mode: One of 'raw', 'pos', 'pos_meaning'
        pos_hint: Optional POS hint
        meaning_hint: Optional meaning hint
    
    Returns:
        Input string to send to translator
    """
    if mode == 'raw':
        return gloss_ru
    elif mode == 'pos':
        pos = pos_hint if pos_hint else 'UNKNOWN'
        return f"{pos} | {gloss_ru}"
    elif mode == 'pos_meaning':
        pos = pos_hint if pos_hint else 'UNKNOWN'
        meaning = meaning_hint if meaning_hint else ''
        return f"{pos} | {gloss_ru} | {meaning}"
    else:
        return gloss_ru


def main():
    parser = argparse.ArgumentParser(description="Translate Russian glosses to English")
    parser.add_argument("--data-dir", type=str, default=str(_DEFAULT_DATA_DIR),
                        help=f"path to data/vepkar/ (default: {_DEFAULT_DATA_DIR})")
    parser.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR),
                        help=f"output directory for translated CSV (default: {_DEFAULT_OUT_DIR})")
    parser.add_argument("--backend", type=str, choices=["marian", "google"], default="marian",
                        help='translation backend: "marian" or "google" (default: marian)')
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size for translation (default: 64)")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help='Device for MarianMT: "cpu" or "cuda" (default: cpu)',
    )
    parser.add_argument(
        "--out-file", type=str, default=None,
        help=(
            "Full path to output CSV file. If provided, overrides --out-dir "
            "and the auto-generated filename. "
            "Example: data/sem_cat/glosses_translated_marian_2026.csv"
        ),
    )
    parser.add_argument(
        "--round-trip", action="store_true", default=False,
        help="also back-translate gloss_en→ru for quality checking (marian only)",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip the first N glosses after cache filtering (default: 0)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N glosses after offset (default: None = all)"
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=False,
        help="Shuffle glosses_to_translate before applying offset/limit"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed used with --shuffle (default: 42)"
    )
    parser.add_argument(
        "--gloss-filter", type=str, default=None,
        help="Optional substring filter applied to gloss_ru before translation"
    )
    parser.add_argument(
        "--translation-input-mode", type=str, choices=["raw", "pos", "pos_meaning"],
        default="raw",
        help="How to prepare input for translator (default: raw)"
    )

    args = parser.parse_args()
    
    # Compute output path
    if args.out_file:
        out_path = pathlib.Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"glosses_translated_{args.backend}.csv"
    
    print(f"Output file: {out_path}")

    # Validate data directory exists
    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        print(f"  Expected structure: {data_dir}/meanings_vep.csv  etc.")
        print(f"  Run from the project root or pass --data-dir explicitly.")
        sys.exit(1)

    # Show warning for Marian model download
    if args.backend == "marian":
        print("MarianMT backend selected. Model 'Helsinki-NLP/opus-mt-ru-en' "
              "will be downloaded on first run (~300 MB). "
              "Subsequent runs use the local HuggingFace cache.")

    # Load meanings
    print("Loading meanings...")
    df_meanings = load_meanings(str(data_dir))

    # Build gloss metadata if using pos or pos_meaning mode
    gloss_metadata_df = None
    gloss_metadata = {}
    if args.translation_input_mode in ['pos', 'pos_meaning']:
        print("Building gloss metadata for context-aware translation...")
        gloss_metadata_df = build_gloss_metadata(df_meanings)
        gloss_metadata = gloss_metadata_df.set_index('gloss_ru').to_dict('index')
        print(f"Built metadata for {len(gloss_metadata)} unique glosses")

    # Extract unique primary glosses
    print("Extracting unique primary glosses...")
    unique_glosses_arr = (
        df_meanings["meaning_ru"]
        .dropna()
        .apply(primary_gloss)
        .pipe(lambda s: s[s.str.len() > 0])
        .unique()
    )
    unique_glosses = unique_glosses_arr.tolist()
    total_unique = len(unique_glosses)
    print(f"Found {total_unique} unique glosses")

    # Load existing cache if it exists
    cached_glosses = set()
    cache_df = pd.DataFrame(columns=["gloss_ru", "gloss_en"])
    if out_path.exists():
        print("Loading existing cache...")
        try:
            cache_df = pd.read_csv(out_path, encoding="utf-8")
            if "gloss_ru" not in cache_df.columns:
                print("WARNING: cache file exists but has no 'gloss_ru' column — ignoring cache.")
                cache_df = pd.DataFrame(columns=["gloss_ru", "gloss_en"])
        except Exception as e:
            print(f"WARNING: could not read cache file ({e}) — starting fresh.")
            cache_df = pd.DataFrame(columns=["gloss_ru", "gloss_en"])
        cached_glosses = set(cache_df["gloss_ru"].dropna().tolist())
        already_cached = len(cached_glosses)
        print(f"Found {already_cached} cached translations")
    else:
        already_cached = 0

    # Determine which glosses need translation
    glosses_to_translate = [g for g in unique_glosses if g not in cached_glosses]
    remaining_after_cache = len(glosses_to_translate)
    print(f"Remaining after cache: {remaining_after_cache}")

    # Apply gloss filter if provided
    if args.gloss_filter:
        glosses_to_translate = [g for g in glosses_to_translate if args.gloss_filter in g]
        print(f"After gloss filter '{args.gloss_filter}': {len(glosses_to_translate)}")

    # Shuffle if requested
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(glosses_to_translate)
        print(f"Shuffled glosses with seed {args.seed}")

    # Apply offset/limit
    offset = args.offset
    limit = args.limit
    
    if offset > 0:
        glosses_to_translate = glosses_to_translate[offset:]
        print(f"After offset {offset}: {len(glosses_to_translate)}")
    
    if limit is not None:
        glosses_to_translate = glosses_to_translate[:limit]
        print(f"After limit {limit}: {len(glosses_to_translate)}")
    
    to_translate_count = len(glosses_to_translate)
    print(f"Selected for this run: {to_translate_count}")

    if to_translate_count == 0:
        print("No new glosses to translate. Exiting.")
        return

    # Initialize translator
    back_translator = None
    if args.backend == "marian":
        translator = MarianTranslator(device=args.device)
        # If round-trip is enabled, initialize back-translator
        if args.round_trip:
            back_translator = MarianTranslator(
                device=args.device,
                model_name="Helsinki-NLP/opus-mt-en-ru"
            )
    else:  # google
        translator = GoogleTranslator()
        # Round-trip is only supported for marian backend
        if args.round_trip:
            print("Warning: --round-trip is only supported with marian backend, ignoring flag.")

    # Translate the remaining glosses
    print(f"Translating with {args.backend} backend (mode: {args.translation_input_mode})...")
    total_written = 0
    total_kept = 0
    total_blank = 0
    total_suspicious = 0

    # Build input texts with context if needed
    input_texts = []
    for gloss in glosses_to_translate:
        if args.translation_input_mode == 'raw':
            input_texts.append(gloss)
        else:
            metadata = gloss_metadata.get(gloss, {})
            pos_hint = metadata.get('dominant_pos')
            meaning_hint = metadata.get('meaning_hint')
            input_text = prepare_translation_input(gloss, args.translation_input_mode, pos_hint, meaning_hint)
            input_texts.append(input_text)

    if args.backend == "marian":
        # Use translate_batch for efficiency with incremental saves
        n = len(glosses_to_translate)
        n_batches = ceil(n / args.batch_size) if n > 0 else 0
        
        # Track if header has been written
        header_written = already_cached > 0

        for batch_idx in range(n_batches):
            batch_glosses = glosses_to_translate[batch_idx * args.batch_size :
                                          (batch_idx + 1) * args.batch_size]
            batch_inputs = input_texts[batch_idx * args.batch_size :
                                (batch_idx + 1) * args.batch_size]
            
            translated_texts = translator.translate_batch(batch_inputs, batch_size=len(batch_inputs))
            
            # If round-trip is enabled, back-translate the results
            if args.round_trip and back_translator is not None:
                back_translated_texts = back_translator.translate_batch(translated_texts, batch_size=len(translated_texts))
            else:
                back_translated_texts = [None] * len(translated_texts)
            
            batch_rows = []
            for i, (gloss_ru, input_text, trans) in enumerate(zip(batch_glosses, batch_inputs, translated_texts)):
                roundtrip_text = back_translated_texts[i] if args.round_trip else None
                keep, qa_flags, qa_score = analyze_translation(gloss_ru, trans, roundtrip_text)
                
                row_data = {
                    "gloss_ru": gloss_ru,
                    "gloss_en": trans if keep else "",
                    "qa_keep": keep,
                    "qa_score": round(qa_score, 2),
                    "qa_flags": ";".join(qa_flags) if qa_flags else "",
                }
                
                # Add back-translation if round-trip is enabled
                if args.round_trip:
                    row_data["gloss_ru_back"] = roundtrip_text if roundtrip_text else ""
                    # Compute roundtrip distance
                    if roundtrip_text:
                        distance = _normalized_edit_distance(gloss_ru, roundtrip_text)
                        row_data["roundtrip_distance"] = round(distance, 3)
                    else:
                        row_data["roundtrip_distance"] = ""
                
                # Add metadata columns if using context mode
                if args.translation_input_mode in ['pos', 'pos_meaning']:
                    metadata = gloss_metadata.get(gloss_ru, {})
                    row_data["pos_hint"] = metadata.get('dominant_pos', '')
                    row_data["meaning_hint"] = metadata.get('meaning_hint', '')
                    row_data["source_count"] = metadata.get('source_count', 0)
                
                batch_rows.append(row_data)
                
                if not keep:
                    total_blank += 1
                elif qa_flags:
                    total_suspicious += 1
                else:
                    total_kept += 1
                    
            batch_df = pd.DataFrame(batch_rows)
            
            # append to CSV; write header only on the very first write of this run
            batch_df.to_csv(
                out_path, mode="a",
                header=not header_written,
                index=False,
                encoding="utf-8"
            )
            if not header_written:
                header_written = True
            total_written += len(batch_rows)
            print(f"  Batch {batch_idx + 1}/{n_batches} saved ({total_written} total written)")
    else:  # google
        # Process with batch_size=1 and delay, with incremental saves every 100 items
        batch_rows = []
        header_written = already_cached > 0
        
        for idx, (gloss, input_text) in enumerate(tqdm(zip(glosses_to_translate, input_texts), total=len(glosses_to_translate), desc="Translating")):
            translated_text = translator.translate(input_text)
            keep, qa_flags, qa_score = analyze_translation(gloss, translated_text)
            
            row_data = {
                'gloss_ru': gloss,
                'gloss_en': translated_text if keep else '',
                'qa_keep': keep,
                'qa_score': round(qa_score, 2),
                'qa_flags': ';'.join(qa_flags) if qa_flags else '',
            }
            # For Google backend, round-trip is not supported
            if args.round_trip:
                row_data["gloss_ru_back"] = ""
                row_data["roundtrip_distance"] = ""
            
            # Add metadata columns if using context mode
            if args.translation_input_mode in ['pos', 'pos_meaning']:
                metadata = gloss_metadata.get(gloss, {})
                row_data["pos_hint"] = metadata.get('dominant_pos', '')
                row_data["meaning_hint"] = metadata.get('meaning_hint', '')
                row_data["source_count"] = metadata.get('source_count', 0)
            
            batch_rows.append(row_data)
            
            if not keep:
                total_blank += 1
            elif qa_flags:
                total_suspicious += 1
            else:
                total_kept += 1
                
            time.sleep(0.3)  # Sleep 0.3s between calls

            # Save every 100 items or at the end
            if len(batch_rows) >= 100 or idx == len(glosses_to_translate) - 1:
                batch_df = pd.DataFrame(batch_rows)
                # append to CSV; write header only on the very first write of this run
                batch_df.to_csv(
                    out_path, mode="a",
                    header=not header_written,
                    index=False,
                    encoding="utf-8"
                )
                if not header_written:
                    header_written = True
                total_written += len(batch_rows)
                print(f"  Saved batch of {len(batch_rows)} items ({total_written} total written)")
                batch_rows = []  # Reset for next batch

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total unique glosses: {total_unique}")
    print(f"Already cached: {already_cached}")
    print(f"Remaining after cache: {remaining_after_cache}")
    print(f"Selected for this run (after filter/subset): {to_translate_count}")
    print(f"Newly translated: {total_written}")
    print(f"  - Kept (good quality): {total_kept}")
    print(f"  - Kept (suspicious, flagged): {total_suspicious}")
    print(f"  - Blank/broken (qa_keep=False): {total_blank}")


if __name__ == "__main__":
    main()
