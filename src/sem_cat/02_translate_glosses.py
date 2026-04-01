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


def _looks_like_proper_name(ru: str, en: str) -> bool:
    """Return True if ru is likely a common noun but en looks like a proper name."""
    # Russian input: no capital letter = common noun or particle
    ru_is_common = ru and ru[0].islower()
    # English output: single capitalized word (proper name pattern)
    en_words = en.strip().split()
    en_is_proper = (
        len(en_words) == 1
        and en_words[0][0].isupper()
        and en_words[0].isalpha()
    )
    return ru_is_common and en_is_proper


def is_valid_translation(ru: str, en: str) -> bool:
    """Return False if translation is empty, too short, or a repetition loop."""
    if not en or not en.strip():
        return False
    # Flag if English output is more than 5x longer than Russian input
    if len(en) > max(80, len(ru) * 5):
        return False
    # Flag if the output is all punctuation/spaces
    if re.fullmatch(r"[\W\s]+", en):
        return False
    # Flag if ru is common noun but en looks like proper name
    if _looks_like_proper_name(ru, en):
        return False
    return True


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
    df = load_meanings(data_dir)

    # Extract unique primary glosses
    print("Extracting unique primary glosses...")
    unique_glosses = (
        df["meaning_ru"]
        .dropna()
        .apply(primary_gloss)
        .pipe(lambda s: s[s.str.len() > 0])
        .unique()
        .tolist()
    )
    total_unique = len(unique_glosses)
    print(f"Found {total_unique} unique glosses")

    # Load existing cache if it exists
    cached_glosses = set()
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
    to_translate_count = len(glosses_to_translate)
    print(f"Need to translate {to_translate_count} glosses")

    if to_translate_count == 0:
        print("No new glosses to translate. Exiting.")
        return

    # Initialize translator
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
    print(f"Translating with {args.backend} backend...")
    total_written = 0
    total_invalid = 0

    if args.backend == "marian":
        # Use translate_batch for efficiency with incremental saves
        n = len(glosses_to_translate)
        n_batches = ceil(n / args.batch_size)
        
        # Track if header has been written
        header_written = already_cached > 0

        for batch_idx in range(n_batches):
            batch = glosses_to_translate[batch_idx * args.batch_size :
                                  (batch_idx + 1) * args.batch_size]
            translated_texts = translator.translate_batch(batch, batch_size=len(batch))
            
            # If round-trip is enabled, back-translate the results
            if args.round_trip:
                back_translated_texts = back_translator.translate_batch(translated_texts, batch_size=len(translated_texts))
            
            batch_rows = []
            for i, (orig, trans) in enumerate(zip(batch, translated_texts)):
                valid = is_valid_translation(orig, trans)
                row_data = {
                    "gloss_ru": orig,
                    "gloss_en": trans if valid else "",
                }
                # Add back-translation if round-trip is enabled
                if args.round_trip:
                    row_data["gloss_ru_back"] = back_translated_texts[i] if valid else ""
                batch_rows.append(row_data)
                if not valid:
                    total_invalid += 1
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
        
        for idx, gloss in enumerate(tqdm(glosses_to_translate, desc="Translating")):
            translated_text = translator.translate(gloss)
            valid = is_valid_translation(gloss, translated_text)
            row_data = {
                'gloss_ru': gloss,
                'gloss_en': translated_text if valid else '',
            }
            # For Google backend, round-trip is not supported, so don't add gloss_ru_back
            batch_rows.append(row_data)
            if not valid:
                total_invalid += 1
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
    print(f"\nSummary:")
    print(f"Total unique glosses: {total_unique}")
    print(f"Already cached: {already_cached}")
    print(f"Newly translated: {total_written}")
    print(f"Invalid/empty translations (flagged): {total_invalid}")


if __name__ == "__main__":
    main()