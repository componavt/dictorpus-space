"""
Translates all unique primary Russian glosses from VepKar meanings files
into English, using a pluggable backend (MarianMT by default).
Results are saved to data/sem_cat/glosses_translated.csv.
Already-cached translations are never re-computed (incremental mode).
"""

import pandas as pd
import argparse
import time
from pathlib import Path
from tqdm import tqdm

from src.sem_cat.utils.vepkar_loader import load_meanings
from src.sem_cat.utils.gloss_normalizer import primary_gloss
from src.sem_cat.translators.marian_translator import MarianTranslator
from src.sem_cat.translators.google_translator import GoogleTranslator


def main():
    parser = argparse.ArgumentParser(description="Translate Russian glosses to English")
    parser.add_argument("--data-dir", type=str, default="../../data/vepkar",
                        help="path to data/vepkar/ (default: ../../data/vepkar)")
    parser.add_argument("--out-file", type=str, default="../../data/sem_cat/glosses_translated.csv",
                        help="output CSV path (default: ../../data/sem_cat/glosses_translated.csv)")
    parser.add_argument("--backend", type=str, choices=["marian", "google"], default="marian",
                        help='translation backend: "marian" or "google" (default: marian)')
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size for translation (default: 64)")

    args = parser.parse_args()

    # Load meanings
    print("Loading meanings...")
    df = load_meanings(args.data_dir)

    # Extract unique primary glosses
    print("Extracting unique primary glosses...")
    unique_glosses = set()
    for _, row in df.iterrows():
        gloss_primary = primary_gloss(row['meaning_ru'])
        if gloss_primary:  # Only add non-empty glosses
            unique_glosses.add(gloss_primary)

    unique_glosses = list(unique_glosses)
    total_unique = len(unique_glosses)
    print(f"Found {total_unique} unique glosses")

    # Load existing cache if it exists
    out_path = Path(args.out_file)
    cached_glosses = set()
    if out_path.exists():
        print("Loading existing cache...")
        cache_df = pd.read_csv(out_path)
        cached_glosses = set(cache_df['gloss_ru'].tolist())
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
        translator = MarianTranslator()
    else:  # google
        translator = GoogleTranslator()

    # Translate the remaining glosses
    print(f"Translating with {args.backend} backend...")
    translated_results = []

    if args.backend == "marian":
        # Use translate_batch for efficiency
        translated_texts = translator.translate_batch(glosses_to_translate, batch_size=args.batch_size)
        for i, original_gloss in enumerate(glosses_to_translate):
            translated_results.append({
                'gloss_ru': original_gloss,
                'gloss_en': translated_texts[i],
                'backend': args.backend
            })
    else:  # google
        # Process with batch_size=1 and delay
        for gloss in tqdm(glosses_to_translate, desc="Translating"):
            translated_text = translator.translate(gloss)
            translated_results.append({
                'gloss_ru': gloss,
                'gloss_en': translated_text,
                'backend': args.backend
            })
            time.sleep(0.3)  # Sleep 0.3s between calls

    # Count failed translations (empty results)
    failed_count = sum(1 for result in translated_results if not result['gloss_en'])

    # Append new results to the cache CSV
    new_results_df = pd.DataFrame(translated_results)
    
    # Write to file (append mode, with header only if file doesn't exist)
    if out_path.exists():
        new_results_df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        new_results_df.to_csv(out_path, mode='w', header=True, index=False)

    # Print summary
    newly_translated = len(translated_results)
    print(f"\nSummary:")
    print(f"Total unique glosses: {total_unique}")
    print(f"Already cached: {already_cached}")
    print(f"Newly translated: {newly_translated}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()