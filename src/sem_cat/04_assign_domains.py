"""
Merges wn_domain labels back into the four VepKar meanings files.
For each meaning row, joins on gloss_primary to get wn_domain and wn_synset.
Outputs one enriched file per language:
    data/sem_cat/results/meanings_{lang}_domains.csv
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Add the parent directory to the path to allow importing from sibling packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.vepkar_loader import load_meanings
from utils.gloss_normalizer import primary_gloss


def main():
    parser = argparse.ArgumentParser(description="Assign WordNet domains to VepKar meanings")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="path to data/vepkar/")
    parser.add_argument("--domains-file", type=str, required=True,
                        help="path to glosses_wn_domains.csv")
    parser.add_argument("--out-dir", type=str, default="../../data/sem_cat/results/",
                        help="output directory (default: ../../data/sem_cat/results/)")

    args = parser.parse_args()

    # Load meanings (all 4 languages) with load_meanings()
    print("Loading meanings...")
    meanings_df = load_meanings(args.data_dir)

    # Compute gloss_primary column using primary_gloss()
    print("Computing primary glosses...")
    meanings_df['gloss_primary'] = meanings_df['meaning_ru'].apply(primary_gloss)

    # Load glosses_wn_domains.csv
    print("Loading domain assignments...")
    domains_df = pd.read_csv(args.domains_file)
    # Keep only columns: gloss_ru, wn_synset, wn_domain
    domains_df = domains_df[['gloss_ru', 'wn_synset', 'wn_domain']]
    # Drop duplicates on gloss_ru (keep first)
    domains_df = domains_df.drop_duplicates(subset=['gloss_ru'], keep='first')

    # Left-join meanings on gloss_primary == gloss_ru
    # Rows with no match get wn_domain = "" and wn_synset = ""
    print("Joining meanings with domains...")
    merged_df = meanings_df.merge(
        domains_df,
        left_on='gloss_primary',
        right_on='gloss_ru',
        how='left'
    )
    
    # Fill NaN values with empty strings
    merged_df['wn_synset'] = merged_df['wn_synset'].fillna("")
    merged_df['wn_domain'] = merged_df['wn_domain'].fillna("")

    # Create output directory if it doesn't exist
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # For each language, save a separate CSV to out-dir
    langs = merged_df['lang'].unique()
    for lang in langs:
        lang_df = merged_df[merged_df['lang'] == lang].copy()
        
        # Select all original columns plus the new ones
        cols_to_save = list(meanings_df.columns) + ['wn_synset', 'wn_domain']
        lang_df_subset = lang_df[cols_to_save]
        
        # Save to file
        output_file = out_dir / f"meanings_{lang}_domains.csv"
        lang_df_subset.to_csv(output_file, index=False)
        
        # Print per-language summary
        total_meanings = len(lang_df)
        meanings_with_domain = len(lang_df[lang_df['wn_domain'] != ""])
        coverage_pct = (meanings_with_domain / total_meanings) * 100 if total_meanings > 0 else 0
        
        print(f"\nLanguage: {lang}")
        print(f"  Total meanings: {total_meanings}")
        print(f"  Meanings with domain assigned: {meanings_with_domain}")
        print(f"  Coverage: {coverage_pct:.2f}%")
        
        # Top-5 wn_domain values for this language
        domain_counts = Counter(lang_df[lang_df['wn_domain'] != ""]['wn_domain'])
        top_5_domains = domain_counts.most_common(5)
        print(f"  Top-5 domains:")
        for domain, count in top_5_domains:
            print(f"    {domain}: {count}")


if __name__ == "__main__":
    main()