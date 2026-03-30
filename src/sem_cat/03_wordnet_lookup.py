"""
For each unique (gloss_en, pos) pair, finds the best NLTK WordNet synset
and maps it to a WordNet Domain label using the wn-domains-3.2 file.
Input:  data/sem_cat/glosses_translated.csv  (from 02_translate_glosses.py)
Output: data/sem_cat/glosses_wn_domains.csv
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path
from collections import Counter
import nltk
from tqdm import tqdm

# Download required NLTK data quietly
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Add the parent directory to the path to allow importing from sibling packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.wn_domains import load_wn_domains, synset_to_key
from nltk.corpus import wordnet as wn


POS_MAP = {
    "NOUN": "n", "VERB": "v",
    "ADJ": "a",  "ADV": "r",
}


def lookup_domain(gloss_en, wn_pos, wn_domains):
    """Find the best WordNet synset for a gloss and map it to a domain."""
    # Try with POS constraint first
    if wn_pos:
        synsets = wn.synsets(gloss_en, pos=wn_pos)
        # If no synsets found with POS constraint, retry without constraint
        if not synsets:
            synsets = wn.synsets(gloss_en)
    else:
        synsets = wn.synsets(gloss_en)
    
    # If still empty: return ("", "factotum") ← fallback domain
    if not synsets:
        return ("", "factotum")
    
    # Take the first (most frequent) synset
    synset = synsets[0]
    
    # Get the synset key
    key = synset_to_key(synset)
    
    # Get domains for this key
    domains = wn_domains.get(key, ["factotum"])
    
    # Return (synset.name(), domains[0]) ← primary domain only
    return (synset.name(), domains[0])


def main():
    parser = argparse.ArgumentParser(description="Map English glosses to WordNet domains")
    parser.add_argument("--translated-file", type=str, required=True,
                        help="path to glosses_translated.csv")
    parser.add_argument("--wn-domains-file", type=str, required=True,
                        help="path to wn-domains-3.2-20070223")
    parser.add_argument("--out-file", type=str, default="../../data/sem_cat/glosses_wn_domains.csv",
                        help="output CSV path (default: ../../data/sem_cat/glosses_wn_domains.csv)")
    parser.add_argument("--pos-file", type=str, default=None,
                        help="optional: path to a CSV with columns [gloss_ru, pos]")

    args = parser.parse_args()

    # Load glosses_translated.csv into a DataFrame (gloss_ru, gloss_en, backend)
    print("Loading translated glosses...")
    df = pd.read_csv(args.translated_file)

    # Load wn_domains dict with load_wn_domains()
    print("Loading WordNet domains...")
    wn_domains = load_wn_domains(args.wn_domains_file)

    # If --pos-file is given, load it and merge on gloss_ru so each row has a POS
    if args.pos_file:
        print("Loading POS data...")
        pos_df = pd.read_csv(args.pos_file)
        df = df.merge(pos_df[['gloss_ru', 'pos']], on='gloss_ru', how='left')
    else:
        # If not given, set POS to None so lookup happens without POS constraint
        df['pos'] = None

    # Prepare results list
    results = []

    # Apply lookup_domain to all rows. Show tqdm progress bar.
    print("Looking up WordNet domains...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing glosses"):
        gloss_en = row['gloss_en']
        
        # Map POS if available
        pos = row['pos']
        if pos and pos in POS_MAP:
            wn_pos = POS_MAP[pos]
        elif pos:  # POS not in POS_MAP → skip WordNet lookup, domain = "factotum"
            results.append({
                'gloss_ru': row['gloss_ru'],
                'gloss_en': row['gloss_en'],
                'backend': row['backend'],
                'wn_synset': "",
                'wn_domain': "factotum"
            })
            continue
        else:
            wn_pos = None
        
        # Perform lookup
        synset_name, domain = lookup_domain(gloss_en, wn_pos, wn_domains)
        
        results.append({
            'gloss_ru': row['gloss_ru'],
            'gloss_en': row['gloss_en'],
            'backend': row['backend'],
            'wn_synset': synset_name,
            'wn_domain': domain
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save output CSV with columns: gloss_ru, gloss_en, backend, wn_synset, wn_domain
    out_path = Path(args.out_file)
    results_df.to_csv(out_path, index=False)

    # Print summary
    total_rows = len(results_df)
    found_synsets = len(results_df[results_df['wn_synset'] != ""])
    not_found_synsets = total_rows - found_synsets
    
    # Top-10 most frequent wn_domain values
    domain_counts = Counter(results_df['wn_domain'])
    top_10_domains = domain_counts.most_common(10)
    
    # % of rows with domain "factotum"
    factotum_pct = (results_df['wn_domain'] == "factotum").sum() / total_rows * 100
    
    print(f"\nSummary:")
    print(f"Total rows: {total_rows}")
    print(f"Rows with synset found: {found_synsets}")
    print(f"Rows with synset not found: {not_found_synsets}")
    print(f"Top-10 most frequent domains:")
    for domain, count in top_10_domains:
        print(f"  {domain}: {count}")
    print(f"% of rows with domain 'factotum': {factotum_pct:.2f}%")


if __name__ == "__main__":
    main()