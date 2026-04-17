"""
For each unique (gloss_en, pos) pair, finds the best NLTK WordNet synset
and maps it to a WordNet Domain label using the wn-domains-3.2 file.
Input:  data/sem_cat/glosses_translated.csv  (from 02_translate_glosses.py)
Output: data/sem_cat/glosses_wn_domains.csv
Columns: gloss_ru, gloss_en, pos, wn_pos, wn_synset, synset_count, lookup_status, wn_domain, qa_skip_reason
"""

import sys
import pathlib
import pandas as pd
import argparse
from pathlib import Path
from collections import Counter
import nltk
from tqdm import tqdm

# Download required NLTK data quietly
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Add project root to sys.path to allow absolute imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Anchor all default paths to the project root (2 levels up from this file).
# This file lives at:  <project_root>/src/sem_cat/03_wordnet_lookup.py
_THIS_FILE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent   # → <project_root>
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "vepkar"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "data" / "sem_cat"

from src.sem_cat.utils.wn_domains import load_wn_domains, synset_to_key
from src.sem_cat.utils.vepkar_loader import load_meanings
from src.sem_cat.utils.gloss_normalizer import primary_gloss
from nltk.corpus import wordnet as wn


POS_MAP = {
    "NOUN": "n",
    "VERB": "v",
    "ADJ":  "a",
    "ADV":  "r",
    "NUM":  None,
    "PROPN": None,
}


def build_gloss_pos_map(df_meanings: pd.DataFrame) -> dict[str, str | None]:
    """
    Build gloss_ru -> dominant_pos mapping from meanings DataFrame.
    Uses primary_gloss extraction.
    """
    df = df_meanings.copy()
    df = df.copy()
    df['primary_gloss'] = df['meaning_ru'].apply(
        lambda x: primary_gloss(x) if pd.notna(x) else ''
    )
    df = df[df['primary_gloss'].str.len() > 0]
    
    gloss_pos_map: dict[str, str | None] = {}
    for gloss, group in df.groupby('primary_gloss'):
        pos_series = group['pos']
        pos_counts = pos_series.dropna().value_counts()
        if len(pos_counts) > 0:
            gloss_pos_map[gloss] = str(pos_counts.index[0])
        else:
            gloss_pos_map[gloss] = None
    
    return gloss_pos_map


def lookup_domain(gloss_en: str, wn_pos: str | None, wn_domains: dict) -> tuple[str, str, int, str]:
    """
    Find the best WordNet synset for a gloss and map it to a domain.
    
    Returns:
        synset_name: str - synset name or ""
        domain: str - domain label
        synset_count: int - number of synsets found
        lookup_status: str - 'found', 'not_found', 'skipped_empty'
    """
    if not gloss_en or not gloss_en.strip():
        return ("", "factotum", 0, "skipped_empty")
    
    # Try with POS constraint first
    if wn_pos:
        synsets = wn.synsets(gloss_en, pos=wn_pos)
        # If no synsets found with POS constraint, retry without constraint
        if not synsets:
            synsets = wn.synsets(gloss_en)
    else:
        synsets = wn.synsets(gloss_en)
    
    synset_count = len(synsets)
    
    # If still empty: return ("", "factotum") ← fallback domain
    if not synsets:
        return ("", "factotum", 0, "not_found")
    
    # Take the first (most frequent) synset
    synset = synsets[0]
    
    # Get the synset key
    key = synset_to_key(synset)
    
    # Get domains for this key
    domains = wn_domains.get(key, ["factotum"])
    
    # Return (synset.name(), domains[0]) ← primary domain only
    return (synset.name(), domains[0], synset_count, "found")


def main():
    parser = argparse.ArgumentParser(description="Map English glosses to WordNet domains")
    parser.add_argument("--translated-file", type=str,
                        default=str(_DEFAULT_OUT_DIR / "glosses_translated_marian.csv"),
                        help="path to glosses_translated_*.csv (default: glosses_translated_marian.csv)")
    parser.add_argument("--wn-domains-file", type=str, required=True,
                        help="path to wn-domains-3.2-20070223")
    parser.add_argument("--out-file", type=str,
                        default=str(_DEFAULT_OUT_DIR / "glosses_wn_domains.csv"),
                        help="output CSV path (default: <project_root>/data/sem_cat/glosses_wn_domains.csv)")
    parser.add_argument("--pos-file", type=str, default=None,
                        help="optional: path to a CSV with columns [gloss_ru, pos]")
    parser.add_argument("--data-dir", type=str, default=str(_DEFAULT_DATA_DIR),
                        help=f"path to data/vepkar/ (default: {_DEFAULT_DATA_DIR})")
    parser.add_argument("--pos-source", type=str, choices=["none", "file", "meanings"],
                        default="meanings",
                        help="source of POS tags: 'none', 'file', or 'meanings' (default: meanings)")
    
    args = parser.parse_args()

    # Load glosses_translated.csv into a DataFrame
    print("Loading translated glosses...")
    df = pd.read_csv(args.translated_file)

    # Load wn_domains dict with load_wn_domains()
    print("Loading WordNet domains...")
    wn_domains = load_wn_domains(args.wn_domains_file)

    # Determine POS source
    if args.pos_source == "file":
        if args.pos_file:
            print("Loading POS data from file...")
            pos_df = pd.read_csv(args.pos_file)
            df = df.merge(pos_df[['gloss_ru', 'pos']], on='gloss_ru', how='left')
        else:
            print("Warning: --pos-source=file but no --pos-file provided. Using meanings.")
            args.pos_source = "meanings"
    
    if args.pos_source == "meanings":
        print("Loading meanings data for POS extraction...")
        data_dir = pathlib.Path(args.data_dir)
        if not data_dir.exists():
            print(f"ERROR: data directory not found: {data_dir}")
            sys.exit(1)
        df_meanings = load_meanings(str(data_dir))
        gloss_pos_map = build_gloss_pos_map(df_meanings)
        df['pos'] = df['gloss_ru'].map(gloss_pos_map)
        print(f"Built POS map for {len(gloss_pos_map)} glosses")
    
    if args.pos_source == "none" or 'pos' not in df.columns:
        df['pos'] = None

    # Preserve QA columns if they exist
    qa_columns = [col for col in ['qa_keep', 'qa_score', 'qa_flags'] if col in df.columns]
    
    # Prepare results list
    results = []
    
    # Cache for deduplicating lookups by (gloss_en, wn_pos)
    lookup_cache = {}

    # Apply lookup_domain to all rows. Show tqdm progress bar.
    print("Looking up WordNet domains...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing glosses"):
        gloss_en = row['gloss_en']
        gloss_ru = row['gloss_ru']
        
        # Map POS if available
        pos_val = row.get('pos')
        pos_str: str | None = str(pos_val) if pd.notna(pos_val) else None
        if pos_str and pos_str in POS_MAP:
            wn_pos: str | None = POS_MAP[pos_str]
        elif pos_str:  # POS not in POS_MAP → skip WordNet lookup, domain = "factotum"
            results.append({
                'gloss_ru': gloss_ru,
                'gloss_en': gloss_en,
                'pos': pos_str,
                'wn_pos': None,
                'wn_synset': "",
                'synset_count': 0,
                'lookup_status': 'skipped_pos',
                'wn_domain': "factotum",
                'qa_skip_reason': ''
            })
            continue
        else:
            wn_pos = None
        
        # Skip WordNet lookup for empty gloss_en
        gloss_en_str: str = str(gloss_en) if pd.notna(gloss_en) else ""
        if not gloss_en_str or not gloss_en_str.strip():
            results.append({
                'gloss_ru': gloss_ru,
                'gloss_en': gloss_en_str,
                'pos': pos_str,
                'wn_pos': wn_pos,
                'wn_synset': "",
                'synset_count': 0,
                'lookup_status': 'skipped_empty',
                'wn_domain': "factotum",
                'qa_skip_reason': 'empty_gloss_en'
            })
            continue
        
        # Check QA skip if qa_keep column exists
        qa_skip_reason = ''
        if 'qa_keep' in df.columns:
            qa_keep_val = row.get('qa_keep')
            if isinstance(qa_keep_val, bool) and qa_keep_val == False:
                qa_skip_reason = 'qa_keep_false'
        
        # Deduplicate lookups by (gloss_en, wn_pos)
        cache_key = (gloss_en_str, wn_pos)
        if cache_key in lookup_cache:
            synset_name, domain, synset_count, lookup_status = lookup_cache[cache_key]
        else:
            synset_name, domain, synset_count, lookup_status = lookup_domain(gloss_en_str, wn_pos, wn_domains)
            lookup_cache[cache_key] = (synset_name, domain, synset_count, lookup_status)
        
        result_row = {
            'gloss_ru': gloss_ru,
            'gloss_en': gloss_en_str,
            'pos': pos_str,
            'wn_pos': wn_pos,
            'wn_synset': synset_name,
            'synset_count': synset_count,
            'lookup_status': lookup_status,
            'wn_domain': domain,
            'qa_skip_reason': qa_skip_reason
        }
        
        # Preserve QA columns if they exist
        for qa_col in qa_columns:
            result_row[qa_col] = row.get(qa_col, '')
        
        results.append(result_row)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save output CSV
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    # Print summary
    total_rows = len(results_df)
    found_synsets = len(results_df[results_df['lookup_status'] == 'found'])
    not_found_synsets = len(results_df[results_df['lookup_status'] == 'not_found'])
    skipped_empty = len(results_df[results_df['lookup_status'] == 'skipped_empty'])
    skipped_pos = len(results_df[results_df['lookup_status'] == 'skipped_pos'])
    
    # Top-10 most frequent wn_domain values
    domain_counts = Counter(results_df['wn_domain'])
    top_10_domains = domain_counts.most_common(10)
    
    # % of rows with domain "factotum"
    factotum_pct = (results_df['wn_domain'] == "factotum").sum() / total_rows * 100
    
    print(f"\nSummary:")
    print(f"Total rows: {total_rows}")
    print(f"Rows with synset found: {found_synsets}")
    print(f"Rows with synset not found: {not_found_synsets}")
    print(f"Rows skipped (empty gloss_en): {skipped_empty}")
    print(f"Rows skipped (unknown POS): {skipped_pos}")
    print(f"Top-10 most frequent domains:")
    for domain, count in top_10_domains:
        print(f"  {domain}: {count}")
    print(f"% of rows with domain 'factotum': {factotum_pct:.2f}%")


if __name__ == "__main__":
    main()
