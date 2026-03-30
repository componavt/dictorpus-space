"""Loads the WordNet Domains 3.2 file into a lookup dict.
File format: each line is  {8digit_offset}-{pos_char}<TAB>{domain_label}
A synset may appear on multiple lines with different domains.
"""

from typing import Dict, List
import nltk


def load_wn_domains(filepath: str) -> dict[str, list[str]]:
    """Read the file, skip comment lines (starting with #) and blank lines.
    Key: synset id string like "00001740-n".
    Value: list of domain label strings (a synset can have multiple).
    Return the dict.
    """
    wn_domains = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and blank lines
            if line.startswith('#') or not line:
                continue
            
            # Parse the line: {8digit_offset}-{pos_char}<TAB>{domain_label}
            parts = line.split('\t')
            if len(parts) >= 2:
                synset_id = parts[0]
                domain_label = parts[1]
                
                if synset_id not in wn_domains:
                    wn_domains[synset_id] = []
                
                wn_domains[synset_id].append(domain_label)
    
    return wn_domains


def synset_to_key(synset) -> str:
    """Given an nltk.corpus.wordnet synset object,
    return the key string "{offset:08d}-{pos_char}"
    where pos_char maps: n→n, v→v, a/s→a, r→r.
    """
    offset = synset.offset()
    pos = synset.pos()
    
    # Map POS tags: n→n, v→v, a/s→a, r→r
    if pos in ['a', 's']:
        pos_char = 'a'
    else:
        pos_char = pos
    
    return f"{offset:08d}-{pos_char}"