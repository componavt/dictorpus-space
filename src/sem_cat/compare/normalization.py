"""Normalization and similarity helpers for English gloss comparison."""

from __future__ import annotations

import re

from src.sem_cat.utils.text_utils import is_blank, token_count
from src.sem_cat.utils.distance import normalized_edit_similarity


def normalize_output_for_comparison(text: str) -> str:
    """Normalize an English gloss for comparison purposes.

    - Lowercase
    - Normalize whitespace
    - Strip surrounding punctuation
    - Remove duplicated spaces
    """
    if is_blank(text):
        return ""

    s = text.lower().strip()
    # Strip surrounding punctuation
    s = s.strip(".,!?;:\"'()-")
    # Normalize internal whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_set(text: str) -> set[str]:
    """Return a set of lowercased tokens."""
    if is_blank(text):
        return set()
    return set(text.lower().split())


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of lowercased token sets."""
    if is_blank(a) or is_blank(b):
        return 0.0
    tokens_a = token_set(a)
    tokens_b = token_set(b)
    if not tokens_a and not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def output_similarity(a: str, b: str) -> float:
    """Combined similarity between two normalized outputs.

    Uses max of edit similarity and token overlap.
    """
    edit_sim = normalized_edit_similarity(a, b)
    overlap = token_overlap(a, b)
    return max(edit_sim, overlap)


def outputs_are_near_match(a: str, b: str, threshold: float = 0.85) -> bool:
    """Return True if two outputs are near-identical."""
    return output_similarity(a, b) >= threshold
