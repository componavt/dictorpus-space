"""Gloss complexity heuristics."""

from __future__ import annotations

import re

from src.sem_cat.utils.text_utils import is_blank, token_count


def compute_gloss_complexity(gloss_ru: str) -> tuple[float, list[str]]:
    """Estimate how inherently difficult a Russian gloss is to translate.

    Returns:
        (complexity_score, complexity_reasons)
        complexity_score is in [0.0, 0.3] range (added to total risk).
    """
    if is_blank(gloss_ru):
        return 0.0, []

    score = 0.0
    reasons: list[str] = []

    tokens = gloss_ru.strip().split()
    n_tokens = len(tokens)

    # Single-word gloss: slightly harder (no context)
    if n_tokens == 1:
        score += 0.05
        reasons.append("singleword_gloss")

    # Very short gloss (1-3 chars): ambiguous
    stripped = gloss_ru.strip()
    if len(stripped) <= 3:
        score += 0.05
        reasons.append("very_short_gloss")

    # Hyphenated / clitic-like items (e.g., "-то", "кое-что")
    if "-" in stripped:
        score += 0.05
        reasons.append("hyphenated_gloss")

    # Probable proper name: title-case Cyrillic single token
    if (
        n_tokens == 1
        and stripped
        and stripped[0].isupper()
        and any("\u0400" <= c <= "\u04FF" for c in stripped)
    ):
        score += 0.05
        reasons.append("probable_proper_name")

    # Punctuation-heavy gloss
    punct_count = sum(1 for c in stripped if c in ".,!?;:\"'()-/\\")
    if punct_count >= 2:
        score += 0.05
        reasons.append("punctuation_heavy_gloss")

    # Particle / function-word indicators
    particle_patterns = {"-то", "-либо", "-нибудь", "кое-", "ни", "ли"}
    for pattern in particle_patterns:
        if pattern in stripped.lower():
            score += 0.05
            reasons.append("particle_or_clitic")
            break

    # Cap at 0.30
    score = min(0.30, score)
    return round(score, 2), reasons
