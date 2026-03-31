"""Normalizes Russian glosses from VepKar meanings files
before WordNet lookup. Handles parentheticals, multi-part glosses, and whitespace.
"""

import re


def strip_parens(text: str) -> str:
    """Remove all (...) parenthetical fragments from text, collapse whitespace."""
    # Remove all parenthetical expressions
    text = re.sub(r'\([^)]*\)', '', text)
    # Collapse multiple whitespace characters into a single space and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def primary_gloss(text: str) -> str:
    """Take the first semicolon-separated part, strip parentheticals, strip."""
    # Take the first semicolon-separated part
    first_part = text.split(';')[0]
    # Strip parentheses
    first_part = strip_parens(first_part)
    # Strip whitespace only
    return first_part.strip()


def all_gloss_parts(text: str) -> list[str]:
    """Split by semicolon, apply strip_parens to each part, strip,
    return non-empty parts.
    """
    parts = text.split(';')
    processed_parts = []
    for part in parts:
        # Strip parentheses
        part = strip_parens(part)
        # Strip whitespace only
        part = part.strip()
        # Add non-empty parts
        if part:
            processed_parts.append(part)
    return processed_parts