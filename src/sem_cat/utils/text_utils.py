"""Shared text analysis helpers for translation QA."""

from __future__ import annotations

import re


def is_blank(value: str | None) -> bool:
    """Return True if value is None, empty, or whitespace-only."""
    if value is None:
        return True
    return str(value).strip() == ""


def normalize_whitespace(text: str) -> str:
    """Collapse internal whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def token_count(text: str) -> int:
    """Count whitespace-separated tokens. Returns 0 for blank."""
    if is_blank(text):
        return 0
    return len(str(text).split())


def contains_ascii_letters(text: str) -> bool:
    """Return True if text contains at least one ASCII letter."""
    return any(c.isalpha() and ord(c) < 128 for c in text)


def is_punctuation_only(text: str) -> bool:
    """Return True if text contains only punctuation and whitespace."""
    if is_blank(text):
        return True
    return bool(re.fullmatch(r"[\W\s]+", text))
