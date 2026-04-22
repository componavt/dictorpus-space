"""Pattern-based flag detectors for translation QA.

Detects sentence-like expansions, token inflation, proper-name
overexpansion, and other suspicious output patterns specific to
dictionary-style gloss translation.
"""

from __future__ import annotations

import re

from src.sem_cat.utils.text_utils import is_blank, token_count


# ---------------------------------------------------------------------------
# Sentence-like expansion prefixes
# ---------------------------------------------------------------------------

SENTENCE_LIKE_PREFIXES: tuple[str, ...] = (
    "it is ",
    "it's ",
    "there is ",
    "there are ",
    "this is ",
    "we can ",
    "it was ",
    "the city of ",
    "it is called ",
    "it is located ",
)


def detect_repetition(text: str) -> bool:
    """Detect obvious repetition loops like 'No, no, no, no...' or '. . . . .'."""
    if is_blank(text):
        return False

    stripped = text.strip()
    tokens = re.findall(r"\w+|[^\w\s]", stripped, re.UNICODE)

    if len(tokens) < 4:
        return False

    from collections import Counter

    token_counts = Counter(tokens)
    most_common_token, most_common_count = token_counts.most_common(1)[0]
    if most_common_count >= int(len(tokens) * 0.7) and len(tokens) >= 4:
        return True

    if len(tokens) >= 6:
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        bigram_counts = Counter(bigrams)
        most_common_bigram, most_common_bigram_count = bigram_counts.most_common(1)[0]
        if most_common_bigram_count >= int(len(bigrams) * 0.6):
            return True

    if len(tokens) >= 8:
        trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]
        trigram_counts = Counter(trigrams)
        most_common_trigram, most_common_trigram_count = trigram_counts.most_common(1)[0]
        if most_common_trigram_count >= int(len(trigrams) * 0.5):
            return True

    return False


def detect_sentence_like_expansion(ru: str, en: str) -> list[str]:
    """Flag outputs that look like short generated sentences for single-word glosses.

    Examples:
        Москва -> It is located in Moscow.
        надежда -> There is hope.
    """
    flags: list[str] = []
    if is_blank(ru) or is_blank(en):
        return flags

    ru_tokens = ru.strip().split()
    en_lower = en.strip().lower()

    if len(ru_tokens) == 1:
        for prefix in SENTENCE_LIKE_PREFIXES:
            if en_lower.startswith(prefix):
                flags.append("sentence_like_singleword_expansion")
                break

    return flags


def detect_length_inflation(ru: str, en: str) -> list[str]:
    """Detect token inflation from short Russian glosses to long English output."""
    flags: list[str] = []
    if is_blank(ru) or is_blank(en):
        return flags

    ru_tokens = token_count(ru)
    en_tokens = token_count(en)

    if ru_tokens == 1 and en_tokens >= 3:
        flags.append("multiword_for_singleword")
    if ru_tokens == 1 and en_tokens >= 4:
        flags.append("token_inflation")
    if len(en) > max(80, len(ru) * 5):
        flags.append("too_long_for_gloss")

    return flags


def _is_single_titlecase_cyrillic(text: str) -> bool:
    """Check if text is a single Cyrillic token starting with uppercase."""
    if is_blank(text) or " " in text.strip():
        return False
    stripped = text.strip()
    if not stripped:
        return False
    has_cyrillic = any("\u0400" <= c <= "\u04FF" for c in stripped)
    return has_cyrillic and stripped[0].isupper()


def detect_name_expansion_patterns(ru: str, en: str) -> list[str]:
    """Flag proper-name overexpansion heuristics.

    Examples:
        Москва -> It is located in Moscow.
        Назарет -> The city of Nazareth
    """
    flags: list[str] = []
    if is_blank(ru) or is_blank(en):
        return flags

    if _is_single_titlecase_cyrillic(ru):
        en_tokens = token_count(en)
        en_lower = en.strip().lower()

        if en_tokens >= 3:
            flags.append("probable_name_overexpansion")
        elif (
            en_lower.startswith("it is ")
            or en_lower.startswith("there is ")
            or en_lower.startswith("the city of ")
            or en_lower.startswith("it is located ")
        ):
            flags.append("probable_name_overexpansion")

    return flags


def detect_placeholder_or_garbage(en: str) -> list[str]:
    """Detect outputs that look like placeholders or broken garbage."""
    flags: list[str] = []
    if is_blank(en):
        return flags

    stripped = en.strip()

    # Output is just quotes around something trivial
    if (stripped.startswith('"') and stripped.endswith('"')) or \
       (stripped.startswith("'") and stripped.endswith("'")):
        inner = stripped[1:-1].strip()
        if len(inner) <= 2:
            flags.append("quoted_trivial_output")

    # Repeated hyphen patterns
    if re.search(r"(-\s*){3,}", stripped):
        flags.append("repeated_hyphens")

    return flags
