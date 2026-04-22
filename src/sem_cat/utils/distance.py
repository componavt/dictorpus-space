"""String distance helpers shared across translation QA and comparison."""

from __future__ import annotations

from .text_utils import is_blank


def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard dynamic programming Levenshtein distance."""
    s1, s2 = str(s1), str(s2)
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len1][len2]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Normalized Levenshtein distance in [0.0, 1.0].

    0.0 = identical, 1.0 = completely different.
    """
    if is_blank(s1) and is_blank(s2):
        return 0.0
    if is_blank(s1) or is_blank(s2):
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    if max_len == 0:
        return 0.0

    return levenshtein_distance(s1, s2) / max_len


def normalized_edit_similarity(a: str, b: str) -> float:
    """1 - normalized_edit_distance (case-insensitive).

    Returns 1.0 if both blank, 0.0 if one blank.
    """
    a, b = str(a).lower(), str(b).lower()
    if is_blank(a) and is_blank(b):
        return 1.0
    if is_blank(a) or is_blank(b):
        return 0.0
    return 1.0 - normalized_edit_distance(a, b)
