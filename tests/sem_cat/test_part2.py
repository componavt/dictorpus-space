"""Part 2 tests for translation QA, text utils, and I/O helpers.

Run with: python3 tests/sem_cat/test_part2.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from src.sem_cat.utils.text_utils import (
    is_blank,
    normalize_whitespace,
    token_count,
    contains_ascii_letters,
    is_punctuation_only,
)
from src.sem_cat.utils.distance import (
    levenshtein_distance,
    normalized_edit_distance,
    normalized_edit_similarity,
)
from src.sem_cat.qa.translation_flags import (
    detect_repetition,
    detect_sentence_like_expansion,
    detect_length_inflation,
    detect_name_expansion_patterns,
    detect_placeholder_or_garbage,
)
from src.sem_cat.qa.translation_qa import (
    analyze_translation,
    QAResult,
    TranslationQAConfig,
)
from src.sem_cat.io.translation_rows import (
    build_translation_row,
    CANONICAL_COLUMNS,
    QA_VERSION,
)
from src.sem_cat.io.translation_cache import (
    load_translation_cache,
    build_cached_gloss_set,
    REQUIRED_CACHE_COLUMNS,
)


def test_is_blank():
    assert is_blank(None) is True
    assert is_blank("") is True
    assert is_blank("   ") is True
    assert is_blank("hello") is False


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"
    assert normalize_whitespace("test") == "test"


def test_token_count():
    assert token_count("") == 0
    assert token_count("hello") == 1
    assert token_count("hello world") == 2
    assert token_count("  a  b  c  ") == 3


def test_contains_ascii_letters():
    assert contains_ascii_letters("hello") is True
    assert contains_ascii_letters("привет") is False
    assert contains_ascii_letters("123") is False
    assert contains_ascii_letters("hello123") is True


def test_is_punctuation_only():
    assert is_punctuation_only("...") is True
    assert is_punctuation_only("!!!") is True
    assert is_punctuation_only("hello") is False
    assert is_punctuation_only("") is True


def test_levenshtein_distance():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("kitten", "sitting") == 3


def test_normalized_edit_distance():
    assert normalized_edit_distance("", "") == 0.0
    assert normalized_edit_distance("abc", "abc") == 0.0
    assert normalized_edit_distance("abc", "") == 1.0
    assert normalized_edit_distance("", "abc") == 1.0


def test_normalized_edit_similarity():
    assert normalized_edit_similarity("", "") == 1.0
    assert normalized_edit_similarity("abc", "abc") == 1.0
    assert normalized_edit_similarity("abc", "") == 0.0


# ---------------------------------------------------------------------------
# Sentence-like expansion detector
# ---------------------------------------------------------------------------

def test_sentence_like_triggers_on_moscow():
    flags = detect_sentence_like_expansion("Москва", "It is located in Moscow.")
    assert "sentence_like_singleword_expansion" in flags


def test_sentence_like_triggers_on_hope():
    flags = detect_sentence_like_expansion("надежда", "There is hope.")
    assert "sentence_like_singleword_expansion" in flags


def test_sentence_like_does_not_trigger_on_montenegro():
    flags = detect_sentence_like_expansion("Черногория", "Montenegro")
    assert "sentence_like_singleword_expansion" not in flags


def test_sentence_like_does_not_trigger_on_hope_word():
    flags = detect_sentence_like_expansion("надежда", "hope")
    assert "sentence_like_singleword_expansion" not in flags


def test_sentence_like_triggers_on_noah():
    flags = detect_sentence_like_expansion("Ной", "It was Noah's day.")
    assert "sentence_like_singleword_expansion" in flags


def test_sentence_like_triggers_on_city_of():
    flags = detect_sentence_like_expansion("Назарет", "The city of Nazareth")
    assert "sentence_like_singleword_expansion" in flags


# ---------------------------------------------------------------------------
# Proper-name overexpansion
# ---------------------------------------------------------------------------

def test_name_overexpansion_moscow():
    flags = detect_name_expansion_patterns("Москва", "It is located in Moscow.")
    assert "probable_name_overexpansion" in flags


# ---------------------------------------------------------------------------
# Repetition detector
# ---------------------------------------------------------------------------

def test_repetition_catches_loops():
    assert detect_repetition("No, no, no, no, no") is True
    assert detect_repetition(". . . . .") is True
    assert detect_repetition("hello world") is False


# ---------------------------------------------------------------------------
# QA analysis
# ---------------------------------------------------------------------------

def test_analyze_empty_output():
    result = analyze_translation("дом", "")
    assert result.qa_keep is False
    assert "empty_translation" in result.qa_flags


def test_analyze_sentence_like_flagged_but_kept():
    result = analyze_translation("дом", "It is located in the house.")
    assert result.qa_keep is True
    assert "sentence_like_singleword_expansion" in result.qa_flags
    assert result.qa_score > 0


def test_analyze_good_translation():
    result = analyze_translation("дом", "house")
    assert result.qa_keep is True
    assert len(result.qa_flags) == 0
    assert result.qa_score == 0.0


def test_qa_result_dataclass():
    result = QAResult(qa_keep=True, qa_score=0.5, qa_flags=["flag1"])
    assert result.qa_keep is True
    assert result.qa_score == 0.5
    assert result.qa_flags == ["flag1"]


# ---------------------------------------------------------------------------
# Cache loader validates required columns
# ---------------------------------------------------------------------------

def test_cache_loader_returns_empty_for_missing_file():
    df = load_translation_cache(pathlib.Path("/nonexistent/path.csv"))
    assert df.empty
    assert "gloss_ru" in df.columns


def test_cache_required_columns_defined():
    assert "gloss_ru" in REQUIRED_CACHE_COLUMNS
    assert "model_key" in REQUIRED_CACHE_COLUMNS
    assert "qa_keep" in REQUIRED_CACHE_COLUMNS


# ---------------------------------------------------------------------------
# Row builder always emits canonical columns
# ---------------------------------------------------------------------------

def test_row_builder_emits_canonical_columns():
    qa_result = QAResult(qa_keep=True, qa_score=0.0)
    row = build_translation_row(
        gloss_ru="дом",
        gloss_en="house",
        qa_result=qa_result,
        model_key="google",
        model_name="google",
        backend_family="google",
        translation_input_mode="raw",
        input_text_used="дом",
    )
    assert list(row.keys()) == CANONICAL_COLUMNS
    assert row["gloss_ru"] == "дом"
    assert row["gloss_en"] == "house"
    assert row["qa_keep"] is True
    assert row["qa_version"] == QA_VERSION
    assert row["model_key"] == "google"


def test_row_builder_with_roundtrip():
    qa_result = QAResult(qa_keep=True, qa_score=0.1, roundtrip_distance=0.2)
    row = build_translation_row(
        gloss_ru="дом",
        gloss_en="house",
        qa_result=qa_result,
        model_key="google",
        model_name="google",
        backend_family="google",
        translation_input_mode="raw",
        input_text_used="дом",
        gloss_ru_back="дом",
    )
    assert row["gloss_ru_back"] == "дом"
    assert row["roundtrip_distance"] == 0.2


# ---------------------------------------------------------------------------
# Length inflation detector
# ---------------------------------------------------------------------------

def test_length_inflation_single_to_multi():
    flags = detect_length_inflation("дом", "a very long description of the house")
    assert "token_inflation" in flags
    assert "multiword_for_singleword" in flags


def test_length_inflation_not_triggered_on_normal():
    flags = detect_length_inflation("дом", "house")
    assert len(flags) == 0


# ---------------------------------------------------------------------------
# Token inflation severity
# ---------------------------------------------------------------------------

def test_token_inflation_3_words():
    flags = detect_length_inflation("дом", "big red house")
    assert "multiword_for_singleword" in flags
    # token_inflation requires 4+ tokens
    assert "token_inflation" not in flags


def test_token_inflation_4_words():
    flags = detect_length_inflation("дом", "a big red house")
    assert "multiword_for_singleword" in flags
    assert "token_inflation" in flags


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_is_blank,
        test_normalize_whitespace,
        test_token_count,
        test_contains_ascii_letters,
        test_is_punctuation_only,
        test_levenshtein_distance,
        test_normalized_edit_distance,
        test_normalized_edit_similarity,
        test_sentence_like_triggers_on_moscow,
        test_sentence_like_triggers_on_hope,
        test_sentence_like_does_not_trigger_on_montenegro,
        test_sentence_like_does_not_trigger_on_hope_word,
        test_sentence_like_triggers_on_noah,
        test_sentence_like_triggers_on_city_of,
        test_name_overexpansion_moscow,
        test_repetition_catches_loops,
        test_analyze_empty_output,
        test_analyze_sentence_like_flagged_but_kept,
        test_analyze_good_translation,
        test_qa_result_dataclass,
        test_cache_loader_returns_empty_for_missing_file,
        test_cache_required_columns_defined,
        test_row_builder_emits_canonical_columns,
        test_row_builder_with_roundtrip,
        test_length_inflation_single_to_multi,
        test_length_inflation_not_triggered_on_normal,
        test_token_inflation_3_words,
        test_token_inflation_4_words,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {test_fn.__name__}: {e}")

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
