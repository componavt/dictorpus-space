"""Tests for step 01: missing-English reuse analysis.

Run with: python3 tests/sem_cat/test_reuse_analysis.py
"""

import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.sem_cat.pipeline.reuse_analysis import (
    is_nonblank_text,
    normalize_existing_en,
    build_pos_gloss_ru_key,
    distinct_existing_en_candidates,
    analyze_missing_en_reuse,
    build_reuse_summary,
    write_reuse_outputs,
    print_reuse_summary,
    ReuseAnalysisResult,
)

# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------


def test_is_nonblank_text():
    assert is_nonblank_text(None) is False
    assert is_nonblank_text("") is False
    assert is_nonblank_text("   ") is False
    assert is_nonblank_text("hello") is True
    assert is_nonblank_text("  hello  ") is True


def test_normalize_existing_en():
    assert normalize_existing_en(None) is None
    assert normalize_existing_en("") is None
    assert normalize_existing_en("   ") is None
    assert normalize_existing_en("hello") == "hello"
    assert normalize_existing_en("  hello  ") == "hello"
    assert normalize_existing_en("  hello   world  ") == "hello world"


def test_build_pos_gloss_ru_key():
    assert build_pos_gloss_ru_key("NOUN", "деревня") == "NOUN::деревня"
    assert build_pos_gloss_ru_key("NOUN", "  деревня  ") == "NOUN::деревня"
    assert build_pos_gloss_ru_key("  NOUN  ", "деревня") == "NOUN::деревня"
    assert build_pos_gloss_ru_key("VERB", "бить") == "VERB::бить"
    assert build_pos_gloss_ru_key(None, None) == "::"


def test_distinct_existing_en_candidates_unambiguous():
    df = pd.DataFrame({
        "meaning_en": ["village", "village", "village"],
    })
    candidates = distinct_existing_en_candidates(df)
    assert candidates == ["village"]


def test_distinct_existing_en_candidates_ambiguous():
    df = pd.DataFrame({
        "meaning_en": ["offence", "insult", "offence"],
    })
    candidates = distinct_existing_en_candidates(df)
    assert candidates == ["insult", "offence"]


def test_distinct_existing_en_candidates_ignores_blank():
    df = pd.DataFrame({
        "meaning_en": ["village", "", None, "village"],
    })
    candidates = distinct_existing_en_candidates(df)
    assert candidates == ["village"]


def test_distinct_existing_en_candidates_sorted_alphabetically():
    df = pd.DataFrame({
        "meaning_en": ["insult", "offence", "insult", "offence"],
    })
    candidates = distinct_existing_en_candidates(df)
    assert candidates == ["insult", "offence"]


def test_distinct_existing_en_candidates_normalizes_whitespace():
    df = pd.DataFrame({
        "meaning_en": ["  village  ", "village", "   village   "],
    })
    candidates = distinct_existing_en_candidates(df)
    assert candidates == ["village"]


def test_distinct_existing_en_candidates_does_not_split_semicolon():
    """One candidate string with semicolon should count as one, not two."""
    df = pd.DataFrame({
        "meaning_en": ["offence; insult"],
    })
    candidates = distinct_existing_en_candidates(df)
    assert candidates == ["offence; insult"]


# ---------------------------------------------------------------------------
# Exact grouping by (pos, primary_gloss_ru)
# ---------------------------------------------------------------------------


def test_grouper_different_pos_same_gloss_are_separate():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "VERB", "primary_gloss_ru": "деревня", "meaning_en": "to village"},
        {"lang": "vep", "lemma": "l3", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": ""},
        {"lang": "vep", "lemma": "l4", "pos": "VERB", "primary_gloss_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    # Each (pos, gloss_ru) is a separate group
    # NOUN + деревня: 1 existing candidate ("village") → missing NOUN goes to unambiguous
    # VERB + деревня: 1 existing candidate ("to village") → missing VERB goes to unambiguous
    assert len(result.missing_en_reusable_unambiguous) == 2
    unamb_keys = set(result.missing_en_reusable_unambiguous["pos_gloss_ru_key"])
    assert unamb_keys == {"NOUN::деревня", "VERB::деревня"}

    assert len(result.missing_en_reusable_ambiguous) == 0
    assert len(result.missing_en_without_reuse) == 0


def test_grouper_same_pos_different_gloss_are_separate():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": ""},
        {"lang": "vep", "lemma": "l3", "pos": "NOUN", "primary_gloss_ru": "дом", "meaning_en": "house"},
        {"lang": "vep", "lemma": "l4", "pos": "NOUN", "primary_gloss_ru": "дом", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    # деревня missing → unambiguous (1 candidate)
    # дом missing → unambiguous (1 candidate)
    # Both groups have 1 candidate, so both missing go to unambiguous
    assert len(result.missing_en_reusable_unambiguous) == 2
    unamb_glosses = set(result.missing_en_reusable_unambiguous["primary_gloss_ru"])
    assert unamb_glosses == {"деревня", "дом"}


# ---------------------------------------------------------------------------
# Unambiguous reuse
# ---------------------------------------------------------------------------


def test_unambiguous_reuse_one_candidate():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": ""},
        {"lang": "olo", "lemma": "l3", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_unambiguous) == 2
    # All rows have 1 candidate and go to unambiguous
    for _, row in result.missing_en_reusable_unambiguous.iterrows():
        assert row["existing_en_candidates"] == "village"
        assert row["existing_en_candidate_count"] == 1
        assert row["suggested_candidate_index"] == 1


def test_unambiguous_reuse_all_missing_no_candidates():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": ""},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_unambiguous) == 0
    # No existing English → goes to no_reuse
    assert len(result.missing_en_without_reuse) == 2


# ---------------------------------------------------------------------------
# Ambiguous reuse
# ---------------------------------------------------------------------------


def test_ambiguous_reuse_two_candidates():
    df = pd.DataFrame([
        {"lang": "krl", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": "offence"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": "insult"},
        {"lang": "olo", "lemma": "l3", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": ""},
        {"lang": "vep", "lemma": "l4", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_ambiguous) == 2
    for _, row in result.missing_en_reusable_ambiguous.iterrows():
        # Candidates should be sorted alphabetically
        assert row["existing_en_candidates"] == "insult || offence"
        assert row["existing_en_candidate_count"] == 2
        assert row["suggested_candidate_index"] == 1


def test_ambiguous_reuse_three_candidates():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "слово", "meaning_en": "word"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "слово", "meaning_en": "word"},
        {"lang": "vep", "lemma": "l3", "pos": "NOUN", "primary_gloss_ru": "слово", "meaning_en": "term"},
        {"lang": "vep", "lemma": "l4", "pos": "NOUN", "primary_gloss_ru": "слово", "meaning_en": "expression"},
        {"lang": "vep", "lemma": "l5", "pos": "NOUN", "primary_gloss_ru": "слово", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_ambiguous) == 1
    row = result.missing_en_reusable_ambiguous.iloc[0]
    # Candidates should be sorted alphabetically
    assert row["existing_en_candidates"] == "expression || term || word"
    assert row["existing_en_candidate_count"] == 3


# ---------------------------------------------------------------------------
# No reuse
# ---------------------------------------------------------------------------


def test_no_reuse_missing_all_have_no_candidates():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "концертный зал", "meaning_en": ""},
        {"lang": "olo", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "концертный зал", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_unambiguous) == 0
    assert len(result.missing_en_reusable_ambiguous) == 0
    # All go to no_reuse
    assert len(result.missing_en_without_reuse) == 2
    assert result.missing_en_without_reuse["pos_gloss_ru_key"].nunique() == 1


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------


def test_whitespace_normalization_counts_as_same_candidate():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": "  village  "},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l3", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    # All "village" values (including whitespace variants) count as 1 distinct candidate
    assert len(result.missing_en_reusable_unambiguous) == 1
    assert result.missing_en_reusable_unambiguous.iloc[0]["existing_en_candidate_count"] == 1


# ---------------------------------------------------------------------------
# Semicolon content not split
# ---------------------------------------------------------------------------


def test_semicolon_not_split_into_multiple_candidates():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "дело", "meaning_en": "offence; insult"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "дело", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    # "offence; insult" is one candidate string
    assert len(result.missing_en_reusable_unambiguous) == 1
    assert result.missing_en_reusable_unambiguous.iloc[0]["existing_en_candidates"] == "offence; insult"
    assert result.missing_en_reusable_unambiguous.iloc[0]["existing_en_candidate_count"] == 1


# ---------------------------------------------------------------------------
# Summary outputs
# ---------------------------------------------------------------------------


def test_unambiguous_summary_one_row_per_group():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": ""},
        {"lang": "olo", "lemma": "l3", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    summary = result.unambiguous_summary
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["pos_gloss_ru_key"] == "NOUN::деревня"
    assert row["task_pos"] == "NOUN"
    assert row["primary_gloss_ru"] == "деревня"
    assert row["existing_en_candidates"] == "village"
    assert row["existing_en_candidate_count"] == 1
    assert row["missing_row_count"] == 2
    assert row["existing_en_row_count"] == 1
    assert "vep" in row["langs"]
    assert "olo" in row["langs"]
    # First lemma might be different based on group order
    assert row["example_lemma"] in ["l1", "l2", "l3"]
    assert row["suggested_candidate_index"] == 1


def test_ambiguous_summary_one_row_per_group():
    df = pd.DataFrame([
        {"lang": "krl", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": "offence"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": "insult"},
        {"lang": "olo", "lemma": "l3", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": ""},
        {"lang": "vep", "lemma": "l4", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    summary = result.ambiguous_summary
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["pos_gloss_ru_key"] == "NOUN::обида"
    assert row["task_pos"] == "NOUN"
    assert row["primary_gloss_ru"] == "обида"
    assert row["existing_en_candidates"] == "insult || offence"
    assert row["existing_en_candidate_count"] == 2
    assert row["missing_row_count"] == 2
    assert row["existing_en_row_count"] == 2
    # Langs are in the group
    assert "vep" in row["langs"]
    assert row["example_lemma"] in ["l1", "l2", "l3", "l4"]
    assert row["suggested_candidate_index"] == 1


# ---------------------------------------------------------------------------
# Empty outputs
# ---------------------------------------------------------------------------


def test_empty_dataframe_creates_valid_outputs():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "primary_gloss_ru": "дом", "meaning_en": "house"},
    ])
    result = analyze_missing_en_reuse(df)

    # No missing English at all
    assert len(result.missing_en_reusable_unambiguous) == 0
    assert len(result.missing_en_reusable_ambiguous) == 0
    assert len(result.missing_en_without_reuse) == 0

    # But summary should still have proper columns
    assert len(result.unambiguous_summary) == 0
    assert len(result.ambiguous_summary) == 0

    # Expected columns present
    expected_unamb_cols = [
        "pos_gloss_ru_key",
        "task_pos",
        "primary_gloss_ru",
        "existing_en_candidates",
        "existing_en_candidate_count",
        "missing_row_count",
        "existing_en_row_count",
        "langs",
        "example_lemma",
        "suggested_candidate_index",
    ]
    assert list(result.unambiguous_summary.columns) == expected_unamb_cols
    assert list(result.ambiguous_summary.columns) == expected_unamb_cols


def test_writer_creates_csvs_with_headers_even_when_empty():
    with tempfile.TemporaryDirectory() as td:
        df = pd.DataFrame([
            {"lang": "vep", "lemma": "l1", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": "village"},
        ])
        result = analyze_missing_en_reuse(df)

        translate_dir = pathlib.Path(td) / "translate"
        write_reuse_outputs(result, translate_dir)

        # All 4 files created with headers
        unamb_path = translate_dir / "missing_en_reusable_unambiguous_pos_gloss_ru.csv"
        amb_path = translate_dir / "missing_en_reusable_ambiguous_pos_gloss_ru.csv"
        unamb_sum_path = translate_dir / "missing_en_reusable_unambiguous_pos_gloss_ru_summary.csv"
        amb_sum_path = translate_dir / "missing_en_reusable_ambiguous_pos_gloss_ru_summary.csv"

        assert unamb_path.exists()
        assert amb_path.exists()
        assert unamb_sum_path.exists()
        assert amb_sum_path.exists()

        # Even empty, they have headers
        unamb_df = pd.read_csv(unamb_path)
        unamb_cols = list(unamb_df.columns)
        assert "pos_gloss_ru_key" in unamb_cols
        assert "task_pos" in unamb_cols
        assert "primary_gloss_ru" in unamb_cols
        assert "lang" in unamb_cols
        assert "lemma" in unamb_cols
        assert "existing_en_candidates" in unamb_cols
        assert "existing_en_candidate_count" in unamb_cols
        assert "missing_row_count_for_pos_gloss_ru" in unamb_cols
        assert "existing_en_row_count_for_pos_gloss_ru" in unamb_cols
        # For row-level output, suggested_candidate_index is present when unambiguous
        # Check if it exists in the columns
        assert len(unamb_df) == 0


# ---------------------------------------------------------------------------
# Full fixture smoke test
# ---------------------------------------------------------------------------


def test_full_fixture_all_cases():
    df = pd.DataFrame([
        # Unambiguous case: one existing EN for NOUN + деревня
        {"lang": "krl", "lemma": "lemma1", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "lemma2", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": ""},
        {"lang": "olo", "lemma": "lemma3", "pos": "NOUN", "primary_gloss_ru": "деревня", "meaning_en": None},

        # Ambiguous case: two existing ENs for NOUN + обида
        {"lang": "krl", "lemma": "lemma4", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": "offence"},
        {"lang": "vep", "lemma": "lemma5", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": "insult"},
        {"lang": "lud", "lemma": "lemma6", "pos": "NOUN", "primary_gloss_ru": "обида", "meaning_en": ""},

        # No reuse case: no existing EN for NOUN + концертный зал
        {"lang": "olo", "lemma": "lemma7", "pos": "NOUN", "primary_gloss_ru": "концертный зал", "meaning_en": ""},

        # Separate group: VERB + деревня (different POS from NOUN + деревня)
        {"lang": "vep", "lemma": "lemma8", "pos": "VERB", "primary_gloss_ru": "деревня", "meaning_en": "to village"},
        {"lang": "vep", "lemma": "lemma9", "pos": "VERB", "primary_gloss_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    # Summary counts
    assert result.stats["pos_gloss_ru_unambiguous_count"] == 2  # NOUN::деревня and VERB::деревня
    assert result.stats["pos_gloss_ru_ambiguous_count"] == 1  # NOUN::обида
    assert result.stats["pos_gloss_ru_without_reuse_count"] == 1  # NOUN::концертный зал


def test_cli_smoke_test():
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        """Run step 01 on a tiny temp data setup and verify the four output files are created."""
        from src.sem_cat.utils.vepkar_loader import load_meanings
        from src.sem_cat.pipeline.vepkar_translation_selection import prepare_meanings_for_translation
        from src.sem_cat.pipeline.reuse_analysis import analyze_missing_en_reuse, write_reuse_outputs

        # Create temp data directory - must have all 4 languages as per vepkar_loader
        data_dir = td_path / "data" / "vepkar"
        data_dir.mkdir(parents=True)

        for lang in ["vep", "olo", "lud", "krl"]:
            df = pd.DataFrame([
                {"id": "1", "lemma_id": "1", "meaning_id": "1", "meaning_num": 1,
                 "lemma": "l1", "lang": lang, "pos": "NOUN", "meaning_ru": "деревня деревня",
                 "meaning_en": "village", "concept_id": "", "category_id": ""},
                {"id": "2", "lemma_id": "1", "meaning_id": "2", "meaning_num": 2,
                 "lemma": "l1", "lang": lang, "pos": "NOUN", "meaning_ru": "деревня деревня",
                 "meaning_en": "", "concept_id": "", "category_id": ""},
            ])
            df.to_csv(data_dir / f"meanings_{lang}.csv", index=False)

        translate_dir = td_path / "data" / "sem_cat" / "2translate"

        # Load and analyze
        df_meanings = load_meanings(str(data_dir))
        work = prepare_meanings_for_translation(df_meanings)
        result = analyze_missing_en_reuse(work)
        write_reuse_outputs(result, translate_dir)

        # Verify all 4 output files exist
        assert (translate_dir / "missing_en_reusable_unambiguous_pos_gloss_ru.csv").exists()
        assert (translate_dir / "missing_en_reusable_ambiguous_pos_gloss_ru.csv").exists()
        assert (translate_dir / "missing_en_reusable_unambiguous_pos_gloss_ru_summary.csv").exists()
        assert (translate_dir / "missing_en_reusable_ambiguous_pos_gloss_ru_summary.csv").exists()

        # Verify unambiguous output has the missing rows (4 languages * 1 missing row each)
        unamb_df = pd.read_csv(translate_dir / "missing_en_reusable_unambiguous_pos_gloss_ru.csv")
        assert len(unamb_df) == 4  # 4 languages, each has 1 missing row for деревня
        # The gloss is "деревня деревня" from the fixture
        assert "деревня деревня" in unamb_df["primary_gloss_ru"].values
        assert unamb_df.iloc[0]["existing_en_candidates"] == "village"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_is_nonblank_text,
        test_normalize_existing_en,
        test_build_pos_gloss_ru_key,
        test_distinct_existing_en_candidates_unambiguous,
        test_distinct_existing_en_candidates_ambiguous,
        test_distinct_existing_en_candidates_ignores_blank,
        test_distinct_existing_en_candidates_sorted_alphabetically,
        test_distinct_existing_en_candidates_normalizes_whitespace,
        test_distinct_existing_en_candidates_does_not_split_semicolon,
        test_grouper_different_pos_same_gloss_are_separate,
        test_grouper_same_pos_different_gloss_are_separate,
        test_unambiguous_reuse_one_candidate,
        test_unambiguous_reuse_all_missing_no_candidates,
        test_ambiguous_reuse_two_candidates,
        test_ambiguous_reuse_three_candidates,
        test_no_reuse_missing_all_have_no_candidates,
        test_whitespace_normalization_counts_as_same_candidate,
        test_semicolon_not_split_into_multiple_candidates,
        test_unambiguous_summary_one_row_per_group,
        test_ambiguous_summary_one_row_per_group,
        test_empty_dataframe_creates_valid_outputs,
        test_writer_creates_csvs_with_headers_even_when_empty,
        test_full_fixture_all_cases,
        test_cli_smoke_test,
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
            import traceback
            traceback.print_exc()

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
