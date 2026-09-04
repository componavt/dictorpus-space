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
    distinct_existing_en_candidates,
    analyze_missing_en_reuse,
    write_reuse_outputs,
    print_reuse_summary,
    ReuseAnalysisResult,
    NO_REUSE_ROW_LEVEL_COLUMNS,
    UNAMBIGUOUS_ROW_LEVEL_COLUMNS,
    AMBIGUOUS_ROW_LEVEL_COLUMNS,
    UNAMBIGUOUS_SUMMARY_COLUMNS,
    AMBIGUOUS_SUMMARY_COLUMNS,
    POS_MEANINGS_RU_COLUMNS,
    CONCEPT_CATEGORY_AUDIT_COLUMNS,
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
# Exact grouping by (pos, meaning_ru)
# ---------------------------------------------------------------------------


def test_grouper_different_pos_same_gloss_are_separate():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "VERB", "meaning_ru": "деревня деревня", "meaning_en": "to village"},
        {"lang": "vep", "lemma": "l3", "pos": "NOUN", "meaning_ru": "деревня деревня", "meaning_en": ""},
        {"lang": "vep", "lemma": "l4", "pos": "VERB", "meaning_ru": "деревня деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    # Each (pos, meaning_ru) is a separate group
    # NOUN + деревня деревня: 1 existing candidate ("village") → missing NOUN goes to unambiguous
    # VERB + деревня деревня: 1 existing candidate ("to village") → missing VERB goes to unambiguous
    assert len(result.missing_en_reusable_unambiguous) == 2
    assert len(result.missing_en_reusable_ambiguous) == 0
    assert len(result.missing_en_without_reuse) == 0


def test_grouper_same_pos_different_full_meanings_are_separate():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "место (под чем-либо)", "meaning_en": "place under something"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "место (под чем-либо)", "meaning_en": ""},
        {"lang": "vep", "lemma": "l3", "pos": "NOUN", "meaning_ru": "место (перед чем-либо)", "meaning_en": "place in front of something"},
        {"lang": "vep", "lemma": "l4", "pos": "NOUN", "meaning_ru": "место (перед чем-либо)", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    # Two distinct full meanings must remain separate
    assert len(result.missing_en_reusable_unambiguous) == 2
    assert set(
        result.missing_en_reusable_unambiguous["meaning_ru"]
    ) == {
        "место (под чем-либо)",
        "место (перед чем-либо)",
    }
    assert result.unambiguous_summary["meaning_ru"].nunique() == 2


# ---------------------------------------------------------------------------
# Unambiguous reuse
# ---------------------------------------------------------------------------


def test_unambiguous_reuse_one_candidate():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"lang": "olo", "lemma": "l3", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_unambiguous) == 2
    for _, row in result.missing_en_reusable_unambiguous.iterrows():
        assert row["existing_en_candidates"] == "village"
        assert row["existing_en_candidate_count"] == 1
        assert "suggested_candidate_index" not in row


def test_unambiguous_reuse_all_missing_no_candidates():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_unambiguous) == 0
    assert len(result.missing_en_without_reuse) == 2


# ---------------------------------------------------------------------------
# Ambiguous reuse
# ---------------------------------------------------------------------------


def test_ambiguous_reuse_two_candidates():
    df = pd.DataFrame([
        {"lang": "krl", "lemma": "l1", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "offence"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
        {"lang": "olo", "lemma": "l3", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
        {"lang": "vep", "lemma": "l4", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_ambiguous) == 2
    for _, row in result.missing_en_reusable_ambiguous.iterrows():
        assert row["existing_en_candidates"] == "insult || offence"
        assert row["existing_en_candidate_count"] == 2
        assert row["suggested_candidate_index"] == 1


def test_ambiguous_reuse_three_candidates():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "слово", "meaning_en": "word"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "слово", "meaning_en": "word"},
        {"lang": "vep", "lemma": "l3", "pos": "NOUN", "meaning_ru": "слово", "meaning_en": "term"},
        {"lang": "vep", "lemma": "l4", "pos": "NOUN", "meaning_ru": "слово", "meaning_en": "expression"},
        {"lang": "vep", "lemma": "l5", "pos": "NOUN", "meaning_ru": "слово", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_ambiguous) == 1
    row = result.missing_en_reusable_ambiguous.iloc[0]
    assert row["existing_en_candidates"] == "expression || term || word"
    assert row["existing_en_candidate_count"] == 3


# ---------------------------------------------------------------------------
# No reuse
# ---------------------------------------------------------------------------


def test_no_reuse_missing_all_have_no_candidates():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "концертный зал", "meaning_en": ""},
        {"lang": "olo", "lemma": "l2", "pos": "NOUN", "meaning_ru": "концертный зал", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_unambiguous) == 0
    assert len(result.missing_en_reusable_ambiguous) == 0
    assert len(result.missing_en_without_reuse) == 2
    assert result.missing_en_without_reuse["pos"].nunique() == 1


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------


def test_whitespace_normalization_counts_as_same_candidate():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "  village  "},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l3", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_unambiguous) == 1
    assert result.missing_en_reusable_unambiguous.iloc[0]["existing_en_candidate_count"] == 1


# ---------------------------------------------------------------------------
# Semicolon content not split
# ---------------------------------------------------------------------------


def test_semicolon_not_split_into_multiple_candidates():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "дело", "meaning_en": "offence; insult"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "дело", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_unambiguous) == 1
    assert result.missing_en_reusable_unambiguous.iloc[0]["existing_en_candidates"] == "offence; insult"
    assert result.missing_en_reusable_unambiguous.iloc[0]["existing_en_candidate_count"] == 1


# ---------------------------------------------------------------------------
# Summary outputs
# ---------------------------------------------------------------------------


def test_unambiguous_summary_one_row_per_group():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"lang": "olo", "lemma": "l3", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    summary = result.unambiguous_summary
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["pos"] == "NOUN"
    assert row["meaning_ru"] == "деревня"
    assert row["existing_en_candidates"] == "village"
    assert row["existing_en_candidate_count"] == 1
    assert row["missing_row_count"] == 2
    assert row["existing_en_row_count"] == 1
    assert "vep" in row["missing_langs"]
    assert "olo" in row["missing_langs"]
    assert "vep" in row["existing_en_langs"]
    assert row["example_missing_lemma"] in ["l2", "l3"]


def test_ambiguous_summary_one_row_per_group():
    df = pd.DataFrame([
        {"lang": "krl", "lemma": "l1", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "offence"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
        {"lang": "olo", "lemma": "l3", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
        {"lang": "vep", "lemma": "l4", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    summary = result.ambiguous_summary
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["pos"] == "NOUN"
    assert row["meaning_ru"] == "обида"
    assert row["existing_en_candidates"] == "insult || offence"
    assert row["existing_en_candidate_count"] == 2
    assert row["missing_row_count"] == 2
    assert row["existing_en_row_count"] == 2
    assert "olo" in row["missing_langs"]
    assert "vep" in row["missing_langs"]
    assert "krl" in row["existing_en_langs"]
    assert "vep" in row["existing_en_langs"]
    assert row["example_missing_lemma"] in ["l3", "l4"]
    assert row["suggested_candidate_index"] == 1


# ---------------------------------------------------------------------------
# Empty outputs
# ---------------------------------------------------------------------------


def test_empty_dataframe_creates_valid_outputs():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "дом", "meaning_en": "house"},
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
    expected_unamb_cols = UNAMBIGUOUS_SUMMARY_COLUMNS
    assert list(result.unambiguous_summary.columns) == expected_unamb_cols
    expected_amb_cols = AMBIGUOUS_SUMMARY_COLUMNS
    assert list(result.ambiguous_summary.columns) == expected_amb_cols


def test_writer_creates_csvs_with_headers_even_when_empty():
    with tempfile.TemporaryDirectory() as td:
        df = pd.DataFrame([
            {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        ])
        result = analyze_missing_en_reuse(df)

        translate_dir = pathlib.Path(td) / "translate"
        write_reuse_outputs(result, translate_dir)

        reusable_dir = translate_dir / "reusable_english"
        # All 6 files created with headers
        assert (reusable_dir / "one_english.csv").exists()
        assert (reusable_dir / "one_english_summary.csv").exists()
        assert (reusable_dir / "several_english.csv").exists()
        assert (reusable_dir / "several_english_summary.csv").exists()
        assert (translate_dir / "needs_translation_no_reuse.csv").exists()
        assert (translate_dir / "pos_meanings_ru.csv").exists()

        # Even empty, they have headers
        unamb_df = pd.read_csv(reusable_dir / "one_english.csv")
        unamb_cols = list(unamb_df.columns)
        assert "pos" in unamb_cols
        assert "meaning_ru" in unamb_cols
        assert "lang" in unamb_cols
        assert "lemma" in unamb_cols
        assert "existing_en_candidates" in unamb_cols
        assert "existing_en_candidate_count" in unamb_cols
        assert "missing_row_count" in unamb_cols
        assert "existing_en_row_count" in unamb_cols
        assert "primary_gloss_ru" not in unamb_cols
        assert "pos_gloss_ru_key" not in unamb_cols
        assert "pos_meaning_ru_key" not in unamb_cols
        assert "missing_row_count_for_pos_gloss_ru" not in unamb_cols
        assert "existing_en_row_count_for_pos_gloss_ru" not in unamb_cols
        assert "task_pos" not in unamb_cols
        assert len(unamb_df) == 0


# ---------------------------------------------------------------------------
# Full fixture smoke test
# ---------------------------------------------------------------------------


def test_full_fixture_all_cases():
    df = pd.DataFrame([
        # Unambiguous case: one existing EN for NOUN + деревня
        {"lang": "krl", "lemma": "lemma1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "lemma2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"lang": "olo", "lemma": "lemma3", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": None},

        # Ambiguous case: two existing ENs for NOUN + обида
        {"lang": "krl", "lemma": "lemma4", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "offence"},
        {"lang": "vep", "lemma": "lemma5", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
        {"lang": "lud", "lemma": "lemma6", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},

        # No reuse case: no existing EN for NOUN + концертный зал
        {"lang": "olo", "lemma": "lemma7", "pos": "NOUN", "meaning_ru": "концертный зал", "meaning_en": ""},

        # Separate group: VERB + деревня (different POS from NOUN + деревня)
        {"lang": "vep", "lemma": "lemma8", "pos": "VERB", "meaning_ru": "деревня", "meaning_en": "to village"},
        {"lang": "vep", "lemma": "lemma9", "pos": "VERB", "meaning_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    # Classification completeness invariant
    assert result.stats["rows_missing_en"] == (
        result.stats["rows_reusable_unambiguous"]
        + result.stats["rows_reusable_ambiguous"]
        + result.stats["rows_missing_en_without_reuse"]
    )

    # Unambiguous output row count matches stats
    assert len(result.missing_en_reusable_unambiguous) == result.stats["rows_reusable_unambiguous"]
    # Ambiguous output row count matches stats
    assert len(result.missing_en_reusable_ambiguous) == result.stats["rows_reusable_ambiguous"]

    # Summary counts
    assert result.stats["unambiguous_group_count"] == 2
    assert result.stats["ambiguous_group_count"] == 1
    assert result.stats["no_reuse_group_count"] == 1


def test_missing_langs_and_existing_en_langs_separated():
    """Verify missing_langs and existing_en_langs are correctly separated."""
    df = pd.DataFrame([
        # Missing row in krl
        {"lang": "krl", "lemma": "l1", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
        # Existing English in vep
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
    ])
    result = analyze_missing_en_reuse(df)

    # With 1 candidate, goes to unambiguous_summary
    summary = result.unambiguous_summary
    assert len(summary) == 1
    row = summary.iloc[0]

    assert row["missing_langs"] == "krl"
    assert row["existing_en_langs"] == "vep"
    assert row["example_missing_lemma"] == "l1"


def test_per_language_stats():
    df = pd.DataFrame([
        {"id": "k1", "meaning_id": "m1", "lemma_id": "l1", "meaning_ru": "деревня деревня", "concept_id": "c1", "category_id": "cat1",
         "lang": "krl", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"id": "k2", "meaning_id": "m2", "lemma_id": "l2", "meaning_ru": "деревня деревня", "concept_id": "c1", "category_id": "cat1",
         "lang": "krl", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"id": "v1", "meaning_id": "m3", "lemma_id": "l3", "meaning_ru": "деревня деревня", "concept_id": "c1", "category_id": "cat1",
         "lang": "vep", "lemma": "l3", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"id": "o1", "meaning_id": "m4", "lemma_id": "l4", "meaning_ru": "деревня деревня", "concept_id": "c1", "category_id": "cat1",
         "lang": "olo", "lemma": "l4", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"id": "lu1", "meaning_id": "m5", "lemma_id": "l5", "meaning_ru": "дом дом", "concept_id": "c2", "category_id": "cat2",
         "lang": "lud", "lemma": "l5", "pos": "NOUN", "meaning_ru": "дом", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert result.per_lang_stats is None


def test_unambiguous_summary_has_no_suggested_candidate_index():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    unamb_summary = result.unambiguous_summary
    assert "suggested_candidate_index" not in unamb_summary.columns
    assert set(unamb_summary.columns) == set(UNAMBIGUOUS_SUMMARY_COLUMNS)


def test_ambiguous_summary_has_suggested_candidate_index():
    df = pd.DataFrame([
        {"lang": "krl", "lemma": "l1", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "offence"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
        {"lang": "olo", "lemma": "l3", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    amb_summary = result.ambiguous_summary
    assert "suggested_candidate_index" in amb_summary.columns
    assert set(amb_summary.columns) == set(AMBIGUOUS_SUMMARY_COLUMNS)
    assert amb_summary.iloc[0]["suggested_candidate_index"] == 1


def test_unambiguous_row_level_output_excludes_summary_placeholder_columns():
    df = pd.DataFrame(
        [
            {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
            {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        ]
    )
    result = analyze_missing_en_reuse(df)
    columns = set(result.missing_en_reusable_unambiguous.columns)

    assert "existing_en_langs_for_summary" not in columns
    assert "missing_langs_for_summary" not in columns
    assert "example_missing_lemma_for_summary" not in columns


def test_ambiguous_row_level_output_excludes_summary_placeholder_columns():
    df = pd.DataFrame(
        [
            {"lang": "krl", "lemma": "l1", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "offence"},
            {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
            {"lang": "olo", "lemma": "l3", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
        ]
    )
    result = analyze_missing_en_reuse(df)
    columns = set(result.missing_en_reusable_ambiguous.columns)

    assert "existing_en_langs_for_summary" not in columns
    assert "missing_langs_for_summary" not in columns
    assert "example_missing_lemma_for_summary" not in columns
    assert "suggested_candidate_index" in columns


def test_row_level_csv_schema_is_exact(tmp_path):
    df = pd.DataFrame(
        [
            {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
            {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
            {"lang": "krl", "lemma": "l3", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "offence"},
            {"lang": "vep", "lemma": "l4", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
            {"lang": "olo", "lemma": "l5", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
        ]
    )
    result = analyze_missing_en_reuse(df)
    write_reuse_outputs(result, tmp_path)

    reusable_dir = tmp_path / "reusable_english"
    unambiguous_output = pd.read_csv(
        reusable_dir / "one_english.csv",
        dtype=str,
        keep_default_na=False,
    )
    ambiguous_output = pd.read_csv(
        reusable_dir / "several_english.csv",
        dtype=str,
        keep_default_na=False,
    )

    assert "primary_gloss_ru" not in unambiguous_output.columns
    assert "pos_gloss_ru_key" not in unambiguous_output.columns
    assert "pos_meaning_ru_key" not in unambiguous_output.columns
    assert "missing_row_count_for_pos_gloss_ru" not in unambiguous_output.columns
    assert "existing_en_row_count_for_pos_gloss_ru" not in unambiguous_output.columns
    assert "suggested_candidate_index" not in unambiguous_output.columns

    assert "primary_gloss_ru" not in ambiguous_output.columns
    assert "pos_gloss_ru_key" not in ambiguous_output.columns
    assert "pos_meaning_ru_key" not in ambiguous_output.columns
    assert "suggested_candidate_index" in ambiguous_output.columns


def test_summary_csv_still_contains_language_and_lemma_fields(tmp_path):
    df = pd.DataFrame(
        [
            {"lang": "krl", "lemma": "l1", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "offence"},
            {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
            {"lang": "olo", "lemma": "l3", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
        ]
    )
    result = analyze_missing_en_reuse(df)
    write_reuse_outputs(result, tmp_path)

    summary_dir = tmp_path / "reusable_english"
    summary_output = pd.read_csv(
        summary_dir / "several_english_summary.csv",
        dtype=str,
        keep_default_na=False,
    )

    assert "missing_langs" in summary_output.columns
    assert "existing_en_langs" in summary_output.columns
    assert "example_missing_lemma" in summary_output.columns
    assert "existing_en_langs_for_summary" not in summary_output.columns
    assert "missing_langs_for_summary" not in summary_output.columns
    assert "example_missing_lemma_for_summary" not in summary_output.columns


def test_row_level_outputs_preserve_identifiers():
    df = pd.DataFrame([
        {"id": "123", "meaning_id": "456", "lemma_id": "789", "lemma": "test_lemma", "lang": "vep",
         "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village", "meaning_ru": "деревня деревня",
         "concept_id": "c1", "category_id": "cat1"},
        {"id": "124", "meaning_id": "457", "lemma_id": "790", "lemma": "test_lemma2", "lang": "vep",
         "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "", "meaning_ru": "деревня деревня",
         "concept_id": "c1", "category_id": "cat1"},
    ])
    result = analyze_missing_en_reuse(df)

    required = {
        "id",
        "meaning_id",
        "lemma_id",
        "lemma",
        "lang",
        "pos",
        "meaning_ru",
    }
    unamb_cols = set(result.missing_en_reusable_unambiguous.columns)
    assert required.issubset(unamb_cols)


def test_row_level_outputs_do_not_leak_internal_fields():
    df = pd.DataFrame([
        {"id": "1", "meaning_id": "m1", "lemma_id": "l1", "meaning_ru": "деревня деревня", "concept_id": "c1", "category_id": "cat1",
         "lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"id": "2", "meaning_id": "m2", "lemma_id": "l2", "meaning_ru": "деревня деревня", "concept_id": "c1", "category_id": "cat1",
         "lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    forbidden = {
        "task_key",
        "has_existing_en",
        "existing_en_norm",
    }
    unamb_cols = set(result.missing_en_reusable_unambiguous.columns)
    assert not (forbidden & unamb_cols)


def test_no_reuse_output_has_correct_schema():
    """Verify no-reuse output has correct schema and no ambiguous-only fields."""
    from pathlib import Path
    import tempfile

    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"lang": "olo", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_reusable_unambiguous) == 0
    assert len(result.missing_en_reusable_ambiguous) == 0
    assert len(result.missing_en_without_reuse) == 2

    with tempfile.TemporaryDirectory() as td:
        translate_dir = Path(td) / "translate"
        write_reuse_outputs(result, translate_dir)

        no_reuse_path = translate_dir / "needs_translation_no_reuse.csv"
        assert no_reuse_path.exists()

        no_reuse_df = pd.read_csv(no_reuse_path)

        assert len(no_reuse_df) == 2

        required_cols = {
            "id", "meaning_id", "lemma_id", "lemma", "lang", "pos", "meaning_ru",
        }
        assert set(no_reuse_df.columns) == required_cols

        forbidden_cols = {
            "suggested_candidate_index",
            "task_key",
            "has_existing_en",
            "existing_en_norm",
            "existing_en_candidates",
            "existing_en_candidate_count",
            "existing_en_row_count",
        }
        assert not (forbidden_cols & set(no_reuse_df.columns))


def test_no_reuse_meaningful_group_size_three_rows():
    """Verify no-reuse group size is correctly computed for a 3-row group."""
    df = pd.DataFrame(
        [
            {
                "id": "1684",
                "meaning_id": "27910",
                "lemma_id": "24423",
                "lemma": "boljom",
                "lang": "vep",
                "pos": "NOUN",
                "meaning_ru": "брусничный напиток",
                "meaning_en": "",
                "concept_id": "",
                "category_id": "",
            },
            {
                "id": "1698",
                "meaning_id": "39759",
                "lemma_id": "34110",
                "lemma": "bolvezi",
                "lang": "vep",
                "pos": "NOUN",
                "meaning_ru": "брусничный напиток",
                "meaning_en": "",
                "concept_id": "",
                "category_id": "",
            },
            {
                "id": "2261",
                "meaning_id": "61455",
                "lemma_id": "53823",
                "lemma": "buoluvezi",
                "lang": "olo",
                "pos": "NOUN",
                "meaning_ru": "брусничный напиток",
                "meaning_en": "",
                "concept_id": "",
                "category_id": "",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)

    no_reuse = result.missing_en_without_reuse

    assert len(no_reuse) == 3
    assert no_reuse["pos"].nunique() == 1

    # Verify not in unambiguous or ambiguous outputs
    assert len(result.missing_en_reusable_unambiguous) == 0
    assert len(result.missing_en_reusable_ambiguous) == 0


def test_no_reuse_empty_output_creates_csv_with_headers():
    """Verify empty no-reuse output still has stable headers."""
    from pathlib import Path
    import tempfile

    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"lang": "olo", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "house"},
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.missing_en_without_reuse) == 0

    with tempfile.TemporaryDirectory() as td:
        translate_dir = Path(td) / "translate"
        write_reuse_outputs(result, translate_dir)

        no_reuse_path = translate_dir / "needs_translation_no_reuse.csv"
        assert no_reuse_path.exists()

        no_reuse_df = pd.read_csv(no_reuse_path)

        assert len(no_reuse_df) == 0

        required_cols = {
            "id", "meaning_id", "lemma_id", "lemma", "lang", "pos", "meaning_ru",
        }
        assert set(no_reuse_df.columns) == required_cols

        assert "suggested_candidate_index" not in no_reuse_df.columns
        assert "primary_gloss_ru" not in no_reuse_df.columns
        assert "pos_gloss_ru_key" not in no_reuse_df.columns
        assert "pos_meaning_ru_key" not in no_reuse_df.columns
        assert "missing_row_count_for_pos_gloss_ru" not in no_reuse_df.columns



def test_no_reuse_csv_schema_and_content():
    """Verify no-reuse CSV has exact schema and content with group size."""
    from pathlib import Path
    import tempfile

    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"lang": "olo", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    with tempfile.TemporaryDirectory() as td:
        translate_dir = Path(td) / "translate"
        write_reuse_outputs(result, translate_dir)

        output_path = translate_dir / "needs_translation_no_reuse.csv"
        output = pd.read_csv(output_path, dtype=str, keep_default_na=False)

        assert output_path.exists()
        assert list(output.columns) == NO_REUSE_ROW_LEVEL_COLUMNS
        assert len(output) == 2

        required_columns = {
            "id",
            "meaning_id",
            "lemma_id",
            "lemma",
            "lang",
            "pos",
            "meaning_ru",
        }
        assert required_columns == set(output.columns)

        forbidden_columns = {
            "existing_en_candidates",
            "existing_en_candidate_count",
            "existing_en_row_count",
            "suggested_candidate_index",
            "task_key",
            "has_existing_en",
            "existing_en_norm",
            "primary_gloss_ru",
            "pos_gloss_ru_key",
            "pos_meaning_ru_key",
            "missing_row_count_for_pos_gloss_ru",
        }
        assert not (forbidden_columns & set(output.columns))


def test_pos_meanings_ru_blank_pos_warning_preservation():
    """Test that blank raw POS triggers warning but is preserved in output."""
    import warnings

    df = pd.DataFrame([
        {
            "id": "1",
            "meaning_id": "m1",
            "lemma_id": "l1",
            "lemma": "test1",
            "lang": "vep",
            "pos": "",
            "meaning_ru": "деревня",
            "meaning_en": "",
            "concept_id": "",
            "category_id": "",
        },
    ])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = analyze_missing_en_reuse(df)

        assert len(w) == 1
        assert "blank raw pos" in str(w[0].message)
        assert "m1" in str(w[0].message)
        assert "l1" in str(w[0].message)

    assert len(result.pos_meanings_ru) == 1
    output_dict = result.pos_meanings_ru.iloc[0].to_dict()
    assert output_dict["meaning_ru"] == "деревня"
    assert output_dict["pos"] == ""

    with tempfile.TemporaryDirectory() as td:
        translate_dir = pathlib.Path(td) / "translate"
        write_reuse_outputs(result, translate_dir)

        output = pd.read_csv(
            translate_dir / "pos_meanings_ru.csv",
            dtype=str,
            keep_default_na=False,
        )
        assert output.iloc[0]["pos"] == ""
        assert output.iloc[0]["meaning_ru"] == "деревня"


def test_pos_meanings_ru_conservation_invariant():
    """Test that classification invariant is preserved with pos_meanings_ru."""
    df = pd.DataFrame([
        {
            "id": "1",
            "meaning_id": "m1",
            "lemma_id": "l1",
            "lemma": "test1",
            "lang": "vep",
            "pos": "NOUN",
            "meaning_ru": "деревня",
            "meaning_en": "",
            "concept_id": "",
            "category_id": "",
        },
        {
            "id": "2",
            "meaning_id": "m2",
            "lemma_id": "l2",
            "lemma": "test2",
            "lang": "vep",
            "pos": "NOUN",
            "meaning_ru": "деревня",
            "meaning_en": "",
            "concept_id": "",
            "category_id": "",
        },
        {
            "id": "3",
            "meaning_id": "m3",
            "lemma_id": "l3",
            "lemma": "test3",
            "lang": "olo",
            "pos": "NOUN",
            "meaning_ru": "деревня",
            "meaning_en": "",
            "concept_id": "",
            "category_id": "",
        },
        {
            "id": "4",
            "meaning_id": "m4",
            "lemma_id": "l4",
            "lemma": "test4",
            "lang": "olo",
            "pos": "NOUN",
            "meaning_ru": "концертный зал",
            "meaning_en": "",
            "concept_id": "",
            "category_id": "",
        },
    ])
    result = analyze_missing_en_reuse(df)

    assert result.stats["rows_missing_en"] == (
        result.stats["rows_reusable_unambiguous"]
        + result.stats["rows_reusable_ambiguous"]
        + result.stats["rows_missing_en_without_reuse"]
        + result.stats["rows_concept_covered_skip"]
        + result.stats["rows_invalid_concept_category_pair"]
    )


def test_pos_meanings_ru_empty_output():
    """Test empty output behavior when all rows have reusable English."""
    with tempfile.TemporaryDirectory() as td:
        df = pd.DataFrame([
            {
                "id": "1",
                "meaning_id": "m1",
                "lemma_id": "l1",
                "lemma": "test1",
                "lang": "vep",
                "pos": "NOUN",
                "meaning_ru": "деревня",
                "meaning_en": "village",
                "concept_id": "c1",
                "category_id": "cat1",
            },
        ])
        result = analyze_missing_en_reuse(df)

        translate_dir = pathlib.Path(td) / "translate"
        write_reuse_outputs(result, translate_dir)

        output_path = translate_dir / "pos_meanings_ru.csv"
        assert output_path.exists()

        output = pd.read_csv(output_path, dtype=str, keep_default_na=False)
        assert list(output.columns) == ["pos", "meaning_ru"]
        assert output.empty

        assert result.stats["rows_missing_en"] == 0
        assert result.stats["rows_missing_en_without_reuse"] == 0


def test_pos_meanings_ru_preserve_distinct_full_meanings():
    """Test that distinct full meanings with same primary_gloss_ru remain separate."""
    df = pd.DataFrame([
        {
            "id": "1",
            "meaning_id": "m1",
            "lemma_id": "l1",
            "lemma": "test1",
            "lang": "vep",
            "pos": "NOUN",
            "meaning_ru": "место (под чем-либо)",
            "primary_gloss_ru": "место",
            "meaning_en": "",
            "concept_id": "",
            "category_id": "",
        },
        {
            "id": "2",
            "meaning_id": "m2",
            "lemma_id": "l2",
            "lemma": "test2",
            "lang": "vep",
            "pos": "NOUN",
            "meaning_ru": "место (перед чем-либо)",
            "primary_gloss_ru": "место",
            "meaning_en": "",
            "concept_id": "",
            "category_id": "",
        },
        {
            "id": "3",
            "meaning_id": "m3",
            "lemma_id": "l3",
            "lemma": "test3",
            "lang": "vep",
            "pos": "NOUN",
            "meaning_ru": "место (вокруг чего-л.)",
            "primary_gloss_ru": "место",
            "meaning_en": "",
            "concept_id": "",
            "category_id": "",
        },
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.pos_meanings_ru) == 3
    meaning_rus = set(result.pos_meanings_ru["meaning_ru"].tolist())
    assert meaning_rus == {
        "место (под чем-либо)",
        "место (перед чем-либо)",
        "место (вокруг чего-л.)",
    }

    assert result.stats["rows_missing_en_without_reuse"] == 3


def test_pos_meanings_ru_exact_schema():
    """Test that pos_meanings_ru has exact schema without forbidden columns."""
    df = pd.DataFrame([
        {
            "id": "1",
            "meaning_id": "m1",
            "lemma_id": "l1",
            "lemma": "test1",
            "lang": "vep",
            "pos": "NOUN",
            "meaning_ru": "деревня",
            "meaning_en": "",
            "concept_id": "c1",
            "category_id": "cat1",
        },
    ])
    result = analyze_missing_en_reuse(df)

    assert list(result.pos_meanings_ru.columns) == ["pos", "meaning_ru"]

    forbidden = {
        "pos_gloss_ru_key",
        "task_pos",
        "primary_gloss_ru",
        "id",
        "meaning_id",
        "lemma_id",
        "lemma",
        "lang",
        "concept_id",
        "category_id",
        "missing_row_count",
        "missing_row_count_for_pos_gloss_ru",
        "missing_langs",
        "example_missing_lemma",
        "existing_en_candidates",
        "existing_en_candidate_count",
        "existing_en_row_count",
        "existing_en_row_count_for_pos_gloss_ru",
        "existing_en_langs",
        "suggested_candidate_index",
        "task_key",
        "has_existing_en",
        "existing_en_norm",
    }
    output_cols = set(result.pos_meanings_ru.columns)
    assert not (forbidden & output_cols)


def test_pos_meanings_ru_deduplicate_identical_pairs():
    """Test that identical (pos, meaning_ru) pairs deduplicate to one row."""
    df = pd.DataFrame([
        {
            "id": "1",
            "meaning_id": "m1",
            "lemma_id": "l1",
            "lemma": "test1",
            "lang": "vep",
            "pos": "NOUN",
            "meaning_ru": "морошковое варенье",
            "primary_gloss_ru": "морошковое варенье",
            "meaning_en": "",
            "concept_id": "",
            "category_id": "",
        },
        {
            "id": "2",
            "meaning_id": "m2",
            "lemma_id": "l2",
            "lemma": "test2",
            "lang": "olo",
            "pos": "NOUN",
            "meaning_ru": "морошковое варенье",
            "primary_gloss_ru": "морошковое варенье",
            "meaning_en": "",
            "concept_id": "",
            "category_id": "",
        },
    ])
    result = analyze_missing_en_reuse(df)

    assert len(result.pos_meanings_ru) == 1
    assert result.pos_meanings_ru.iloc[0].to_dict() == {
        "pos": "NOUN",
        "meaning_ru": "морошковое варенье",
    }

    assert result.stats["rows_missing_en_without_reuse"] == 2


def test_no_reuse_csv_schema_snapshot(tmp_path):
    """Lock the exact 7-column schema of needs_translation_no_reuse.csv."""
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "meaning_id": "10",
                "lemma_id": "100",
                "lemma": "test_lemma",
                "lang": "vep",
                "pos": "NOUN",
                "meaning_ru": "тестовое значение",
                "meaning_en": "",
                "concept_id": "",
                "category_id": "",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)
    write_reuse_outputs(result, tmp_path)

    output = pd.read_csv(
        tmp_path / "needs_translation_no_reuse.csv",
        dtype=str,
        keep_default_na=False,
    )

    expected_columns = [
        "id",
        "meaning_id",
        "lemma_id",
        "lemma",
        "lang",
        "pos",
        "meaning_ru",
    ]
    assert list(output.columns) == expected_columns


def test_pos_meanings_ru_matches_no_reuse_raw_fields(tmp_path):
    """Verify pos_meanings_ru is derived from the same no_reuse rows."""
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "meaning_id": "10",
                "lemma_id": "100",
                "lemma": "l1",
                "lang": "vep",
                "pos": "NOUN",
                "meaning_ru": "морошковое варенье",
                "primary_gloss_ru": "морошковое варенье",
                "meaning_en": "",
                "concept_id": "",
                "category_id": "",
            },
            {
                "id": "2",
                "meaning_id": "11",
                "lemma_id": "101",
                "lemma": "l2",
                "lang": "olo",
                "pos": "NOUN",
                "meaning_ru": "морошковое варенье",
                "primary_gloss_ru": "морошковое варенье",
                "meaning_en": "",
                "concept_id": "",
                "category_id": "",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)
    write_reuse_outputs(result, tmp_path)

    no_reuse = pd.read_csv(
        tmp_path / "needs_translation_no_reuse.csv", dtype=str, keep_default_na=False
    )
    pos_meanings = pd.read_csv(
        tmp_path / "pos_meanings_ru.csv", dtype=str, keep_default_na=False
    )

    expected_pairs = set(zip(no_reuse["pos"], no_reuse["meaning_ru"]))
    actual_pairs = set(zip(pos_meanings["pos"], pos_meanings["meaning_ru"]))

    assert expected_pairs == actual_pairs
    assert len(pos_meanings) == len(expected_pairs)


def test_reusable_english_files_moved_and_renamed(tmp_path):
    """Verify reusable-English files are moved to subfolder and renamed."""
    df = pd.DataFrame(
        [
            {"lang": "krl", "lemma": "l1", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "offence"},
            {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
            {"lang": "olo", "lemma": "l3", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
        ]
    )
    result = analyze_missing_en_reuse(df)
    write_reuse_outputs(result, tmp_path)

    reusable_dir = tmp_path / "reusable_english"
    assert reusable_dir.is_dir()
    assert (reusable_dir / "one_english.csv").exists()
    assert (reusable_dir / "one_english_summary.csv").exists()
    assert (reusable_dir / "several_english.csv").exists()
    assert (reusable_dir / "several_english_summary.csv").exists()

    assert not (tmp_path / "missing_en_reusable_unambiguous_pos_gloss_ru.csv").exists()
    assert not (tmp_path / "missing_en_reusable_ambiguous_pos_gloss_ru.csv").exists()
    assert not (tmp_path / "missing_en_reusable_unambiguous_pos_gloss_ru_summary.csv").exists()
    assert not (tmp_path / "missing_en_reusable_ambiguous_pos_gloss_ru_summary.csv").exists()

    assert (tmp_path / "needs_translation_no_reuse.csv").exists()
    assert (tmp_path / "pos_meanings_ru.csv").exists()


def test_reusable_english_files_created_even_when_empty(tmp_path):
    """Verify reusable-English files are created even when no reusable rows exist."""
    df = pd.DataFrame(
        [
            {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "тест", "meaning_en": ""},
        ]
    )
    result = analyze_missing_en_reuse(df)
    write_reuse_outputs(result, tmp_path)

    reusable_dir = tmp_path / "reusable_english"
    assert reusable_dir.is_dir()

    for filename in [
        "one_english.csv",
        "one_english_summary.csv",
        "several_english.csv",
        "several_english_summary.csv",
    ]:
        path = reusable_dir / filename
        assert path.exists()
        content = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert content.empty


def test_one_english_and_several_english_content_unchanged(tmp_path):
    """Verify reusable-English output content is unchanged, only location changed."""
    df = pd.DataFrame(
        [
            {"lang": "krl", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
            {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
            {"lang": "krl", "lemma": "l3", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "offence"},
            {"lang": "vep", "lemma": "l4", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
            {"lang": "olo", "lemma": "l5", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
        ]
    )
    result = analyze_missing_en_reuse(df)
    write_reuse_outputs(result, tmp_path)

    one_english = pd.read_csv(
        tmp_path / "reusable_english" / "one_english.csv", dtype=str, keep_default_na=False
    )
    several_english = pd.read_csv(
        tmp_path / "reusable_english" / "several_english.csv", dtype=str, keep_default_na=False
    )

    assert len(one_english) == len(result.missing_en_reusable_unambiguous)
    assert len(several_english) == len(result.missing_en_reusable_ambiguous)
    assert "suggested_candidate_index" in several_english.columns
    assert "suggested_candidate_index" not in one_english.columns


def test_no_reuse_empty_file_schema():
    """Verify empty no-reuse output has exact header columns."""
    from pathlib import Path
    import tempfile

    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"lang": "olo", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "house"},
    ])
    result = analyze_missing_en_reuse(df)

    with tempfile.TemporaryDirectory() as td:
        translate_dir = Path(td) / "translate"
        write_reuse_outputs(result, translate_dir)

        output_path = translate_dir / "needs_translation_no_reuse.csv"
        assert output_path.exists()

        output = pd.read_csv(output_path, dtype=str, keep_default_na=False)
        assert list(output.columns) == NO_REUSE_ROW_LEVEL_COLUMNS
        assert output.empty


def test_classification_invariant_preserved():
    """Verify rows_missing_en equals sum of unambiguous + ambiguous + no_reuse."""
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"lang": "vep", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"lang": "olo", "lemma": "l3", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"lang": "lud", "lemma": "l4", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "offence"},
        {"lang": "krl", "lemma": "l5", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": "insult"},
        {"lang": "vep", "lemma": "l6", "pos": "NOUN", "meaning_ru": "обида", "meaning_en": ""},
        {"lang": "olo", "lemma": "l7", "pos": "NOUN", "meaning_ru": "дом", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    rows_missing_en = result.stats["rows_missing_en"]
    rows_reusable_unambiguous = result.stats["rows_reusable_unambiguous"]
    rows_reusable_ambiguous = result.stats["rows_reusable_ambiguous"]
    rows_missing_en_without_reuse = result.stats["rows_missing_en_without_reuse"]

    assert rows_missing_en == (
        rows_reusable_unambiguous
        + rows_reusable_ambiguous
        + rows_missing_en_without_reuse
    )

    actual_unamb = len(result.missing_en_reusable_unambiguous)
    actual_amb = len(result.missing_en_reusable_ambiguous)
    actual_no_reuse = len(result.missing_en_without_reuse)

    assert rows_missing_en == actual_unamb + actual_amb + actual_no_reuse


def test_translatable_row_still_classified_normally():
    """A row with blank meaning_en, blank concept_id, blank category_id must still flow into existing logic."""
    df = pd.DataFrame(
        [
            {
                "id": "1", "meaning_id": "10", "lemma_id": "100", "lemma": "l1",
                "lang": "vep", "pos": "NOUN", "meaning_ru": "деревня",
                "meaning_en": "",
                "concept_id": "", "category_id": "",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)

    assert len(result.concept_category_without_english) == 0
    assert len(result.invalid_concept_category_pairs) == 0
    assert len(result.missing_en_without_reuse) == 1


def test_group_counts_use_exact_pos_and_meaning_ru_pairs():
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "meaning_id": "1",
                "lemma_id": "1",
                "lemma": "l1",
                "lang": "vep",
                "pos": "NOUN",
                "meaning_ru": "место (под чем-либо)",
                "meaning_en": "",
                "concept_id": "",
                "category_id": "",
            },
            {
                "id": "2",
                "meaning_id": "2",
                "lemma_id": "2",
                "lemma": "l2",
                "lang": "vep",
                "pos": "NOUN",
                "meaning_ru": "место (перед чем-либо)",
                "meaning_en": "",
                "concept_id": "",
                "category_id": "",
            },
        ]
    )

    result = analyze_missing_en_reuse(df)

    assert result.stats["rows_missing_en_without_reuse"] == 2
    assert result.stats["no_reuse_group_count"] == 2


def test_all_group_counts_count_exact_pairs_not_pos_tags():
    df = pd.DataFrame([
        {"lang": "vep", "lemma": "l1", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": "village"},
        {"lang": "olo", "lemma": "l2", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"lang": "krl", "lemma": "l3", "pos": "NOUN", "meaning_ru": "деревня", "meaning_en": ""},
        {"lang": "vep", "lemma": "l4", "pos": "NOUN", "meaning_ru": "дом", "meaning_en": "house"},
        {"lang": "olo", "lemma": "l5", "pos": "NOUN", "meaning_ru": "дом", "meaning_en": ""},
        {"lang": "vep", "lemma": "l6", "pos": "NOUN", "meaning_ru": "концертный зал", "meaning_en": ""},
        {"lang": "olo", "lemma": "l7", "pos": "NOUN", "meaning_ru": "концертный зал", "meaning_en": ""},
    ])
    result = analyze_missing_en_reuse(df)

    assert result.stats["unambiguous_group_count"] == 2
    assert result.stats["ambiguous_group_count"] == 0
    assert result.stats["no_reuse_group_count"] == 1


def test_concept_covered_row_fully_excluded():
    """A row with both concept_id and category_id filled must be excluded from all reuse outputs."""
    df = pd.DataFrame(
        [
            {
                "id": "1", "meaning_id": "10", "lemma_id": "100", "lemma": "l1",
                "lang": "vep", "pos": "NOUN", "meaning_ru": "стрелять",
                "meaning_en": "",
                "concept_id": "951", "category_id": "B353",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)

    assert len(result.concept_category_without_english) == 1
    assert len(result.missing_en_reusable_unambiguous) == 0
    assert len(result.missing_en_reusable_ambiguous) == 0
    assert len(result.missing_en_without_reuse) == 0
    assert "стрелять" not in result.pos_meanings_ru["meaning_ru"].tolist()


def test_invalid_pair_row_fully_excluded():
    """A row with exactly one of concept_id/category_id filled must be excluded and audited."""
    df = pd.DataFrame(
        [
            {
                "id": "1", "meaning_id": "10", "lemma_id": "100", "lemma": "l1",
                "lang": "vep", "pos": "NOUN", "meaning_ru": "берег",
                "meaning_en": "",
                "concept_id": "1293", "category_id": "",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)

    assert len(result.invalid_concept_category_pairs) == 1
    assert len(result.concept_category_without_english) == 0
    assert len(result.missing_en_reusable_unambiguous) == 0
    assert len(result.missing_en_reusable_ambiguous) == 0
    assert len(result.missing_en_without_reuse) == 0
    assert "берег" not in result.pos_meanings_ru["meaning_ru"].tolist()


def test_invalid_pair_reverse_case():
    """Test reverse case: category_id filled, concept_id blank."""
    df = pd.DataFrame(
        [
            {
                "id": "1", "meaning_id": "10", "lemma_id": "100", "lemma": "l1",
                "lang": "vep", "pos": "NOUN", "meaning_ru": "берег",
                "meaning_en": "",
                "concept_id": "", "category_id": "B353",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)

    assert len(result.invalid_concept_category_pairs) == 1
    assert len(result.concept_category_without_english) == 0
    assert len(result.missing_en_reusable_unambiguous) == 0
    assert len(result.missing_en_reusable_ambiguous) == 0
    assert len(result.missing_en_without_reuse) == 0


def test_concept_covered_does_not_act_as_donor():
    """A concept-covered row should not act as a reuse donor for other rows in its group."""
    df = pd.DataFrame(
        [
            {
                "id": "1", "meaning_id": "10", "lemma_id": "100", "lemma": "l1",
                "lang": "vep", "pos": "NOUN", "meaning_ru": "тест",
                "meaning_en": "",
                "concept_id": "1", "category_id": "A1",
            },
            {
                "id": "2", "meaning_id": "11", "lemma_id": "101", "lemma": "l2",
                "lang": "olo", "pos": "NOUN", "meaning_ru": "тест",
                "meaning_en": "",
                "concept_id": "", "category_id": "",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)

    assert len(result.concept_category_without_english) == 1
    assert len(result.missing_en_without_reuse) == 1
    assert result.missing_en_without_reuse.iloc[0]["meaning_id"] == "11"


def test_rows_with_existing_meaning_en_not_checked():
    """Rows with non-empty meaning_en should never be evaluated for concept/category consistency."""
    df = pd.DataFrame(
        [
            {
                "id": "1", "meaning_id": "10", "lemma_id": "100", "lemma": "l1",
                "lang": "vep", "pos": "NOUN", "meaning_ru": "дом",
                "meaning_en": "house",
                "concept_id": "1", "category_id": "",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)

    assert len(result.invalid_concept_category_pairs) == 0
    assert len(result.concept_category_without_english) == 0


def test_concept_category_audit_files_schema_and_empty(tmp_path):
    """Test schema and empty-file behavior for the two new audit files."""
    df = pd.DataFrame(
        [
            {
                "id": "1", "meaning_id": "10", "lemma_id": "100", "lemma": "l1",
                "lang": "vep", "pos": "NOUN", "meaning_ru": "тест",
                "meaning_en": "",
                "concept_id": "", "category_id": "",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)
    write_reuse_outputs(result, tmp_path)

    for filename in [
        "concept_category_without_english.csv",
        "invalid_concept_category_pairs.csv",
    ]:
        path = tmp_path / filename
        assert path.exists()
        content = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(content.columns) == [
            "id", "meaning_id", "lemma_id", "lemma", "lang",
            "pos", "meaning_ru", "concept_id", "category_id",
        ]
        assert content.empty


def test_five_term_classification_invariant():
    """Test five-term classification invariant including concept-covered and invalid-pair rows."""
    df = pd.DataFrame(
        [
            {"id": "1", "meaning_id": "1", "lemma_id": "1", "lemma": "l1", "lang": "vep", "pos": "NOUN", "meaning_ru": "а", "meaning_en": "village", "concept_id": "", "category_id": ""},
            {"id": "2", "meaning_id": "2", "lemma_id": "2", "lemma": "l2", "lang": "olo", "pos": "NOUN", "meaning_ru": "а", "meaning_en": "", "concept_id": "", "category_id": ""},
            {"id": "3", "meaning_id": "3", "lemma_id": "3", "lemma": "l3", "lang": "krl", "pos": "NOUN", "meaning_ru": "б", "meaning_en": "offence", "concept_id": "", "category_id": ""},
            {"id": "4", "meaning_id": "4", "lemma_id": "4", "lemma": "l4", "lang": "vep", "pos": "NOUN", "meaning_ru": "б", "meaning_en": "insult", "concept_id": "", "category_id": ""},
            {"id": "5", "meaning_id": "5", "lemma_id": "5", "lemma": "l5", "lang": "olo", "pos": "NOUN", "meaning_ru": "б", "meaning_en": "", "concept_id": "", "category_id": ""},
            {"id": "6", "meaning_id": "6", "lemma_id": "6", "lemma": "l6", "lang": "vep", "pos": "VERB", "meaning_ru": "стрелять", "meaning_en": "", "concept_id": "951", "category_id": "B353"},
            {"id": "7", "meaning_id": "7", "lemma_id": "7", "lemma": "l7", "lang": "krl", "pos": "NOUN", "meaning_ru": "берег", "meaning_en": "", "concept_id": "1293", "category_id": ""},
        ]
    )
    result = analyze_missing_en_reuse(df)
    stats = result.stats

    assert stats["rows_missing_en"] == (
        stats["rows_reusable_unambiguous"]
        + stats["rows_reusable_ambiguous"]
        + stats["rows_missing_en_without_reuse"]
        + stats["rows_concept_covered_skip"]
        + stats["rows_invalid_concept_category_pair"]
    )
    assert stats["rows_concept_covered_skip"] == 1
    assert stats["rows_invalid_concept_category_pair"] == 1


def test_cli_smoke_test():
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        """Run step 01 on a tiny temp data setup and verify the output files are created."""
        from src.sem_cat.utils.vepkar_loader import load_meanings
        from src.sem_cat.pipeline.meaning_preparation import prepare_meanings_for_reuse_and_translation
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
        work = prepare_meanings_for_reuse_and_translation(df_meanings)
        result = analyze_missing_en_reuse(work)
        write_reuse_outputs(result, translate_dir)

        # Verify all 8 output files exist
        assert (translate_dir / "needs_translation_no_reuse.csv").exists()
        assert (translate_dir / "pos_meanings_ru.csv").exists()
        assert (translate_dir / "concept_category_without_english.csv").exists()
        assert (translate_dir / "invalid_concept_category_pairs.csv").exists()
        reusable_dir = translate_dir / "reusable_english"
        assert reusable_dir.exists()
        assert (reusable_dir / "one_english.csv").exists()
        assert (reusable_dir / "one_english_summary.csv").exists()
        assert (reusable_dir / "several_english.csv").exists()
        assert (reusable_dir / "several_english_summary.csv").exists()

        # Verify old file names do not exist at root
        assert not (translate_dir / "missing_en_reusable_unambiguous_pos_gloss_ru.csv").exists()
        assert not (translate_dir / "missing_en_reusable_ambiguous_pos_gloss_ru.csv").exists()
        assert not (translate_dir / "missing_en_reusable_unambiguous_pos_gloss_ru_summary.csv").exists()
        assert not (translate_dir / "missing_en_reusable_ambiguous_pos_gloss_ru_summary.csv").exists()

        # Verify unambiguous output has the missing rows (4 languages * 1 missing row each)
        unamb_df = pd.read_csv(reusable_dir / "one_english.csv")
        assert len(unamb_df) == 4  # 4 languages, each has 1 missing row for деревня
        # The meaning_ru is "деревня деревня" from the fixture
        assert "деревня деревня" in unamb_df["meaning_ru"].values
        assert unamb_df.iloc[0]["existing_en_candidates"] == "village"


# ---------------------------------------------------------------------------
# Schema snapshot tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_is_nonblank_text,
        test_normalize_existing_en,
        test_distinct_existing_en_candidates_unambiguous,
        test_distinct_existing_en_candidates_ambiguous,
        test_distinct_existing_en_candidates_ignores_blank,
        test_distinct_existing_en_candidates_sorted_alphabetically,
        test_distinct_existing_en_candidates_normalizes_whitespace,
        test_distinct_existing_en_candidates_does_not_split_semicolon,
        test_grouper_different_pos_same_gloss_are_separate,
        test_grouper_same_pos_different_full_meanings_are_separate,
        test_unambiguous_reuse_one_candidate,
        test_unambiguous_reuse_all_missing_no_candidates,
        test_ambiguous_reuse_two_candidates,
        test_ambiguous_reuse_three_candidates,
        test_no_reuse_missing_all_have_no_candidates,
        test_whitespace_normalization_counts_as_same_candidate,
        test_semicolon_not_split_into_multiple_candidates,
        test_unambiguous_summary_one_row_per_group,
        test_ambiguous_summary_one_row_per_group,
        test_ambiguous_row_level_output_excludes_summary_placeholder_columns,
        test_row_level_csv_schema_is_exact,
        test_summary_csv_still_contains_language_and_lemma_fields,
        test_empty_dataframe_creates_valid_outputs,
        test_writer_creates_csvs_with_headers_even_when_empty,
        test_full_fixture_all_cases,
        test_cli_smoke_test,
        test_missing_langs_and_existing_en_langs_separated,
        test_per_language_stats,
        test_unambiguous_summary_has_no_suggested_candidate_index,
        test_ambiguous_summary_has_suggested_candidate_index,
        test_unambiguous_rows_have_no_suggested_candidate_index,
        test_ambiguous_rows_have_suggested_candidate_index,
        test_row_level_outputs_preserve_identifiers,
        test_row_level_outputs_do_not_leak_internal_fields,
        test_no_reuse_output_has_correct_schema,
        test_no_reuse_empty_output_creates_csv_with_headers,
        test_no_reuse_meaningful_group_size_three_rows,
        test_no_reuse_csv_schema_and_content,
        test_no_reuse_empty_file_schema,
        test_classification_invariant_preserved,
        test_no_reuse_csv_schema_snapshot,
        test_pos_meanings_ru_matches_no_reuse_raw_fields,
        test_pos_meanings_ru_deduplicate_identical_pairs,
        test_pos_meanings_ru_preserve_distinct_full_meanings,
        test_pos_meanings_ru_exact_schema,
        test_pos_meanings_ru_empty_output,
        test_pos_meanings_ru_blank_pos_warning_preservation,
        test_pos_meanings_ru_conservation_invariant,
        test_reusable_english_files_moved_and_renamed,
        test_reusable_english_files_created_even_when_empty,
        test_one_english_and_several_english_content_unchanged,
        test_row_level_schemas_exclude_task_pos_and_concept_category,
        test_audit_files_still_have_concept_and_category,
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


def test_row_level_schemas_exclude_task_pos_and_concept_category(tmp_path):
    df = pd.DataFrame(
        [
            {
                "id": "1", "meaning_id": "1", "lemma_id": "1", "lemma": "l1",
                "lang": "vep", "pos": "NOUN", "meaning_ru": "тест1",
                "meaning_en": "", "concept_id": "", "category_id": "",
            },
            {
                "id": "2", "meaning_id": "2", "lemma_id": "2", "lemma": "l2",
                "lang": "olo", "pos": "NOUN", "meaning_ru": "тест2",
                "meaning_en": "village", "concept_id": "", "category_id": "",
            },
            {
                "id": "3", "meaning_id": "3", "lemma_id": "3", "lemma": "l3",
                "lang": "krl", "pos": "NOUN", "meaning_ru": "тест2",
                "meaning_en": "", "concept_id": "", "category_id": "",
            },
            {
                "id": "4", "meaning_id": "4", "lemma_id": "4", "lemma": "l4",
                "lang": "vep", "pos": "NOUN", "meaning_ru": "тест2",
                "meaning_en": "hamlet", "concept_id": "", "category_id": "",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)
    write_reuse_outputs(result, tmp_path)

    no_reuse = pd.read_csv(
        tmp_path / "needs_translation_no_reuse.csv", dtype=str, keep_default_na=False
    )
    one_english = pd.read_csv(
        tmp_path / "reusable_english" / "one_english.csv", dtype=str, keep_default_na=False
    )
    several_english = pd.read_csv(
        tmp_path / "reusable_english" / "several_english.csv", dtype=str, keep_default_na=False
    )

    for output in (no_reuse, one_english, several_english):
        assert "task_pos" not in output.columns
        assert "concept_id" not in output.columns
        assert "category_id" not in output.columns
        assert "pos" in output.columns
        assert "meaning_ru" in output.columns

    assert list(no_reuse.columns) == [
        "id", "meaning_id", "lemma_id", "lemma", "lang", "pos", "meaning_ru",
    ]


def test_audit_files_still_have_concept_and_category(tmp_path):
    df = pd.DataFrame(
        [
            {
                "id": "1", "meaning_id": "1", "lemma_id": "1", "lemma": "l1",
                "lang": "vep", "pos": "VERB", "meaning_ru": "стрелять",
                "meaning_en": "", "concept_id": "951", "category_id": "B353",
            },
            {
                "id": "2", "meaning_id": "2", "lemma_id": "2", "lemma": "l2",
                "lang": "krl", "pos": "NOUN", "meaning_ru": "берег",
                "meaning_en": "", "concept_id": "1293", "category_id": "",
            },
        ]
    )
    result = analyze_missing_en_reuse(df)
    write_reuse_outputs(result, tmp_path)

    concept_covered = pd.read_csv(
        tmp_path / "concept_category_without_english.csv", dtype=str, keep_default_na=False
    )
    invalid_pair = pd.read_csv(
        tmp_path / "invalid_concept_category_pairs.csv", dtype=str, keep_default_na=False
    )

    assert list(concept_covered.columns) == [
        "id", "meaning_id", "lemma_id", "lemma", "lang",
        "pos", "meaning_ru", "concept_id", "category_id",
    ]
    assert list(invalid_pair.columns) == [
        "id", "meaning_id", "lemma_id", "lemma", "lang",
        "pos", "meaning_ru", "concept_id", "category_id",
    ]
    assert len(concept_covered) == 1
    assert len(invalid_pair) == 1
