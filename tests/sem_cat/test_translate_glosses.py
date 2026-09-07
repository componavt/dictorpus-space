"""Tests for Step-02 translation workflow with pos_meaning_ru task file."""

import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.sem_cat.io.translation_rows import (
    build_translation_row,
    CANONICAL_COLUMNS,
)
from src.sem_cat.io.pos_meaning_ru_reader import read_pos_meaning_ru_tasks
import pytest

from src.sem_cat.io.translation_cache import (
    load_translation_cache,
    CANONICAL_COLUMNS,
    build_cached_identity_set,
)
from src.sem_cat.qa.translation_qa import (
    analyze_translation,
    QAResult,
    TranslationQAConfig,
)
from src.sem_cat.pipeline.vepkar_translation_selection import (
    TranslationTaskMetadata,
    build_translation_tasks_from_pos_meaning_ru,
)


def test_new_translation_row_schema_columns():
    """New translation row should have exact canonical columns in required order."""
    qa_result = QAResult(qa_keep=True, qa_score=0.0)
    row = build_translation_row(
        pos="NOUN",
        meaning_ru="дом",
        meaning_en="house",
        qa_result=qa_result,
    )
    expected_columns = [
        "pos", "meaning_ru", "meaning_en",
        "qa_keep", "qa_score", "qa_flags",
        "meaning_ru_back", "roundtrip_distance",
    ]
    assert list(row.keys()) == expected_columns


def test_new_translation_row_has_required_identity_fields():
    """New translation row must have pos and meaning_ru as task identity."""
    qa_result = QAResult(qa_keep=True, qa_score=0.0)
    row = build_translation_row(
        pos="NOUN",
        meaning_ru="дом",
        meaning_en="house",
        qa_result=qa_result,
    )
    assert row["pos"] == "NOUN"
    assert row["meaning_ru"] == "дом"
    assert row["meaning_en"] == "house"


def test_new_translation_row_no_legacy_fields():
    """New translation row must NOT contain legacy fields."""
    qa_result = QAResult(qa_keep=True, qa_score=0.0)
    row = build_translation_row(
        pos="NOUN",
        meaning_ru="дом",
        meaning_en="house",
        qa_result=qa_result,
    )
    
    legacy_fields = ["model_key", "model_name", "backend_family", "translation_input_mode",
                     "input_text_used", "qa_version", "is_single_word_ru", 
                     "input_token_count", "output_token_count", "task_key", "task_pos", 
                     "gloss_ru", "gloss_en", "gloss_ru_back", "primary_gloss_ru", 
                     "pos_hint", "meaning_hint", "sourcecount"]
    for field in legacy_fields:
        assert field not in row, f"Legacy field {field} should not be present"


def test_new_translation_row_token_metadata():
    """Token counts are no longer included in the compact schema."""
    qa_result = QAResult(qa_keep=True, qa_score=0.0)
    row = build_translation_row(
        pos="PART",
        meaning_ru="а",
        meaning_en="a",
        qa_result=qa_result,
    )
    assert "input_token_count" not in row
    assert "output_token_count" not in row
    assert "is_single_word_ru" not in row


def test_new_translation_row_roundtrip_qa():
    """Round-trip QA fields should use meaning_ru_back and roundtrip_distance."""
    qa_result = QAResult(qa_keep=True, qa_score=0.1, roundtrip_distance=0.2)
    row = build_translation_row(
        pos="NOUN",
        meaning_ru="дом",
        meaning_en="house",
        qa_result=qa_result,
        meaning_ru_back="дом",
        roundtrip_distance=0.2,
    )
    assert row["meaning_ru_back"] == "дом"
    assert row["roundtrip_distance"] == 0.2


def test_new_translation_row_blank_meaning_en():
    """Blank meaning_en should be allowed without corrupting schema."""
    qa_result = QAResult(qa_keep=False, qa_score=1.0)
    row = build_translation_row(
        pos="NOUN",
        meaning_ru="дом",
        meaning_en="",
        qa_result=qa_result,
    )
    assert row["meaning_en"] == ""
    assert row["qa_keep"] is False


def test_cache_loads_exact_schema(tmp_path):
    """Cache with exact canonical schema should load successfully."""
    df = pd.DataFrame([
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "house", 
         "qa_keep": "True", "qa_score": "0.0", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
        {"pos": "VERB", "meaning_ru": "читать", "meaning_en": "read",
         "qa_keep": "True", "qa_score": "0.1", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
    ])
    path = tmp_path / "cache.csv"
    df.to_csv(path, index=False)
    
    result = load_translation_cache(path, expected_model_key="google")
    
    assert result.state == "valid"
    assert "pos" in result.df.columns
    assert "meaning_ru" in result.df.columns
    assert "meaning_en" in result.df.columns


def test_cache_rejects_extra_columns(tmp_path):
    """Cache with extra columns should be rejected."""
    df = pd.DataFrame([
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "house", 
         "qa_keep": "True", "qa_score": "0.0", "qa_flags": "", 
         "meaning_ru_back": "", "roundtrip_distance": "", "extra_col": "x"},
    ])
    path = tmp_path / "extra_cache.csv"
    df.to_csv(path, index=False)
    
    result = load_translation_cache(path, expected_model_key="google")
    
    assert result.state == "malformed"
    assert "extra" in result.reason.lower() or "canonical" in result.reason.lower()


def test_cache_identity_is_pos_meaning_ru_pair(tmp_path):
    """Cache identity must be exact (pos, meaning_ru) pair."""
    df = pd.DataFrame([
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "house", 
         "qa_keep": "True", "qa_score": "0.0", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
        {"pos": "VERB", "meaning_ru": "дом", "meaning_en": "to house",
         "qa_keep": "True", "qa_score": "0.1", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
        {"pos": "NOUN", "meaning_ru": "дома", "meaning_en": "houses",
         "qa_keep": "True", "qa_score": "0.2", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
    ])
    path = tmp_path / "cache.csv"
    df.to_csv(path, index=False)
    
    result = load_translation_cache(path, expected_model_key="google")
    assert result.state == "valid"
    
    cached_set = build_cached_identity_set(result.df)
    assert ("NOUN", "дом") in cached_set
    assert ("VERB", "дом") in cached_set
    assert ("NOUN", "дома") in cached_set
    assert len(cached_set) == 3


def test_cache_deduplicates_by_pos_meaning_ru(tmp_path):
    """Duplicate (pos, meaning_ru) rows should be deduplicated keeping best qa_keep then lowest qa_score."""
    df = pd.DataFrame([
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "house", 
         "qa_keep": "True", "qa_score": "0.0", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "home",
         "qa_keep": "False", "qa_score": "0.1", "qa_flags": "suspicious", "meaning_ru_back": "", "roundtrip_distance": ""},
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "residence",
         "qa_keep": "True", "qa_score": "0.2", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
    ])
    path = tmp_path / "cache_with_dupes.csv"
    df.to_csv(path, index=False)
    
    result = load_translation_cache(path, expected_model_key="google")
    assert result.state == "valid"
    
    cached_set = build_cached_identity_set(result.df)
    assert len(cached_set) == 1


def test_cache_keeps_qa_keep_true_over_false(tmp_path):
    """When duplicates exist, qa_keep=True is preferred over qa_keep=False."""
    df = pd.DataFrame([
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "house", 
         "qa_keep": "False", "qa_score": "0.0", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "home",
         "qa_keep": "True", "qa_score": "1.0", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
    ])
    path = tmp_path / "cache_with_dupes.csv"
    df.to_csv(path, index=False)
    
    result = load_translation_cache(path, expected_model_key="google")
    assert result.state == "valid"
    
    cached_df = result.df
    assert len(cached_df) == 1
    assert cached_df.iloc[0]["meaning_en"] == "home"
    assert cached_df.iloc[0]["qa_keep"] == "True"


def test_cache_keeps_lowest_qa_score_when_equal_keep(tmp_path):
    """Among qa_keep=True rows, lowest qa_score is preferred."""
    df = pd.DataFrame([
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "house", 
         "qa_keep": "True", "qa_score": "0.5", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "home",
         "qa_keep": "True", "qa_score": "0.1", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
    ])
    path = tmp_path / "cache_with_dupes.csv"
    df.to_csv(path, index=False)
    
    result = load_translation_cache(path, expected_model_key="google")
    assert result.state == "valid"
    
    cached_df = result.df
    assert len(cached_df) == 1
    assert cached_df.iloc[0]["meaning_en"] == "home"
    assert cached_df.iloc[0]["qa_score"] == "0.1"


def test_builder_from_pos_meaning_ru():
    """build_translation_tasks_from_pos_meaning_ru should produce tasks with (pos, meaning_ru)."""
    df = pd.DataFrame({
        "pos": ["NOUN", "VERB", "PART"],
        "meaning_ru": ["дом", "читать", "а"],
    })
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    assert len(tasks) == 3
    assert tasks[0].pos == "NOUN"
    assert tasks[0].meaning_ru == "дом"
    assert tasks[1].pos == "VERB"
    assert tasks[1].meaning_ru == "читать"
    assert tasks[2].pos == "PART"
    assert tasks[2].meaning_ru == "а"
    
    for task in tasks:
        assert not hasattr(task, "task_key")
        assert not hasattr(task, "primary_gloss_ru")
        assert not hasattr(task, "meaning_hint")
        assert not hasattr(task, "sourcecount")


def test_qa_receives_meaning_ru_not_gloss():
    """QA analyze_translation should receive complete meaning_ru."""
    result = analyze_translation(
        "морошковое варенье",
        "juniper jam",
        "морошковое варенье",
        config=TranslationQAConfig()
    )
    assert isinstance(result, QAResult)


def test_reader_accepts_task_file_format():
    """pos_meaning_ru_reader should accept pos,meaning_ru CSV format."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\nPART,а\nNOUN,морошковое варенье\n")
        
        df = read_pos_meaning_ru_tasks(path)
        
        assert list(df.columns) == ["pos", "meaning_ru"]
        assert len(df) == 2
        assert df.iloc[0]["pos"] == "PART"
        assert df.iloc[0]["meaning_ru"] == "а"
        assert df.iloc[1]["pos"] == "NOUN"
        assert df.iloc[1]["meaning_ru"] == "морошковое варенье"


def test_reader_rejects_legacy_columns():
    """pos_meaning_ru_reader should reject files with task_key or gloss_ru columns."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "legacy.csv"
        path.write_text("pos,meaning_ru,task_key\nNOUN,дом,NOUN::дом\n")
        
        try:
            read_pos_meaning_ru_tasks(path)
            assert False, "Expected ValueError for legacy columns"
        except ValueError as e:
            assert "invalid columns" in str(e).lower() or "expected" in str(e).lower()


def test_reader_rejects_blank_pos():
    """pos_meaning_ru_reader should reject blank pos values."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "blank_pos.csv"
        path.write_text("pos,meaning_ru\n,NOUN\n")
        
        try:
            read_pos_meaning_ru_tasks(path)
            assert False, "Expected ValueError for blank pos"
        except ValueError as e:
            assert "blank pos" in str(e).lower()


def test_reader_rejects_blank_meaning_ru():
    """pos_meaning_ru_reader should reject blank meaning_ru values."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "blank_meaning.csv"
        path.write_text("pos,meaning_ru\nNOUN,\n")
        
        try:
            read_pos_meaning_ru_tasks(path)
            assert False, "Expected ValueError for blank meaning_ru"
        except ValueError as e:
            assert "blank meaning_ru" in str(e).lower()


def test_reader_rejects_duplicate_pairs():
    """pos_meaning_ru_reader should reject duplicate (pos, meaning_ru) pairs."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "dupes.csv"
        path.write_text("pos,meaning_ru\nNOUN,дом\nNOUN,дом\n")
        
        try:
            read_pos_meaning_ru_tasks(path)
            assert False, "Expected ValueError for duplicates"
        except ValueError as e:
            assert "duplicate" in str(e).lower()
            assert "pos=" in str(e)
            assert "meaning_ru=" in str(e)


def test_empty_task_file_exits_gracefully():
    """Empty task file (header only) should be accepted and produce no tasks."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "empty.csv"
        path.write_text("pos,meaning_ru\n")
        
        df = read_pos_meaning_ru_tasks(path)
        
        assert list(df.columns) == ["pos", "meaning_ru"]
        assert len(df) == 0
        
        tasks = build_translation_tasks_from_pos_meaning_ru(df)
        assert tasks == []


if __name__ == "__main__":
    tests = [
        test_new_translation_row_schema_columns,
        test_new_translation_row_has_required_identity_fields,
        test_new_translation_row_no_legacy_fields,
        test_new_translation_row_token_metadata,
        test_new_translation_row_roundtrip_qa,
        test_new_translation_row_blank_meaning_en,
        test_cache_loads_new_schema,
        test_cache_rejects_legacy_schema,
        test_cache_identity_is_pos_meaning_ru_pair,
        test_cache_deduplicates_by_pos_meaning_ru,
        test_builder_from_pos_meaning_ru,
        test_qa_receives_meaning_ru_not_gloss,
        test_reader_accepts_task_file_format,
        test_reader_rejects_legacy_columns,
        test_reader_rejects_blank_pos,
        test_reader_rejects_blank_meaning_ru,
        test_reader_rejects_duplicate_pairs,
        test_empty_task_file_exits_gracefully,
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
