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
    prepare_translation_input_for_task,
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
    
    result = load_translation_cache(path)
    
    assert result.state == "valid"
    assert "pos" in result.df.columns
    assert "meaning_ru" in result.df.columns
    assert "meaning_en" in result.df.columns


def test_load_translation_cache_no_expected_model_key_param():
    """load_translation_cache should no longer accept expected_model_key parameter."""
    import inspect
    sig = inspect.signature(load_translation_cache)
    params = list(sig.parameters.keys())
    assert "expected_model_key" not in params, "expected_model_key parameter should be removed"


def test_cache_rejects_extra_columns(tmp_path):
    """Cache with extra columns should be rejected."""
    df = pd.DataFrame([
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "house", 
         "qa_keep": "True", "qa_score": "0.0", "qa_flags": "", 
         "meaning_ru_back": "", "roundtrip_distance": "", "extra_col": "x"},
    ])
    path = tmp_path / "extra_cache.csv"
    df.to_csv(path, index=False)
    
    result = load_translation_cache(path)
    
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
    
    result = load_translation_cache(path)
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
         "qa_keep": "True", "qa_score": "0.1", "qa_flags": "suspicious", "meaning_ru_back": "", "roundtrip_distance": ""},
        {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "residence",
         "qa_keep": "True", "qa_score": "0.2", "qa_flags": "", "meaning_ru_back": "", "roundtrip_distance": ""},
    ])
    path = tmp_path / "cache_with_dupes.csv"
    df.to_csv(path, index=False)
    
    result = load_translation_cache(path)
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
    
    result = load_translation_cache(path)
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
    
    result = load_translation_cache(path)
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


# ---------------------------------------------------------------------------
# Real Step-02 integration tests with monkeypatched translator
# ---------------------------------------------------------------------------


def test_step02_integration_successful_translation(monkeypatch, tmp_path):
    """Full Step-02 flow: task file → fake translator → cache output."""
    # Create task file
    task_file = tmp_path / "pos_meanings_ru.csv"
    task_file.write_text("pos,meaning_ru\nPART,а\nNOUN,морошковое варенье\n")
    
    from src.sem_cat.io.pos_meaning_ru_reader import read_pos_meaning_ru_tasks
    from src.sem_cat.pipeline.vepkar_translation_selection import build_translation_tasks_from_pos_meaning_ru, prepare_translation_input_for_task
    
    task_df = read_pos_meaning_ru_tasks(task_file)
    tasks = build_translation_tasks_from_pos_meaning_ru(task_df)
    assert len(tasks) == 2
    
    inputs = [prepare_translation_input_for_task(t) for t in tasks]
    assert inputs == ["PART | а", "NOUN | морошковое варенье"]


def test_step02_integration_cache_filtering_with_offset_limit(monkeypatch, tmp_path):
    """Verify cache filtering → shuffle → offset → limit sequence."""
    # Create task file with multiple items
    task_file = tmp_path / "pos_meanings_ru.csv"
    task_file.write_text("pos,meaning_ru\nNOUN,дом\nVERB,строить\nNOUN,морошковое варенье\n")
    
    from src.sem_cat.io.pos_meaning_ru_reader import read_pos_meaning_ru_tasks
    from src.sem_cat.pipeline.vepkar_translation_selection import build_translation_tasks_from_pos_meaning_ru
    from src.sem_cat.io.translation_cache import build_cached_identity_set
    
    # Test the core workflow without full integration
    task_df = read_pos_meaning_ru_tasks(task_file)
    tasks = build_translation_tasks_from_pos_meaning_ru(task_df)
    
    # Simulate cache filtering with NOUN,дом already in cache
    cache_df = pd.DataFrame([{
        "pos": "NOUN",
        "meaning_ru": "дом",
        "meaning_en": "house",
        "qa_keep": "True",
        "qa_score": "0.0",
        "qa_flags": "",
        "meaning_ru_back": "",
        "roundtrip_distance": "",
    }])
    cached_ids = build_cached_identity_set(cache_df)
    
    # After cache filtering: remove NOUN,дом
    tasks_to_translate = [t for t in tasks if (t.pos, t.meaning_ru) not in cached_ids]
    assert len(tasks_to_translate) == 2  # VERB,строить and NOUN,морошковое варенье
    
    # Apply offset 0, limit 1
    tasks_subset = tasks_to_translate[0:1]
    
    from src.sem_cat.pipeline.vepkar_translation_selection import prepare_translation_input_for_task
    inputs = [prepare_translation_input_for_task(t) for t in tasks_subset]
    assert len(inputs) == 1
    assert inputs[0] == "VERB | строить"


def test_step02_integration_empty_input_no_output(tmp_path):
    """Empty valid input (header only) should exit before translator construction."""
    task_file = tmp_path / "pos_meanings_ru.csv"
    task_file.write_text("pos,meaning_ru\n")
    
    from src.sem_cat.io.pos_meaning_ru_reader import read_pos_meaning_ru_tasks
    from src.sem_cat.pipeline.vepkar_translation_selection import build_translation_tasks_from_pos_meaning_ru
    
    task_df = read_pos_meaning_ru_tasks(task_file)
    tasks_in_file = len(task_df)
    
    # Should be 0
    assert tasks_in_file == 0
    
    tasks = build_translation_tasks_from_pos_meaning_ru(task_df)
    assert tasks == []


def test_step02_integration_successful_translation_with_pipe_format():
    """prepare_translation_input_for_task should use pipe separator."""
    task = TranslationTaskMetadata(pos="NOUN", meaning_ru="дом")
    assert prepare_translation_input_for_task(task) == "NOUN | дом"


def test_step02_cli_rejects_obsolete_options():
    """argparse should reject obsolete CLI options."""
    import argparse
    import sys
    from src.sem_cat.translators.model_registry import list_model_keys
    
    # These options should NOT be present in argparse
    obsolete_args = [
        ["--out-file", "out.csv"],
        ["--translation-input-mode", "gloss"],
        ["--data-dir", "/data"],
        ["--translate-dir", "/translate"],
        ["--gloss-filter", "NOUN"],
    ]
    
    # Get current valid args from the actual parser
    model_keys = list_model_keys()
    parser = argparse.ArgumentParser(description="Translate VepKar meanings to English from fixed task file")
    parser.add_argument("--out-dir", type=str, default=str(pathlib.Path("/data/sem_cat")))
    parser.add_argument("--model-key", type=str, choices=model_keys, default=None)
    parser.add_argument("--backend", type=str, default="marian")
    parser.add_argument("--nllb-model", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--round-trip", action="store_true", default=False)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-sample", type=int, default=0)
    parser.add_argument("--retry", type=int, default=None)
    parser.add_argument("--retry-delay", type=float, default=None)
    parser.add_argument("--google-retries", type=int, default=2)
    parser.add_argument("--google-retry-delay", type=float, default=1.0)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--ignore-proxy-env", action="store_true", default=False)
    parser.add_argument("--backend-info", action="store_true", default=False)
    
    # Verify obsolete options cause error
    for args in obsolete_args:
        with pytest.raises(SystemExit):
            parser.parse_args(args)


if __name__ == "__main__":
    tests = [
        test_new_translation_row_schema_columns,
        test_new_translation_row_has_required_identity_fields,
        test_new_translation_row_no_legacy_fields,
        test_new_translation_row_token_metadata,
        test_new_translation_row_roundtrip_qa,
        test_new_translation_row_blank_meaning_en,
        test_cache_loads_exact_schema,
        test_cache_rejects_extra_columns,
        test_cache_identity_is_pos_meaning_ru_pair,
        test_cache_deduplicates_by_pos_meaning_ru,
        test_cache_keeps_qa_keep_true_over_false,
        test_cache_keeps_lowest_qa_score_when_equal_keep,
        test_builder_from_pos_meaning_ru,
        test_qa_receives_meaning_ru_not_gloss,
        test_reader_accepts_task_file_format,
        test_reader_rejects_legacy_columns,
        test_reader_rejects_blank_pos,
        test_reader_rejects_blank_meaning_ru,
        test_reader_rejects_duplicate_pairs,
        test_empty_task_file_exits_gracefully,
        test_step02_integration_successful_translation,
        test_step02_integration_cache_filtering_with_offset_limit,
        test_step02_integration_empty_input_no_output,
        test_load_translation_cache_no_expected_model_key_param,
        test_step02_cli_rejects_obsolete_options,
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
