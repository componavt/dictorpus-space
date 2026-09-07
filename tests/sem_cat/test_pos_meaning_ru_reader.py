"""Tests for pos_meaning_ru_reader module."""

import sys
import pathlib
import tempfile
import csv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import pytest

from src.sem_cat.io.pos_meaning_ru_reader import (
    read_pos_meaning_ru_tasks,
    POS_MEANINGS_RU_COLUMNS,
)
from src.sem_cat.pipeline.vepkar_translation_selection import (
    build_translation_tasks_from_pos_meaning_ru,
)


def test_valid_nonempty_file():
    """Valid non-empty file is read correctly with preserved order."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\nPART,а\nNOUN,морошковое варенье\nNOUN,\"место (под чем-либо)\"\n")
        
        df = read_pos_meaning_ru_tasks(path)
        
        assert list(df.columns) == ["pos", "meaning_ru"]
        assert len(df) == 3
        assert df.iloc[0]["pos"] == "PART"
        assert df.iloc[0]["meaning_ru"] == "а"
        assert df.iloc[1]["pos"] == "NOUN"
        assert df.iloc[1]["meaning_ru"] == "морошковое варенье"
        assert df.iloc[2]["pos"] == "NOUN"
        assert df.iloc[2]["meaning_ru"] == "место (под чем-либо)"


def test_valid_empty_file():
    """Valid empty file (only header) is accepted."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\n")
        
        df = read_pos_meaning_ru_tasks(path)
        
        assert list(df.columns) == ["pos", "meaning_ru"]
        assert len(df) == 0


def test_file_not_found():
    """Missing file raises ValueError with clear message."""
    path = pathlib.Path("/nonexistent/path/to/file.csv")
    
    with pytest.raises(ValueError) as exc_info:
        read_pos_meaning_ru_tasks(path)
    
    assert "Translation task file does not exist:" in str(exc_info.value)
    assert str(path) in str(exc_info.value)


def test_missing_pos_column():
    """Missing pos column raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("meaning_ru\ndом\n")
        
        with pytest.raises(ValueError) as exc_info:
            read_pos_meaning_ru_tasks(path)
        
        assert "Translation task file has invalid columns" in str(exc_info.value)


def test_missing_meaning_ru_column():
    """Missing meaning_ru column raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos\nNOUN\n")
        
        with pytest.raises(ValueError) as exc_info:
            read_pos_meaning_ru_tasks(path)
        
        assert "Translation task file has invalid columns" in str(exc_info.value)


def test_extra_column():
    """Extra column raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru,extra\nNOUN,дом,x\n")
        
        with pytest.raises(ValueError) as exc_info:
            read_pos_meaning_ru_tasks(path)
        
        assert "Translation task file has invalid columns" in str(exc_info.value)


def test_wrong_column_order():
    """Wrong column order raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("meaning_ru,pos\ndом,NOUN\n")
        
        with pytest.raises(ValueError) as exc_info:
            read_pos_meaning_ru_tasks(path)
        
        assert "Translation task file has invalid columns" in str(exc_info.value)


def test_blank_pos():
    """Blank pos raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\n,NOUN\n")
        
        with pytest.raises(ValueError) as exc_info:
            read_pos_meaning_ru_tasks(path)
        
        assert "Translation task file contains blank pos at row 2" in str(exc_info.value)


def test_whitespace_only_pos():
    """Whitespace-only pos raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\n   ,NOUN\n")
        
        with pytest.raises(ValueError) as exc_info:
            read_pos_meaning_ru_tasks(path)
        
        assert "Translation task file contains blank pos at row 2" in str(exc_info.value)


def test_blank_meaning_ru():
    """Blank meaning_ru raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\nNOUN,\n")
        
        with pytest.raises(ValueError) as exc_info:
            read_pos_meaning_ru_tasks(path)
        
        assert "Translation task file contains blank meaning_ru at row 2" in str(exc_info.value)


def test_whitespace_only_meaning_ru():
    """Whitespace-only meaning_ru raises ValueError."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\nNOUN,   \n")
        
        with pytest.raises(ValueError) as exc_info:
            read_pos_meaning_ru_tasks(path)
        
        assert "Translation task file contains blank meaning_ru at row 2" in str(exc_info.value)


def test_duplicate_rows():
    """Duplicate (pos, meaning_ru) rows raise ValueError."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\nNOUN,дом\nNOUN,дом\n")
        
        with pytest.raises(ValueError) as exc_info:
            read_pos_meaning_ru_tasks(path)
        
        assert "Translation task file contains duplicate (pos, meaning_ru) task" in str(exc_info.value)
        assert "pos='NOUN'" in str(exc_info.value)
        assert "meaning_ru='дом'" in str(exc_info.value)


def test_same_pos_different_meaning_allowed():
    """Same POS with different meaning_ru is allowed."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\nNOUN,дом\nVERB,дом\n")
        
        df = read_pos_meaning_ru_tasks(path)
        
        assert len(df) == 2
        assert df.iloc[0]["pos"] == "NOUN"
        assert df.iloc[0]["meaning_ru"] == "дом"
        assert df.iloc[1]["pos"] == "VERB"
        assert df.iloc[1]["meaning_ru"] == "дом"


def test_path_as_string():
    """Accepts file path as string."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\nNOUN,дом\n")
        
        df = read_pos_meaning_ru_tasks(str(path))
        
        assert list(df.columns) == ["pos", "meaning_ru"]
        assert len(df) == 1


def test_column_schema_constant():
    """POS_MEANINGS_RU_COLUMNS constant matches expected schema."""
    assert POS_MEANINGS_RU_COLUMNS == ["pos", "meaning_ru"]


def test_build_translation_tasks_from_pos_meaning_ru():
    """build_translation_tasks_from_pos_meaning_ru converts DataFrame to task list."""
    df = pd.DataFrame({
        "pos": ["NOUN", "VERB", "PART"],
        "meaning_ru": ["дом", "читать", "а"],
    })
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    assert len(tasks) == 3
    
    assert tasks[0].pos == "NOUN"
    assert tasks[0].meaning_ru == "дом"
    assert not hasattr(tasks[0], "task_key")
    
    assert tasks[1].pos == "VERB"
    assert tasks[1].meaning_ru == "читать"
    assert not hasattr(tasks[1], "task_key")
    
    assert tasks[2].pos == "PART"
    assert tasks[2].meaning_ru == "а"
    assert not hasattr(tasks[2], "task_key")


def test_build_translation_tasks_preserves_order():
    """Task list preserves file row order."""
    df = pd.DataFrame({
        "pos": ["NOUN", "VERB", "PART"],
        "meaning_ru": ["дом", "читать", "а"],
    })
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    keys = [(t.pos, t.meaning_ru) for t in tasks]
    assert keys == [("NOUN", "дом"), ("VERB", "читать"), ("PART", "а")]


def test_build_translation_tasks_from_empty():
    """build_translation_tasks_from_pos_meaning_ru on empty DataFrame returns empty list."""
    df = pd.DataFrame(columns=["pos", "meaning_ru"])
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    assert tasks == []


def test_build_translation_tasks_empty_dataframe_no_task_key():
    """Empty DataFrame yields empty list (no task_key field to check)."""
    df = pd.DataFrame()
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    assert tasks == []


def test_build_translation_tasks_from_multiple_rows():
    """DataFrame with multiple valid rows produces same number of task objects."""
    df = pd.DataFrame({
        "pos": ["NOUN", "VERB", "PART", "ADJ"],
        "meaning_ru": ["дом", "читать", "а", "красный"],
    })
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    assert len(tasks) == 4
    assert tasks[0].pos == "NOUN"
    assert tasks[0].meaning_ru == "дом"
    assert tasks[1].pos == "VERB"
    assert tasks[1].meaning_ru == "читать"
    assert tasks[2].pos == "PART"
    assert tasks[2].meaning_ru == "а"
    assert tasks[3].pos == "ADJ"
    assert tasks[3].meaning_ru == "красный"
    for task in tasks:
        assert not hasattr(task, "task_key")


def test_build_translation_tasks_preserves_meaning_with_parens():
    """Meaning with parentheses is preserved correctly."""
    df = pd.DataFrame({
        "pos": ["NOUN"],
        "meaning_ru": ["место (под чем-либо)"],
    })
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    assert len(tasks) == 1
    assert tasks[0].pos == "NOUN"
    assert tasks[0].meaning_ru == "место (под чем-либо)"
    assert not hasattr(tasks[0], "task_key")


def test_build_translation_tasks_no_task_key_field():
    """Tasks should not have task_key field."""
    df = pd.DataFrame({
        "pos": ["NOUN"],
        "meaning_ru": ["дом"],
    })
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    assert len(tasks) == 1
    assert not hasattr(tasks[0], "task_key")
    assert not hasattr(tasks[0], "primary_gloss_ru")
    assert not hasattr(tasks[0], "meaning_hint")
    assert not hasattr(tasks[0], "sourcecount")


def test_build_translation_tasks_distinct_pos_same_meaning():
    """Same meaning_ru with distinct POS values yields two distinct task objects."""
    df = pd.DataFrame({
        "pos": ["NOUN", "VERB"],
        "meaning_ru": ["дом", "дом"],
    })
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    assert len(tasks) == 2
    assert tasks[0].pos == "NOUN"
    assert tasks[0].meaning_ru == "дом"
    assert tasks[1].pos == "VERB"
    assert tasks[1].meaning_ru == "дом"
    # The two tasks are distinct by (pos, meaning_ru)
    assert (tasks[0].pos, tasks[0].meaning_ru) != (tasks[1].pos, tasks[1].meaning_ru)


def test_build_translation_tasks_from_reader_output():
    """Integration: reader output -> build_translation_tasks_from_pos_meaning_ru."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\nNOUN,дом\nVERB,читать\n")
        
        reader_df = read_pos_meaning_ru_tasks(path)
        tasks = build_translation_tasks_from_pos_meaning_ru(reader_df)
        
        assert len(tasks) == 2
        assert tasks[0].pos == "NOUN"
        assert tasks[0].meaning_ru == "дом"
        assert tasks[1].pos == "VERB"
        assert tasks[1].meaning_ru == "читать"


if __name__ == "__main__":
    import os
    
    pytest.main([__file__, "-v"])
