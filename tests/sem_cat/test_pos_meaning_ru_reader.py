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
    serialize_task_key,
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
    
    assert tasks[0].task_key == "NOUN::дом"
    assert tasks[0].meaning_ru == "дом"
    assert tasks[0].pos == "NOUN"
    
    assert tasks[1].task_key == "VERB::читать"
    assert tasks[1].meaning_ru == "читать"
    assert tasks[1].pos == "VERB"
    
    assert tasks[2].task_key == "PART::а"
    assert tasks[2].meaning_ru == "а"
    assert tasks[2].pos == "PART"


def test_build_translation_tasks_preserves_order():
    """Task list preserves file row order."""
    df = pd.DataFrame({
        "pos": ["NOUN", "VERB", "PART"],
        "meaning_ru": ["дом", "читать", "а"],
    })
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    task_keys = [t.task_key for t in tasks]
    assert task_keys == ["NOUN::дом", "VERB::читать", "PART::а"]


def test_build_translation_tasks_from_empty():
    """build_translation_tasks_from_pos_meaning_ru on empty DataFrame returns empty list."""
    df = pd.DataFrame(columns=["pos", "meaning_ru"])
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    assert tasks == []


def test_build_translation_tasks_preserves_meaning_with_parens():
    """Meaning with parentheses is preserved correctly."""
    df = pd.DataFrame({
        "pos": ["NOUN"],
        "meaning_ru": ["место (под чем-либо)"],
    })
    
    tasks = build_translation_tasks_from_pos_meaning_ru(df)
    
    assert len(tasks) == 1
    assert tasks[0].task_key == "NOUN::место (под чем-либо)"
    assert tasks[0].meaning_ru == "место (под чем-либо)"


def test_build_translation_tasks_from_reader_output():
    """Integration: reader output -> build_translation_tasks_from_pos_meaning_ru."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.csv"
        path.write_text("pos,meaning_ru\nNOUN,дом\nVERB,читать\n")
        
        reader_df = read_pos_meaning_ru_tasks(path)
        tasks = build_translation_tasks_from_pos_meaning_ru(reader_df)
        
        assert len(tasks) == 2
        assert tasks[0].task_key == "NOUN::дом"
        assert tasks[1].task_key == "VERB::читать"


if __name__ == "__main__":
    import os
    
    pytest.main([__file__, "-v"])
