"""Part 3 tests for multi-model translation comparison pipeline."""

import sys
import pathlib
import csv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import tempfile
import os

import pandas as pd

from src.sem_cat.compare.loading import (
    parse_translation_arg,
    load_single_model,
    merge_all_models,
)
from src.sem_cat.compare.output_tables import build_comparison_df


def _make_csv(path, rows):
    """Write a minimal translation CSV for testing."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_load_and_prefix_columns():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "test.csv")
        _make_csv(csv_path, [
            {"pos": "NOUN", "meaning_ru": "дом", "meaning_en": "house", "qa_keep": "True",
             "qa_score": "0.0", "qa_flags": "", "meaning_ru_back": "дом", "roundtrip_distance": "0.1"},
        ])
        df = load_single_model(pathlib.Path(csv_path), "google")
        assert "google_en" in df.columns
        assert "google_keep" in df.columns
        assert "google_score" in df.columns
        assert "google_flags" in df.columns
        assert "google_ru" in df.columns
        assert "google_rt" in df.columns
        assert df.iloc[0]["google_en"] == "house"


def test_merge_two_models_shared():
    df1 = pd.DataFrame({
        "pos": ["NOUN", "NOUN"],
        "meaning_ru": ["дом", "машина"],
        "google_en": ["house", "car"],
        "google_keep": ["True", "True"],
    })
    df2 = pd.DataFrame({
        "pos": ["NOUN", "VERB"],
        "meaning_ru": ["дом", "жить"],
        "helsinki_en": ["house", "live"],
        "helsinki_keep": ["True", "True"],
    })
    merged = merge_all_models({"google": df1, "helsinki": df2})
    assert len(merged) == 3
    assert "pos" in merged.columns
    assert "meaning_ru" in merged.columns
    assert "google_en" in merged.columns
    assert "helsinki_en" in merged.columns
    shared_row = merged[(merged["pos"] == "NOUN") & (merged["meaning_ru"] == "дом")].iloc[0]
    assert shared_row["google_en"] == "house"
    assert shared_row["helsinki_en"] == "house"


def test_merge_missing_model_appears_blank():
    df1 = pd.DataFrame({
        "pos": ["NOUN"],
        "meaning_ru": ["дом"],
        "google_en": ["house"],
        "google_keep": ["True"],
    })
    df2 = pd.DataFrame({
        "pos": ["VERB"],
        "meaning_ru": ["жить"],
        "helsinki_en": ["live"],
        "helsinki_keep": ["True"],
    })
    merged = merge_all_models({"google": df1, "helsinki": df2})
    assert len(merged) == 2
    google_row = merged[merged["meaning_ru"] == "дом"].iloc[0]
    assert google_row["google_en"] == "house"
    helsinki_val = google_row["helsinki_en"]
    assert pd.isna(helsinki_val) or helsinki_val == ""


def test_build_comparison_df_preserves_order():
    df1 = pd.DataFrame({
        "pos": ["NOUN", "VERB"],
        "meaning_ru": ["дом", "жить"],
        "google_en": ["house", "live"],
        "google_keep": ["True", "True"],
    })
    df2 = pd.DataFrame({
        "pos": ["NOUN", "VERB"],
        "meaning_ru": ["дом", "жить"],
        "helsinki_en": ["house", "live"],
        "helsinki_keep": ["True", "True"],
    })
    merged = merge_all_models({"google": df1, "helsinki": df2})
    out_df = build_comparison_df(merged)
    assert list(out_df.columns[:2]) == ["pos", "meaning_ru"]
    assert "google_en" in out_df.columns
    assert "helsinki_en" in out_df.columns


def test_parse_translation_arg_valid():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"gloss_ru,gloss_en\n")
        tmp_path = f.name
    try:
        mk, path = parse_translation_arg(f"google={tmp_path}")
        assert mk == "google"
        assert str(path) == tmp_path
    finally:
        os.unlink(tmp_path)


def test_parse_translation_arg_no_equals():
    try:
        parse_translation_arg("google")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_parse_translation_arg_empty_key():
    try:
        parse_translation_arg("=/tmp/test.csv")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_merge_empty():
    merged = merge_all_models({})
    assert list(merged.columns) == ["pos", "meaning_ru"]
    assert len(merged) == 0


if __name__ == "__main__":
    tests = [
        test_load_and_prefix_columns,
        test_merge_two_models_shared,
        test_merge_missing_model_appears_blank,
        test_build_comparison_df_preserves_order,
        test_parse_translation_arg_valid,
        test_parse_translation_arg_no_equals,
        test_parse_translation_arg_empty_key,
        test_merge_empty,
    ]
    
    import pytest
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
