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
import pandas as pd
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
    result = load_translation_cache(pathlib.Path("/nonexistent/path.csv"))
    assert result.state == "missing"
    assert result.df.empty
    assert "gloss_ru" in result.df.columns


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
        test_paths_config_loads_default,
        test_paths_config_resolves_repo_relative_paths,
        test_paths_config_raises_on_missing_key,
        test_build_concepts_wdh_returns_flat_lookup_schema,
        test_build_concepts_wdh_handles_missing_category_wdh,
        test_save_and_load_concepts_wdh_flat_schema,
        test_load_concepts_wdh_accepts_legacy_extra_columns,
        test_load_concepts_wdh_requires_minimum_columns,
        test_normalize_wdh_strips_and_sorts,
        test_concepts_wdh_wdh_normalization_on_build,
        test_normalize_loaded_task_key_normalizes_tab_format,
        test_build_ambiguous_task_summary_normalizes_legacy_task_keys,
        test_load_translation_cache_normalizes_task_key,
        test_cache_load_handles_task_key_str_fallback,
        test_step03_load_normalizes_task_key,
        test_step03_merge_uses_normalized_task_key,
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


# ---------------------------------------------------------------------------
# 21. Path config loader tests
# ---------------------------------------------------------------------------


def test_paths_config_loads_default():
    """Default config should load successfully from sem_cat_paths.toml."""
    from src.sem_cat.paths_config import (
        load_sem_cat_paths,
        SemCatPaths,
    )
    import os

    old_cwd = os.getcwd()
    try:
        os.chdir(pathlib.Path(__file__).resolve().parents[2])
        cfg = load_sem_cat_paths()
        assert isinstance(cfg, SemCatPaths)
        assert cfg.wn_domains.name == "00_wn-domains-3.2-20070223"
        assert cfg.concepts_catalog.name == "concepts_with_english_1445.csv"
        assert cfg.concepts_wdh.name == "concepts_wdh.tsv"
    finally:
        os.chdir(old_cwd)


def test_paths_config_resolves_repo_relative_paths():
    """Repo-relative paths should be resolved to absolute paths from project root."""
    from src.sem_cat.paths_config import load_sem_cat_paths

    import os

    old_cwd = os.getcwd()
    try:
        os.chdir(pathlib.Path(__file__).resolve().parents[2])
        cfg = load_sem_cat_paths()
        assert cfg.concepts_catalog.is_absolute()
        assert "data" in cfg.concepts_catalog.parts
        assert "concepts" in cfg.concepts_catalog.parts
        assert "concepts_with_english_1445.csv" in cfg.concepts_catalog.parts
    finally:
        os.chdir(old_cwd)


def test_paths_config_raises_on_missing_key():
    """Missing required key should raise ValueError with clear message."""
    import tempfile

    from src.sem_cat.paths_config import load_sem_cat_paths

    with tempfile.TemporaryDirectory() as td:
        config_file = pathlib.Path(td) / "test_config.toml"
        config_file.write_text(
            """[paths]
wn_domains = "data/sem_cat/test"
concepts_catalog = "data/sem_cat/test.csv"
"""
        )
        try:
            load_sem_cat_paths(str(config_file))
            assert False, "Expected ValueError for missing key"
        except ValueError as e:
            assert "concept_categories_wdh" in str(e) or "concepts_wdh" in str(e)


def test_build_concepts_wdh_returns_flat_lookup_schema():
    """build_concepts_wdh should return only the new flat schema."""
    import pandas as pd
    from src.sem_cat.utils.concept_wdh import build_concepts_wdh

    cat = pd.DataFrame(
        {
            "category_id": ["A11", "B355"],
            "wdh": ["astronomy", "industry"],
        }
    )
    concepts = pd.DataFrame(
        {
            "category_id": ["A11", "B355"],
            "pos": ["NOUN", "VERB"],
            "concept_id": ["1", "1004"],
            "concept_ru": ["небо", "ковать (железо)"],
            "concept_en": ["sky", "to forge (iron)"],
        }
    )

    out = build_concepts_wdh(cat, concepts)

    assert list(out.columns) == [
        "category_id",
        "pos",
        "concept_id",
        "concept_ru",
        "concept_en",
        "wdh",
    ]
    assert out["wdh"].tolist() == ["astronomy", "industry"]
    assert "wdh_source" not in out.columns
    assert "wdh_confidence" not in out.columns
    assert "wdh_note" not in out.columns


def test_build_concepts_wdh_handles_missing_category_wdh():
    """Missing WDH for a category should result in empty wdh."""
    import pandas as pd
    from src.sem_cat.utils.concept_wdh import build_concepts_wdh

    cat = pd.DataFrame(
        {
            "category_id": ["A11"],
            "wdh": ["astronomy"],
        }
    )
    concepts = pd.DataFrame(
        {
            "category_id": ["A11", "B999"],
            "pos": ["NOUN", "NOUN"],
            "concept_id": ["1", "2"],
            "concept_ru": ["небо", "пустота"],
            "concept_en": ["sky", "emptiness"],
        }
    )

    out = build_concepts_wdh(cat, concepts)

    assert out.loc[0, "wdh"] == "astronomy"
    assert out.loc[1, "wdh"] == ""
    assert out["concept_id"].tolist() == ["1", "2"]


def test_save_and_load_concepts_wdh_flat_schema(tmp_path):
    """Save/load roundtrip should preserve the flat schema."""
    import pandas as pd
    from src.sem_cat.utils.concept_wdh import save_concepts_wdh
    from src.sem_cat.utils.meaning_propagation import load_concepts_wdh

    df = pd.DataFrame(
        {
            "category_id": ["A11"],
            "pos": ["NOUN"],
            "concept_id": ["1"],
            "concept_ru": ["небо"],
            "concept_en": ["sky"],
            "wdh": ["astronomy"],
        }
    )
    path = tmp_path / "concepts_wdh.tsv"
    save_concepts_wdh(df, str(path))
    loaded = load_concepts_wdh(str(path))

    assert list(loaded.columns) == list(df.columns)
    assert loaded.iloc[0]["concept_id"] == "1"
    assert loaded.iloc[0]["wdh"] == "astronomy"


def test_load_concepts_wdh_accepts_legacy_extra_columns(tmp_path):
    """Loader should tolerate older files with extra provenance columns."""
    import pandas as pd

    from src.sem_cat.utils.meaning_propagation import load_concepts_wdh

    df = pd.DataFrame(
        {
            "category_id": ["A11"],
            "pos": ["NOUN"],
            "concept_id": ["1"],
            "concept_ru": ["небо"],
            "concept_en": ["sky"],
            "wdh": ["astronomy"],
            "wdh_source": ["inherited_from_category"],
            "wdh_confidence": ["medium"],
            "wdh_note": ["WDH inherited from category A11"],
        }
    )
    path = tmp_path / "legacy.tsv"
    df.to_csv(path, sep="\t", index=False)
    loaded = load_concepts_wdh(str(path))

    assert loaded.iloc[0]["concept_id"] == "1"
    assert loaded.iloc[0]["wdh"] == "astronomy"


def test_load_concepts_wdh_requires_minimum_columns(tmp_path):
    """Loader should reject files missing required columns."""
    import pandas as pd
    from src.sem_cat.utils.meaning_propagation import load_concepts_wdh

    df = pd.DataFrame(
        {
            "category_id": ["A11"],
            "pos": ["NOUN"],
            "concept_id": ["1"],
            "concept_ru": ["небо"],
            "concept_en": ["sky"],
        }
    )
    path = tmp_path / "incomplete.tsv"
    df.to_csv(path, sep="\t", index=False)

    try:
        load_concepts_wdh(str(path))
        assert False, "Expected ValueError for missing wdh column"
    except ValueError as e:
        assert "wdh" in str(e)


def test_normalize_wdh_strips_and_sorts():
    """Normalize WDH should strip, lowercase, and sort comma-separated values."""
    from src.sem_cat.utils.concept_wdh import _normalize_wdh

    assert _normalize_wdh("astronomy, physics") == "astronomy, physics"
    assert _normalize_wdh("  PHYSICS  ,   ASTRONOMY  ") == "astronomy, physics"
    assert _normalize_wdh("") == ""
    assert _normalize_wdh("astronomy") == "astronomy"
    assert _normalize_wdh(None) == ""


def test_concepts_wdh_wdh_normalization_on_build():
    """WDH normalization should happen during build_concepts_wdh."""
    import pandas as pd
    from src.sem_cat.utils.concept_wdh import build_concepts_wdh

    cat = pd.DataFrame(
        {
            "category_id": ["A11"],
            "wdh": "  PHYSICS,   ASTRONOMY  ",
        }
    )
    concepts = pd.DataFrame(
        {
            "category_id": ["A11"],
            "pos": ["NOUN"],
            "concept_id": ["1"],
            "concept_ru": ["небо"],
            "concept_en": ["sky"],
        }
    )

    out = build_concepts_wdh(cat, concepts)

    assert out.iloc[0]["wdh"] == "astronomy, physics"


# ---------------------------------------------------------------------------
# 22. WDH label statistics tests (atomic-label counting)
# ---------------------------------------------------------------------------


def test_explode_wdh_labels_basic_splitting():
    """Split comma-separated labels and return flat list."""
    from src.sem_cat.utils.concept_wdh import _explode_wdh_labels
    import pandas as pd

    s = pd.Series([
        "person",
        "factotum, person",
        "physiology, psychological_features, psychology",
    ])
    labels = _explode_wdh_labels(s)
    assert sorted(labels) == [
        "factotum",
        "person",
        "person",
        "physiology",
        "psychological_features",
        "psychology",
    ]


def test_explode_wdh_labels_ignores_blank_and_none():
    """None, empty and whitespace-only values are ignored."""
    from src.sem_cat.utils.concept_wdh import _explode_wdh_labels
    import pandas as pd

    s = pd.Series([None, "", "  ", "person, factotum"])
    labels = _explode_wdh_labels(s)
    assert sorted(labels) == ["factotum", "person"]


def test_explode_wdh_labels_deduplicates_within_row():
    """Duplicate labels in same row are deduplicated."""
    from src.sem_cat.utils.concept_wdh import _explode_wdh_labels
    import pandas as pd

    s = pd.Series(["person, person, factotum"])
    labels = _explode_wdh_labels(s)
    assert sorted(labels) == ["factotum", "person"]


def test_collect_wdh_label_stats_counts_atomic_labels():
    """collect_wdh_label_stats counts atomic labels, not combinations."""
    from src.sem_cat.utils.concept_wdh import collect_wdh_label_stats
    import pandas as pd

    s = pd.Series([
        "person",
        "factotum, person",
        "physiology, psychological_features, psychology",
    ])
    unique_count, top = collect_wdh_label_stats(s)
    assert unique_count == 5
    assert top[:5] == [
        ("person", 2),
        ("factotum", 1),
        ("physiology", 1),
        ("psychological_features", 1),
        ("psychology", 1),
    ]


def test_collect_wdh_label_stats_ignores_blank_and_duplicates():
    """Blank values ignored, duplicates within row counted once."""
    from src.sem_cat.utils.concept_wdh import collect_wdh_label_stats
    import pandas as pd

    s = pd.Series([None, "", "  ", "person, person, factotum"])
    unique_count, top = collect_wdh_label_stats(s)
    assert unique_count == 2
    # Alphabetical order for ties: factotum < person
    assert top == [("factotum", 1), ("person", 1)]


def test_collect_wdh_label_stats_deterministic_tie_ordering():
    """Same counts are sorted alphabetically for stability."""
    from src.sem_cat.utils.concept_wdh import collect_wdh_label_stats
    import pandas as pd

    s = pd.Series(["zebra", "apple", "charlie", "banana"])
    unique_count, top = collect_wdh_label_stats(s)
    assert unique_count == 4
    assert top == [
        ("apple", 1),
        ("banana", 1),
        ("charlie", 1),
        ("zebra", 1),
    ]


def test_collect_wdh_label_stats_case_preserved():
    """Atomic label counting preserves original case."""
    from src.sem_cat.utils.concept_wdh import collect_wdh_label_stats
    import pandas as pd

    s = pd.Series(["Person", "PERSON, factotum"])
    unique_count, top = collect_wdh_label_stats(s)
    assert unique_count == 3
    assert top[0] == ("PERSON", 1)
    assert top[1] == ("Person", 1)
    assert top[2] == ("factotum", 1)


def test_collect_wdh_label_stats_combined_case():
    """Real-world mix of cases."""
    from src.sem_cat.utils.concept_wdh import collect_wdh_label_stats
    import pandas as pd

    s = pd.Series([
        "person",
        "factotum, person",
        "",
        "astronomy, physics",
    ])
    unique_count, top = collect_wdh_label_stats(s)
    assert unique_count == 4
    assert top[0] == ("person", 2)
    assert top[1] == ("astronomy", 1)
    assert top[2] == ("factotum", 1)
    assert top[3] == ("physics", 1)


# ---------------------------------------------------------------------------
# 23. Step-06 path resolution tests
# ---------------------------------------------------------------------------


def test_step06_resolve_no_overrides():
    """No CLI overrides -> use config defaults."""
    import types
    import importlib

    from src.sem_cat.paths_config import SemCatPaths

    mod = importlib.import_module("src.sem_cat.06_concepts_wdh")

    config = types.SimpleNamespace(
        concept_categories_wdh=pathlib.Path("/cfg/cat.tsv"),
        concepts_catalog=pathlib.Path("/cfg/concepts.csv"),
        concepts_wdh=pathlib.Path("/cfg/out.tsv"),
    )

    class Args:
        cat_wdh = None
        concepts = None
        out_file = None

    args = Args()
    cat_wdh, concepts, out = mod.resolve_step06_paths(args, config)
    assert cat_wdh == pathlib.Path("/cfg/cat.tsv")
    assert concepts == pathlib.Path("/cfg/concepts.csv")
    assert out == pathlib.Path("/cfg/out.tsv")


def test_step06_resolve_concepts_override():
    """--concepts override only -> only concepts path changes."""
    import types
    import importlib

    from src.sem_cat.paths_config import SemCatPaths

    mod = importlib.import_module("src.sem_cat.06_concepts_wdh")

    config = types.SimpleNamespace(
        concept_categories_wdh=pathlib.Path("/cfg/cat.tsv"),
        concepts_catalog=pathlib.Path("/cfg/concepts.csv"),
        concepts_wdh=pathlib.Path("/cfg/out.tsv"),
    )

    class Args:
        cat_wdh = None
        concepts = "/override/concepts.csv"
        out_file = None

    args = Args()
    cat_wdh, concepts, out = mod.resolve_step06_paths(args, config)
    assert cat_wdh == pathlib.Path("/cfg/cat.tsv")
    assert str(concepts) == "/override/concepts.csv"
    assert out == pathlib.Path("/cfg/out.tsv")


def test_step06_resolve_out_file_override():
    """--out-file override only -> only output path changes."""
    import types
    import importlib

    from src.sem_cat.paths_config import SemCatPaths

    mod = importlib.import_module("src.sem_cat.06_concepts_wdh")

    config = types.SimpleNamespace(
        concept_categories_wdh=pathlib.Path("/cfg/cat.tsv"),
        concepts_catalog=pathlib.Path("/cfg/concepts.csv"),
        concepts_wdh=pathlib.Path("/cfg/out.tsv"),
    )

    class Args:
        cat_wdh = None
        concepts = None
        out_file = "/override/out.tsv"

    args = Args()
    cat_wdh, concepts, out = mod.resolve_step06_paths(args, config)
    assert cat_wdh == pathlib.Path("/cfg/cat.tsv")
    assert concepts == pathlib.Path("/cfg/concepts.csv")
    assert str(out) == "/override/out.tsv"


# ---------------------------------------------------------------------------
# 25. Transformation regression tests
# ---------------------------------------------------------------------------


def test_normalize_loaded_task_key_normalizes_tab_format():
    """Legacy tab-separated task keys should normalize to :: format."""
    from src.sem_cat.compare.loading import normalize_loaded_task_key

    assert normalize_loaded_task_key("NOUN\tобида") == "NOUN::обида"
    assert normalize_loaded_task_key("NOUN::обида") == "NOUN::обида"
    assert normalize_loaded_task_key(None) is None
    assert normalize_loaded_task_key("") is None
    assert normalize_loaded_task_key("   ") is None


def test_build_ambiguous_task_summary_normalizes_legacy_task_keys():
    """build_ambiguous_task_summary should normalize task keys before grouping."""
    import pandas as pd
    from src.sem_cat.io.translation_cache import normalize_loaded_task_key

    df = pd.DataFrame(
        [
            {
                "task_key": "NOUN\tобида",
                "task_pos": "NOUN",
                "primary_gloss_ru": "обида",
                "existing_en_candidates": "offence || offence; insult",
                "existing_en_candidate_count": 2,
                "lemma": "abidaine",
                "lang": "vep",
            },
            {
                "task_key": "NOUN::обида",
                "task_pos": "NOUN",
                "primary_gloss_ru": "обида",
                "existing_en_candidates": "offence || offence; insult",
                "existing_en_candidate_count": 2,
                "lemma": "abid",
                "lang": "olo",
            },
        ]
    )

    from src.sem_cat.io.translation_cache import normalize_loaded_task_key as nltk

    if df.empty:
        out = pd.DataFrame()
    else:
        out = df.copy()
        if "task_key" in out.columns:
            out["task_key"] = out["task_key"].map(nltk)

        summaryParts = []
        for _, group in out.groupby(["task_key", "task_pos", "primary_gloss_ru", "existing_en_candidates", "existing_en_candidate_count"], dropna=False, sort=False):
            from src.sem_cat.pipeline.vepkar_translation_selection import compute_suggested_candidate_index
            suggested_idx = compute_suggested_candidate_index(
                str(group["existing_en_candidates"].iloc[0]) if "existing_en_candidates" in group.columns else ""
            )
            summaryParts.append(
                {
                    "task_key": group["task_key"].iloc[0] if "task_key" in group.columns else "",
                    "task_pos": group["task_pos"].iloc[0] if "task_pos" in group.columns else "",
                    "primary_gloss_ru": group["primary_gloss_ru"].iloc[0] if "primary_gloss_ru" in group.columns else "",
                    "existing_en_candidates": str(group["existing_en_candidates"].iloc[0]) if "existing_en_candidates" in group.columns else "",
                    "existing_en_candidate_count": int(group["existing_en_candidate_count"].iloc[0]) if "existing_en_candidate_count" in group.columns else 0,
                    "suggested_candidate_index": suggested_idx if suggested_idx is not None else "",
                    "missing_row_count": len(group),
                    "example_lemma": str(group["lemma"].iloc[0]) if "lemma" in group.columns and not group["lemma"].isna().any() else "",
                    "langs": " || ".join(sorted(set(str(x) for x in group["lang"].dropna().tolist()))) if "lang" in group.columns else "",
                }
            )
        out = pd.DataFrame(summaryParts)

    assert len(out) == 1
    assert out.loc[0, "task_key"] == "NOUN::обида"
    assert out.loc[0, "suggested_candidate_index"] == 1


def test_load_translation_cache_normalizes_task_key(tmp_path):
    """Cache loading should normalize task keys from legacy formats."""
    import os
    df = pd.DataFrame(
        [
            {"gloss_ru": "дом", "gloss_en": "house", "qa_keep": "True", "qa_score": "0.0", "qa_flags": "", "model_key": "google", "task_key": "NOUN\tдом"},
            {"gloss_ru": "машина", "gloss_en": "car", "qa_keep": "True", "qa_score": "0.1", "qa_flags": "", "model_key": "google", "task_key": "NOUN::машина"},
        ]
    )
    path = tmp_path / "cache.csv"
    df.to_csv(path, index=False)
    
    from src.sem_cat.io.translation_cache import load_translation_cache
    result = load_translation_cache(path, expected_model_key="google")
    
    assert "task_key" in result.df.columns
    assert "NOUN::дом" in result.df["task_key"].values
    assert "NOUN\tдом" not in result.df["task_key"].values


def test_cache_load_handles_task_key_str_fallback(tmp_path):
    """Cache loader should fall back to task_key_str if task_key is missing."""
    import pandas as pd
    df = pd.DataFrame(
        [
            {"gloss_ru": "дом", "gloss_en": "house", "qa_keep": "True", "qa_score": "0.0", "qa_flags": "", "model_key": "google", "task_key_str": "NOUN\tдом"},
        ]
    )
    path = tmp_path / "legacy_cache.csv"
    df.to_csv(path, index=False)
    
    from src.sem_cat.io.translation_cache import load_translation_cache
    result = load_translation_cache(path, expected_model_key="google")
    
    assert "task_key" in result.df.columns
    assert "NOUN::дом" in result.df["task_key"].values


def test_step03_load_normalizes_task_key(tmp_path):
    """Step 03 loading should normalize task keys before merge decision."""
    df = pd.DataFrame(
        [
            {"gloss_ru": "дом", "gloss_en": "house", "qa_keep": "True", "qa_score": "0.0", "qa_flags": "", "model_key": "google", "task_key": "NOUN\tдом"},
        ]
    )
    path = tmp_path / "model.csv"
    df.to_csv(path, index=False)
    
    from src.sem_cat.compare.loading import load_single_model
    result = load_single_model(path, "google")
    
    # task_key and task_pos are preserved without prefix
    assert "task_key" in result.columns
    assert "NOUN::дом" in result["task_key"].values


def test_step03_merge_uses_normalized_task_key(tmp_path):
    """Step 03 merge should correctly match normalized task keys."""
    from src.sem_cat.compare.loading import load_single_model, merge_all_models
    
    df1 = pd.DataFrame(
        [
            {"gloss_ru": "дом", "gloss_en": "house", "qa_keep": "True", "qa_score": "0.0", "qa_flags": "", "model_key": "google", "task_key": "NOUN\tдом"},
        ]
    )
    df2 = pd.DataFrame(
        [
            {"gloss_ru": "дом", "gloss_en": "house", "qa_keep": "True", "qa_score": "0.0", "qa_flags": "", "model_key": "google", "task_key": "NOUN::дом"},
        ]
    )
    
    path1 = tmp_path / "m1.csv"
    path2 = tmp_path / "m2.csv"
    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)
    
    m1 = load_single_model(path1, "google")
    m2 = load_single_model(path2, "google")
    
    merged = merge_all_models({"google": m1, "google2": m2})
    
    # With normalized keys, both should appear in the same row
    assert len(merged) == 1
    # task_key is preserved without prefix
    assert "task_key" in merged.columns
    assert "NOUN::дом" in merged["task_key"].values
    assert merged["gloss_ru_x"].iloc[0] == "дом"
    assert merged["gloss_ru_y"].iloc[0] == "дом"



def test_step06_resolve_all_overrides():
    """All CLI overrides -> all respected."""
    import types
    import importlib

    from src.sem_cat.paths_config import SemCatPaths

    mod = importlib.import_module("src.sem_cat.06_concepts_wdh")

    config = types.SimpleNamespace(
        concept_categories_wdh=pathlib.Path("/cfg/cat.tsv"),
        concepts_catalog=pathlib.Path("/cfg/concepts.csv"),
        concepts_wdh=pathlib.Path("/cfg/out.tsv"),
    )

    class Args:
        cat_wdh = "/override/cat.tsv"
        concepts = "/override/concepts.csv"
        out_file = "/override/out.tsv"

    args = Args()
    cat_wdh, concepts, out = mod.resolve_step06_paths(args, config)
    assert str(cat_wdh) == "/override/cat.tsv"
    assert str(concepts) == "/override/concepts.csv"
    assert str(out) == "/override/out.tsv"
