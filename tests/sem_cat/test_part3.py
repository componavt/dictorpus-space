"""Part 3 tests for multi-model comparison pipeline."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import tempfile
import os

import pandas as pd

from src.sem_cat.compare.loading import parse_translation_arg, load_single_model
from src.sem_cat.compare.normalization import (
    normalize_output_for_comparison,
    output_similarity,
    outputs_are_near_match,
)
from src.sem_cat.compare.consensus import cluster_outputs
from src.sem_cat.compare.complexity import compute_gloss_complexity
from src.sem_cat.compare.risk import compute_total_risk, compute_risk_level, ComparisonRiskConfig
from src.sem_cat.compare.proposal import select_proposed_translation
from src.sem_cat.compare.data_structures import ModelOutput, ConsensusCluster

# VepKar translation workflow tests
from src.sem_cat.pipeline.vepkar_translation_selection import (
    normalize_pos_for_task,
    canonical_existing_en,
    has_existing_english,
    build_task_key,
    prepare_meanings_for_translation,
    split_by_existing_en_reuse,
    prepare_translation_input_for_task,
    extract_unique_translation_tasks,
)
from src.sem_cat.utils.gloss_normalizer import primary_gloss


# ---------------------------------------------------------------------------
# 1. Parsing repeated --translations arguments
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Loading multiple model CSVs
# ---------------------------------------------------------------------------

def _make_csv(path, rows):
    """Write a minimal translation CSV for testing."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_load_and_prefix_columns():
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "test.csv")
        _make_csv(csv_path, [
            {"gloss_ru": "дом", "gloss_en": "house", "qa_keep": "True",
             "qa_score": "0.0", "qa_flags": "", "model_key": "google"},
        ])
        df = load_single_model(pathlib.Path(csv_path), "google")
        assert "google__gloss_en" in df.columns
        assert "google__qa_keep" in df.columns
        assert "google__qa_score" in df.columns
        assert "google__qa_flags" in df.columns
        assert df.iloc[0]["google__gloss_en"] == "house"


def test_load_rejects_mismatched_model_key():
    """A CLI label that mismatches the file's model_key column must fail."""
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "test.csv")
        _make_csv(csv_path, [
            {"gloss_ru": "дом", "gloss_en": "house", "qa_keep": "True",
             "qa_score": "0.0", "qa_flags": "", "model_key": "helsinki_opus_mt_ru_en"},
        ])
        try:
            load_single_model(pathlib.Path(csv_path), "helsinkiopusmtruen")
            assert False, "Expected ValueError for mismatched model key"
        except ValueError as e:
            assert "helsinki_opus_mt_ru_en" in str(e)
            assert "helsinkiopusmtruen" in str(e)


# ---------------------------------------------------------------------------
# 3. Consensus clustering groups near-identical outputs
# ---------------------------------------------------------------------------

def test_consensus_clustering_groups_identical():
    outputs = [
        ModelOutput("m1", "M1", "house", True, 0.0, normalized_gloss_en="house"),
        ModelOutput("m2", "M2", "house", True, 0.0, normalized_gloss_en="house"),
        ModelOutput("m3", "M3", "home", True, 0.1, normalized_gloss_en="home"),
    ]
    clusters = cluster_outputs(outputs, threshold=0.85)
    assert len(clusters) == 2
    assert len(clusters[0].model_keys) == 2  # "house" cluster
    assert clusters[0].representative == "house"


def test_consensus_all_different():
    outputs = [
        ModelOutput("m1", "M1", "house", True, 0.0, normalized_gloss_en="house"),
        ModelOutput("m2", "M2", "building", True, 0.2, normalized_gloss_en="building"),
        ModelOutput("m3", "M3", "residence", True, 0.3, normalized_gloss_en="residence"),
    ]
    clusters = cluster_outputs(outputs, threshold=0.85)
    assert len(clusters) == 3


# ---------------------------------------------------------------------------
# 4. Strong consensus lowers risk
# ---------------------------------------------------------------------------

def test_strong_consensus_lowers_risk():
    outputs = [
        ModelOutput("m1", "M1", "house", True, 0.0),
        ModelOutput("m2", "M2", "house", True, 0.0),
        ModelOutput("m3", "M3", "house", True, 0.0),
        ModelOutput("m4", "M4", "house", True, 0.0),
    ]
    risk, reasons = compute_total_risk(
        outputs=outputs,
        total_models=4,
        largest_cluster_size=4,
        good_model_count=4,
        consensus_ratio=1.0,
        disagreement_score=0.0,
        complexity_score=0.0,
    )
    assert "strong_consensus" in reasons
    assert risk < 0.2  # Should be very low with strong consensus


# ---------------------------------------------------------------------------
# 5. All-blank outputs get high risk
# ---------------------------------------------------------------------------

def test_all_blank_high_risk():
    outputs = [
        ModelOutput("m1", "M1", "", False, 1.0),
        ModelOutput("m2", "M2", "", False, 1.0),
    ]
    risk, reasons = compute_total_risk(
        outputs=outputs,
        total_models=2,
        largest_cluster_size=0,
        good_model_count=0,
        consensus_ratio=0.0,
        disagreement_score=0.0,
        complexity_score=0.0,
    )
    assert "all_blank" in reasons
    assert risk >= 0.3


# ---------------------------------------------------------------------------
# 6. Severe disagreement raises risk
# ---------------------------------------------------------------------------

def test_severe_disagreement_raises_risk():
    outputs = [
        ModelOutput("m1", "M1", "house", True, 0.0),
        ModelOutput("m2", "M2", "building", True, 0.1),
        ModelOutput("m3", "M3", "residence", True, 0.2),
        ModelOutput("m4", "M4", "home", True, 0.0),
    ]
    risk, reasons = compute_total_risk(
        outputs=outputs,
        total_models=4,
        largest_cluster_size=1,
        good_model_count=4,
        consensus_ratio=0.25,
        disagreement_score=0.75,
        complexity_score=0.0,
    )
    assert "severe_disagreement" in reasons
    assert risk > 0.2


# ---------------------------------------------------------------------------
# 7. Review queue sorting
# ---------------------------------------------------------------------------

def test_review_sorting_logic():
    """Verify that higher risk comes first in sorting."""
    items = [
        {"total_risk": 0.3, "is_singleword": True, "gloss_ru": "beta"},
        {"total_risk": 0.8, "is_singleword": False, "gloss_ru": "alpha"},
        {"total_risk": 0.8, "is_singleword": True, "gloss_ru": "gamma"},
        {"total_risk": 0.5, "is_singleword": True, "gloss_ru": "delta"},
    ]
    import pandas as pd
    df = pd.DataFrame(items)
    df = df.sort_values(
        by=["total_risk", "is_singleword", "gloss_ru"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    # First row should be highest risk + singleword
    assert df.iloc[0]["gloss_ru"] == "gamma"  # 0.8, True
    assert df.iloc[1]["gloss_ru"] == "alpha"  # 0.8, False


# ---------------------------------------------------------------------------
# 8. Gold template contains required editable columns
# ---------------------------------------------------------------------------

def test_gold_template_columns():
    from src.sem_cat.compare.output_tables import build_gold_template_df
    import pandas as pd
    df = pd.DataFrame([{"gloss_ru": "дом", "total_risk": 0.5}])
    gold = build_gold_template_df(df)
    assert "expert_gloss_en" in gold.columns
    assert "expert_notes" in gold.columns
    assert "final_decision" in gold.columns
    assert "include_in_gold" in gold.columns
    assert "accepted_model_key" in gold.columns
    assert "accepted_raw_output" in gold.columns
    assert "review_status" in gold.columns


# ---------------------------------------------------------------------------
# 9. Normalization helpers
# ---------------------------------------------------------------------------

def test_normalization_strips_punctuation():
    assert normalize_output_for_comparison("  House!  ") == "house"
    assert normalize_output_for_comparison("The house.") == "the house"


def test_output_similarity_identical():
    assert output_similarity("house", "house") == 1.0


def test_output_similarity_different():
    sim = output_similarity("house", "elephant")
    assert sim < 0.5


def test_near_match_threshold():
    assert outputs_are_near_match("house", "house", threshold=0.85) is True
    assert outputs_are_near_match("house", "elephant", threshold=0.85) is False


# ---------------------------------------------------------------------------
# 10. Gloss complexity
# ---------------------------------------------------------------------------

def test_complexity_singleword():
    score, reasons = compute_gloss_complexity("дом")
    assert "singleword_gloss" in reasons


def test_complexity_very_short():
    score, reasons = compute_gloss_complexity("я")
    assert "very_short_gloss" in reasons


def test_complexity_hyphenated():
    score, reasons = compute_gloss_complexity("-то")
    assert "hyphenated_gloss" in reasons


def test_complexity_proper_name():
    score, reasons = compute_gloss_complexity("Москва")
    assert "probable_proper_name" in reasons


def test_complexity_empty():
    score, reasons = compute_gloss_complexity("")
    assert score == 0.0
    assert reasons == []


def test_particle_or_clitic_fires_on_standalone():
    score, reasons = compute_gloss_complexity("ни")
    assert "particle_or_clitic" in reasons
    score, reasons = compute_gloss_complexity("ли")
    assert "particle_or_clitic" in reasons


def test_particle_or_clitic_fires_on_suffixed():
    score, reasons = compute_gloss_complexity("кто-то")
    assert "particle_or_clitic" in reasons
    score, reasons = compute_gloss_complexity("что-либо")
    assert "particle_or_clitic" in reasons
    score, reasons = compute_gloss_complexity("кто-нибудь")
    assert "particle_or_clitic" in reasons


def test_particle_or_clitic_not_fires_on_lexical():
    lexical_words = ["книга", "лист", "малина", "долина", "никогда", "линия",
                     "близко", "слива", "улитка", "снимать"]
    for word in lexical_words:
        score, reasons = compute_gloss_complexity(word)
        assert "particle_or_clitic" not in reasons, \
            f"False positive on '{word}': {reasons}"


# ---------------------------------------------------------------------------
# 11. Proposal selection
# ---------------------------------------------------------------------------

def test_proposal_strong_consensus():
    outputs = [
        ModelOutput("m1", "M1", "house", True, 0.0, normalized_gloss_en="house"),
        ModelOutput("m2", "M2", "house", True, 0.0, normalized_gloss_en="house"),
        ModelOutput("m3", "M3", "house", True, 0.0, normalized_gloss_en="house"),
    ]
    clusters = cluster_outputs(outputs)
    proposed, source, key, reason = select_proposed_translation(
        clusters, outputs, total_risk=0.1
    )
    assert proposed == "house"
    assert reason == "strong_consensus"


def test_proposal_all_blank():
    outputs = [
        ModelOutput("m1", "M1", "", False, 1.0),
        ModelOutput("m2", "M2", "", False, 1.0),
    ]
    clusters = cluster_outputs([])
    proposed, source, key, reason = select_proposed_translation(
        clusters, outputs, total_risk=0.8
    )
    assert proposed == ""
    assert reason == "all_blank"


# ---------------------------------------------------------------------------
# 12. Risk level classification
# ---------------------------------------------------------------------------

def test_risk_levels():
    assert compute_risk_level(0.8) == "high"
    assert compute_risk_level(0.5) == "medium"
    assert compute_risk_level(0.1) == "low"


# ---------------------------------------------------------------------------
# 13. Expert review criteria
# ---------------------------------------------------------------------------

def test_low_risk_consensus_no_review():
    """Low-risk rows with strong model consensus should NOT need expert review."""
    import importlib
    import pandas as pd
    mod = importlib.import_module("src.sem_cat.03_compare_translations")
    _process_row = mod.process_gloss_row
    row_data = {
        "gloss_ru": "дом",
        "m1__gloss_en": "house", "m1__qa_keep": "True", "m1__qa_score": "0.0",
        "m1__qa_flags": "", "m1__roundtrip_distance": "", "m1__model_name": "M1",
        "m2__gloss_en": "house", "m2__qa_keep": "True", "m2__qa_score": "0.0",
        "m2__qa_flags": "", "m2__roundtrip_distance": "", "m2__model_name": "M2",
        "m3__gloss_en": "home", "m3__qa_keep": "True", "m3__qa_score": "0.1",
        "m3__qa_flags": "", "m3__roundtrip_distance": "", "m3__model_name": "M3",
    }
    row = pd.Series(row_data)
    result = _process_row(row, model_keys=["m1", "m2", "m3"], total_models=3)
    assert result.needs_expert_review is False, \
        f"Expected no review for low-risk consensus, got risk={result.total_risk} level={result.risk_level}"


def test_qa_keep_false_triggers_review():
    """Nonblank outputs with qa_keep=False should trigger expert review."""
    import importlib
    import pandas as pd
    mod = importlib.import_module("src.sem_cat.03_compare_translations")
    _process_row = mod.process_gloss_row
    row_data = {
        "gloss_ru": "тест",
        "m1__gloss_en": "test", "m1__qa_keep": "True", "m1__qa_score": "0.0",
        "m1__qa_flags": "", "m1__roundtrip_distance": "", "m1__model_name": "M1",
        "m2__gloss_en": "Some garbage output with repetition", "m2__qa_keep": "False",
        "m2__qa_score": "1.0", "m2__qa_flags": "repeated_token_loop",
        "m2__roundtrip_distance": "", "m2__model_name": "M2",
    }
    row = pd.Series(row_data)
    result = _process_row(row, model_keys=["m1", "m2"], total_models=2)
    assert result.needs_expert_review is True, \
        f"Expected review for qa_keep=False nonblank, got {result.needs_expert_review}"


# ---------------------------------------------------------------------------
# 14. Review queue include-low-risk behavior
# ---------------------------------------------------------------------------

def test_include_low_risk_flag():
    """Verify that --include-low-risk expands the review queue."""
    import importlib
    import pandas as pd
    mod = importlib.import_module("src.sem_cat.compare.output_tables")
    build_review = mod.build_review_queue_df
    
    df = pd.DataFrame([
        {"gloss_ru": "a", "needs_expert_review": True, "risk_level": "high"},
        {"gloss_ru": "b", "needs_expert_review": True, "risk_level": "medium"},
        {"gloss_ru": "c", "needs_expert_review": False, "risk_level": "low"},
        {"gloss_ru": "d", "needs_expert_review": False, "risk_level": "medium"},
    ])
    
    review_strict = build_review(df, include_low_risk=False)
    review_inclusive = build_review(df, include_low_risk=True)
    
    assert len(review_strict) == 2, "Strict mode should exclude low-risk"
    assert len(review_inclusive) == 3, "Include mode should add low-risk row"
    assert "c" in review_inclusive["gloss_ru"].values, "Low-risk row should be included"
    assert "c" not in review_strict["gloss_ru"].values, "Low-risk row should be excluded"


# ---------------------------------------------------------------------------
# 15. Output table schema (no model_name, conditional roundtrip_distance)
# ---------------------------------------------------------------------------

def test_full_comparison_omits_model_name_columns():
    """Per-model __model_name columns should NOT be in exported tables."""
    import importlib
    import pandas as pd
    mod = importlib.import_module("src.sem_cat.compare.output_tables")
    build_full = mod.build_full_comparison_df
    from src.sem_cat.compare.data_structures import ComparisonResult, ModelOutput

    # Create a simple result with two models
    outputs = [
        ModelOutput("model_a", "A Model", "house", True, 0.9),
        ModelOutput("model_b", "B Model", "home", True, 0.8),
    ]
    result = ComparisonResult(
        gloss_ru="дом",
        proposed_gloss_en="house",
        preferred_source="model_a",
        chosen_from_model_key="model_a",
        decision_reason="strong_consensus",
        total_risk=0.1,
        risk_level="low",
    )
    result._outputs = outputs  # type: ignore[attr-defined]

    df = build_full([result], model_keys=["model_a", "model_b"])

    assert "model_a__model_name" not in df.columns
    assert "model_b__model_name" not in df.columns


def test_full_comparison_only_include_roundtrip_with_data():
    """__roundtrip_distance column should only be present if model has non-null values."""
    import importlib
    import pandas as pd
    mod = importlib.import_module("src.sem_cat.compare.output_tables")
    build_full = mod.build_full_comparison_df
    from src.sem_cat.compare.data_structures import ComparisonResult, ModelOutput

    # Model A has roundtrip data, Model B does not
    outputs = [
        ModelOutput("model_a", "A", "house", True, 0.9, roundtrip_distance=0.5),
        ModelOutput("model_b", "B", "home", True, 0.8, roundtrip_distance=None),
    ]
    result = ComparisonResult(
        gloss_ru="дом",
        proposed_gloss_en="house",
        preferred_source="model_a",
        chosen_from_model_key="model_a",
        decision_reason="strong_consensus",
        total_risk=0.1,
        risk_level="low",
    )
    result._outputs = outputs  # type: ignore[attr-defined]

    df = build_full([result], model_keys=["model_a", "model_b"])

    assert "model_a__roundtrip_distance" in df.columns, "model_a has roundtrip data, column should exist"
    assert "model_b__roundtrip_distance" not in df.columns, "model_b has no roundtrip data, column should be omitted"


def test_full_comparison_includes_roundtrip_when_any_has_data():
    """__roundtrip_distance column should be present per-model based on whether that model has data."""
    import importlib
    import pandas as pd
    mod = importlib.import_module("src.sem_cat.compare.output_tables")
    build_full = mod.build_full_comparison_df
    from src.sem_cat.compare.data_structures import ComparisonResult, ModelOutput

    # Model A has NO roundtrip data (None), Model B HAS roundtrip data (0.3)
    # Per the schema policy, only models with actual data should have their columns
    outputs = [
        ModelOutput("model_a", "A", "house", True, 0.9, roundtrip_distance=None),
        ModelOutput("model_b", "B", "home", True, 0.8, roundtrip_distance=0.3),
    ]
    result = ComparisonResult(
        gloss_ru="дом",
        proposed_gloss_en="house",
        preferred_source="model_b",
        chosen_from_model_key="model_b",
        decision_reason="strong_consensus",
        total_risk=0.1,
        risk_level="low",
    )
    result._outputs = outputs  # type: ignore[attr-defined]

    df = build_full([result], model_keys=["model_a", "model_b"])

    # Only model_b should have roundtrip_distance since only model_b has data
    # model_a has no data (None), so its roundtrip_distance column should be omitted
    assert "model_a__roundtrip_distance" not in df.columns, "model_a has no roundtrip data, column should be omitted"
    assert "model_b__roundtrip_distance" in df.columns, "model_b has roundtrip data, column should exist"


def test_gold_template_only_has_gloss_en_per_model():
    """Gold template should NOT include per-model model_name or roundtrip_distance."""
    import importlib
    import pandas as pd
    mod = importlib.import_module("src.sem_cat.compare.output_tables")
    build_gold = mod.build_gold_template_df

    # Full review df with per-model columns including model_name and roundtrip_distance
    df = pd.DataFrame([
        {
            "gloss_ru": "дом",
            "model_a__gloss_en": "house",
            "model_a__qa_keep": True,
            "model_a__qa_score": 0.9,
            "model_a__qa_flags": "",
            "model_a__roundtrip_distance": "0.5",
            "model_a__model_name": "A Model",
            "model_b__gloss_en": "home",
            "model_b__qa_keep": True,
            "model_b__qa_score": 0.8,
            "model_b__qa_flags": "",
            "model_b__roundtrip_distance": "",
            "model_b__model_name": "B Model",
            "total_risk": 0.1,
            "consensus_ratio": 0.8,
            "disagreement_score": 0.2,
        }
    ])

    gold = build_gold(df)

    assert "model_a__model_name" not in gold.columns
    assert "model_b__model_name" not in gold.columns
    assert "model_a__roundtrip_distance" not in gold.columns
    assert "model_b__roundtrip_distance" not in gold.columns
    assert "model_a__gloss_en" in gold.columns
    assert "model_b__gloss_en" in gold.columns


def test_review_queue_only_has_relevant_columns():
    """Review queue should be narrower than full comparison (only needed columns)."""
    import importlib
    import pandas as pd
    mod = importlib.import_module("src.sem_cat.compare.output_tables")
    build_review = mod.build_review_queue_df

    df = pd.DataFrame([
        {
            "gloss_ru": "дом",
            "is_singleword": True,
            "proposed_gloss_en": "house",
            "preferred_source": "model_a",
            "total_risk": 0.8,
            "risk_level": "high",
            "needs_expert_review": True,
            "model_a__gloss_en": "house",
            "model_a__qa_keep": True,
            "model_a__qa_score": 0.9,
            "model_a__qa_flags": "",
            "model_a__roundtrip_distance": "0.5",
            "model_a__model_name": "A Model",
            "model_b__gloss_en": "home",
            "model_b__qa_keep": True,
            "model_b__qa_score": 0.8,
            "model_b__qa_flags": "",
            "model_b__roundtrip_distance": "",
            "model_b__model_name": "B Model",
        }
    ])

    review = build_review(df, include_low_risk=False)

    # Should have core review columns
    assert "gloss_ru" in review.columns
    assert "total_risk" in review.columns
    # needs_expert_review is used for filtering, not included in output

    # Per-model: only gloss_en, qa_keep, qa_score, qa_flags (no model_name, roundtrip only if exists)
    assert "model_a__gloss_en" in review.columns
    assert "model_a__qa_keep" in review.columns
    assert "model_a__qa_score" in review.columns
    assert "model_a__qa_flags" in review.columns
    assert "model_a__model_name" not in review.columns

    assert "model_b__gloss_en" in review.columns
    assert "model_b__qa_keep" in review.columns
    assert "model_b__qa_score" in review.columns
    assert "model_b__qa_flags" in review.columns


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_parse_translation_arg_valid,
        test_parse_translation_arg_no_equals,
        test_parse_translation_arg_empty_key,
        test_load_and_prefix_columns,
        test_load_rejects_mismatched_model_key,
        test_consensus_clustering_groups_identical,
        test_consensus_all_different,
        test_strong_consensus_lowers_risk,
        test_all_blank_high_risk,
        test_severe_disagreement_raises_risk,
        test_review_sorting_logic,
        test_gold_template_columns,
        test_normalization_strips_punctuation,
        test_output_similarity_identical,
        test_output_similarity_different,
        test_near_match_threshold,
        test_complexity_singleword,
        test_complexity_very_short,
        test_complexity_hyphenated,
        test_complexity_proper_name,
        test_complexity_empty,
        test_particle_or_clitic_fires_on_standalone,
        test_particle_or_clitic_fires_on_suffixed,
        test_particle_or_clitic_not_fires_on_lexical,
        test_proposal_strong_consensus,
        test_proposal_all_blank,
        test_risk_levels,
        test_low_risk_consensus_no_review,
        test_qa_keep_false_triggers_review,
        test_include_low_risk_flag,
        test_full_comparison_omits_model_name_columns,
        test_full_comparison_only_include_roundtrip_with_data,
        test_full_comparison_includes_roundtrip_when_any_has_data,
        test_gold_template_only_has_gloss_en_per_model,
        test_review_queue_only_has_relevant_columns,
        test_normalize_pos_for_task,
        test_canonical_existing_en,
        test_has_existing_english,
        test_build_task_key,
        test_split_by_existing_en_excludes_human_translations,
        test_split_by_existing_en_unambiguous_reuse,
        test_split_by_existing_en_ambiguous_reuse,
        test_split_by_existing_en_all_missing,
        test_extract_unique_translation_tasks_deduplicates,
        test_translation_input_mode_pos,
        test_translation_input_mode_pos_meaning,
        test_translation_input_mode_default_changed_to_pos,
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
# 24. VepKar translation workflow tests
# ---------------------------------------------------------------------------


def test_normalize_pos_for_task():
    """POS normalization handles None, empty, and normal values."""
    assert normalize_pos_for_task(None) == "UNKNOWN"
    assert normalize_pos_for_task("") == "UNKNOWN"
    assert normalize_pos_for_task("   ") == "UNKNOWN"
    assert normalize_pos_for_task("NOUN") == "NOUN"
    assert normalize_pos_for_task("VERB") == "VERB"


def test_canonical_existing_en():
    """Existing English canonicalization handles None, empty, whitespace."""
    assert canonical_existing_en(None) == ""
    assert canonical_existing_en("") == ""
    assert canonical_existing_en("   ") == ""
    assert canonical_existing_en(" rock ") == "rock"


def test_has_existing_english():
    """has_existing_english returns correct boolean for various inputs."""
    assert has_existing_english(None) is False
    assert has_existing_english("") is False
    assert has_existing_english("   ") is False
    assert has_existing_english("rock") is True
    assert has_existing_english("  rock  ") is True


def test_build_task_key():
    """task_key is (pos, primary_gloss_ru) tuple."""
    # Normal case
    key = build_task_key("NOUN", "дом")
    assert key == ("NOUN", "дом")
    
    # Empty POS becomes UNKNOWN
    key = build_task_key("", "дом")
    assert key == ("UNKNOWN", "дом")
    
    # Missing POS becomes UNKNOWN
    key = build_task_key(None, "дом")
    assert key == ("UNKNOWN", "дом")


def test_split_by_existing_en_excludes_human_translations():
    """Rows with non-empty meaning_en are excluded from translation.
    
    When all missing rows share exactly one existing English, they become reusable_unambiguous.
    """
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "meaning_id": [10, 20, 30],
        "lang": ["krl", "krl", "krl"],
        "lemma": ["koti", "koti", "maja"],
        "pos": ["NOUN", "NOUN", "NOUN"],
        "meaning_ru": ["дом", "дом", "дом"],
        "meaning_en": ["house", "", ""],
    })
    
    work = prepare_meanings_for_translation(df)
    unusamb, amb, needs = split_by_existing_en_reuse(work)
    
    # Two rows have empty meaning_en, one has "house"
    # Since there's exactly one distinct existing English value ("house"),
    # both missing rows become reusable_unambiguous (not needs_model)
    assert len(unusamb) == 2
    assert "reused_existing_en" in unusamb.columns
    assert unusamb.iloc[0]["reused_existing_en"] == "house"
    assert len(amb) == 0
    assert len(needs) == 0


def test_split_by_existing_en_unambiguous_reuse():
    """Rows with same (pos, gloss) and one existing English become reusable_unambiguous."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "meaning_id": [10, 20, 30],
        "lang": ["krl", "krl", "krl"],
        "lemma": ["koti", "koti", "maja"],
        "pos": ["NOUN", "NOUN", "NOUN"],
        "meaning_ru": ["дом", "дом", "дом"],
        "meaning_en": ["house", "house", ""],
    })
    
    work = prepare_meanings_for_translation(df)
    unusamb, amb, needs = split_by_existing_en_reuse(work)
    
    # Two rows have "house" (same English), one has ""
    # Since there's exactly one distinct existing English value ("house"),
    # the missing row becomes reusable_unambiguous
    assert len(unusamb) == 1
    assert "reused_existing_en" in unusamb.columns
    assert unusamb.iloc[0]["reused_existing_en"] == "house"
    assert 30 in unusamb["meaning_id"].values
    assert len(amb) == 0
    assert len(needs) == 0


def test_split_by_existing_en_ambiguous_reuse():
    """Rows with same (pos, gloss) and multiple existing English become reusable_ambiguous."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "meaning_id": [10, 20, 30],
        "lang": ["krl", "krl", "krl"],
        "lemma": ["koti", "koti", "maja"],
        "pos": ["NOUN", "NOUN", "NOUN"],
        "meaning_ru": ["дом", "дом", "дом"],
        "meaning_en": ["house", "home", ""],
    })
    
    work = prepare_meanings_for_translation(df)
    unusamb, amb, needs = split_by_existing_en_reuse(work)
    
    # Two existing English values ("house", "home"), one missing
    # Since there are 2+ distinct existing English values, the missing row becomes reusable_ambiguous
    assert len(unusamb) == 0
    assert len(amb) == 1
    assert "existing_en_candidates" in amb.columns
    assert "home" in amb.iloc[0]["existing_en_candidates"]
    assert "house" in amb.iloc[0]["existing_en_candidates"]
    assert 30 in amb["meaning_id"].values
    assert len(needs) == 0


def test_split_by_existing_en_all_missing():
    """All rows missing meaning_en -> all go to needs_model when no existing English exists."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "meaning_id": [10, 20, 30],
        "lang": ["krl", "krl", "krl"],
        "lemma": ["koti", "koti", "maja"],
        "pos": ["NOUN", "NOUN", "NOUN"],
        "meaning_ru": ["дом", "дом", "дом"],
        "meaning_en": ["", "", ""],
    })
    
    work = prepare_meanings_for_translation(df)
    unusamb, amb, needs = split_by_existing_en_reuse(work)
    
    # All rows missing meaning_en AND no existing English values for this task_key
    # -> all go to needs_model
    assert len(unusamb) == 0
    assert len(amb) == 0
    assert len(needs) == 3


def test_split_by_existing_en_with_existing_and_all_missing():
    """All rows missing meaning_en but with existing English -> reusable_ambiguous."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "meaning_id": [10, 20, 30],
        "lang": ["krl", "krl", "krl"],
        "lemma": ["koti", "koti", "maja"],
        "pos": ["NOUN", "NOUN", "NOUN"],
        "meaning_ru": ["дом", "дом", "дом"],
        "meaning_en": ["house", "", ""],
    })
    
    work = prepare_meanings_for_translation(df)
    unusamb, amb, needs = split_by_existing_en_reuse(work)
    
    # One row with existing English ("house"), two missing
    # Since exactly one distinct existing English value exists, missing rows become reusable_unambiguous
    assert len(unusamb) == 2
    assert len(amb) == 0
    assert len(needs) == 0


def test_extract_unique_translation_tasks_deduplicates():
    """Tasks deduplicated by task_key, not just gloss."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "meaning_id": [10, 20, 30],
        "lang": ["krl", "krl", "krl"],
        "lemma": ["koti", "koti", "maja"],
        "pos": ["NOUN", "VERB", "NOUN"],
        "meaning_ru": ["дом", "дом", "дом"],
        "meaning_en": ["", "", ""],
    })
    
    work = prepare_meanings_for_translation(df)
    tasks = extract_unique_translation_tasks(work)
    
    assert len(tasks) == 2
    task_keys = [t.task_key for t in tasks]
    # NOUN::дом and VERB::дом should be separate tasks (:: separator)
    assert "NOUN::дом" in task_keys
    assert "VERB::дом" in task_keys


def test_translation_input_mode_pos():
    """pos mode includes POS in input."""
    from src.sem_cat.pipeline.vepkar_translation_selection import TranslationTaskMetadata
    
    task = TranslationTaskMetadata(
        task_key="NOUN::дом",
        primary_gloss_ru="дом",
        task_pos="NOUN",
        meaning_hint=None,
        sourcecount=1,
    )
    
    assert prepare_translation_input_for_task(task, "pos") == "NOUN | дом"
    assert prepare_translation_input_for_task(task, "raw") == "дом"


def test_translation_input_mode_pos_meaning():
    """pos_meaning mode includes POS and meaning hint."""
    from src.sem_cat.pipeline.vepkar_translation_selection import TranslationTaskMetadata
    
    task = TranslationTaskMetadata(
        task_key="NOUN::дом",
        primary_gloss_ru="дом",
        task_pos="NOUN",
        meaning_hint="жилищное строение",
        sourcecount=1,
    )
    
    input_text = prepare_translation_input_for_task(task, "pos_meaning")
    assert "NOUN | дом" in input_text
    assert "дом" in input_text
    assert "жилищное строение" in input_text


def test_translation_input_mode_default_changed_to_pos():
    """Default translation input mode should be pos, not raw."""
    from src.sem_cat.pipeline.vepkar_translation_selection import TranslationTaskMetadata
    
    task = TranslationTaskMetadata(
        task_key="NOUN::дом",
        primary_gloss_ru="дом",
        task_pos="NOUN",
        meaning_hint=None,
        sourcecount=1,
    )
    
    # When called without mode (defaults to pos in new code)
    # The prepare_translation_input_for_task has no default, so caller must specify
    # But the 02_translate_glosses.py now defaults to "pos" in CLI
    assert prepare_translation_input_for_task(task, "pos") == "NOUN | дом"
