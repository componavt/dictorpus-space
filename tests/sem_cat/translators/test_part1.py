"""Part 1 tests for the translation architecture.

Lightweight checks that do not require downloading models or GPU.
Run with: python3 -m pytest tests/sem_cat/translators/test_part1.py -v
"""

import sys
import pathlib

# Add project root to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent))

import pytest

from src.sem_cat.translators.model_registry import (
    MODEL_REGISTRY,
    ModelSpec,
    get_model_spec,
    list_model_keys,
    resolve_legacy_args_to_model_key,
)
from src.sem_cat.translators.generation_presets import (
    GLOSS_STRICT_PRESET,
    DEFAULT_PRESET,
    get_generation_preset,
)
from src.sem_cat.translators.base import Translator
from src.sem_cat.translators.google_translator import GoogleTranslator
from src.sem_cat.translators.factory import build_translator, build_reverse_translator


# ---------------------------------------------------------------------------
# 1. Model registry contains all 6 model keys
# ---------------------------------------------------------------------------

EXPECTED_KEYS = [
    "google",
    "helsinki_opus_mt_ru_en",
    "nllb_distilled_1_3b",
    "nllb_1_3b",
    "nllb_3_3b",
    "wmt19_ru_en",
]


class TestModelRegistry:
    def test_all_six_keys_present(self):
        keys = list_model_keys()
        for key in EXPECTED_KEYS:
            assert key in keys, f"Missing model key: {key}"

    def test_no_extra_keys(self):
        keys = list_model_keys()
        assert len(keys) == 6, f"Expected 6 keys, got {len(keys)}: {keys}"

    def test_get_model_spec_returns_spec(self):
        for key in EXPECTED_KEYS:
            spec = get_model_spec(key)
            assert isinstance(spec, ModelSpec)
            assert spec.model_key == key

    def test_get_model_spec_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown model key"):
            get_model_spec("nonexistent_model")

    def test_backend_families(self):
        assert get_model_spec("google").backend_family == "google"
        assert get_model_spec("helsinki_opus_mt_ru_en").backend_family == "hf_seq2seq"
        assert get_model_spec("nllb_distilled_1_3b").backend_family == "nllb"
        assert get_model_spec("nllb_1_3b").backend_family == "nllb"
        assert get_model_spec("nllb_3_3b").backend_family == "nllb"
        assert get_model_spec("wmt19_ru_en").backend_family == "hf_seq2seq"

    def test_wmt19_no_roundtrip(self):
        spec = get_model_spec("wmt19_ru_en")
        assert spec.supports_roundtrip is False
        assert spec.reverse_model_name is None

    def test_google_supports_roundtrip(self):
        spec = get_model_spec("google")
        assert spec.supports_roundtrip is True
        assert spec.reverse_model_name == "google"

    def test_nllb_specs_have_reverse(self):
        for key in ["nllb_distilled_1_3b", "nllb_1_3b", "nllb_3_3b"]:
            spec = get_model_spec(key)
            assert spec.supports_roundtrip is True
            assert spec.reverse_model_name is not None
            assert spec.reverse_src_lang is not None
            assert spec.reverse_tgt_lang is not None


# ---------------------------------------------------------------------------
# 2. Legacy resolver maps old CLI args correctly
# ---------------------------------------------------------------------------

class TestLegacyResolver:
    def test_google(self):
        assert resolve_legacy_args_to_model_key("google", None) == "google"

    def test_marian(self):
        assert resolve_legacy_args_to_model_key("marian", None) == "helsinki_opus_mt_ru_en"

    def test_nllb_distilled(self):
        assert resolve_legacy_args_to_model_key(
            "nllb", "facebook/nllb-200-distilled-1.3B"
        ) == "nllb_distilled_1_3b"

    def test_nllb_1_3b(self):
        assert resolve_legacy_args_to_model_key(
            "nllb", "facebook/nllb-200-1.3B"
        ) == "nllb_1_3b"

    def test_nllb_3_3b(self):
        assert resolve_legacy_args_to_model_key(
            "nllb", "facebook/nllb-200-3.3B"
        ) == "nllb_3_3b"

    def test_nllb_default(self):
        # Default NLLB model should resolve to distilled
        assert resolve_legacy_args_to_model_key("nllb", None) == "nllb_distilled_1_3b"

    def test_unknown_nllb_model(self):
        with pytest.raises(ValueError, match="Unknown NLLB model"):
            resolve_legacy_args_to_model_key("nllb", "unknown/model")

    def test_default_fallback(self):
        # No backend specified -> default to helsinki
        assert resolve_legacy_args_to_model_key(None, None) == "helsinki_opus_mt_ru_en"


# ---------------------------------------------------------------------------
# 3. Translator factory returns the correct class family
# ---------------------------------------------------------------------------

class TestFactory:
    def test_google_factory(self):
        spec = get_model_spec("google")
        translator = build_translator(spec)
        assert isinstance(translator, GoogleTranslator)
        assert translator.model_key == "google"

    def test_google_reverse_factory(self):
        spec = get_model_spec("google")
        reverse = build_reverse_translator(spec)
        assert reverse is not None
        assert isinstance(reverse, GoogleTranslator)

    def test_wmt19_no_reverse(self):
        spec = get_model_spec("wmt19_ru_en")
        reverse = build_reverse_translator(spec)
        assert reverse is None

    def test_nllb_reverse_factory(self):
        spec = get_model_spec("nllb_distilled_1_3b")
        # This will try to load the model; skip if torch not available
        try:
            reverse = build_reverse_translator(spec)
            assert reverse is not None
            assert reverse.model_key == "nllb_distilled_1_3b_reverse"
        except ImportError:
            pytest.skip("PyTorch not available")


# ---------------------------------------------------------------------------
# 4. HFSeq2SeqTranslator accepts custom generation kwargs
# ---------------------------------------------------------------------------

class TestHFSeq2SeqTranslator:
    def test_accepts_custom_kwargs(self):
        """Verify that HFSeq2SeqTranslator can be constructed with custom generation kwargs."""
        from src.sem_cat.translators.hf_seq2seq_translator import HFSeq2SeqTranslator
        # We can't actually load the model without downloading it,
        # but we can verify the constructor accepts the parameter.
        # This test will be skipped if transformers isn't installed.
        try:
            # Just verify the import works and class exists
            assert HFSeq2SeqTranslator is not None
            # Verify the __init__ signature accepts generation_kwargs
            import inspect
            sig = inspect.signature(HFSeq2SeqTranslator.__init__)
            assert "generation_kwargs" in sig.parameters
        except ImportError:
            pytest.skip("transformers not installed")


# ---------------------------------------------------------------------------
# 5. NLLBTranslator builds generation parameters without conflict
# ---------------------------------------------------------------------------

class TestNLLBGeneration:
    def test_generation_config_max_length_cleared(self):
        """Verify that NLLBTranslator clears max_length from generation config."""
        from src.sem_cat.translators.nllb_translator import NLLBTranslator
        try:
            translator = NLLBTranslator(
                model_key="nllb_distilled_1_3b",
                model_name="facebook/nllb-200-distilled-1.3B",
            )
            # The generation config should have max_length set to 20
            # to avoid conflict with max_new_tokens
            assert translator.model.generation_config.max_length == 20
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_gloss_strict_preset_has_max_new_tokens(self):
        """Verify the gloss_strict preset uses max_new_tokens, not max_length."""
        preset = get_generation_preset("gloss_strict")
        assert "max_new_tokens" in preset
        assert "max_length" not in preset


# ---------------------------------------------------------------------------
# 6. GoogleTranslator returns None on final failure instead of ""
# ---------------------------------------------------------------------------

class TestGoogleTranslator:
    def test_returns_none_on_empty_input(self):
        translator = GoogleTranslator()
        assert translator.translate("") is None
        assert translator.translate("   ") is None

    def test_batch_size_attribute(self):
        translator = GoogleTranslator()
        assert translator.default_batch_size == 1

    def test_model_key_attribute(self):
        translator = GoogleTranslator()
        assert translator.model_key == "google"
        assert translator.model_name == "google"
        assert translator.supports_roundtrip is True


# ---------------------------------------------------------------------------
# 7. Generation presets
# ---------------------------------------------------------------------------

class TestGenerationPresets:
    def test_gloss_strict_preset(self):
        preset = get_generation_preset("gloss_strict")
        assert preset["max_new_tokens"] == 12
        assert preset["num_beams"] == 4
        assert preset["do_sample"] is False

    def test_default_preset(self):
        preset = get_generation_preset("default")
        assert preset["max_new_tokens"] == 64

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown generation preset"):
            get_generation_preset("nonexistent")

    def test_returns_copy(self):
        preset1 = get_generation_preset("gloss_strict")
        preset2 = get_generation_preset("gloss_strict")
        assert preset1 is not preset2


# ---------------------------------------------------------------------------
# 8. Base translator API
# ---------------------------------------------------------------------------

class TestBaseTranslator:
    def test_translate_returns_optional_str(self):
        """Verify that translate() returns str | None."""
        from src.sem_cat.translators.google_translator import GoogleTranslator
        translator = GoogleTranslator()
        result = translator.translate("")
        assert result is None

    def test_default_batch_implementation(self):
        """Verify that default translate_batch uses translate() in a loop."""
        from src.sem_cat.translators.google_translator import GoogleTranslator
        translator = GoogleTranslator()
        results = translator.translate_batch(["", "test", ""])
        assert len(results) == 3
        assert results[0] is None
        assert results[2] is None
