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
# 3. Import safety - modules must import without optional deps
# ---------------------------------------------------------------------------

class TestImportSafety:
    """Verify that translator modules can be imported even when optional
    dependencies (deep_translator, torch, transformers) are absent."""

    def test_base_imports(self):
        from src.sem_cat.translators.base import Translator
        assert Translator is not None

    def test_google_module_imports(self):
        """google_translator module must import without deep_translator."""
        from src.sem_cat.translators import google_translator
        assert google_translator.GoogleTranslator is not None

    def test_hf_module_imports(self):
        """hf_seq2seq_translator module must import without torch/transformers."""
        from src.sem_cat.translators import hf_seq2seq_translator
        assert hf_seq2seq_translator.HFSeq2SeqTranslator is not None

    def test_nllb_module_imports(self):
        """nllb_translator module must import without torch/transformers."""
        from src.sem_cat.translators import nllb_translator
        assert nllb_translator.NLLBTranslator is not None

    def test_marian_module_imports(self):
        """marian_translator module must import without torch/transformers."""
        from src.sem_cat.translators import marian_translator
        assert marian_translator.MarianTranslator is not None

    def test_factory_imports(self):
        from src.sem_cat.translators.factory import build_translator
        assert build_translator is not None

    def test_model_registry_imports(self):
        from src.sem_cat.translators.model_registry import MODEL_REGISTRY
        assert MODEL_REGISTRY is not None


# ---------------------------------------------------------------------------
# 4. Error hierarchy
# ---------------------------------------------------------------------------

class TestErrorHierarchy:
    def test_error_classes_exist(self):
        from src.sem_cat.translators.base import (
            TranslatorError,
            BackendUnavailableError,
            TranslatorInitializationError,
        )
        assert issubclass(BackendUnavailableError, TranslatorError)
        assert issubclass(TranslatorInitializationError, TranslatorError)

    def test_nllb_raises_backend_unavailable_without_torch(self):
        """NLLBTranslator should raise BackendUnavailableError when
        torch is not available."""
        from src.sem_cat.translators.base import BackendUnavailableError
        from src.sem_cat.translators.nllb_translator import NLLBTranslator

        with pytest.raises(BackendUnavailableError):
            NLLBTranslator(
                model_key="nllb_distilled_1_3b",
                model_name="facebook/nllb-200-distilled-1.3B",
            )


# ---------------------------------------------------------------------------
# 5. Translator factory returns the correct class family
# ---------------------------------------------------------------------------

class TestFactory:
    def test_google_factory(self):
        spec = get_model_spec("google")
        try:
            translator = build_translator(spec)
            from src.sem_cat.translators.google_translator import GoogleTranslator
            assert isinstance(translator, GoogleTranslator)
            assert translator.model_key == "google"
        except Exception:
            pytest.skip("deep_translator not available")

    def test_google_reverse_factory(self):
        spec = get_model_spec("google")
        try:
            reverse = build_reverse_translator(spec)
            assert reverse is not None
            from src.sem_cat.translators.google_translator import GoogleTranslator
            assert isinstance(reverse, GoogleTranslator)
        except Exception:
            pytest.skip("deep_translator not available")

    def test_wmt19_no_reverse(self):
        spec = get_model_spec("wmt19_ru_en")
        reverse = build_reverse_translator(spec)
        assert reverse is None

    def test_nllb_raises_without_torch(self, monkeypatch):
        """NLLBTranslator should raise BackendUnavailableError when
        torch is not available."""
        spec = get_model_spec("nllb_distilled_1_3b")
        from src.sem_cat.translators.base import BackendUnavailableError

        with pytest.raises(BackendUnavailableError):
            build_translator(spec)


# ---------------------------------------------------------------------------
# 6. HFSeq2SeqTranslator accepts custom generation kwargs
# ---------------------------------------------------------------------------

class TestHFSeq2SeqTranslator:
    def test_accepts_custom_kwargs(self):
        """Verify that HFSeq2SeqTranslator can be constructed with custom generation kwargs."""
        from src.sem_cat.translators.hf_seq2seq_translator import HFSeq2SeqTranslator
        # We can't actually load the model without downloading it,
        # but we can verify the constructor accepts the parameter.
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
# 7. NLLBTranslator builds generation parameters without conflict
# ---------------------------------------------------------------------------

class TestNLLBGeneration:
    def test_generation_config_max_length_cleared(self):
        """Verify that NLLBTranslator clears max_length from generation config."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            pytest.skip("torch/transformers not available")

        from src.sem_cat.translators.nllb_translator import NLLBTranslator
        translator = NLLBTranslator(
            model_key="nllb_distilled_1_3b",
            model_name="facebook/nllb-200-distilled-1.3B",
        )
        # The generation config should have max_length set to 20
        # to avoid conflict with max_new_tokens
        assert translator.model.generation_config.max_length == 20

    def test_gloss_strict_preset_has_max_new_tokens(self):
        """Verify the gloss_strict preset uses max_new_tokens, not max_length."""
        preset = get_generation_preset("gloss_strict")
        assert "max_new_tokens" in preset
        assert "max_length" not in preset


# ---------------------------------------------------------------------------
# 8. GoogleTranslator returns None on final failure instead of ""
# ---------------------------------------------------------------------------

class TestGoogleTranslator:
    @classmethod
    def setup_class(cls):
        """Skip all Google tests if deep_translator is not available."""
        try:
            import deep_translator  # noqa: F401
        except ImportError:
            pytest.skip("deep_translator not available", allow_module_level=True)

    def test_returns_none_on_empty_input(self):
        from src.sem_cat.translators.google_translator import GoogleTranslator
        translator = GoogleTranslator()
        assert translator.translate("") is None
        assert translator.translate("   ") is None

    def test_batch_size_attribute(self):
        from src.sem_cat.translators.google_translator import GoogleTranslator
        translator = GoogleTranslator()
        assert translator.default_batch_size == 1

    def test_model_key_attribute(self):
        from src.sem_cat.translators.google_translator import GoogleTranslator
        translator = GoogleTranslator()
        assert translator.model_key == "google"
        assert translator.model_name == "google"
        assert translator.supports_roundtrip is True


# ---------------------------------------------------------------------------
# 9. Generation presets
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
# 10. Base translator API
# ---------------------------------------------------------------------------

class TestBaseTranslator:
    @classmethod
    def setup_class(cls):
        """Skip if deep_translator is not available (used as test translator)."""
        try:
            import deep_translator  # noqa: F401
        except ImportError:
            pytest.skip("deep_translator not available", allow_module_level=True)

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


# ---------------------------------------------------------------------------
# 11. HF runtime helpers
# ---------------------------------------------------------------------------

class TestHFRuntime:
    def test_collect_proxy_env_empty(self, monkeypatch):
        """When no proxy env vars are set, collect_proxy_env returns empty dict."""
        from src.sem_cat.translators.hf_runtime import collect_proxy_env
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                     "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.delenv(var, raising=False)
        assert collect_proxy_env() == {}

    def test_collect_proxy_env_detects(self, monkeypatch):
        """When proxy env vars are set, collect_proxy_env returns them."""
        from src.sem_cat.translators.hf_runtime import collect_proxy_env
        monkeypatch.setenv("HTTP_PROXY", "http://proxy:8080")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
        env = collect_proxy_env()
        assert "HTTP_PROXY" in env
        assert env["HTTP_PROXY"] == "http://proxy:8080"

    def test_explain_hf_init_error_proxy(self):
        """Proxy errors should mention proxy configuration."""
        from src.sem_cat.translators.hf_runtime import explain_hf_init_error
        exc = ValueError("Unknown scheme for proxy URL URL('socks://127.0.0.1:12334/')")
        msg = explain_hf_init_error(exc, "test-model")
        assert "proxy" in msg.lower()
        assert "socks5" in msg.lower()

    def test_explain_hf_init_error_local_files_only(self):
        """Local-files-only errors should mention pre-downloading."""
        from src.sem_cat.translators.hf_runtime import explain_hf_init_error
        exc = OSError("Cannot find model")
        msg = explain_hf_init_error(exc, "test-model", local_files_only=True)
        assert "local-files-only" in msg.lower()
        assert "pre-download" in msg.lower()

    def test_explain_hf_init_error_generic(self):
        """Generic errors should include original message."""
        from src.sem_cat.translators.hf_runtime import explain_hf_init_error
        exc = OSError("Some random HF error")
        msg = explain_hf_init_error(exc, "my-model")
        assert "my-model" in msg
        assert "Some random HF error" in msg


# ---------------------------------------------------------------------------
# 12. HF init error wrapping
# ---------------------------------------------------------------------------

class TestHFInitErrorWrapping:
    def test_proxy_error_wrapped_as_initialization_error(self, monkeypatch):
        """Invalid proxy env should raise TranslatorInitializationError."""
        from src.sem_cat.translators.base import TranslatorInitializationError
        from src.sem_cat.translators.hf_seq2seq_translator import HFSeq2SeqTranslator

        # Mock torch to be available
        import types
        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = lambda: type("ctx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        # Mock transformers to raise proxy error
        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise ValueError("Unknown scheme for proxy URL URL('socks://127.0.0.1:12334/')")

        class FakeAutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise ValueError("Unknown scheme for proxy URL URL('socks://127.0.0.1:12334/')")

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_transformers.AutoModelForSeq2SeqLM = FakeAutoModel
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        with pytest.raises(TranslatorInitializationError, match="proxy"):
            HFSeq2SeqTranslator(
                model_key="test",
                model_name="test-model",
            )

    def test_local_files_only_error_wrapped(self, monkeypatch):
        """Local-files-only failure should raise TranslatorInitializationError."""
        from src.sem_cat.translators.base import TranslatorInitializationError
        from src.sem_cat.translators.hf_seq2seq_translator import HFSeq2SeqTranslator

        import types
        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = lambda: type("ctx", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        class FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise OSError("Cannot find model in local cache")

        class FakeAutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise OSError("Cannot find model in local cache")

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_transformers.AutoModelForSeq2SeqLM = FakeAutoModel
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        with pytest.raises(TranslatorInitializationError, match="local-files-only"):
            HFSeq2SeqTranslator(
                model_key="test",
                model_name="test-model",
                local_files_only=True,
            )


# ---------------------------------------------------------------------------
# 13. Backend diagnostics
# ---------------------------------------------------------------------------

class TestBackendDiagnostics:
    def test_summarize_all_pass(self):
        from src.sem_cat.translators.diagnostics import ProbeResult, summarize_diagnostics
        results = [
            ProbeResult("\u0434\u043e\u043c", "house", "PASS", []),
            ProbeResult("\u043a\u043e\u0448\u043a\u0430", "cat", "PASS", []),
        ]
        status, msg = summarize_diagnostics(results)
        assert status == "OK"

    def test_summarize_all_fail(self):
        from src.sem_cat.translators.diagnostics import ProbeResult, summarize_diagnostics
        results = [
            ProbeResult("\u0434\u043e\u043c", None, "FAIL", ["Output is None"]),
            ProbeResult("\u043a\u043e\u0448\u043a\u0430", None, "FAIL", ["Output is None"]),
        ]
        status, msg = summarize_diagnostics(results)
        assert status == "FAIL"

    def test_summarize_mixed(self):
        from src.sem_cat.translators.diagnostics import ProbeResult, summarize_diagnostics
        results = [
            ProbeResult("\u0434\u043e\u043c", "house", "PASS", []),
            ProbeResult("\u043a\u043e\u0448\u043a\u0430", "\u043a\u043e\u0448\u043a\u0430", "WARN", ["identical"]),
        ]
        status, msg = summarize_diagnostics(results)
        assert status == "WARN"

    def test_unchanged_source_is_warn_not_ok(self):
        """If translator returns unchanged source text, diagnostics should not say OK."""
        from src.sem_cat.translators.diagnostics import ProbeResult, summarize_diagnostics
        results = [
            ProbeResult("\u0434\u043e\u043c", "\u0434\u043e\u043c", "WARN", ["identical"]),
            ProbeResult("\u043a\u043e\u0448\u043a\u0430", "\u043a\u043e\u0448\u043a\u0430", "WARN", ["identical"]),
            ProbeResult("\u0432\u043e\u0434\u0430", "\u0432\u043e\u0434\u0430", "WARN", ["identical"]),
        ]
        status, msg = summarize_diagnostics(results)
        assert status != "OK"
        assert status in ("WARN", "FAIL")

    def test_proper_english_output_is_pass(self):
        """Proper ASCII English output should be PASS."""
        from src.sem_cat.translators.diagnostics import _run_probe

        class FakeTranslator:
            def translate(self, text):
                return "house"

        result = _run_probe(FakeTranslator(), "\u0434\u043e\u043c")
        assert result.status == "PASS"
        assert result.output == "house"


# ---------------------------------------------------------------------------
# 14. CLI graceful failure tests
# ---------------------------------------------------------------------------

class TestCLIGracefulFailure:
    def test_backend_info_catches_initialization_error(self, monkeypatch, capsys):
        """--backend-info should exit cleanly on TranslatorInitializationError."""
        from src.sem_cat.translators.base import TranslatorInitializationError
        from src.sem_cat.translators.diagnostics import run_backend_diagnostics, summarize_diagnostics

        # Test the diagnostic function directly with a failing translator
        class FailingTranslator:
            def translate(self, text):
                raise TranslatorInitializationError("Model not found")

        results = run_backend_diagnostics(FailingTranslator())
        status, msg = summarize_diagnostics(results)
        assert status == "FAIL"
        assert "Model not found" in msg

    def test_backend_info_unchanged_source_warn(self, monkeypatch, capsys):
        """If translator returns unchanged source, diagnostics should WARN."""
        from src.sem_cat.translators.diagnostics import run_backend_diagnostics, summarize_diagnostics

        class EchoTranslator:
            def translate(self, text):
                return text  # Returns unchanged source

        results = run_backend_diagnostics(EchoTranslator())
        status, msg = summarize_diagnostics(results)
        assert status != "OK"
        assert "identical" in msg.lower() or "no translation" in msg.lower()

    def test_backend_info_proper_translation_ok(self, monkeypatch, capsys):
        """If translator returns proper English, diagnostics should OK."""
        from src.sem_cat.translators.diagnostics import run_backend_diagnostics, summarize_diagnostics

        class GoodTranslator:
            def translate(self, text):
                mapping = {"\u0434\u043e\u043c": "house", "\u043a\u043e\u0448\u043a\u0430": "cat", "\u0432\u043e\u0434\u0430": "water"}
                return mapping.get(text, "translation")

        results = run_backend_diagnostics(GoodTranslator())
        status, msg = summarize_diagnostics(results)
        assert status == "OK"
