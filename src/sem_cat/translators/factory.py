"""Translator factory.

Builds translator instances from ModelSpec definitions.
All backend-specific instantiation logic lives inside the translator classes;
the factory simply dispatches to the correct constructor.
"""

from __future__ import annotations

from .base import Translator
from .generation_presets import get_generation_preset
from .model_registry import ModelSpec


def build_translator(
    spec: ModelSpec,
    device: str = "cpu",
    retry: int = 3,
    delay: float = 1.0,
    local_files_only: bool = False,
    cache_dir: str | None = None,
) -> Translator:
    """Build a translator from a ModelSpec.

    Args:
        spec: Model specification from the registry.
        device: "cpu" or "cuda" for local models.
        retry: Number of retries for Google backend.
        delay: Delay between retries for Google backend.
        local_files_only: If True, HF/NLLB models only use local cache.
        cache_dir: Optional custom cache directory for HF/NLLB models.

    Returns:
        Translator instance.

    Raises:
        ValueError: If backend_family is unknown.
        BackendUnavailableError: If required dependencies are missing.
        TranslatorInitializationError: If model/client loading fails.
    """
    gen_kwargs = get_generation_preset(spec.generation_preset)

    if spec.backend_family == "google":
        from .google_translator import GoogleTranslator

        return GoogleTranslator(
            source=spec.src_lang,
            target=spec.tgt_lang,
            retry=retry,
            delay=delay,
            model_key=spec.model_key,
            model_name=spec.model_name,
        )

    if spec.backend_family == "hf_seq2seq":
        from .hf_seq2seq_translator import HFSeq2SeqTranslator

        return HFSeq2SeqTranslator(
            model_key=spec.model_key,
            model_name=spec.model_name,
            device=device,
            tokenizer_max_length=spec.tokenizer_max_length,
            default_batch_size=spec.default_batch_size,
            generation_kwargs=gen_kwargs,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

    if spec.backend_family == "nllb":
        from .nllb_translator import NLLBTranslator

        return NLLBTranslator(
            model_key=spec.model_key,
            model_name=spec.model_name,
            src_lang=spec.src_lang,
            tgt_lang=spec.tgt_lang,
            device=device,
            tokenizer_max_length=spec.tokenizer_max_length,
            default_batch_size=spec.default_batch_size,
            generation_kwargs=gen_kwargs,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

    raise ValueError(f"Unknown backend family: {spec.backend_family!r}")


def build_reverse_translator(
    spec: ModelSpec,
    device: str = "cpu",
    retry: int = 3,
    delay: float = 1.0,
    local_files_only: bool = False,
    cache_dir: str | None = None,
) -> Translator | None:
    """Build a reverse translator for back-translation.

    Returns None if reverse translation is not supported for this model.

    Args:
        spec: Model specification from the registry.
        device: "cpu" or "cuda" for local models.
        retry: Number of retries for Google backend.
        delay: Delay between retries for Google backend.
        local_files_only: If True, HF/NLLB models only use local cache.
        cache_dir: Optional custom cache directory for HF/NLLB models.

    Returns:
        Reverse translator instance, or None if unsupported.
    """
    if not spec.supports_roundtrip:
        return None

    if spec.reverse_model_name is None:
        return None

    gen_kwargs = get_generation_preset(spec.generation_preset)

    if spec.backend_family == "google":
        from .google_translator import GoogleTranslator

        return GoogleTranslator(
            source=spec.reverse_src_lang or spec.tgt_lang,
            target=spec.reverse_tgt_lang or spec.src_lang,
            retry=retry,
            delay=delay,
            model_key=f"{spec.model_key}_reverse",
            model_name=spec.reverse_model_name,
        )

    if spec.backend_family == "hf_seq2seq":
        from .hf_seq2seq_translator import HFSeq2SeqTranslator

        return HFSeq2SeqTranslator(
            model_key=f"{spec.model_key}_reverse",
            model_name=spec.reverse_model_name,
            device=device,
            tokenizer_max_length=spec.tokenizer_max_length,
            default_batch_size=spec.default_batch_size,
            generation_kwargs=gen_kwargs,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

    if spec.backend_family == "nllb":
        from .nllb_translator import NLLBTranslator

        return NLLBTranslator(
            model_key=f"{spec.model_key}_reverse",
            model_name=spec.reverse_model_name,
            src_lang=spec.reverse_src_lang or spec.tgt_lang,
            tgt_lang=spec.reverse_tgt_lang or spec.src_lang,
            device=device,
            tokenizer_max_length=spec.tokenizer_max_length,
            default_batch_size=spec.default_batch_size,
            generation_kwargs=gen_kwargs,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

    return None
