"""NLLB (No Language Left Behind) translation backend.

Uses facebook/nllb-200 models via HuggingFace transformers.
NLLB supports 200+ languages with high quality.

NLLB uses BCP-47 + script codes, not ISO 639-1:
  Russian:  rus_Cyrl    English: eng_Latn
  Finnish:  fin_Latn    Estonian: est_Latn
For a full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md

torch and transformers are lazily imported so that the module is
import-safe even when those heavy dependencies are absent.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .base import BackendUnavailableError, Translator, TranslatorInitializationError
from .generation_presets import get_generation_preset
from .hf_runtime import load_hf_model

logger = logging.getLogger(__name__)


class NLLBTranslator(Translator):
    """NLLB translator using direct model/tokenizer API.

    All heavy dependencies (torch, transformers) are loaded at
    instantiation time, not at module import time.
    """

    DEFAULT_MODEL = "facebook/nllb-200-distilled-1.3B"

    def __init__(
        self,
        model_key: str,
        model_name: str,
        src_lang: str = "rus_Cyrl",
        tgt_lang: str = "eng_Latn",
        device: str = "cpu",
        tokenizer_max_length: int = 128,
        default_batch_size: int = 32,
        generation_kwargs: dict[str, Any] | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        ignore_proxy_env: bool = False,
    ) -> None:
        """Initialize NLLB model and tokenizer.

        Args:
            model_key: Registry key (e.g. 'nllb_distilled_1_3b')
            model_name: HuggingFace model name
            src_lang: Source language code (e.g. rus_Cyrl)
            tgt_lang: Target language code (e.g. eng_Latn)
            device: "cpu" or "cuda"
            tokenizer_max_length: Maximum sequence length for tokenization
            default_batch_size: Default batch size for translate_batch()
            generation_kwargs: Override generation parameters
            local_files_only: If True, only use locally cached files.
            cache_dir: Optional custom cache directory.
            ignore_proxy_env: If True, temporarily unset proxy env vars during loading.
        """
        # Lazy import heavy dependencies
        try:
            import torch as _torch
        except ImportError as e:
            raise BackendUnavailableError(
                "NLLBTranslator requires PyTorch. "
                "Install it with: pip install torch"
            ) from e
        self.torch = _torch

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise BackendUnavailableError(
                "NLLBTranslator requires the 'transformers' package. "
                "Install it with: pip install transformers"
            ) from e

        self.model_key = model_key
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer_max_length = tokenizer_max_length
        self.default_batch_size = default_batch_size
        self.device = device
        self.supports_roundtrip = True
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self.ignore_proxy_env = ignore_proxy_env

        if generation_kwargs is not None:
            self.generation_kwargs = generation_kwargs
        else:
            self.generation_kwargs = get_generation_preset("gloss_strict")

        logger.info("Loading NLLBTranslator: %s | %s->%s | device=%s",
                     model_name, src_lang, tgt_lang, device)

        self.tokenizer, self.model = load_hf_model(
            model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            device=device,
            ignore_proxy_env=ignore_proxy_env,
            torch=self.torch,
            AutoTokenizer=AutoTokenizer,
            AutoModelForSeq2SeqLM=AutoModelForSeq2SeqLM,
            tokenizer_kwargs={"src_lang": src_lang},
        )

        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        # Clear max_length from generation config to avoid conflict with max_new_tokens.
        if hasattr(self.model.generation_config, "max_length"):
            self.model.generation_config.max_length = 20

    def _tokenize_and_generate(
        self,
        texts: list[str],
    ) -> list[str]:
        """Shared tokenization + generation + decoding logic."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer_max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                **self.generation_kwargs,
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def translate(self, text: str) -> str | None:
        """Translate a single string.

        Returns:
            Translated text or None if translation fails or input is blank.
        """
        try:
            if not text or not text.strip():
                return None

            decoded = self._tokenize_and_generate([text])
            translated = decoded[0]
            return translated.strip() if translated.strip() else None
        except Exception as e:
            logger.warning("NLLB translate error: %s", e)
            return None

    def translate_batch(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> list[str | None]:
        """Translate a list of texts in batch.

        Returns:
            List of translated texts (or None for failed items) in same order.
        """
        if not texts:
            return []

        effective_batch_size = batch_size if batch_size is not None else self.default_batch_size
        results: list[str | None] = []

        for i in range(0, len(texts), effective_batch_size):
            batch_slice = list(texts[i:i + effective_batch_size])

            try:
                decoded = self._tokenize_and_generate(batch_slice)
                results.extend([t.strip() if t.strip() else None for t in decoded])
            except Exception as e:
                logger.warning("NLLB translate_batch error: %s", e)
                results.extend([None] * len(batch_slice))

        return results
