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

from collections.abc import Sequence
from typing import Any

from .base import BackendUnavailableError, Translator
from .generation_presets import get_generation_preset


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

        if generation_kwargs is not None:
            self.generation_kwargs = generation_kwargs
        else:
            self.generation_kwargs = get_generation_preset("gloss_strict")

        print(f"NLLBTranslator: {model_name} | {src_lang}->{tgt_lang} | device={device}")
        print("First run will download model from HuggingFace. Subsequent runs use local cache.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)

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
            print(f"NLLB translate error: {e}")
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
                print(f"NLLB translate_batch error: {e}")
                results.extend([None] * len(batch_slice))

        return results
