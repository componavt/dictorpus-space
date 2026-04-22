"""Generic HuggingFace seq2seq translator.

Handles encoder-decoder models such as:
- Helsinki-NLP/opus-mt-ru-en
- Helsinki-NLP/opus-mt-en-ru
- facebook/wmt19-ru-en
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import Translator


class HFSeq2SeqTranslator(Translator):
    """Generic HuggingFace seq2seq translator."""

    def __init__(
        self,
        model_key: str,
        model_name: str,
        device: str = "cpu",
        tokenizer_max_length: int = 64,
        default_batch_size: int = 32,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_key = model_key
        self.model_name = model_name
        self.device = device
        self.tokenizer_max_length = tokenizer_max_length
        self.default_batch_size = default_batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.supports_roundtrip = True

        print(f"HFSeq2SeqTranslator: {model_name} | device={device}")
        print("First run will download model from HuggingFace. Subsequent runs use local cache.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)

    def translate(self, text: str) -> str | None:
        """Translate a single string."""
        try:
            if not text or not text.strip():
                return None

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer_max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_kwargs,
                )

            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated.strip() if translated.strip() else None
        except Exception as e:
            print(f"HFSeq2SeqTranslator translate error: {e}")
            return None

    def translate_batch(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> list[str | None]:
        """Translate a list of texts in batch."""
        if not texts:
            return []

        effective_batch_size = batch_size if batch_size is not None else self.default_batch_size
        results: list[str | None] = []

        for i in range(0, len(texts), effective_batch_size):
            batch_slice = list(texts[i:i + effective_batch_size])

            try:
                inputs = self.tokenizer(
                    batch_slice,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer_max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **self.generation_kwargs,
                    )

                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend([t.strip() if t.strip() else None for t in decoded])

            except Exception as e:
                print(f"HFSeq2SeqTranslator translate_batch error: {e}")
                results.extend([None] * len(batch_slice))

        return results
