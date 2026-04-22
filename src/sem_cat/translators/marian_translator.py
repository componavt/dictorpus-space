"""MarianMT-based translator (thin wrapper around HFSeq2SeqTranslator).

Default model: Helsinki-NLP/opus-mt-ru-en (~300 MB)
"""

from __future__ import annotations

from .hf_seq2seq_translator import HFSeq2SeqTranslator


class MarianTranslator(HFSeq2SeqTranslator):
    """MarianMT-based Russian-to-English translator.

    This is a thin compatibility wrapper around HFSeq2SeqTranslator.
    """

    MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"

    def __init__(
        self,
        device: str = "cpu",
        model_name: str | None = None,
    ) -> None:
        super().__init__(
            model_key="helsinki_opus_mt_ru_en",
            model_name=model_name or self.MODEL_NAME,
            device=device,
            tokenizer_max_length=64,
            default_batch_size=64,
            generation_kwargs={
                "max_new_tokens": 16,
                "num_beams": 4,
                "no_repeat_ngram_size": 2,
                "repetition_penalty": 1.2,
                "length_penalty": 0.8,
                "early_stopping": True,
            },
        )
