"""NLLB (No Language Left Behind) translation backend.
Uses facebook/nllb-200-distilled-1.3B model via HuggingFace transformers.
NLLB supports 200+ languages with high quality.

# NLLB uses BCP-47 + script codes, not ISO 639-1:
# Russian:  rus_Cyrl    English: eng_Latn
# Finnish:  fin_Latn    Estonian: est_Latn
# For a full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
"""

from typing import List, Optional
from transformers import pipeline, Pipeline
from .base import Translator


class NLLBTranslator(Translator):
    """NLLB translator using HuggingFace pipeline API."""
    
    DEFAULT_MODEL = "facebook/nllb-200-distilled-1.3B"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        src_lang: str = "rus_Cyrl",
        tgt_lang: str = "eng_Latn",
        device: str = "cpu",
        max_length: int = 256,
        batch_size: int = 32,
    ) -> None:
        """Initialize NLLB translation pipeline.
        
        Args:
            model_name: HuggingFace model name (default: facebook/nllb-200-distilled-1.3B)
            src_lang: Source language code (default: rus_Cyrl for Russian)
            tgt_lang: Target language code (default: eng_Latn for English)
            device: "cpu" or "cuda"
            max_length: Maximum sequence length for translation
            batch_size: Batch size for translate_batch()
        """
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._batch_size = batch_size
        
        # Convert device string to pipeline device id
        device_id = 0 if device == "cuda" else -1
        
        print(f"NLLBTranslator: {model_name} | {src_lang}→{tgt_lang} | device={device}")
        print("First run will download ~5 GB from HuggingFace. Subsequent runs use local cache.")
        
        self._pipe: Pipeline = pipeline(
            "translation",
            model=model_name,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=device_id,
            max_length=max_length,
        )
    
    def translate(self, text: str) -> Optional[str]:
        """Translate a single string.
        
        Args:
            text: Input text to translate
            
        Returns:
            Translated text or None if translation fails
        """
        try:
            result = self._pipe(text)
            translated = result[0]["translation_text"]
            return translated if translated.strip() else None
        except Exception as e:
            print(f"NLLB translate error: {e}")
            return None
    
    def translate_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Translate a list of texts in batch.
        
        Args:
            texts: List of input texts to translate
            
        Returns:
            List of translated texts (or None for failed items) in same order
        """
        if not texts:
            return []
        
        try:
            results = self._pipe(texts, batch_size=self._batch_size)
            # results is list[{"translation_text": str}]
            translated = [r["translation_text"] if r and r.get("translation_text") else None for r in results]
            return translated
        except Exception as e:
            print(f"NLLB translate_batch error: {e}")
            # Return list of None for all inputs on error
            return [None] * len(texts)
