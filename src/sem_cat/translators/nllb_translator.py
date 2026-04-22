"""NLLB (No Language Left Behind) translation backend.
Uses facebook/nllb-200-distilled-1.3B model via HuggingFace transformers.
NLLB supports 200+ languages with high quality.

# NLLB uses BCP-47 + script codes, not ISO 639-1:
# Russian:  rus_Cyrl    English: eng_Latn
# Finnish:  fin_Latn    Estonian: est_Latn
# For a full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
"""

from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .base import Translator


class NLLBTranslator(Translator):
    """NLLB translator using direct model/tokenizer API."""
    
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
        """Initialize NLLB model and tokenizer.
        
        Args:
            model_name: HuggingFace model name (default: facebook/nllb-200-distilled-1.3B)
            src_lang: Source language code (default: rus_Cyrl for Russian)
            tgt_lang: Target language code (default: eng_Latn for English)
            device: "cpu" or "cuda"
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for translate_batch()
        """
        try:
            import torch as _torch
        except ImportError as e:
            raise ImportError(
                "NLLBTranslator requires PyTorch. Install it in the active virtualenv with:\n"
                "  pip install torch"
            ) from e
        self.torch = _torch
        
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self._batch_size = batch_size
        self.device = device
        
        print(f"NLLBTranslator: {model_name} | {src_lang}→{tgt_lang} | device={device}")
        print("First run will download ~5 GB from HuggingFace. Subsequent runs use local cache.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
    
    def translate(self, text: str) -> Optional[str]:
        """Translate a single string.
        
        Args:
            text: Input text to translate
            
        Returns:
            Translated text or None if translation fails
        """
        try:
            if not text or not text.strip():
                return None
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.forced_bos_token_id,
                    max_new_tokens=64,
                    num_beams=4,
                )
            
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated.strip() if translated.strip() else None
        except Exception as e:
            print(f"NLLB translate error: {e}")
            return None
    
    def translate_batch(self, texts: List[str], batch_size: int | None = None) -> List[Optional[str]]:
        """Translate a list of texts in batch.
        
        Args:
            texts: List of input texts to translate
            batch_size: Optional batch size override (uses self._batch_size if None)
            
        Returns:
            List of translated texts (or None for failed items) in same order
        """
        if not texts:
            return []
        
        effective_batch_size = batch_size if batch_size is not None else self._batch_size
        results: List[Optional[str]] = []
        
        for i in range(0, len(texts), effective_batch_size):
            batch_slice = texts[i:i + effective_batch_size]
            
            try:
                inputs = self.tokenizer(
                    batch_slice,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with self.torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.forced_bos_token_id,
                        max_new_tokens=64,
                        num_beams=4,
                    )
                
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend([t.strip() if t.strip() else None for t in decoded])
                
            except Exception as e:
                print(f"NLLB translate_batch error: {e}")
                results.extend([None] * len(batch_slice))
        
        return results
