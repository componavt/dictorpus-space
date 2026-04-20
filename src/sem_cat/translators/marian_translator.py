"""Local translation backend using Helsinki-NLP/opus-mt-tc-big-ru-en
(MarianMT via HuggingFace transformers). No API key, no rate limits.
Optimal for translating all 42k unique VepKar glosses offline.

# Upgraded from opus-mt-ru-en (transformer-align, ~300 MB)
# to opus-mt-tc-big-ru-en (transformer-big, ~600 MB).
# Same API, better quality on ambiguous single-word glosses.
# If the big model fails to download, pass model_name="Helsinki-NLP/opus-mt-ru-en"
# to fall back to the original model.
"""

from typing import List
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from .base import Translator


class MarianTranslator(Translator):
    # Default model: opus-mt-tc-big-ru-en (transformer-big, ~600 MB)
    # Fallback: opus-mt-ru-en (transformer-align, ~300 MB)
    MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-ru-en"
    # Back-translation model: opus-mt-en-ru (tc-big variant not available for en→ru)
    BACK_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"

    def __init__(self, device: str = "cpu", model_name: str = None, back_model_name: str = None):
        """Load tokenizer and model on init. Log model name and device."""
        self.device = device
        self.MODEL_NAME = model_name if model_name else self.MODEL_NAME
        self.BACK_MODEL_NAME = back_model_name if back_model_name else self.BACK_MODEL_NAME
        print(f"Loading model {self.MODEL_NAME} on device {device}")
        self.tokenizer = MarianTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = MarianMTModel.from_pretrained(self.MODEL_NAME, use_safetensors=True)
        self.model.to(self.device)
        print(f"Model {self.MODEL_NAME} loaded successfully on {device}")

    def translate(self, text: str) -> str:
        """Translate single string."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=3.0,
                early_stopping=True,
            )
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

    def translate_batch(self, texts: List[str], batch_size: int = 64) -> List[str]:
        """Translate a list in batches of batch_size. Show tqdm progress bar.
        Return list of translated strings in the same order.
        """
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=64
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=40,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    repetition_penalty=3.0,
                    early_stopping=True,
                )
            
            for output in outputs:
                translated = self.tokenizer.decode(output, skip_special_tokens=True)
                results.append(translated)
        
        return results
