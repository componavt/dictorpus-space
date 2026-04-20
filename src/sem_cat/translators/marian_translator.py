"""Local translation backend using Helsinki-NLP/opus-mt-ru-en
(MarianMT via HuggingFace transformers). No API key, no rate limits.
Optimal for translating all 42k unique VepKar glosses offline.

Default model: Helsinki-NLP/opus-mt-ru-en (~300 MB)
Reverse model: Helsinki-NLP/opus-mt-en-ru (for back-translation)
"""

from typing import List
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from .base import Translator


class MarianTranslator(Translator):
    """MarianMT-based Russian-to-English translator."""
    
    MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"

    def __init__(self, device: str = "cpu", model_name: str | None = None):
        """Load tokenizer and model on init. Log model name and device.
        
        Args:
            device: "cpu" or "cuda"
            model_name: Optional override for model name (default: MODEL_NAME)
        """
        self.device = device
        self.model_name = model_name or self.MODEL_NAME
        print(f"Loading model {self.model_name} on device {device}")
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        print(f"Model {self.model_name} loaded successfully on {device}")

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
