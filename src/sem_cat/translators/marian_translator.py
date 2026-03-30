"""Local translation backend using Helsinki-NLP/opus-mt-ru-en
(MarianMT via HuggingFace transformers). No API key, no rate limits.
Optimal for translating all 42k unique VepKar glosses offline.
"""

from typing import List
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from .base import Translator


class MarianTranslator(Translator):
    MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"

    def __init__(self, device: str = "cpu"):
        """Load tokenizer and model on init. Log model name and device."""
        self.device = device
        print(f"Loading model {self.MODEL_NAME} on device {device}")
        self.tokenizer = MarianTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = MarianMTModel.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        print(f"Model {self.MODEL_NAME} loaded successfully on {device}")

    def translate(self, text: str) -> str:
        """Translate single string. Use max_length=64."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=64)
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

    def translate_batch(self, texts: List[str], batch_size: int = 64) -> List[str]:
        """Translate a list in batches of batch_size. Show tqdm progress bar.
        Return list of translated strings in the same order.
        """
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=64)
            
            for output in outputs:
                translated = self.tokenizer.decode(output, skip_special_tokens=True)
                results.append(translated)
        
        return results