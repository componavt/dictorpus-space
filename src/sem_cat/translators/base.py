"""Abstract base class for Russian-to-English gloss translators."""

from abc import ABC, abstractmethod
from typing import List


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str) -> str:
        """Translate a single Russian string to English."""
        pass

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Default: call translate() in a loop. Subclasses may override for efficiency."""
        return [self.translate(text) for text in texts]