"""Abstract base class for gloss translators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Translator(ABC):
    """Base translator with metadata contract.

    Subclasses must set these class/instance attributes:
        model_key: str - Registry key (e.g. 'google', 'nllb_distilled_1_3b')
        model_name: str - Human-readable model identifier
        supports_roundtrip: bool - Whether round-trip translation is supported
        default_batch_size: int - Default batch size for translate_batch()
    """

    model_key: str = "unknown"
    model_name: str = "unknown"
    supports_roundtrip: bool = False
    default_batch_size: int = 1

    @abstractmethod
    def translate(self, text: str) -> str | None:
        """Translate a single input string.

        Returns:
            Translated text, or None if translation fails or input is blank.
        """

    def translate_batch(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> list[str | None]:
        """Default batch implementation using translate() in a loop.

        Subclasses may override for more efficient batch processing.

        Args:
            texts: Sequence of input strings.
            batch_size: Ignored by default implementation; subclasses may use it.

        Returns:
            List of translated strings (or None for failures) in same order.
        """
        return [self.translate(text) for text in texts]
