"""Abstract base class and error types for gloss translators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class TranslatorError(Exception):
    """Base exception for translator-related errors."""


class BackendUnavailableError(TranslatorError):
    """Raised when a required backend dependency is not installed."""


class TranslatorInitializationError(TranslatorError):
    """Raised when a translator fails to initialize (e.g. bad config, missing model)."""


class TranslationFailedError(TranslatorError):
    """Raised when a translation attempt fails irrecoverably."""


class Translator(ABC):
    """Base translator with a consistent metadata and behavior contract.

    Subclasses must set these instance attributes during __init__:
        model_key: str - Registry key (e.g. 'google', 'nllb_distilled_1_3b')
        model_name: str - Human-readable model identifier
        supports_roundtrip: bool - Whether round-trip translation is supported
        default_batch_size: int - Default batch size for translate_batch()

    Behavioral contract:
        - translate(text) returns str | None.
          None is returned for blank input or failed translation.
        - translate_batch(texts) returns list[str | None] in the same order.
        - No empty strings are returned on failure; use None instead.
        - Missing optional dependencies must NOT cause import-time failures.
          Instead, raise BackendUnavailableError at instantiation time.
    """

    model_key: str = "unknown"
    model_name: str = "unknown"
    supports_roundtrip: bool = False
    default_batch_size: int = 1

    @abstractmethod
    def translate(self, text: str) -> str | None:
        """Translate a single input string.

        Args:
            text: The source text to translate.

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
