"""Google Translate backend using deep_translator.

Suitable for small batches and spot-checking. Includes retry with backoff.
deep_translator is lazily imported so that the module is import-safe
even when the dependency is absent.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from .base import BackendUnavailableError, Translator


class GoogleTranslator(Translator):
    """Google Translate adapter using deep_translator.

    Google is treated as a first-class translator with the same interface
    as all other backends. Internally it processes items one at a time
    due to API limitations.
    """

    def __init__(
        self,
        source: str = "ru",
        target: str = "en",
        retry: int = 3,
        delay: float = 1.0,
        model_key: str = "google",
        model_name: str = "google",
    ) -> None:
        self.model_key = model_key
        self.model_name = model_name
        self.source = source
        self.target = target
        self.retry = retry
        self.delay = delay
        self.default_batch_size = 1
        self.supports_roundtrip = True

        try:
            from deep_translator import GoogleTranslator as DeepGoogleTranslator
        except ImportError as e:
            raise BackendUnavailableError(
                "GoogleTranslator requires the 'deep_translator' package. "
                "Install it with: pip install deep_translator"
            ) from e

        self._client = DeepGoogleTranslator(source=source, target=target)

    def translate(self, text: str) -> str | None:
        """Translate a single string with retry.

        Returns None on final failure instead of empty string.
        """
        if not text or not text.strip():
            return None

        for attempt in range(self.retry + 1):
            try:
                result = self._client.translate(text)
                if result and result.strip():
                    return result.strip()
                return None
            except Exception as e:
                if attempt < self.retry:
                    time.sleep(self.delay)
                    # Recreate client on failure to recover from stale state
                    try:
                        from deep_translator import (
                            GoogleTranslator as DeepGoogleTranslator,
                        )
                        self._client = DeepGoogleTranslator(
                            source=self.source, target=self.target
                        )
                    except Exception:
                        pass
                else:
                    return None
        return None

    def translate_batch(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> list[str | None]:
        """Process one at a time with delay (Google API limitation)."""
        _ = batch_size
        results: list[str | None] = []
        for text in texts:
            results.append(self.translate(text))
            if text and text.strip():
                time.sleep(0.3)
        return results
