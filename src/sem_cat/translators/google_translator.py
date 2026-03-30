"""Google Translate backend using deep_translator.
Suitable for small batches and spot-checking. Includes retry with backoff.
"""

import time
from typing import List
from deep_translator import GoogleTranslator as DeepGoogleTranslator
from .base import Translator


class GoogleTranslator(Translator):
    def __init__(self, source="ru", target="en", retry=3, delay=1.0):
        self.source = source
        self.target = target
        self.retry = retry
        self.delay = delay

    def translate(self, text: str) -> str:
        """Use deep_translator.GoogleTranslator.
        Retry up to self.retry times on exception, sleep self.delay between retries.
        Return empty string on final failure (do not raise).
        """
        for attempt in range(self.retry + 1):
            try:
                translator = DeepGoogleTranslator(source=self.source, target=self.target)
                result = translator.translate(text)
                return result if result is not None else ""
            except Exception as e:
                if attempt < self.retry:
                    time.sleep(self.delay)
                else:
                    return ""  # Return empty string on final failure