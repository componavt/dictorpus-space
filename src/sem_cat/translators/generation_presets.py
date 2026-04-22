"""Generation presets for translation models.

Each preset is a dict of kwargs passed to model.generate().
Designed for short dictionary-style gloss translation.
"""

from __future__ import annotations

from typing import Any


GLOSS_STRICT_PRESET: dict[str, Any] = {
    "max_new_tokens": 12,
    "num_beams": 4,
    "no_repeat_ngram_size": 2,
    "repetition_penalty": 1.3,
    "length_penalty": 0.8,
    "early_stopping": True,
    "do_sample": False,
}

DEFAULT_PRESET: dict[str, Any] = {
    "max_new_tokens": 64,
    "num_beams": 4,
    "do_sample": False,
}

PRESETS: dict[str, dict[str, Any]] = {
    "gloss_strict": GLOSS_STRICT_PRESET,
    "default": DEFAULT_PRESET,
}


def get_generation_preset(name: str) -> dict[str, Any]:
    """Return a copy of the generation preset by name.

    Raises:
        ValueError: If preset name is not found.
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(
            f"Unknown generation preset: {name!r}. Available: {available}"
        )
    return dict(PRESETS[name])
