"""Translation backends for the VepKar semantic categorization pipeline."""

from .base import (
    Translator,
    TranslatorError,
    BackendUnavailableError,
    TranslatorInitializationError,
    TranslationFailedError,
)
from .model_registry import ModelSpec, get_model_spec, list_model_keys
from .factory import build_translator, build_reverse_translator

__all__ = [
    "Translator",
    "TranslatorError",
    "BackendUnavailableError",
    "TranslatorInitializationError",
    "TranslationFailedError",
    "ModelSpec",
    "get_model_spec",
    "list_model_keys",
    "build_translator",
    "build_reverse_translator",
]
