"""Translation backends for the VepKar semantic categorization pipeline."""

from .base import (
    Translator,
    TranslatorError,
    BackendUnavailableError,
    TranslatorInitializationError,
)
from .model_registry import ModelSpec, get_model_spec, list_model_keys
from .factory import build_translator, build_reverse_translator
from .hf_runtime import collect_proxy_env, explain_hf_init_error

__all__ = [
    "Translator",
    "TranslatorError",
    "BackendUnavailableError",
    "TranslatorInitializationError",
    "ModelSpec",
    "get_model_spec",
    "list_model_keys",
    "build_translator",
    "build_reverse_translator",
    "collect_proxy_env",
    "explain_hf_init_error",
]
