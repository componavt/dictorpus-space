"""HuggingFace runtime helpers.

Provides proxy environment detection, actionable error messages for
model initialization failures, and shared loading logic for HF-based
translators.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def collect_proxy_env() -> dict[str, str]:
    """Return a dict of proxy-related environment variables that are set."""
    return {k: v for k in PROXY_ENV_VARS if (v := os.environ.get(k)) is not None}


def explain_hf_init_error(
    exc: Exception,
    model_name: str,
    *,
    local_files_only: bool = False,
    cache_dir: str | None = None,
) -> str:
    """Produce a human-readable explanation for a HuggingFace init failure.

    Args:
        exc: The original exception caught during model/tokenizer loading.
        model_name: The HuggingFace model identifier being loaded.
        local_files_only: Whether local-files-only mode was active.
        cache_dir: Optional custom cache directory path.

    Returns:
        An actionable error message string.
    """
    text = str(exc)
    proxy_env = collect_proxy_env()

    # Proxy-related errors
    if "Unknown scheme for proxy URL" in text or "proxy" in text.lower():
        proxy_vars = ", ".join(f"{k}={v!r}" for k, v in proxy_env.items()) if proxy_env else "none"
        return (
            f"Failed to initialize HuggingFace model '{model_name}' because proxy "
            f"configuration is invalid. Detected proxy variables: {proxy_vars}. "
            "If you use a SOCKS proxy, prefer 'socks5://' instead of 'socks://', "
            "or unset the proxy variables before running the script."
        )

    # Local files only mode
    if local_files_only:
        cache_info = f" in cache_dir={cache_dir!r}" if cache_dir else ""
        return (
            f"Failed to initialize HuggingFace model '{model_name}' in local-files-only mode. "
            f"No usable local cache was found{cache_info}. "
            "Pre-download the model with 'huggingface-cli download <model>' "
            "or disable --local-files-only."
        )

    # Offline / connection errors
    if "offline" in text.lower() or "connection" in text.lower() or "404" in text:
        return (
            f"Failed to initialize HuggingFace model '{model_name}'. "
            f"The model may not be available locally and network access failed. "
            f"Original error: {text}"
        )

    # Generic
    return (
        f"Failed to initialize HuggingFace model '{model_name}'. "
        f"Original error: {text}"
    )


def load_hf_model(
    model_name: str,
    *,
    local_files_only: bool = False,
    cache_dir: str | None = None,
    device: str = "cpu",
    torch: Any,
    AutoTokenizer: Any,
    AutoModelForSeq2SeqLM: Any,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Load a HuggingFace tokenizer and model with proper error wrapping.

    Args:
        model_name: HuggingFace model identifier.
        local_files_only: If True, only use locally cached files.
        cache_dir: Optional custom cache directory.
        device: Target device ("cpu" or "cuda").
        torch: The torch module (already imported).
        AutoTokenizer: Tokenizer class from transformers.
        AutoModelForSeq2SeqLM: Model class from transformers.
        tokenizer_kwargs: Extra kwargs for tokenizer loading (e.g. src_lang).

    Returns:
        Tuple of (tokenizer, model).

    Raises:
        TranslatorInitializationError: If model/tokenizer loading fails.
    """
    from .base import TranslatorInitializationError

    common_kwargs: dict[str, Any] = {}
    if local_files_only:
        common_kwargs["local_files_only"] = True
    if cache_dir:
        common_kwargs["cache_dir"] = cache_dir

    tok_kwargs = {**(tokenizer_kwargs or {}), **common_kwargs}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    except Exception as e:
        msg = explain_hf_init_error(
            e, model_name, local_files_only=local_files_only, cache_dir=cache_dir
        )
        raise TranslatorInitializationError(msg) from e

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **common_kwargs)
    except Exception as e:
        msg = explain_hf_init_error(
            e, model_name, local_files_only=local_files_only, cache_dir=cache_dir
        )
        raise TranslatorInitializationError(msg) from e

    model = model.to(device)
    return tokenizer, model
