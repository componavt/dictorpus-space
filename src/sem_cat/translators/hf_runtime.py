"""HuggingFace runtime helpers.

Provides proxy environment detection, actionable error messages for
model initialization failures, shared loading logic for HF-based
translators, and a context manager for temporarily unsetting proxy vars.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
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


def identify_bad_proxy_vars(proxy_env: dict[str, str]) -> list[tuple[str, str]]:
    """Identify which proxy variables are likely malformed.

    Returns a list of (name, value) pairs for variables that use
    unsupported schemes like bare 'socks://' instead of 'socks5://'.
    """
    bad = []
    for name, value in proxy_env.items():
        if value.startswith("socks://") and not value.startswith("socks5://"):
            bad.append((name, value))
    return bad


@contextmanager
def temporarily_unset_env(var_names: tuple[str, ...]):
    """Context manager that temporarily removes specified env vars.

    Restores them to their original values on exit.
    """
    saved = {name: os.environ.get(name) for name in var_names if name in os.environ}
    try:
        for name in var_names:
            os.environ.pop(name, None)
        yield
    finally:
        for name in var_names:
            if name in saved:
                os.environ[name] = saved[name]
            else:
                os.environ.pop(name, None)


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
    bad_proxies = identify_bad_proxy_vars(proxy_env)

    # Proxy-related errors
    if "Unknown scheme for proxy URL" in text or "proxy" in text.lower():
        if bad_proxies:
            bad_desc = ", ".join(f"{n}={v!r}" for n, v in bad_proxies)
            proxy_hint = (
                f"The following variables use an unsupported proxy scheme: {bad_desc}. "
                "Use 'socks5://' instead of 'socks://', or unset them."
            )
        else:
            all_proxy = ", ".join(f"{k}={v!r}" for k, v in proxy_env.items())
            proxy_hint = (
                f"Detected proxy variables: {all_proxy}. "
                "One of them may have an invalid URL scheme."
            )

        shell_workaround = (
            "Shell workaround for Linux/macOS:\n"
            "  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY "
            "-u http_proxy -u https_proxy -u all_proxy \\\n"
            f"  python3 -m src.sem_cat.02_translate_glosses --model-key <key>"
        )

        return (
            f"Failed to initialize HuggingFace model '{model_name}' because proxy "
            f"configuration is invalid.\n{proxy_hint}\n{shell_workaround}"
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
    ignore_proxy_env: bool = False,
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
        ignore_proxy_env: If True, temporarily unset proxy env vars during loading.
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

    def _load() -> tuple[Any, Any]:
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

    if ignore_proxy_env:
        with temporarily_unset_env(PROXY_ENV_VARS):
            return _load()
    return _load()


def load_hf_model_causal(
    model_name: str,
    *,
    local_files_only: bool = False,
    cache_dir: str | None = None,
    device: str = "cpu",
    ignore_proxy_env: bool = False,
    trust_remote_code: bool = False,
    torch: Any,
    AutoTokenizer: Any,
    AutoModelForCausalLM: Any,
) -> tuple[Any, Any]:
    """Load a HuggingFace tokenizer and causal LM model.

    Args:
        model_name: HuggingFace model identifier.
        local_files_only: If True, only use locally cached files.
        cache_dir: Optional custom cache directory.
        device: Target device ("cpu" or "cuda").
        ignore_proxy_env: If True, temporarily unset proxy env vars during loading.
        trust_remote_code: If True, allow execution of model code from the HF repo.
            Required by some models (e.g. tencent/Hy-MT2-30B-A3B).
        torch: The torch module (already imported).
        AutoTokenizer: Tokenizer class from transformers.
        AutoModelForCausalLM: Model class from transformers.

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
    if trust_remote_code:
        common_kwargs["trust_remote_code"] = True

    def _load() -> tuple[Any, Any]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **common_kwargs)
        except Exception as e:
            msg = explain_hf_init_error(
                e, model_name, local_files_only=local_files_only, cache_dir=cache_dir
            )
            raise TranslatorInitializationError(msg) from e

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
        except Exception as e:
            msg = explain_hf_init_error(
                e, model_name, local_files_only=local_files_only, cache_dir=cache_dir
            )
            raise TranslatorInitializationError(msg) from e

        model = model.to(device)
        return tokenizer, model

    if ignore_proxy_env:
        with temporarily_unset_env(PROXY_ENV_VARS):
            return _load()
    return _load()
