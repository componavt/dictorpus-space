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


def _contains_any(text: str, needles: list[str]) -> bool:
    """Check if text contains any of the needles (case-insensitive)."""
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def explain_hf_causal_init_error(
    exc: Exception,
    model_name: str,
    *,
    local_files_only: bool = False,
    cache_dir: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> str:
    """Produce a human-readable explanation for a HuggingFace causal LM init failure.

    Args:
        exc: The original exception caught during model/tokenizer loading.
        model_name: The HuggingFace model identifier being loaded.
        local_files_only: Whether local-files-only mode was active.
        cache_dir: Optional custom cache directory path.
        load_in_4bit: Whether 4-bit quantization was requested.
        load_in_8bit: Whether 8-bit quantization was requested.

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
            f"Failed to initialize HuggingFace causal model '{model_name}' because proxy "
            f"configuration is invalid.\n{proxy_hint}\n{shell_workaround}"
        )

    # Local files only mode
    if local_files_only:
        cache_info = f" in cache_dir={cache_dir!r}" if cache_dir else ""
        return (
            f"Failed to initialize HuggingFace causal model '{model_name}' in local-files-only mode. "
            f"No usable local cache was found{cache_info}. "
            "Pre-download the model with 'huggingface-cli download <model>' "
            "or disable --local-files-only."
        )

    # Offline / connection errors
    if "offline" in text.lower() or "connection" in text.lower() or "404" in text:
        return (
            f"Failed to initialize HuggingFace causal model '{model_name}'. "
            f"The model may not be available locally and network access failed. "
            f"Original error: {text}"
        )

    # Check for accelerate requirements
    if _contains_any(text, ["requires `accelerate`", "requires accelerate"]):
        return (
            f"Failed to initialize HuggingFace causal model '{model_name}'. "
            "The selected loading mode requires the `accelerate` package. "
            "Install it with: pip install accelerate\n\n"
            f"Original error: {text}"
        )

    # Check for bitsandbytes/CUDA runtime issues (quantization-related)
    load_quantized = load_in_4bit or load_in_8bit
    if load_quantized:
        bnb_error_indicators = [
            "bitsandbytes",
            "cuda setup error",
            "libnvjitlink",
            "cannot open shared object file",
            "native code method attempted to call",
        ]
        if any(indicator in text.lower() for indicator in bnb_error_indicators):
            return (
                f"Failed to initialize HuggingFace causal model '{model_name}'. "
                "4-bit/8-bit quantized loading requires a working bitsandbytes + CUDA runtime setup.\n\n"
                "Detected a bitsandbytes / CUDA runtime failure. "
                "This often means the Python package is present, but a required CUDA shared library is missing.\n\n"
                "Recommended actions:\n"
                "1. Run: python -m bitsandbytes\n"
                "2. Verify the required CUDA runtime/toolkit is installed\n"
                "3. Ensure the CUDA lib directory is visible in LD_LIBRARY_PATH\n"
                "4. If supported, rerun without quantization (--quantization none)\n\n"
                f"Original error: {text}"
            )

    # Generic
    return (
        f"Failed to initialize HuggingFace causal model '{model_name}'. "
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
    torch_dtype: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    torch: Any,
    AutoTokenizer: Any,
    AutoModelForCausalLM: Any,
) -> tuple[Any, Any]:
    """Load a HuggingFace tokenizer and causal LM model with optional quantization.

    Args:
        model_name: HuggingFace model identifier.
        local_files_only: If True, only use locally cached files.
        cache_dir: Optional custom cache directory.
        device: Target device ("cpu" or "cuda").
        ignore_proxy_env: If True, temporarily unset proxy env vars during loading.
        trust_remote_code: If True, allow execution of model code from the HF repo.
            Required by some models (e.g. tencent/Hy-MT2-30B-A3B).
        torch_dtype: Optional torch dtype string ("float16", "bfloat16", "float32", etc.).
        load_in_4bit: If True, load model in 4-bit quantized mode.
        load_in_8bit: If True, load model in 8-bit quantized mode.
        torch: The torch module (already imported).
        AutoTokenizer: Tokenizer class from transformers.
        AutoModelForCausalLM: Model class from transformers.

    Returns:
        Tuple of (tokenizer, model).

    Raises:
        TranslatorInitializationError: If model/tokenizer loading fails.
        BackendUnavailableError: If quantization requested but bitsandbytes unavailable.
    """
    from .base import BackendUnavailableError, TranslatorInitializationError

    common_kwargs: dict[str, Any] = {}
    if local_files_only:
        common_kwargs["local_files_only"] = True
    if cache_dir:
        common_kwargs["cache_dir"] = cache_dir
    if trust_remote_code:
        common_kwargs["trust_remote_code"] = True
    if torch_dtype:
        if torch_dtype == "float16":
            common_kwargs["torch_dtype"] = torch.float16
        elif torch_dtype == "bfloat16":
            common_kwargs["torch_dtype"] = torch.bfloat16
        elif torch_dtype == "float32":
            common_kwargs["torch_dtype"] = torch.float32
        else:
            common_kwargs["torch_dtype"] = torch_dtype

    # Handle quantization with BitsAndBytesConfig
    load_quantized = load_in_4bit or load_in_8bit
    if load_quantized:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise BackendUnavailableError(
                f"Quantized loading requested but 'bitsandbytes' is not installed. "
                f"Install it with: pip install bitsandbytes"
            ) from e

        bnb_kwargs: dict[str, Any] = {}
        if load_in_4bit:
            bnb_kwargs["load_in_4bit"] = True
            bnb_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            bnb_kwargs["bnb_4bit_use_double_quant"] = True
            bnb_kwargs["bnb_4bit_quant_type"] = "nf4"
        elif load_in_8bit:
            bnb_kwargs["load_in_8bit"] = True

        bnb_config = BitsAndBytesConfig(**bnb_kwargs)
        common_kwargs["quantization_config"] = bnb_config

    # For CUDA, use device_map="auto" with quantization to avoid OOM
    if device == "cuda" and load_quantized:
        common_kwargs["device_map"] = "auto"
    # For CPU loading with quantization, explicitly reject it
    elif device == "cpu" and load_quantized:
        raise TranslatorInitializationError(
            f"Quantized loading is not supported on CPU. "
            f"Remove --device cpu or disable quantization."
        )

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
            msg = explain_hf_causal_init_error(
                e, model_name,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            raise TranslatorInitializationError(msg) from e

        # Skip manual .to(device) when device_map="auto" handles placement
        # Quantized models with device_map should NOT be moved manually
        if "device_map" not in common_kwargs:
            model = model.to(device)
        return tokenizer, model

    if ignore_proxy_env:
        with temporarily_unset_env(PROXY_ENV_VARS):
            return _load()
    return _load()
