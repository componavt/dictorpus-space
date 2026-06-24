"""HuggingFace causal LM translator for decoder-only translation models.

Handles models such as:
- Unbabel/Tower-Plus-9B            (chat-template, instruction-style)
- tencent/Hy-MT2-30B-A3B          (plain prompt, trust_remote_code)
- haoranxu/ALMA-7B-R              (plain prompt, translation-focused)

These are decoder-only (causal) LMs, not encoder-decoder seq2seq models.
They require AutoModelForCausalLM and a prompt template for translation.

torch and transformers are lazily imported so that the module is
import-safe even when those heavy dependencies are absent.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Sequence
from typing import Any

from .base import (
    BackendUnavailableError,
    Translator,
    TranslatorInitializationError,
    TranslatorRuntimeError,
)
from .hf_runtime import load_hf_model_causal

logger = logging.getLogger(__name__)

_TRANSIENT_ERROR_PATTERNS = (
    "out of memory",
    "oom",
    "timeout",
    "connection",
    "network",
)

_PROMPT_TEMPLATES: dict[str, str] = {
    "tower_chatml": (
        "Translate the following text from {src_lang} into {tgt_lang}.\n"
        "{src_lang}: {text}"
    ),
    "hy_chat": (
        "Translate the following text from {src_lang} to {tgt_lang}:\n"
        "{src_lang}: {text}\n"
        "{tgt_lang}:"
    ),
    "alma_plain": (
        "Translate this from {src_lang} to {tgt_lang}:\n"
        "{src_lang}: {text}\n"
        "{tgt_lang}:"
    ),
}

_DEFAULT_TEMPLATE = (
    "Translate the following text from {src_lang} to {tgt_lang}.\n"
    "{src_lang}: {text}\n"
    "{tgt_lang}:"
)


def _log_translate_error(backend: str, method: str, exc: Exception) -> None:
    err_msg = str(exc)
    is_transient = any(
        pattern in err_msg.lower() for pattern in _TRANSIENT_ERROR_PATTERNS
    )
    if is_transient:
        logger.warning("%s %s transient error: %s", backend, method, exc)
    else:
        tb_summary = traceback.format_exc()
        tb_lines = tb_summary.strip().split("\n")
        short_tb = "\n".join(tb_lines[-4:]) if len(tb_lines) > 4 else tb_summary
        logger.error(
            "%s %s unexpected error: %s\n%s",
            backend, method, exc, short_tb,
        )


def _is_fatal_generation_error(exc: Exception) -> bool:
    text = str(exc).lower()
    fatal_patterns = [
        "expected all tensors to be on the same device",
        "input_ids is on cpu",
        "different from other tensors on cuda",
        "cuda out of memory",
        "device-side assert",
        "cublas",
        "cuda error",
        "device map",
        "device placement",
    ]
    return any(p in text for p in fatal_patterns)


def _summarize_generation_device_state(model, device: str, load_quantized: bool) -> str:
    emb_getter = getattr(model, "get_input_embeddings", None)
    emb_dev = None
    if callable(emb_getter):
        emb = emb_getter()
        weight = getattr(emb, "weight", None)
        if weight is not None:
            emb_dev = getattr(weight, "device", None)

    hf_map = getattr(model, "hf_device_map", None)
    map_preview = None
    if isinstance(hf_map, dict):
        items = list(hf_map.items())[:5]
        map_preview = ", ".join(f"{k}:{v}" for k, v in items)

    if emb_dev is not None:
        inferred = str(emb_dev)
    elif isinstance(hf_map, dict):
        for value in hf_map.values():
            if value not in (None, "cpu", "disk"):
                inferred = str(value)
                break
        else:
            inferred = "cpu"
    else:
        inferred = str(device)

    load_quantized_str = "4bit" if getattr(model, "_load_in_4bit", False) else ("8bit" if getattr(model, "_load_in_8bit", False) else "none")

    return (
        f"requested_device={device}, "
        f"inferred_input_device={inferred}, "
        f"quantization={load_quantized_str}, "
        f"has_hf_device_map={bool(hf_map)}, "
        f"hf_device_map_preview={map_preview or 'none'}"
    )


class HFCausalTranslator(Translator):
    """HuggingFace causal LM translator.

    Uses AutoModelForCausalLM with a prompt template for translation.
    Supports both plain-text prompts and chat-template formatting.
    All heavy dependencies are loaded at instantiation time, not at
    module import time.
    """

    def __init__(
        self,
        model_key: str,
        model_name: str,
        device: str = "cpu",
        tokenizer_max_length: int = 256,
        default_batch_size: int = 8,
        generation_kwargs: dict[str, Any] | None = None,
        local_files_only: bool = False,
        cache_dir: str | None = None,
        ignore_proxy_env: bool = False,
        src_lang: str = "Russian",
        tgt_lang: str = "English",
        prompt_style: str | None = None,
        use_chat_template: bool = False,
        trust_remote_code: bool = False,
        torch_dtype: str | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_key = model_key
        self.model_name = model_name
        self.device = device
        self.tokenizer_max_length = tokenizer_max_length
        self.default_batch_size = default_batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.supports_roundtrip = False
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self.ignore_proxy_env = ignore_proxy_env
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._prompt_style = prompt_style
        self._use_chat_template = use_chat_template
        self._prompt_template = _PROMPT_TEMPLATES.get(
            prompt_style or "", _DEFAULT_TEMPLATE
        )
        
        # Quantization config (used for summary output)
        self._torch_dtype = torch_dtype
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit

        try:
            import torch as _torch
        except ImportError as e:
            raise BackendUnavailableError(
                "HFCausalTranslator requires PyTorch. "
                "Install it with: pip install torch"
            ) from e
        self.torch = _torch

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise BackendUnavailableError(
                "HFCausalTranslator requires the 'transformers' package. "
                "Install it with: pip install transformers"
            ) from e

        logger.info(
            "Loading HFCausalTranslator: %s | device=%s | prompt_style=%s",
            model_name, device, prompt_style,
        )

        self.tokenizer, self.model = load_hf_model_causal(
            model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            device=device,
            ignore_proxy_env=ignore_proxy_env,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch=self.torch,
            AutoTokenizer=AutoTokenizer,
            AutoModelForCausalLM=AutoModelForCausalLM,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Print one-time model load summary
        self._print_model_load_summary()

    def _print_model_load_summary(self) -> None:
        """Print a human-readable summary of model loading configuration once at startup."""
        # Determine quantization mode
        load_quantized = self._load_in_4bit or self._load_in_8bit
        if self._load_in_4bit:
            quant_mode = "4bit"
        elif self._load_in_8bit:
            quant_mode = "8bit"
        else:
            quant_mode = "none"
        
        effective_load_mode = "quantized-" + quant_mode if load_quantized else "normal"
        
        # Determine device_map status
        uses_device_map = "auto" if load_quantized else "none"
        
        # bitsandbytes path
        bnb_path = "yes" if load_quantized else "no"
        
        # round-trip status
        roundtrip = "yes" if self.supports_roundtrip else "no"
        
        print()
        print("-" * 60)
        print("MODEL LOAD SUMMARY")
        print("-" * 60)
        print(f"Model key:                  {self.model_key}")
        print(f"Model name:                 {self.model_name}")
        print(f"Backend family:             hf_causal")
        print(f"Requested device:           {self.device}")
        print(f"Effective loading mode:     {effective_load_mode}")
        print(f"Quantization:               {quant_mode}")
        dtype_display = self._torch_dtype or "auto"
        print(f"torch dtype:                {dtype_display}")
        print(f"device_map:                 {uses_device_map}")
        print(f"default trust_remote_code:  False")  # Will be True if passed in
        print(f"Default batch size:         {self.default_batch_size}")
        print(f"Tokenizer max length:       {self.tokenizer_max_length}")
        print(f"Round-trip supported:       {roundtrip}")
        print(f"local_files_only:           {self.local_files_only}")
        print(f"ignore_proxy_env:           {self.ignore_proxy_env}")
        print(f"bitsandbytes path:          {bnb_path}")
        print()

    def _build_prompt_text(self, text: str) -> str:
        return self._prompt_template.format(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            text=text,
        )

    def _build_prompt(self, text: str) -> str:
        prompt_text = self._build_prompt_text(text)
        if self._use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                logger.warning(
                    "chat template application failed for %s, falling back to plain prompt",
                    self.model_name,
                )
        return prompt_text

    def _tokenize_and_generate(self, texts: list[str]) -> list[str]:
        prompts = [self._build_prompt(t) for t in texts]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer_max_length,
        )

        load_quantized = self._load_in_4bit or self._load_in_8bit
        target_device = self._get_inference_device()
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_kwargs,
            )

        input_lengths = inputs["input_ids"].shape[1]
        results: list[str] = []
        for i in range(outputs.shape[0]):
            generated_ids = outputs[i, input_lengths:]
            decoded = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            results.append(decoded.strip())
        return results

    def _get_inference_device(self) -> str | self.torch.device:
        emb_getter = getattr(self.model, "get_input_embeddings", None)
        if callable(emb_getter):
            emb = emb_getter()
            weight = getattr(emb, "weight", None)
            if weight is not None:
                return getattr(weight, "device", self.device)

        hf_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_map, dict):
            for value in hf_map.values():
                if value not in (None, "cpu", "disk"):
                    return self.torch.device(value) if isinstance(value, str) else value

        return self.torch.device(self.device) if isinstance(self.device, str) else self.device

    def translate(self, text: str) -> str | None:
        try:
            if not text or not text.strip():
                return None

            decoded = self._tokenize_and_generate([text])
            translated = decoded[0]
            return translated if translated else None
        except Exception as e:
            if _is_fatal_generation_error(e):
                device_state = _summarize_generation_device_state(
                    self.model, self.device, self._load_in_4bit or self._load_in_8bit
                )
                raise TranslatorRuntimeError(
                    f"HFCausalTranslator generation failed. "
                    f"model={self.model_name!r}; "
                    f"{device_state}; "
                    f"original_error={e}"
                ) from e
            _log_translate_error("HFCausalTranslator", "translate", e)
            return None

    def translate_batch(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> list[str | None]:
        if not texts:
            return []

        effective_batch_size = (
            batch_size if batch_size is not None else self.default_batch_size
        )
        results: list[str | None] = []

        for i in range(0, len(texts), effective_batch_size):
            batch_slice = list(texts[i:i + effective_batch_size])

            try:
                decoded = self._tokenize_and_generate(batch_slice)
                results.extend([t if t else None for t in decoded])
            except Exception as e:
                if _is_fatal_generation_error(e):
                    device_state = _summarize_generation_device_state(
                        self.model, self.device, self._load_in_4bit or self._load_in_8bit
                    )
                    raise TranslatorRuntimeError(
                        f"HFCausalTranslator batch generation failed for model "
                        f"{self.model_name!r}: {e}"
                    ) from e
                _log_translate_error("HFCausalTranslator", "translate_batch", e)
                results.extend([None] * len(batch_slice))

        return results
