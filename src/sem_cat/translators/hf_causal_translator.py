"""HuggingFace causal LM translator for decoder-only translation models.

Handles models such as:
- Unbabel/TowerInstruct-13B-v0.1  (chat-template, instruction-style)
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

from .base import BackendUnavailableError, Translator, TranslatorInitializationError
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
            torch=self.torch,
            AutoTokenizer=AutoTokenizer,
            AutoModelForCausalLM=AutoModelForCausalLM,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

    def translate(self, text: str) -> str | None:
        try:
            if not text or not text.strip():
                return None

            decoded = self._tokenize_and_generate([text])
            translated = decoded[0]
            return translated if translated else None
        except Exception as e:
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
                _log_translate_error("HFCausalTranslator", "translate_batch", e)
                results.extend([None] * len(batch_slice))

        return results
