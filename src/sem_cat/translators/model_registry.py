"""Model registry for translation backends.

Defines ModelSpec dataclass and a registry of supported translation models.
Also provides helpers for resolving legacy CLI arguments to model keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BackendFamily = Literal["google", "hf_seq2seq", "nllb", "hf_causal"]


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a translation model."""
    model_key: str
    backend_family: BackendFamily
    model_name: str
    src_lang: str
    tgt_lang: str
    reverse_model_name: str | None = None
    reverse_src_lang: str | None = None
    reverse_tgt_lang: str | None = None
    default_batch_size: int = 32
    tokenizer_max_length: int = 64
    generation_preset: str = "gloss_strict"
    supports_roundtrip: bool = True

    # Causal LM specific (ignored for non-causal backends)
    prompt_style: str | None = None
    use_chat_template: bool = False
    trust_remote_code: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "google": ModelSpec(
        model_key="google",
        backend_family="google",
        model_name="google",
        src_lang="ru",
        tgt_lang="en",
        reverse_model_name="google",
        reverse_src_lang="en",
        reverse_tgt_lang="ru",
        default_batch_size=1,
        tokenizer_max_length=0,
        generation_preset="default",
        supports_roundtrip=True,
    ),
    "helsinki_opus_mt_ru_en": ModelSpec(
        model_key="helsinki_opus_mt_ru_en",
        backend_family="hf_seq2seq",
        model_name="Helsinki-NLP/opus-mt-ru-en",
        src_lang="ru",
        tgt_lang="en",
        reverse_model_name="Helsinki-NLP/opus-mt-en-ru",
        reverse_src_lang="en",
        reverse_tgt_lang="ru",
        default_batch_size=64,
        tokenizer_max_length=64,
        generation_preset="gloss_strict",
        supports_roundtrip=True,
    ),
    "nllb_3_3b": ModelSpec(
        model_key="nllb_3_3b",
        backend_family="nllb",
        model_name="facebook/nllb-200-3.3B",
        src_lang="rus_Cyrl",
        tgt_lang="eng_Latn",
        reverse_model_name="facebook/nllb-200-3.3B",
        reverse_src_lang="eng_Latn",
        reverse_tgt_lang="rus_Cyrl",
        default_batch_size=8,
        tokenizer_max_length=128,
        generation_preset="gloss_strict",
        supports_roundtrip=True,
    ),
    "tower_plus_9b": ModelSpec(
        model_key="tower_plus_9b",
        backend_family="hf_causal",
        model_name="Unbabel/Tower-Plus-9B",
        src_lang="Russian",
        tgt_lang="English",
        reverse_model_name=None,
        reverse_src_lang=None,
        reverse_tgt_lang=None,
        default_batch_size=4,
        tokenizer_max_length=256,
        generation_preset="gloss_strict",
        supports_roundtrip=False,
        prompt_style="tower_chatml",
        use_chat_template=True,
    ),
    "hy_mt2_30b_a3b": ModelSpec(
        model_key="hy_mt2_30b_a3b",
        backend_family="hf_causal",
        model_name="tencent/Hy-MT2-30B-A3B",
        src_lang="Russian",
        tgt_lang="English",
        reverse_model_name=None,
        reverse_src_lang=None,
        reverse_tgt_lang=None,
        default_batch_size=4,
        tokenizer_max_length=256,
        generation_preset="gloss_strict",
        supports_roundtrip=False,
        prompt_style="hy_chat",
        trust_remote_code=True,
    ),
    "alma_7b_r": ModelSpec(
        model_key="alma_7b_r",
        backend_family="hf_causal",
        model_name="haoranxu/ALMA-7B-R",
        src_lang="Russian",
        tgt_lang="English",
        reverse_model_name=None,
        reverse_src_lang=None,
        reverse_tgt_lang=None,
        default_batch_size=4,
        tokenizer_max_length=256,
        generation_preset="gloss_strict",
        supports_roundtrip=False,
        prompt_style="alma_plain",
    ),
}


def get_model_spec(model_key: str) -> ModelSpec:
    """Return the ModelSpec for a given model key.

    Raises:
        ValueError: If model_key is not in the registry.
    """
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model key: {model_key!r}. Available: {available}"
        )
    return MODEL_REGISTRY[model_key]


def list_model_keys() -> list[str]:
    """Return all registered model keys."""
    return list(MODEL_REGISTRY.keys())


def resolve_legacy_args_to_model_key(
    backend: str | None,
    nllb_model: str | None,
) -> str:
    """Resolve legacy --backend and --nllb-model arguments to a model key.

    Supports old usage patterns:
        --backend google                              -> google
        --backend marian                              -> helsinki_opus_mt_ru_en
        --backend nllb --nllb-model facebook/nllb-200-3.3B -> nllb_3_3b
    """
    nllb_model_map = {
        "facebook/nllb-200-3.3B": "nllb_3_3b",
    }

    if backend == "google":
        return "google"
    if backend == "marian":
        return "helsinki_opus_mt_ru_en"
    if backend == "nllb":
        model = nllb_model or "facebook/nllb-200-3.3B"
        if model in nllb_model_map:
            return nllb_model_map[model]
        raise ValueError(
            f"Unknown NLLB model: {model!r}. "
            f"Supported: {', '.join(nllb_model_map.keys())}"
        )
    # Default for backward compatibility
    return "helsinki_opus_mt_ru_en"
