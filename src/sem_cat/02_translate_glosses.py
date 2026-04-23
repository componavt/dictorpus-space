"""
Translates all unique primary Russian glosses from VepKar meanings files
into English, using a pluggable model from the translator registry.
Results are saved to data/sem_cat/02_glosses_translated_{model_key}.csv.
Already-cached translations are never re-computed (incremental mode).
"""

import sys
import pathlib
import argparse
import time
import random
from math import ceil

import pandas as pd

# Add project root to sys.path to allow absolute imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Anchor all default paths to the project root (2 levels up from this file).
_THIS_FILE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "vepkar"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "data" / "sem_cat"

from src.sem_cat.utils.vepkar_loader import load_meanings
from src.sem_cat.translators.model_registry import (
    get_model_spec,
    list_model_keys,
    resolve_legacy_args_to_model_key,
)
from src.sem_cat.translators.factory import build_translator, build_reverse_translator
from src.sem_cat.translators.base import Translator
from src.sem_cat.pipeline.translation_input import (
    extract_unique_primary_glosses,
    build_gloss_metadata_map,
    prepare_translation_input,
    GlossMetadata,
)
from src.sem_cat.qa.translation_qa import (
    analyze_translation,
    TranslationQAConfig,
    QAResult,
)
from src.sem_cat.io.translation_cache import (
    load_translation_cache,
    build_cached_gloss_set,
    count_cached_rows,
)
from src.sem_cat.io.translation_rows import (
    build_translation_row,
    CANONICAL_COLUMNS,
    QA_VERSION,
)


def _print_summary(
    total_unique: int,
    already_cached: int,
    remaining_after_cache: int,
    to_translate_count: int,
    total_written: int,
    total_kept: int,
    total_suspicious: int,
    total_blank: int,
    total_roundtrip: int,
    flag_counts: dict[str, int],
) -> None:
    """Print final summary statistics."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total unique glosses:           {total_unique}")
    print(f"Already cached:                 {already_cached}")
    print(f"Remaining after cache:          {remaining_after_cache}")
    print(f"Selected for this run:          {to_translate_count}")
    print(f"Newly translated:               {total_written}")
    print(f"  - Kept (good quality):        {total_kept}")
    print(f"  - Kept (suspicious, flagged): {total_suspicious}")
    print(f"  - Unusable (qa_keep=False):   {total_blank}")
    print(f"  - With round-trip:            {total_roundtrip}")

    if flag_counts:
        print("  QA flag breakdown:")
        for flag, count in sorted(flag_counts.items()):
            print(f"    - {flag}: {count}")


def main() -> None:
    model_keys = list_model_keys()

    parser = argparse.ArgumentParser(description="Translate Russian glosses to English")
    parser.add_argument("--data-dir", type=str, default=str(_DEFAULT_DATA_DIR),
                        help=f"path to data/vepkar/ (default: {_DEFAULT_DATA_DIR})")
    parser.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR),
                        help=f"output directory for translated CSV (default: {_DEFAULT_OUT_DIR})")
    parser.add_argument("--model-key", type=str, choices=model_keys, default=None,
                        help=f"translation model key (default: resolved from --backend)")
    parser.add_argument("--backend", type=str, choices=["marian", "google", "nllb"], default="marian",
                        help='legacy: translation backend (default: marian). Prefer --model-key.')
    parser.add_argument(
        "--nllb-model", type=str, default="facebook/nllb-200-distilled-1.3B",
        help="legacy: NLLB model name (used with --backend nllb). Prefer --model-key.",
    )
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size for translation (default: 64)")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help='Device for local HuggingFace models: "cpu" or "cuda" (default: cpu)',
    )
    parser.add_argument(
        "--out-file", type=str, default=None,
        help=(
            "Full path to output CSV file. If provided, overrides --out-dir "
            "and the auto-generated filename."
        ),
    )
    parser.add_argument(
        "--round-trip", action="store_true", default=False,
        help="also back-translate gloss_en -> ru for quality checking",
    )
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N glosses after cache filtering (default: 0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N glosses after offset (default: None = all)")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Shuffle glosses_to_translate before applying offset/limit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used with --shuffle (default: 42)")
    parser.add_argument("--gloss-filter", type=str, default=None,
                        help="Optional substring filter applied to gloss_ru before translation")
    parser.add_argument(
        "--translation-input-mode", type=str, choices=["raw", "pos", "pos_meaning"],
        default="raw",
        help="How to prepare input for translator (default: raw)",
    )
    parser.add_argument("--debug-sample", type=int, default=0,
                        help="Print raw translation output for first N items (default: 0 = off)")
    # Generic retry flags with backward-compatible aliases
    parser.add_argument("--retry", type=int, default=None,
                        help="Number of retry attempts for failed responses (default: backend-specific)")
    parser.add_argument("--retry-delay", type=float, default=None,
                        help="Additional sleep in seconds before each retry (default: backend-specific)")
    # Backward-compatible aliases
    parser.add_argument("--google-retries", type=int, default=2,
                        help="legacy alias for --retry (default: 2)")
    parser.add_argument("--google-retry-delay", type=float, default=1.0,
                        help="legacy alias for --retry-delay (default: 1.0)")
    parser.add_argument("--backend-info", action="store_true",
                        help="Print model configuration, run a single test translation, then exit")

    args = parser.parse_args()

    # Resolve retry/delay: generic flags take precedence over legacy aliases
    retry = args.retry if args.retry is not None else args.google_retries
    delay = args.retry_delay if args.retry_delay is not None else args.google_retry_delay

    # 1. Resolve model spec
    resolved_model_key = args.model_key or resolve_legacy_args_to_model_key(
        backend=args.backend,
        nllb_model=args.nllb_model,
    )
    spec = get_model_spec(resolved_model_key)

    # 2. Compute output path
    if args.out_file:
        out_path = pathlib.Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"02_glosses_translated_{resolved_model_key}.csv"

    print(f"Model key: {resolved_model_key}")
    print(f"Model name: {spec.model_name}")
    print(f"Backend family: {spec.backend_family}")
    print(f"Output file: {out_path}")

    # 3. Handle --backend-info
    if args.backend_info:
        print(f"\nRunning test translation: 'dom' -> expected 'house'")
        try:
            test_translator = build_translator(
                spec,
                device=args.device,
                retry=retry,
                delay=delay,
            )
        except (ImportError, ValueError) as e:
            print(f"Model unavailable: {e}")
            sys.exit(1)
        try:
            test_result = test_translator.translate("dom")
            if test_result and test_result.strip():
                status = "OK"
            else:
                status = "EMPTY"
        except Exception as e:
            test_result = f"ERROR: {type(e).__name__}: {e}"
            status = "ERROR"
        print(f"Test input:  dom")
        print(f"Test output: {test_result!r}")
        print(f"Status: {status}")
        return

    # 4. Validate data directory
    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    # 5. Load meanings
    print("Loading meanings...")
    df_meanings = load_meanings(str(data_dir))

    # 6. Build gloss metadata map if needed
    needs_metadata = args.translation_input_mode in ("pos", "pos_meaning")
    metadata_map: dict[str, GlossMetadata] = {}
    if needs_metadata:
        print("Building gloss metadata for context-aware translation...")
        metadata_map = build_gloss_metadata_map(df_meanings)
        print(f"Built metadata for {len(metadata_map)} unique glosses")

    # 7. Extract unique primary glosses
    print("Extracting unique primary glosses...")
    unique_glosses = extract_unique_primary_glosses(df_meanings)
    total_unique = len(unique_glosses)
    print(f"Found {total_unique} unique glosses")

    # 8. Load cache and filter already translated
    cache_df = load_translation_cache(out_path, expected_model_key=resolved_model_key)
    cached_glosses = build_cached_gloss_set(cache_df)
    already_cached = count_cached_rows(cache_df)
    if already_cached > 0:
        print(f"Found {already_cached} cached translations")

    glosses_to_translate = [g for g in unique_glosses if g not in cached_glosses]
    remaining_after_cache = len(glosses_to_translate)
    print(f"Remaining after cache: {remaining_after_cache}")

    # 9. Apply gloss filter / shuffle / offset / limit
    if args.gloss_filter:
        glosses_to_translate = [g for g in glosses_to_translate if args.gloss_filter in g]
        print(f"After gloss filter '{args.gloss_filter}': {len(glosses_to_translate)}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(glosses_to_translate)
        print(f"Shuffled glosses with seed {args.seed}")

    if args.offset > 0:
        glosses_to_translate = glosses_to_translate[args.offset:]
        print(f"After offset {args.offset}: {len(glosses_to_translate)}")

    if args.limit is not None:
        glosses_to_translate = glosses_to_translate[:args.limit]
        print(f"After limit {args.limit}: {len(glosses_to_translate)}")

    to_translate_count = len(glosses_to_translate)
    print(f"Selected for this run: {to_translate_count}")

    if to_translate_count == 0:
        print("No new glosses to translate. Exiting.")
        return

    # 10. Build translators
    try:
        translator = build_translator(
            spec,
            device=args.device,
            retry=retry,
            delay=delay,
        )
    except (ImportError, ValueError) as e:
        print(f"Failed to build translator: {e}")
        sys.exit(1)

    back_translator: Translator | None = None
    if args.round_trip and spec.supports_roundtrip:
        back_translator = build_reverse_translator(
            spec,
            device=args.device,
            retry=retry,
            delay=delay,
        )
        if back_translator is not None:
            print(f"Round-trip enabled: back-translator built ({back_translator.model_key})")
        else:
            print("Round-trip requested but reverse model unavailable for this spec.")

    # 11. Build input texts
    input_texts: list[str] = []
    for gloss in glosses_to_translate:
        meta = metadata_map.get(gloss) if needs_metadata else None
        input_text = prepare_translation_input(gloss, args.translation_input_mode, meta)
        input_texts.append(input_text)

    # 12. Translate in batches
    print(f"Translating with {resolved_model_key} (mode: {args.translation_input_mode})...")

    effective_batch_size = spec.default_batch_size if spec.default_batch_size else args.batch_size
    n = len(glosses_to_translate)
    n_batches = ceil(n / effective_batch_size) if n > 0 else 0

    header_written = already_cached > 0
    qa_config = TranslationQAConfig()

    total_written = 0
    total_kept = 0
    total_suspicious = 0
    total_blank = 0
    total_roundtrip = 0
    flag_counts: dict[str, int] = {}

    for batch_idx in range(n_batches):
        batch_glosses = glosses_to_translate[
            batch_idx * effective_batch_size: (batch_idx + 1) * effective_batch_size
        ]
        batch_inputs = input_texts[
            batch_idx * effective_batch_size: (batch_idx + 1) * effective_batch_size
        ]

        # Forward translation - use the common interface
        raw_batch = translator.translate_batch(batch_inputs, batch_size=len(batch_inputs))
        translated_texts = [t or "" for t in raw_batch]

        # Back-translation - use the common interface
        if back_translator is not None and args.round_trip:
            raw_back = back_translator.translate_batch(
                translated_texts, batch_size=len(translated_texts)
            )
            back_translated = [t or "" for t in raw_back]
        else:
            back_translated = [None] * len(translated_texts)

        # Build rows and write
        batch_rows = []
        for idx, (gloss_ru, input_text, trans) in enumerate(
            zip(batch_glosses, batch_inputs, translated_texts)
        ):
            roundtrip_text = back_translated[idx] if args.round_trip else None

            qa_result = analyze_translation(gloss_ru, trans, roundtrip_text, config=qa_config)

            meta = metadata_map.get(gloss_ru) if needs_metadata else None

            row = build_translation_row(
                gloss_ru=gloss_ru,
                gloss_en=trans,
                qa_result=qa_result,
                model_key=resolved_model_key,
                model_name=spec.model_name,
                backend_family=spec.backend_family,
                translation_input_mode=args.translation_input_mode,
                input_text_used=input_text,
                pos_hint=meta.dominant_pos if meta else None,
                meaning_hint=meta.meaning_hint if meta else None,
                source_count=meta.source_count if meta else None,
                gloss_ru_back=roundtrip_text,
            )
            batch_rows.append(row)

            if not qa_result.qa_keep:
                total_blank += 1
            elif qa_result.qa_flags:
                total_suspicious += 1
            else:
                total_kept += 1

            if roundtrip_text:
                total_roundtrip += 1

            for flag in qa_result.qa_flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        batch_df = pd.DataFrame(batch_rows, columns=CANONICAL_COLUMNS)
        batch_df.to_csv(out_path, mode="a", header=not header_written, index=False, encoding="utf-8")
        if not header_written:
            header_written = True
        total_written += len(batch_rows)
        print(f"  Batch {batch_idx + 1}/{n_batches} saved ({total_written} total written)")

    # 13. Print summary
    _print_summary(
        total_unique=total_unique,
        already_cached=already_cached,
        remaining_after_cache=remaining_after_cache,
        to_translate_count=to_translate_count,
        total_written=total_written,
        total_kept=total_kept,
        total_suspicious=total_suspicious,
        total_blank=total_blank,
        total_roundtrip=total_roundtrip,
        flag_counts=flag_counts,
    )


if __name__ == "__main__":
    main()
