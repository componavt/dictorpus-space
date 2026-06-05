"""
Translates all unique primary Russian glosses from VepKar meanings files
into English, using a pluggable model from the translator registry.
Results are saved to data/sem_cat/02_glosses_translated_{model_key}.csv.
Already-cached translations are never re-computed (incremental mode).
"""

import sys
import pathlib
import argparse
import random
from dataclasses import dataclass
from math import ceil
from typing import Literal

import pandas as pd

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
from src.sem_cat.translators.base import (
    BackendUnavailableError,
    Translator,
    TranslatorInitializationError,
)
from src.sem_cat.translators.diagnostics import (
    run_backend_diagnostics,
    summarize_diagnostics,
)
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


@dataclass(frozen=True)
class ReverseSetupResult:
    """Status of reverse translator initialization."""
    translator: Translator | None
    status: Literal["ready", "unsupported", "init_failed"]
    message: str | None = None


def _setup_reverse_translator(
    spec,
    device: str,
    retry: int,
    delay: float,
    local_files_only: bool,
    cache_dir: str | None,
    ignore_proxy_env: bool,
) -> ReverseSetupResult:
    """Attempt to build a reverse translator and return explicit status."""
    if not spec.supports_roundtrip or spec.reverse_model_name is None:
        return ReverseSetupResult(
            translator=None,
            status="unsupported",
            message="The model spec does not support round-trip translation.",
        )

    try:
        translator = build_reverse_translator(
            spec,
            device=device,
            retry=retry,
            delay=delay,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            ignore_proxy_env=ignore_proxy_env,
        )
    except (BackendUnavailableError, TranslatorInitializationError) as e:
        return ReverseSetupResult(
            translator=None,
            status="init_failed",
            message=f"Failed to initialize reverse translator: {e}",
        )

    if translator is None:
        return ReverseSetupResult(
            translator=None,
            status="unsupported",
            message="The model spec does not support round-trip translation.",
        )

    return ReverseSetupResult(
        translator=translator,
        status="ready",
        message=f"Round-trip enabled: back-translator built ({translator.model_key})",
    )


def _print_summary(
    total_unique: int,
    already_cached: int,
    remaining_after_cache: int,
    to_translate_count: int,
    total_written: int,
    total_kept: int,
    total_suspicious: int,
    total_rejected: int,
    total_empty_output: int,
    total_rejected_nonblank: int,
    total_roundtrip: int,
    flag_counts: dict[str, int],
    blanks_path: str | None = None,
) -> None:
    """Print final summary statistics."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total unique glosses:           {total_unique}")
    print(f"Already cached (skipped):       {already_cached}")
    print(f"Remaining after cache:          {remaining_after_cache}")
    print(f"Selected for this run:          {to_translate_count}")
    print(f"  (cache filter is applied first, then --offset/--limit)")
    print(f"Newly translated:               {total_written}")
    print(f"  - Kept (good quality):        {total_kept}")
    print(f"  - Kept (suspicious, flagged): {total_suspicious}")
    print(f"  - Rejected (qa_keep=False):   {total_rejected}")
    if total_rejected > 0:
        print(f"      • Empty output:           {total_empty_output}")
        print(f"      • Rejected but nonblank:  {total_rejected_nonblank}")
    print(f"  - With round-trip:            {total_roundtrip}")

    if flag_counts:
        print("  QA flag breakdown:")
        for flag, count in sorted(flag_counts.items()):
            print(f"    - {flag}: {count}")

    if total_empty_output > 0:
        print(f"\nEmpty output rows written to: {blanks_path}")


def _run_backend_info(
    spec,
    device: str,
    retry: int,
    delay: float,
    local_files_only: bool,
    cache_dir: str | None,
    ignore_proxy_env: bool,
) -> None:
    """Run backend diagnostics and print a readable summary."""
    print(f"\nRunning backend diagnostics for '{spec.model_name}'...")

    try:
        translator = build_translator(
            spec,
            device=device,
            retry=retry,
            delay=delay,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            ignore_proxy_env=ignore_proxy_env,
        )
    except (BackendUnavailableError, TranslatorInitializationError) as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    results = run_backend_diagnostics(translator)
    overall_status, message = summarize_diagnostics(results)

    print(f"\nDiagnostics: {overall_status}")
    print(message)

    if overall_status == "FAIL":
        sys.exit(1)
    elif overall_status == "WARN":
        print("\nTranslator is usable but produced suspicious output on some probes.")
    else:
        print("\nTranslator is working correctly.")


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
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for translation. Overrides model default if provided.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help='Device for local HuggingFace models: "cpu" or "cuda" (default: cpu)',
    )
    parser.add_argument(
        "--out-file", type=str, default=None,
        help=(
            "Full path to output CSV file. If provided, overrides --out-dir "
            "and the auto-generated filename. "
            "Blank/None translations are excluded from this file and written "
            "to <out_file>.blanks.csv instead."
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
    # HuggingFace model loading options
    parser.add_argument("--local-files-only", action="store_true", default=False,
                        help="Only use locally cached HF models (no network download)")
    parser.add_argument("--hf-cache-dir", type=str, default=None,
                        help="Custom cache directory for HuggingFace models")
    parser.add_argument("--ignore-proxy-env", action="store_true", default=False,
                        help="Temporarily unset proxy env vars during HF/NLLB model loading")
    parser.add_argument("--backend-info", action="store_true",
                        help="Run backend diagnostics with probe translations, then exit")

    args = parser.parse_args()

    # Validate batch size
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

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
        _run_backend_info(
            spec, args.device, retry, delay,
            args.local_files_only, args.hf_cache_dir,
            args.ignore_proxy_env,
        )
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
            local_files_only=args.local_files_only,
            cache_dir=args.hf_cache_dir,
            ignore_proxy_env=args.ignore_proxy_env,
        )
    except (BackendUnavailableError, TranslatorInitializationError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Set up reverse translator with explicit status
    reverse_result = ReverseSetupResult(translator=None, status="unsupported")
    if args.round_trip:
        reverse_result = _setup_reverse_translator(
            spec,
            device=args.device,
            retry=retry,
            delay=delay,
            local_files_only=args.local_files_only,
            cache_dir=args.hf_cache_dir,
            ignore_proxy_env=args.ignore_proxy_env,
        )

    if reverse_result.status == "ready":
        print(reverse_result.message)
    elif reverse_result.status == "unsupported":
        if args.round_trip:
            print(f"WARNING: {reverse_result.message}")
    elif reverse_result.status == "init_failed":
        print(f"WARNING: {reverse_result.message}")
        print("Continuing without round-trip QA.")

    back_translator = reverse_result.translator

    # 11. Build input texts
    input_texts: list[str] = []
    for gloss in glosses_to_translate:
        meta = metadata_map.get(gloss) if needs_metadata else None
        input_text = prepare_translation_input(gloss, args.translation_input_mode, meta)
        input_texts.append(input_text)

    # 12. Translate in batches
    print(f"Translating with {resolved_model_key} (mode: {args.translation_input_mode})...")

    # Batch size precedence:
    #   - Google backend: always 1 (API limitation)
    #   - Explicit CLI --batch-size: wins if positive
    #   - spec.default_batch_size: model default
    #   - Fallback: 1
    if spec.backend_family == "google":
        effective_batch_size = 1
    elif args.batch_size is not None and args.batch_size > 0:
        effective_batch_size = args.batch_size
    else:
        effective_batch_size = spec.default_batch_size or 1
    n = len(glosses_to_translate)
    n_batches = ceil(n / effective_batch_size) if n > 0 else 0

    header_written = already_cached > 0
    qa_config = TranslationQAConfig()

    total_written = 0
    total_kept = 0
    total_suspicious = 0
    total_rejected = 0
    total_empty_output = 0
    total_rejected_nonblank = 0
    total_roundtrip = 0
    flag_counts: dict[str, int] = {}

    for batch_idx in range(n_batches):
        batch_glosses = glosses_to_translate[
            batch_idx * effective_batch_size: (batch_idx + 1) * effective_batch_size
        ]
        batch_inputs = input_texts[
            batch_idx * effective_batch_size: (batch_idx + 1) * effective_batch_size
        ]

        # Forward translation - keep None alive
        raw_batch = translator.translate_batch(batch_inputs, batch_size=len(batch_inputs))

        # Back-translation - keep None alive
        if back_translator is not None:
            raw_back = back_translator.translate_batch(
                [t if t else "" for t in raw_batch],
                batch_size=len(raw_batch),
            )
            back_translated = list(raw_back)
        else:
            back_translated = [None] * len(raw_batch)

        # Build rows and write
        batch_rows = []
        for idx, (gloss_ru, input_text, trans) in enumerate(
            zip(batch_glosses, batch_inputs, raw_batch)
        ):
            roundtrip_text = back_translated[idx] if back_translator is not None else None

            qa_result = analyze_translation(gloss_ru, trans, roundtrip_text, config=qa_config)

            meta = metadata_map.get(gloss_ru) if needs_metadata else None

            row = build_translation_row(
                gloss_ru=gloss_ru,
                gloss_en=trans if trans else "",
                qa_result=qa_result,
                model_key=resolved_model_key,
                model_name=spec.model_name,
                backend_family=spec.backend_family,
                translation_input_mode=args.translation_input_mode,
                input_text_used=input_text,
                pos_hint=meta.dominant_pos if meta else None,
                meaning_hint=meta.meaning_hint if meta else None,
                source_count=meta.source_count if meta else None,
                gloss_ru_back=roundtrip_text if roundtrip_text else "",
            )
            batch_rows.append(row)

            if not qa_result.qa_keep:
                total_rejected += 1
                if not trans or not str(trans).strip():
                    total_empty_output += 1
                else:
                    total_rejected_nonblank += 1
            elif qa_result.qa_flags:
                total_suspicious += 1
            else:
                total_kept += 1

            if roundtrip_text:
                total_roundtrip += 1

            for flag in qa_result.qa_flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        batch_df = pd.DataFrame(batch_rows, columns=CANONICAL_COLUMNS)

        # Split rows: only save non-blank translations to the main cache.
        # Blank/None gloss_en values go to a sidecar .blanks.csv to prevent
        # them from being treated as cached successes on the next run.
        good_rows = [r for r in batch_rows if r.get("gloss_en", "").strip()]
        blank_rows = [r for r in batch_rows if not r.get("gloss_en", "").strip()]

        if good_rows:
            good_df = pd.DataFrame(good_rows, columns=CANONICAL_COLUMNS)
            good_df.to_csv(out_path, mode="a", header=not header_written, index=False, encoding="utf-8")
            if not header_written:
                header_written = True

        if blank_rows:
            blanks_path = out_path.with_suffix(out_path.suffix + ".blanks.csv")
            blanks_df = pd.DataFrame(blank_rows, columns=CANONICAL_COLUMNS)
            write_header = not blanks_path.exists()
            blanks_df.to_csv(blanks_path, mode="a", header=write_header, index=False, encoding="utf-8")
        total_written += len(batch_rows)
        print(f"  Batch {batch_idx + 1}/{n_batches} saved ({total_written} total written)")

    # 13. Print summary
    blanks_path = out_path.with_suffix(out_path.suffix + ".blanks.csv")
    _print_summary(
        total_unique=total_unique,
        already_cached=already_cached,
        remaining_after_cache=remaining_after_cache,
        to_translate_count=to_translate_count,
        total_written=total_written,
        total_kept=total_kept,
        total_suspicious=total_suspicious,
        total_rejected=total_rejected,
        total_empty_output=total_empty_output,
        total_rejected_nonblank=total_rejected_nonblank,
        total_roundtrip=total_roundtrip,
        flag_counts=flag_counts,
        blanks_path=str(blanks_path) if total_empty_output > 0 else None,
    )


if __name__ == "__main__":
    main()
