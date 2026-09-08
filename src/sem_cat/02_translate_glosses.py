"""
Translates VepKar meanings from a fixed task file.
Reads pos_meanings_ru.csv and produces model-specific translation CSV.

Workflow:
  pos_meanings_ru.csv
    -> strict reader
    -> TranslationTaskMetadata(pos, meaning_ru)
    -> cache filtering by (pos, meaning_ru)
    -> optional shuffle
    -> offset
    -> limit
    -> translation
    -> QA using meaning_ru
    -> model-specific translation CSV

Output cache file uses schema:
  pos, meaning_ru, meaning_en, qa_keep, qa_score, qa_flags, meaning_ru_back, roundtrip_distance
"""

import sys
import pathlib
import argparse
import dataclasses
import random
import re
import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

_THIS_FILE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent

DEFAULT_TASKS_FILE = _PROJECT_ROOT / "data" / "sem_cat" / "2translate" / "pos_meanings_ru.csv"

from src.sem_cat.translators.model_registry import (
    get_model_spec,
    list_model_keys,
    resolve_legacy_args_to_model_key,
    ModelSpec,
)
from src.sem_cat.translators.factory import build_translator, build_reverse_translator
from src.sem_cat.translators.base import (
    BackendUnavailableError,
    Translator,
    TranslatorInitializationError,
    TranslatorRuntimeError,
)
from src.sem_cat.translators.diagnostics import (
    run_backend_diagnostics,
    summarize_diagnostics,
)
from src.sem_cat.qa.translation_qa import (
    analyze_translation,
    TranslationQAConfig,
    QAResult,
)
from src.sem_cat.io.pos_meaning_ru_reader import read_pos_meaning_ru_tasks
from src.sem_cat.io.translation_cache import (
    load_translation_cache,
    count_cached_rows,
    TranslationCacheLoadResult,
    build_cached_identity_set,
)
from src.sem_cat.io.translation_rows import (
    build_translation_row,
    CANONICAL_COLUMNS,
)
from src.sem_cat.pipeline.vepkar_translation_selection import (
    TranslationTaskMetadata,
    prepare_translation_input_for_task,
    build_translation_tasks_from_pos_meaning_ru,
)

_POS_PREFIX_RE = re.compile(r"^(NOUN|VERB|ADJ|ADV|PROPN|PRON|NUM|PART|INTJ|ADP|AUX|CCONJ|SCONJ|DET)\s+[|:-]?\s*(.+)$")


def strip_pos_echo_prefix(text: str) -> str:
    """Strip POS prefix echo from translation output.
    
    Some models repeat the POS label as a prefix in the English output.
    This helper detects and removes that pattern while preserving
    legitimate content that starts with uppercase words.
    """
    s = str(text or "").strip()
    if not s:
        return s
    m = _POS_PREFIX_RE.match(s)
    if not m:
        return s
    cleaned = m.group(2).strip()
    return cleaned or s


def _setup_reverse_translator(
    spec,
    device: str,
    retry: int,
    delay: float,
    local_files_only: bool,
    cache_dir: str | None,
    ignore_proxy_env: bool,
) -> "ReverseSetupResult":
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
    tasks_in_file: int,
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
) -> None:
    """Print final translation processing summary."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Translation processing:")
    print(f"  Tasks in fixed input file:    {tasks_in_file}")
    print(f"  Already cached (skipped):     {already_cached}")
    print(f"  Remaining after cache:        {remaining_after_cache}")
    print(f"  Selected for this run:        {to_translate_count}")
    print(f"  Newly translated:             {total_written}")
    print(f"    - Kept (good quality):      {total_kept}")
    print(f"    - Kept (suspicious):        {total_suspicious}")
    print(f"    - Rejected (qa_keep=False): {total_rejected}")
    if total_rejected > 0:
        print(f"        • Empty output:         {total_empty_output}")
        print(f"        • Rejected nonblank:    {total_rejected_nonblank}")
    print(f"    - With round-trip:          {total_roundtrip}")

    if flag_counts:
        print("  QA flag breakdown:")
        for flag, count in sorted(flag_counts.items()):
            print(f"    - {flag}: {count}")


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


def _causal_generation_preflight(
    translator: Translator,
    prepared_inputs: list[str],
    model_key: str,
    backend_family: str,
    *,
    effective_batch_size: int,
) -> None:
    """Run a causal generation preflight probe to catch runtime failures early."""
    if backend_family != "hf_causal":
        return

    nonempty = [x for x in prepared_inputs if x and x.strip()]
    if not nonempty:
        return

    target_probe_size = max(1, effective_batch_size)
    sample = nonempty[:target_probe_size]
    probe_batch_size = min(target_probe_size, len(sample))

    try:
        outputs = translator.translate_batch(sample, batch_size=probe_batch_size)
    except TranslatorRuntimeError as e:
        print(f"ERROR: Causal generation preflight failed for {model_key!r}: {e}")
        sys.exit(1)

    if not any(o and o.strip() for o in outputs):
        raise TranslatorRuntimeError(
            f"Causal backend {model_key!r} loaded, but preflight generation produced "
            "no non-empty outputs. Aborting before full run."
        )


def _should_abort_for_early_empty_run(
    *,
    backend_family: str,
    batches_seen: int,
    total_written: int,
    total_empty_output: int,
    total_kept: int,
    min_rows: int = 16,
    max_batches: int = 4,
    empty_ratio_threshold: float = 0.95,
) -> bool:
    """Detect pathological early-empty runs and abort with clear message."""
    if backend_family != "hf_causal":
        return False
    if batches_seen > max_batches:
        return False
    if total_written < min_rows:
        return False
    if total_kept > 0:
        return False
    if total_empty_output == 0:
        return False
    return (total_empty_output / total_written) >= empty_ratio_threshold


@dataclass(frozen=True)
class ReverseSetupResult:
    """Status of reverse translator initialization."""
    translator: Translator | None
    status: Literal["ready", "unsupported", "init_failed"]
    message: str | None = None


def main() -> None:
    model_keys = list_model_keys()

    parser = argparse.ArgumentParser(
        description="Translate VepKar meanings to English from fixed task file"
    )
    parser.add_argument("--out-dir", type=str, default=str(_PROJECT_ROOT / "data" / "sem_cat"),
                        help=f"output directory for translated CSV")
    parser.add_argument("--model-key", type=str, choices=model_keys, default=None,
                        help=f"translation model key (default: resolved from --backend)")
    parser.add_argument("--backend", type=str, choices=["marian", "google", "nllb"], default="marian",
                        help='legacy: translation backend (default: marian). Prefer --model-key.')
    parser.add_argument(
        "--nllb-model", type=str, default="facebook/nllb-200-3.3B",
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
        "--round-trip", action="store_true", default=False,
        help="also back-translate meaning_en -> ru for quality checking",
    )
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N tasks after cache filtering (default: 0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N tasks after offset (default: None = all)")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Shuffle tasks before applying offset/limit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used with --shuffle (default: 42)")
    parser.add_argument("--debug-sample", type=int, default=0,
                        help="Print raw translation output for first N items (default: 0 = off)")
    parser.add_argument("--retry", type=int, default=None,
                        help="Number of retry attempts for failed responses (default: backend-specific)")
    parser.add_argument("--retry-delay", type=float, default=None,
                        help="Additional sleep in seconds before each retry (default: backend-specific)")
    parser.add_argument("--google-retries", type=int, default=2,
                        help="legacy alias for --retry (default: 2)")
    parser.add_argument("--google-retry-delay", type=float, default=1.0,
                        help="legacy alias for --retry-delay (default: 1.0)")
    parser.add_argument("--local-files-only", action="store_true", default=False,
                        help="Only use locally cached HF models (no network download)")
    parser.add_argument("--hf-cache-dir", type=str, default=None,
                        help="Custom cache directory for HuggingFace models")
    parser.add_argument("--ignore-proxy-env", action="store_true", default=False,
                        help="Temporarily unset proxy env vars during HF/NLLB model loading")
    parser.add_argument("--backend-info", action="store_true",
                        help="Run backend diagnostics with probe translations, then exit")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["none", "4bit", "8bit"],
        help="Quantization mode for hf_causal models (default: registry setting)",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        help="Variant/override for model name (e.g., '4bit', '8bit') - only for hf_causal models",
    )

    args = parser.parse_args()

    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    retry = args.retry if args.retry is not None else args.google_retries
    delay = args.retry_delay if args.retry_delay is not None else args.google_retry_delay

    resolved_model_key = args.model_key or resolve_legacy_args_to_model_key(
        backend=args.backend,
        nllb_model=args.nllb_model,
    )
    spec = get_model_spec(resolved_model_key)
    
    if args.quantization is not None and spec.backend_family == "hf_causal":
        if args.quantization == "4bit":
            spec = ModelSpec(**{**spec.__dict__, "load_in_4bit": True, "load_in_8bit": False})
        elif args.quantization == "8bit":
            spec = ModelSpec(**{**spec.__dict__, "load_in_4bit": False, "load_in_8bit": True})
        elif args.quantization == "none":
            spec = ModelSpec(**{**spec.__dict__, "load_in_4bit": False, "load_in_8bit": False})
    elif args.quantization is not None and spec.backend_family != "hf_causal":
        print(f"ERROR: --quantization is only valid for hf_causal models, not {spec.backend_family}")
        sys.exit(1)
    
    if args.model_variant is not None and spec.backend_family == "hf_causal":
        print(f"NOTE: Using model variant override: {args.model_variant}")
        spec = ModelSpec(**{**spec.__dict__, "model_name": args.model_variant})
    elif args.model_variant is not None and spec.backend_family != "hf_causal":
        print(f"ERROR: --model-variant is only valid for hf_causal models, not {spec.backend_family}")
        sys.exit(1)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"02_meanings_translated_{resolved_model_key}.csv"

    print(f"Model key: {resolved_model_key}")
    print(f"Model name: {spec.model_name}")
    print(f"Backend family: {spec.backend_family}")
    print(f"Output file: {out_path}")

    if args.backend_info:
        _run_backend_info(
            spec, args.device, retry, delay,
            args.local_files_only, args.hf_cache_dir,
            args.ignore_proxy_env,
        )
        return

    tasks_file = DEFAULT_TASKS_FILE
    print(f"Translation tasks file: {tasks_file}")
    
    if not tasks_file.exists():
        print(f"ERROR: Tasks file not found: {tasks_file}")
        sys.exit(1)
    
    print("Loading translation tasks...")
    task_df = read_pos_meaning_ru_tasks(tasks_file)
    tasks_in_file = len(task_df)
    print(f"  Loaded {tasks_in_file} tasks")
    
    if tasks_in_file == 0:
        print("No tasks to translate. Exiting.")
        return
    
    print("Building translation tasks...")
    tasks = build_translation_tasks_from_pos_meaning_ru(task_df)
    print(f"  Built {len(tasks)} task objects")
    
    print(f"Resolved model key: {resolved_model_key}")
    print("Loading and validating translation cache...")
    cache_result = load_translation_cache(out_path)
    cache_df = cache_result.df
    
    writer_mode: Literal["start_new_file", "append_to_existing_valid_file", "abort_due_to_malformed_existing_file"]
    
    if cache_result.state == "missing":
        writer_mode = "start_new_file"
    elif cache_result.state == "malformed":
        writer_mode = "abort_due_to_malformed_existing_file"
    else:
        writer_mode = "append_to_existing_valid_file"
    
    if writer_mode == "abort_due_to_malformed_existing_file":
        detected_cols = cache_result.columns or []
        print(f"\nFATAL: Output file exists but is malformed: {out_path}")
        print(f"  Detected columns: {list(detected_cols)}")
        print(f"  Validation error: {cache_result.reason}")
        print("  Likely cause: a previous run wrote data rows before the CSV header,")
        print("  or the file uses the obsolete gloss-based schema.")
        print("  Action: Remove or rename the file, then rerun.")
        sys.exit(1)
    
    need_header_for_good_output = writer_mode == "start_new_file"

    print("Filtering by cache...")
    cached_ids = build_cached_identity_set(cache_df) if not cache_df.empty else set()
    tasks_to_translate = [t for t in tasks if (t.pos, t.meaning_ru) not in cached_ids]
    remaining_after_cache = len(tasks_to_translate)
    cached_count = len(tasks) - remaining_after_cache
    print(f"  Already cached: {cached_count}")
    print(f"  Remaining after cache: {remaining_after_cache}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(tasks_to_translate)
        print(f"  Shuffled tasks with seed {args.seed}")

    if args.offset > 0:
        tasks_to_translate = tasks_to_translate[args.offset:]
        print(f"  After offset {args.offset}: {len(tasks_to_translate)}")

    if args.limit is not None:
        tasks_to_translate = tasks_to_translate[:args.limit]
        print(f"  After limit {args.limit}: {len(tasks_to_translate)}")

    to_translate_count = len(tasks_to_translate)
    print(f"  Selected for this run: {to_translate_count}")

    if to_translate_count == 0:
        print("No new tasks to translate. Exiting.")
        return

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

    if spec.backend_family == "google":
        effective_batch_size = 1
    elif args.batch_size is not None and args.batch_size > 0:
        effective_batch_size = args.batch_size
    else:
        effective_batch_size = spec.default_batch_size or 1

    print(f"Translating with {resolved_model_key}...")
    print(f"Effective batch size: {effective_batch_size}")

    input_texts: list[str] = []
    for task in tasks_to_translate:
        input_text = prepare_translation_input_for_task(task)
        input_texts.append(input_text)

    _causal_generation_preflight(
        translator,
        input_texts,
        resolved_model_key,
        spec.backend_family,
        effective_batch_size=effective_batch_size,
    )

    n = len(tasks_to_translate)
    n_batches = math.ceil(n / effective_batch_size) if n > 0 else 0
    
    file_exists = out_path.exists()
    file_size = out_path.stat().st_size if file_exists else 0
    
    print("\nOutput/cache status")
    print(f"  Output path: {out_path}")
    print(f"  Exists: {'yes' if file_exists else 'no'}")
    if file_exists:
        print(f"  Size bytes: {file_size}")
    print(f"  Cache status: {cache_result.state}")
    print(f"  Detected columns: {list(cache_result.columns) if cache_result.columns else []}")
    print(f"  Valid cache rows: {len(cache_df)}")
    print(f"  Cached task count: {len(cached_ids)}")
    print(f"  Writer mode: {writer_mode}")
    print(f"  First good batch writes header: {need_header_for_good_output}")
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
        batch_tasks = tasks_to_translate[
            batch_idx * effective_batch_size: (batch_idx + 1) * effective_batch_size
        ]
        batch_inputs = input_texts[
            batch_idx * effective_batch_size: (batch_idx + 1) * effective_batch_size
        ]

        raw_batch = translator.translate_batch(batch_inputs, batch_size=len(batch_inputs))

        if back_translator is not None:
            raw_back = back_translator.translate_batch(
                [t if t else "" for t in raw_batch],
                batch_size=len(raw_batch),
            )
            back_translated = list(raw_back)
        else:
            back_translated = [None] * len(raw_batch)

        batch_rows = []
        for task, input_text, trans in zip(batch_tasks, batch_inputs, raw_batch):
            batch_idx_in_tasks = list(batch_tasks).index(task)
            roundtrip_text = back_translated[batch_idx_in_tasks] if back_translator is not None else None

            trans_clean = strip_pos_echo_prefix(trans if trans else "")
            qa_result = analyze_translation(
                task.meaning_ru,
                trans_clean,
                roundtrip_text,
                config=qa_config
            )

            row = build_translation_row(
                pos=task.pos,
                meaning_ru=task.meaning_ru,
                meaning_en=trans_clean,
                qa_result=qa_result,
                meaning_ru_back=roundtrip_text if roundtrip_text else "",
                roundtrip_distance=qa_result.roundtrip_distance,
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

        good_rows = [r for r in batch_rows if r.get("meaning_en", "").strip()]
        blank_rows = [r for r in batch_rows if not r.get("meaning_en", "").strip()]
        
        print(f"  Batch schema check: rows={len(batch_rows)}, blank_pos={sum(1 for r in batch_rows if not str(r.get('pos', '')).strip())}, blank_meaning_ru={sum(1 for r in batch_rows if not str(r.get('meaning_ru', '')).strip())}, blank_meaning_en={sum(1 for r in good_rows if not str(r.get('meaning_en', '')).strip())}")
        if batch_rows:
            print("   - preview:")
            for preview in batch_rows[:3]:
                print(f"     - {preview.get('pos', '')} | {preview.get('meaning_ru', '')} => {preview.get('meaning_en', '')}")

        if good_rows:
            good_df = pd.DataFrame(good_rows, columns=CANONICAL_COLUMNS)
            good_df.to_csv(out_path, mode="a", header=need_header_for_good_output, index=False, encoding="utf-8")
            if need_header_for_good_output:
                need_header_for_good_output = False

        if blank_rows:
            blanks_path = out_path.with_suffix(out_path.suffix + ".blanks.csv")
            blanks_df = pd.DataFrame(blank_rows, columns=CANONICAL_COLUMNS)
            write_header = not blanks_path.exists()
            blanks_df.to_csv(blanks_path, mode="a", header=write_header, index=False, encoding="utf-8")
        
        total_written += len(batch_rows)
        print(f"  Batch {batch_idx + 1}/{n_batches} saved ({total_written} total written)")

        if _should_abort_for_early_empty_run(
            backend_family=spec.backend_family,
            batches_seen=batch_idx + 1,
            total_written=total_written,
            total_empty_output=total_empty_output,
            total_kept=total_kept,
        ):
            raise TranslatorRuntimeError(
                f"Causal backend {resolved_model_key!r} produced near-100% empty "
                f"output ({total_empty_output}/{total_written} = "
                f"{100*total_empty_output/total_written:.1f}%) after {batch_idx + 1} "
                f"batches with {total_kept} kept items. This indicates generation is "
                f"broken, not genuine blank translations. Aborting."
            )

    blanks_path = out_path.with_suffix(out_path.suffix + ".blanks.csv")
    _print_summary(
        tasks_in_file=tasks_in_file,
        already_cached=cached_count,
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
    )


if __name__ == "__main__":
    main()
