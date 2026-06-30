"""
Translates only rows from VepKar meanings files that need translation.
Skips rows with existing human English (meaning_en).
Detects reusable translations by (pos, primary_gloss_ru).
Only translates truly unique tasks.

Results saved to data/sem_cat/02_glosses_translated_{model_key}.csv and
helper files in data/sem_cat/2translate/
"""

import sys
import pathlib
import argparse
import random
from dataclasses import dataclass
from math import ceil
from typing import Literal

import pandas as pd

_THIS_FILE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "vepkar"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "data" / "sem_cat"
_DEFAULT_TRANSLATE_DIR = _DEFAULT_OUT_DIR / "2translate"

from src.sem_cat.utils.vepkar_loader import load_meanings
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
from src.sem_cat.pipeline.vepkar_translation_selection import (
    prepare_meanings_for_translation,
    split_by_existing_en_reuse,
    extract_unique_translation_tasks,
    build_task_metadata_map,
    prepare_translation_input_for_task,
    TranslationTaskMetadata,
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
    total_rows: int,
    existing_human_en: int,
    raw_needs_mt: int,
    reusable_unambiguous: int,
    reusable_ambiguous: int,
    needs_model: int,
    unique_tasks: int,
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
    print(f"VepKar coverage (per-language):")
    print(f"  Total rows:                   {total_rows}")
    print(f"  Existing human English:       {existing_human_en}")
    print(f"  Raw needs MT:                 {raw_needs_mt}")
    print(f"  Reusable (unambiguous):       {reusable_unambiguous}")
    print(f"  Reusable (ambiguous):         {reusable_ambiguous}")
    print(f"  Needs model translation:      {needs_model}")
    print(f"  Unique translation tasks:     {unique_tasks}")
    print()
    print(f"Translation processing:")
    print(f"  Already cached (skipped):     {already_cached}")
    print(f"  Remaining after cache:        {remaining_after_cache}")
    print(f"  Selected for this run:        {to_translate_count}")
    print(f"  (cache filter applied after reuse analysis, then --offset/--limit)")
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

    if total_empty_output > 0:
        print(f"\nEmpty output rows written to: {blanks_path}")


def _print_coverage_summary(
    per_lang_stats: dict[str, dict],
    total_rows: int,
    existing_human_en: int,
    raw_needs_mt: int,
    reusable_unambiguous: int,
    reusable_ambiguous: int,
    needs_model: int,
    unique_tasks: int,
) -> None:
    """Print per-language and global coverage summary."""
    print(f"\n{'=' * 60}")
    print("VepKar translation coverage")
    print(f"{'=' * 60}")
    
    for lang in ["krl", "lud", "olo", "vep"]:
        stats = per_lang_stats[lang]
        print(f"\nmeanings_{lang}.csv")
        print(f"  total_rows              {stats['total_rows']}")
        print(f"  existing_human_en       {stats['existing_human_en']}")
        print(f"  raw_needs_mt            {stats['raw_needs_mt']}")
        print(f"  reusable_unambiguous    {stats['reusable_unambiguous']}")
        print(f"  reusable_ambiguous      {stats['reusable_ambiguous']}")
        print(f"  needs_model             {stats['needs_model']}")
        print(f"  unique_tasks            {stats['unique_tasks']}")
    
    print(f"\nALL FILES")
    print(f"  total_rows              {total_rows}")
    print(f"  existing_human_en       {existing_human_en}")
    print(f"  raw_needs_mt            {raw_needs_mt}")
    print(f"  reusable_unambiguous    {reusable_unambiguous}")
    print(f"  reusable_ambiguous      {reusable_ambiguous}")
    print(f"  needs_model             {needs_model}")
    print(f"  unique_tasks            {unique_tasks}")


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


def save_translate_helper_files(
    df_krl: pd.DataFrame,
    df_lud: pd.DataFrame,
    df_olo: pd.DataFrame,
    df_vep: pd.DataFrame,
    ambiguous_df: pd.DataFrame,
    translate_dir: pathlib.Path,
) -> None:
    """Save helper CSV files for translation workflow."""
    translate_dir.mkdir(parents=True, exist_ok=True)
    
    if not df_krl.empty:
        # Use task_key_str for CSV output (more stable than tuple)
        if "task_key" in df_krl.columns and "task_key_str" in df_krl.columns:
            df_krl = df_krl.copy()
            df_krl["task_key"] = df_krl["task_key_str"]
        df_krl.to_csv(translate_dir / "meanings_krl_to_translate.csv", index=False)
    if not df_lud.empty:
        if "task_key" in df_lud.columns and "task_key_str" in df_lud.columns:
            df_lud = df_lud.copy()
            df_lud["task_key"] = df_lud["task_key_str"]
        df_lud.to_csv(translate_dir / "meanings_lud_to_translate.csv", index=False)
    if not df_olo.empty:
        if "task_key" in df_olo.columns and "task_key_str" in df_olo.columns:
            df_olo = df_olo.copy()
            df_olo["task_key"] = df_olo["task_key_str"]
        df_olo.to_csv(translate_dir / "meanings_olo_to_translate.csv", index=False)
    if not df_vep.empty:
        if "task_key" in df_vep.columns and "task_key_str" in df_vep.columns:
            df_vep = df_vep.copy()
            df_vep["task_key"] = df_vep["task_key_str"]
        df_vep.to_csv(translate_dir / "meanings_vep_to_translate.csv", index=False)
    
    if not ambiguous_df.empty:
        if "task_key" in ambiguous_df.columns and "task_key_str" in ambiguous_df.columns:
            ambiguous_df = ambiguous_df.copy()
            ambiguous_df["task_key"] = ambiguous_df["task_key_str"]
        ambiguous_df.to_csv(translate_dir / "ambiguous_existing_en_by_task.csv", index=False)
    
    if not ambiguous_df.empty:
        ambiguous_task_df = _build_ambiguous_task_summary(ambiguous_df)
        ambiguous_task_df.to_csv(translate_dir / "ambiguous_existing_en_by_task_summary.csv", index=False)


def _build_ambiguous_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build task-level summary for ambiguous reuse cases.
    
    Returns a DataFrame with one row per unique task_key that has ambiguous existing English,
    instead of one row per missing meaning.
    """
    summaryParts = []
    for _, group in df.groupby(["task_key", "task_pos", "primary_gloss_ru", "existing_en_candidates", "existing_en_candidate_count"], dropna=False, sort=False):
        summary = {
            "task_key": str(group["task_key"].iloc[0]) if "task_key" in group.columns else "",
            "task_pos": str(group["task_pos"].iloc[0]) if "task_pos" in group.columns else "",
            "primary_gloss_ru": str(group["primary_gloss_ru"].iloc[0]) if "primary_gloss_ru" in group.columns else "",
            "existing_en_candidates": str(group["existing_en_candidates"].iloc[0]) if "existing_en_candidates" in group.columns else "",
            "existing_en_candidate_count": int(group["existing_en_candidate_count"].iloc[0]) if "existing_en_candidate_count" in group.columns else 0,
            "missing_row_count": len(group),
            "example_lemma": str(group["lemma"].iloc[0]) if "lemma" in group.columns and not group["lemma"].isna().any() else "",
            "langs": " || ".join(sorted(set(str(x) for x in group["lang"].dropna().tolist()))) if "lang" in group.columns else "",
        }
        summaryParts.append(summary)
    
    return pd.DataFrame(summaryParts)


def main() -> None:
    model_keys = list_model_keys()

    parser = argparse.ArgumentParser(
        description="Translate VepKar meanings to English (skips existing human translations)"
    )
    parser.add_argument("--data-dir", type=str, default=str(_DEFAULT_DATA_DIR),
                        help=f"path to data/vepkar/ (default: {_DEFAULT_DATA_DIR})")
    parser.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR),
                        help=f"output directory for translated CSV (default: {_DEFAULT_OUT_DIR})")
    parser.add_argument("--translate-dir", type=str, default=str(_DEFAULT_TRANSLATE_DIR),
                        help=f"output directory for translation helpers (default: {_DEFAULT_TRANSLATE_DIR})")
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
                        help="Skip the first N tasks after cache filtering (default: 0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N tasks after offset (default: None = all)")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Shuffle tasks before applying offset/limit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used with --shuffle (default: 42)")
    parser.add_argument("--gloss-filter", type=str, default=None,
                        help="Optional substring filter applied to primary_gloss_ru before translation")
    parser.add_argument(
        "--translation-input-mode", type=str, choices=["raw", "pos", "pos_meaning"],
        default="pos",
        help="How to prepare input for translator (default: pos)",
    )
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

    if args.out_file:
        out_path = pathlib.Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"02_glosses_translated_{resolved_model_key}.csv"

    translate_dir = pathlib.Path(args.translate_dir)
    translate_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model key: {resolved_model_key}")
    print(f"Model name: {spec.model_name}")
    print(f"Backend family: {spec.backend_family}")
    print(f"Output file: {out_path}")
    print(f"Translate helper dir: {translate_dir}")

    if args.backend_info:
        _run_backend_info(
            spec, args.device, retry, delay,
            args.local_files_only, args.hf_cache_dir,
            args.ignore_proxy_env,
        )
        return

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    print("Loading meanings...")
    df_meanings = load_meanings(str(data_dir))
    
    print("Preparing meanings for translation...")
    work = prepare_meanings_for_translation(df_meanings)
    print(f"Total rows with non-empty primary gloss: {len(work)}")
    
    print("Computing reuse analysis by (pos, primary_gloss_ru)...")
    reusable_unambiguous_df, reusable_ambiguous_df, needs_model_df = split_by_existing_en_reuse(work)
    
    print(f"  Reusable (unambiguous): {len(reusable_unambiguous_df)}")
    print(f"  Reusable (ambiguous):   {len(reusable_ambiguous_df)}")
    print(f"  Needs model:            {len(needs_model_df)}")
    
    per_lang_stats = {}
    for lang in ["krl", "lud", "olo", "vep"]:
        lang_work = work[work["lang"] == lang] if "lang" in work.columns else work
        lang_has_en = lang_work[lang_work["has_existing_en"]] if "lang" in lang_work.columns else lang_work
        lang_missing = lang_work[~lang_work["has_existing_en"]] if "lang" in lang_work.columns else lang_work
        
        lang_reuse_unamb = reusable_unambiguous_df[
            reusable_unambiguous_df["lang"] == lang
        ] if "lang" in reusable_unambiguous_df.columns else pd.DataFrame()
        lang_reuse_amb = reusable_ambiguous_df[
            reusable_ambiguous_df["lang"] == lang
        ] if "lang" in reusable_ambiguous_df.columns else pd.DataFrame()
        lang_needs = needs_model_df[
            needs_model_df["lang"] == lang
        ] if "lang" in needs_model_df.columns else pd.DataFrame()
        
        per_lang_stats[lang] = {
            "total_rows": len(lang_work),
            "existing_human_en": len(lang_has_en),
            "raw_needs_mt": len(lang_missing),
            "reusable_unambiguous": len(lang_reuse_unamb),
            "reusable_ambiguous": len(lang_reuse_amb),
            "needs_model": len(lang_needs),
            "unique_tasks": len(lang_needs["task_key"].unique()) if not lang_needs.empty else 0,
        }

    total_rows = len(work)
    existing_human_en = len(work[work["has_existing_en"]])
    raw_needs_mt = len(work[~work["has_existing_en"]])
    reusable_unambiguous = len(reusable_unambiguous_df)
    reusable_ambiguous = len(reusable_ambiguous_df)
    needs_model = len(needs_model_df)
    
    unique_tasks_df = needs_model_df.drop_duplicates(subset=["task_key"]) if not needs_model_df.empty else pd.DataFrame()
    unique_tasks = len(unique_tasks_df)
    
    _print_coverage_summary(
        per_lang_stats,
        total_rows, existing_human_en, raw_needs_mt,
        reusable_unambiguous, reusable_ambiguous, needs_model, unique_tasks,
    )
    
    print("\nSaving helper files...")
    
    save_translate_helper_files(
        df_krl=needs_model_df[needs_model_df["lang"] == "krl"] if "lang" in needs_model_df.columns else pd.DataFrame(),
        df_lud=needs_model_df[needs_model_df["lang"] == "lud"] if "lang" in needs_model_df.columns else pd.DataFrame(),
        df_olo=needs_model_df[needs_model_df["lang"] == "olo"] if "lang" in needs_model_df.columns else pd.DataFrame(),
        df_vep=needs_model_df[needs_model_df["lang"] == "vep"] if "lang" in needs_model_df.columns else pd.DataFrame(),
        ambiguous_df=reusable_ambiguous_df,
        translate_dir=translate_dir,
    )
    
    print(f"  Saved ambiguous_existing_en_by_task.csv")
    
    print("Extracting unique translation tasks...")
    tasks = extract_unique_translation_tasks(needs_model_df)
    total_tasks = len(tasks)
    print(f"Found {total_tasks} unique translation tasks")

    print("Building task metadata map...")
    task_metadata_map = build_task_metadata_map(needs_model_df)
    
    print("Loading translation cache...")
    cache_df = load_translation_cache(out_path, expected_model_key=resolved_model_key)
    cached_tasks = set()
    if not cache_df.empty and "task_key" in cache_df.columns:
        cached_tasks = set(cache_df["task_key"].dropna().tolist())
        print(f"Found {len(cached_tasks)} cached task keys")
    else:
        print("Cache has no task_key column - will not skip any cached tasks")

    tasks_to_translate = [t for t in tasks if t.task_key not in cached_tasks]
    remaining_after_cache = len(tasks_to_translate)
    print(f"Remaining after cache: {remaining_after_cache}")

    if args.gloss_filter:
        tasks_to_translate = [
            t for t in tasks_to_translate
            if args.gloss_filter in t.primary_gloss_ru
        ]
        print(f"After gloss filter '{args.gloss_filter}': {len(tasks_to_translate)}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(tasks_to_translate)
        print(f"Shuffled tasks with seed {args.seed}")

    if args.offset > 0:
        tasks_to_translate = tasks_to_translate[args.offset:]
        print(f"After offset {args.offset}: {len(tasks_to_translate)}")

    if args.limit is not None:
        tasks_to_translate = tasks_to_translate[:args.limit]
        print(f"After limit {args.limit}: {len(tasks_to_translate)}")

    to_translate_count = len(tasks_to_translate)
    print(f"Selected for this run: {to_translate_count}")

    if to_translate_count == 0:
        print("No new tasks to translation. Exiting.")
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

    print(f"Translating with {resolved_model_key} (mode: {args.translation_input_mode})...")
    print(f"Effective batch size: {effective_batch_size}")

    input_texts: list[str] = []
    for task in tasks_to_translate:
        input_text = prepare_translation_input_for_task(task, args.translation_input_mode)
        input_texts.append(input_text)

    _causal_generation_preflight(
        translator,
        input_texts,
        resolved_model_key,
        spec.backend_family,
        effective_batch_size=effective_batch_size,
    )

    n = len(tasks_to_translate)
    n_batches = ceil(n / effective_batch_size) if n > 0 else 0

    header_written = not out_path.exists() or count_cached_rows(cache_df) > 0
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
            roundtrip_text = back_translated[
                list(batch_tasks).index(task)
            ] if back_translator is not None else None

            qa_result = analyze_translation(
                task.primary_gloss_ru,
                trans if trans else "",
                roundtrip_text,
                config=qa_config
            )

            row = build_translation_row(
                gloss_ru=task.primary_gloss_ru,
                gloss_en=trans if trans else "",
                qa_result=qa_result,
                model_key=resolved_model_key,
                model_name=spec.model_name,
                backend_family=spec.backend_family,
                translation_input_mode=args.translation_input_mode,
                input_text_used=input_text,
                pos_hint=task.task_pos,
                meaning_hint=task.meaning_hint,
                source_count=task.sourcecount,
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
        total_rows=total_rows,
        existing_human_en=existing_human_en,
        raw_needs_mt=raw_needs_mt,
        reusable_unambiguous=reusable_unambiguous,
        reusable_ambiguous=reusable_ambiguous,
        needs_model=needs_model,
        unique_tasks=unique_tasks,
        already_cached=len(cached_tasks),
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
