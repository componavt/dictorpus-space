# sem_cat · Semantic Categorization for VepKar

```text
   ┌───────────────────────────────┐
   │  lemma:   liib               │
   │  pos:     NOUN               │
   │  gloss_ru: хлеб              │
   └───────────────────────────────┘
                 │
                 ▼
        "What domain is this about?"
                 │
                 ▼
      WordNet Domains + concept-level WDH
```

This sub-project builds a **semantic categorization pipeline** for the VepKar dictionary.

- **Input:** VepKar per-meaning exports for four language varieties.
- **Processing:** normalize Russian glosses, translate to English, look up WordNet synsets and domains, propagate concept-level WDH, and audit gaps.
- **Output:** semantic labels attached to dictionary **meanings** rather than only lemmas.

The goal is practical, not magical:
take noisy lexicographic data, add useful semantic structure, and keep the pipeline inspectable.

---

## What lives here?

```text
data/vepkar/meanings_*.csv
         │
         ▼
   unique Russian glosses
         │
         ▼
   Step 02: translation cache
         │
         ▼
   Step 03: compare models
         │
         ▼
   Step 04: WordNet lookup
         │
         ▼
   Step 05: assign domains to meanings
         │
         ├──────────────────────────────────────────────┐
         ▼                                              │
   gloss-based semantic labels                          │
                                                        │
data/sem_cat/concept_categories/*.tsv                   │
data/sem_cat/concepts/*.csv                             │
         │                                              │
         ▼                                              │
   Step 06: concept-level WDH                           │
         │                                              │
         ▼                                              │
   Step 07: propagate WDH to meanings                   │
         │                                              │
         ▼                                              │
   Step 08: gap audit  ◄────────────────────────────────┘
```

Two semantic paths coexist:

1. **Gloss-based path** — translation → WordNet → WordNet Domains.
2. **Concept-aware path** — concept categories → concept-level WDH → propagation to meanings.

The concept-aware path does **not** replace the gloss-based path.
It complements it, and step 07 records conflicts when the two disagree.

---

## Environment setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Download NLTK resources once per environment:

```bash
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Smoke tests

Before a long run, check the backends. It is cheaper to discover drama here than forty minutes into a batch.

```bash
python3 -m src.sem_cat.02_translate_glosses --model-key google --backend-info
python3 -m src.sem_cat.02_translate_glosses --model-key helsinki_opus_mt_ru_en --backend-info
python3 -m src.sem_cat.02_translate_glosses --model-key nllb_3_3b --backend-info
python3 -m src.sem_cat.02_translate_glosses --model-key tower_plus_9b --backend-info
```

`--backend-info` is a quick preflight check. It loads the selected backend, runs a few tiny probe translations, prints an `OK / WARN / FAIL` summary, and exits without touching your real data files.

If HuggingFace loading is grumpy because of proxy variables, try:

```bash
python3 -m src.sem_cat.02_translate_glosses \
  --model-key helsinki_opus_mt_ru_en \
  --ignore-proxy-env \
  --device cuda \
  --limit 20
```

The translator layer uses a **registry + factory** design.
Optional dependencies are loaded lazily, so import-time failures should not crash unrelated parts of the pipeline.

---

## Translation models

A small but important distinction:

- **Model key** = the canonical CLI / registry identifier used by the code.
- **Model name** = the actual backend model or service name.

For example, `helsinki_opus_mt_ru_en` is the registry key.
The actual model behind it is `Helsinki-NLP/opus-mt-ru-en`.

```text
    Registry / CLI keys        │ Actual model / backend            │ Backend family │ Round-trip
    ─────────────────────────────────────────────────────────────────────────────────────────────
    google                     │ GoogleTranslator service          │ google         │ yes
    helsinki_opus_mt_ru_en     │ Helsinki-NLP/opus-mt-ru-en        │ hf_seq2seq     │ yes
    nllb_3_3b                  │ facebook/nllb-200-3.3B           │ nllb           │ yes
    tower_plus_9b              │ Unbabel/Tower-Plus-9B             │ hf_causal      │ no
    hy_mt2_30b_a3b             │ tencent/Hy-MT2-30B-A3B           │ hf_causal      │ no
    alma_7b_r                  │ haoranxu/ALMA-7B-R               │ hf_causal      │ no
    ─────────────────────────────────────────────────────────────────────────────────────────────
```

| CLI / registry key | Actual model / backend | Backend family | Round-trip | Notes |
|---|---|---|---|---|
| `google` | GoogleTranslator service | `google` | ✓ | Handy baseline for quick checks and small runs |
| `helsinki_opus_mt_ru_en` | `Helsinki-NLP/opus-mt-ru-en` | `hf_seq2seq` | ✓ | MarianMT baseline; fast and practical |
| `nllb_3_3b` | `facebook/nllb-200-3.3B` | `nllb` | ✓ | Largest registered NLLB option |
| `tower_plus_9b` | `Unbabel/Tower-Plus-9B` | `hf_causal` | ✗ | Instruction-style causal LM; 4-bit quantized by default (~4.5 GB VRAM) |
| `hy_mt2_30b_a3b` | `tencent/Hy-MT2-30B-A3B` | `hf_causal` | ✗ | MoE translation model; requires `trust_remote_code` |
| `alma_7b_r` | `haoranxu/ALMA-7B-R` | `hf_causal` | ✗ | Translation-focused ALMA-R model; RU→EN only |

The repository supports four backend families:
- **`google`** – GoogleTranslator API service.
- **`hf_seq2seq`** – HuggingFace encoder-decoder seq2seq models (MarianMT style). Loaded via `AutoModelForSeq2SeqLM`.
- **`nllb`** – NLLB-200 models via HuggingFace with forced BOS language tokens.
- **`hf_causal`** – Decoder-only causal LMs loaded via `AutoModelForCausalLM`. These use prompt/chat-style translation and often rely on quantized loading on consumer GPUs.

Quantization for causal models reduces VRAM usage:
- Full precision: ~14-26 GB for 7B-13B models (may OOM on consumer GPUs)
- 4-bit: ~4-6 GB (recommended for 16 GB VRAM cards)
- 8-bit: ~8-12 GB

Many causal models have quantization enabled in the registry defaults:
- `tower_plus_9b` loads in 4-bit quantized mode by default
- Others can be overridden with `--quantization {none,4bit,8bit}`

---

## Step 02 — VepKar-aware translation

```text
VepKar meanings_*.csv (RU glosses)
    │
    ├─ "дом"
    ├─ "кошка"
    ├─ "вкусная калитка"
    ▼
    ├─ Skip rows with meaning_en already set (human-ready)
    ├─ For rows with empty meaning_en:
    │   ├─ Group by (pos, primary_gloss_ru)
    │   ├─ Detect reusable English variants
    │   │   ├─ Exactly 1 variant → reusable_unambiguous (no translation needed)
    │   │   └─ 2+ variants → reusable_ambiguous (export for expert review)
    │   └─ Remaining → needs_translation
    │
    └─ Translate only unique tasks from needs_translation
           │
           ├─ google
           ├─ helsinki_opus_mt_ru_en
           ├─ nllb_3_3b
           ├─ tower_plus_9b
           ├─ hy_mt2_30b_a3b
           └─ alma_7b_r
           ▼
EN gloss cache + QA metadata + task metadata
```

**What it does**

- Loads VepKar meanings files (`meanings_krl.csv`, `meanings_lud.csv`, `meanings_olo.csv`, `meanings_vep.csv`).
- Skips rows with existing human English in `meaning_en` (non-empty after simple strip).
- Groups remaining work by `(pos, primary_gloss_ru)`.
- Performs reuse analysis:
  - **reusable_unambiguous**: exactly one distinct existing English value → reuse without translation
  - **reusable_ambiguous**: two or more distinct existing English values → export for expert review
  - **needs_translation**: no existing English → true translation task
- Prints per-language and overall coverage statistics.
- Writes helper CSVs in `data/sem_cat/2translate/`.
- Translates only unique tasks from the `needs_translation` subset.
- `--offset` and `--limit` apply after skip/reuse/cache filtering.

**Main commands**

```bash
# Quick smoke runs
python3 -m src.sem_cat.02_translate_glosses \
  --model-key google \
  --limit 50 \
  --out-file data/sem_cat/02_glosses_translated_google_smoke.csv

python3 -m src.sem_cat.02_translate_glosses \
  --model-key helsinki_opus_mt_ru_en \
  --device cpu \
  --limit 50 \
  --out-file data/sem_cat/02_glosses_translated_helsinki_opus_mt_ru_en_smoke.csv

python3 -m src.sem_cat.02_translate_glosses \
  --model-key nllb_3_3b \
  --device cpu \
  --limit 50 \
  --out-file data/sem_cat/02_glosses_translated_nllb_3_3b_smoke.csv
```



**Important behavior**

- The step is **VepKar-aware**: skips rows with existing human English before any model translation.
- Reuse analysis groups by `(pos, primary_gloss_ru)` to detect exact reuse opportunities.
- **reusable_unambiguous**: rows that can reuse a single existing English translation are NOT sent to models.
- **reusable_ambiguous**: rows with conflicting existing English are exported for expert review.
- Only truly unique tasks from `needs_translation` subset go to models.
- `--offset` and `--limit` apply after skip/reuse filtering.
- Google backend always uses effective batch size **1**.
- The main CSV keeps **nonblank outputs even when `qa_keep=False`**; that is intentional, because raw nonblank failures are still useful for review and for step 03.
- The sidecar file `02_glosses_translated_<model_key>.csv.blanks.csv` is for rows whose English output is actually empty or blank.
- The summary printed by step 02 now includes per-language coverage statistics (see example below).

**Output file handling and schema**

- Step 02 writes a CSV header exactly once when creating a new output file.
- Before appending to an existing output file, Step 02 validates the file structure.
- If the target file exists but is malformed (for example, data rows appear before the header, or required columns like `gloss_ru` are missing), Step 02 aborts with a clear error message explaining what is wrong.
- Translated rows must include these required schema fields: `gloss_ru`, `gloss_en`, `task_key`, `task_pos`, `primary_gloss_ru`. Rows missing any of these are rejected before write.
- Helper file generation reflects only rows that truly need translation; files are only written when there are actual rows to include.
- POS echo prefix detection: if a model echoes the POS label into the English output (e.g., `NOUN concert hall`), Step 02 strips that prefix before QA and write.

**Typical output**

```text
data/sem_cat/02_glosses_translated_<model_key>.csv
data/sem_cat/02_glosses_translated_<model_key>.csv.blanks.csv
data/sem_cat/2translate/meanings_krl_to_translate.csv
data/sem_cat/2translate/meanings_lud_to_translate.csv
data/sem_cat/2translate/meanings_olo_to_translate.csv
data/sem_cat/2translate/meanings_vep_to_translate.csv
data/sem_cat/2translate/ambiguous_existing_en_by_task.csv
data/sem_cat/2translate/ambiguous_existing_en_by_task_summary.csv

`ambiguous_existing_en_by_task_summary.csv` includes `suggested_candidate_index`,
a 1-based pointer to the candidate that upstream review tools should show first.
```

**Columns you will care about**

- `gloss_ru`
- `gloss_en`
- `task_key`
- `task_pos`
- `primary_gloss_ru`
- `qa_keep`
- `qa_score`
- `qa_flags`
- `model_key`
- `model_name`
- `backend_family`
- `translation_input_mode`
- `gloss_ru_back`
- `roundtrip_distance`

`task_key` is the task-level identity used for cache deduplication and multi-model comparison.
    In CSV outputs it is serialized as `task_pos::primary_gloss_ru`, for example `NOUN::обида`.
    It distinguishes tasks by `(task_pos, primary_gloss_ru)`, so identical gloss strings with different POS are not merged accidentally.

- `task_key` is the task-level identity used for cache deduplication and multi-model comparison.
  In CSV outputs it is serialized as `task_pos::primary_gloss_ru`, for example `NOUN::обида`.
  It distinguishes tasks by `(task_pos, primary_gloss_ru)`, so identical gloss strings with different POS are not merged accidentally.

- `task_pos` records the part of speech for the task.

- `ambiguous_existing_en_by_task_summary.csv` includes `suggested_candidate_index`,
  a 1-based pointer to the candidate that downstream review tools should show first.

- Current outputs write `task_key` and `task_pos` explicitly. Reader paths still tolerate legacy files that use
  tab-separated task keys or the older `task_key_str` column.

### hf_causal models and quantization

`hf_causal` models such as `tower_plus_9b` are decoder-only LMs loaded via `AutoModelForCausalLM`.
    They use prompt/chat-style translation and often rely on quantized loading on consumer GPUs.

**Quantization options**

For `hf_causal` models, you can override the default quantization mode:

- `--quantization 4bit` loads the model in 4-bit quantized mode (recommended for 16 GB VRAM or less)
- `--quantization 8bit` loads the model in 8-bit quantized mode
- `--quantization none` forces full-precision loading. If --quantization is omitted, the registry default is used; for tower_plus_9b that default is 4-bit.

Quantization significantly reduces VRAM usage:
- Full precision 7B-13B models: 14-26 GB VRAM
- 4-bit quantized: ~4-6 GB VRAM
- 8-bit quantized: ~8-12 GB VRAM

**Example causal model commands**

```bash
# Default causal model (registry default quantization)
python3 -m src.sem_cat.02_translate_glosses --model-key tower_plus_9b

# 4-bit quantized (least VRAM)
python3 -m src.sem_cat.02_translate_glosses --model-key tower_plus_9b --quantization 4bit --device cuda

# 8-bit quantized
python3 -m src.sem_cat.02_translate_glosses --model-key tower_plus_9b --quantization 8bit --device cuda

# Full precision (only if you have enough VRAM)
python3 -m src.sem_cat.02_translate_glosses --model-key tower_plus_9b --quantization none --device cuda
```

**Model variant override**

You can also override the model name directly for experimental variants:

```bash
python3 -m src.sem_cat.02_translate_glosses --model-key tower_plus_9b \
  --model-variant Unbabel/Tower-Plus-9B --quantization 4bit
```

**Model load summary**

When any model initializes, a one-time summary is printed to console:
- Model key and name
- Backend family and device
- Quantization mode
- torch dtype
- device_map status
- trust_remote_code
- batch size and tokenizer settings
- round-trip support

This helps debug memory issues and verify your load configuration.

---

## Step 03 — Multi-model comparison

```text
google ─────────────────┐
helsinki_opus_mt_ru_en ┤
nllb_3_3b ──────────────┤
tower_plus_9b ──────────┼──► merge by task_key (or gloss_ru for legacy)
hy_mt2_30b_a3b ─────────┤
alma_7b_r ──────────────┘
                    │
                    ▼
     consensus clusters + disagreement score
         + QA signals + gloss complexity
         + total risk score
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   expert review queue    gold template
   (stricter than         (for curation /
    "all disagreements")   adjudication)
```

**What it does**

- Merges translation outputs from multiple models by `task_key` when available.
- Falls back to `gloss_ru` for legacy files that lack `task_key`.
- Computes comparison and risk features.
- Produces a full comparison table, an expert review queue, and a gold template.

**Main command**

```bash
python3 -m src.sem_cat.03_compare_translations \
  --translations google=data/sem_cat/02_glosses_translated_google.csv \
  --translations helsinki_opus_mt_ru_en=data/sem_cat/02_glosses_translated_helsinki_opus_mt_ru_en.csv \
  --translations nllb_3_3b=data/sem_cat/02_glosses_translated_nllb_3_3b.csv
```

**Larger example**

```bash
python3 -m src.sem_cat.03_compare_translations \
  --translations google=data/sem_cat/02_glosses_translated_google.csv \
  --translations helsinki_opus_mt_ru_en=data/sem_cat/02_glosses_translated_helsinki_opus_mt_ru_en.csv \
  --translations nllb_3_3b=data/sem_cat/02_glosses_translated_nllb_3_3b.csv \
  --translations tower_plus_9b=data/sem_cat/02_glosses_translated_tower_plus_9b.csv \
  --translations alma_7b_r=data/sem_cat/02_glosses_translated_alma_7b_r.csv
```

**Important behavior**

- Use the **real registry key** in `--translations model_key=path.csv`.
- If a file contains a single distinct `model_key` that disagrees with the CLI label, the script should reject the mismatch rather than quietly soldier on.
- The full comparison file is exhaustive.
- The review queue is intentionally stricter than “all disagreements”.
- Rows are especially likely to enter review when they are medium/high risk, when proposal selection still has no clear winner, or when a model produced a nonblank output with `qa_keep=False`.

This is the stage where multiple systems line up, clear their throats, and disagree with admirable professionalism.

**Output**

```text
data/sem_cat/03_translation_comparison_full.csv
data/sem_cat/03_translation_review_queue.csv
data/sem_cat/03_translation_gold_template.csv
```

The gold template includes workflow fields that are useful during expert adjudication:

- `accepted_model_key`
- `accepted_raw_output`
- `review_status`

---

## Steps 04–05 — WordNet lookup and domain assignment

```text
English gloss
    │
    ▼
[WordNet synset lookup]
    │
    ▼
WN synset + WN domain
    │
    ▼
merge back into meanings_*.csv
```

### Step 04 — `04_wordnet_lookup.py`

**What it does**

- Reads translated glosses from step 02 or step 03.
- Looks up WordNet synsets.
- Maps synsets to WordNet Domains.

**Command**

```bash
python3 -m src.sem_cat.04_wordnet_lookup \
  --translated-file data/sem_cat/03_translation_comparison_full.csv \
  --wn-domains-file data/sem_cat/00_wn-domains-3.2-20070223
```

**Typical output**

```text
data/sem_cat/04_glosses_wn_domains.csv
```

### Step 05 — `05_assign_domains.py`

**What it does**

- Merges gloss-level WordNet Domain labels back into per-meaning VepKar exports.

**Command**

```bash
python3 -m src.sem_cat.05_assign_domains \
  --data-dir data/vepkar \
  --domains-file data/sem_cat/04_glosses_wn_domains.csv \
  --out-dir data/sem_cat/results
```

**Typical output**

```text
data/sem_cat/results/meanings_vep_domains.csv
data/sem_cat/results/meanings_olo_domains.csv
data/sem_cat/results/meanings_lud_domains.csv
data/sem_cat/results/meanings_krl_domains.csv
```

**Practical note**

If step 04 produces too much `factotum`, do not panic.
Panic is reserved for silent schema drift, corrupted joins, and cheerful lies told by CSV headers.

---

## Steps 06–08 — Concept-level WDH and gap audit

```text
category WDH + concept catalog
            │
            ▼
    Step 06: concept-level WDH (concepts_wdh.tsv)
            │
            ▼
    Step 07: propagate WDH to meanings (meanings_*_concept_wdh.csv)
            │
            ▼
    Step 08: audit_*.csv
```

### Step 06 — Build concept-level WDH

```text
category_id ──► WDH
      │
      ▼
concept_id ───► inherited domain
```

**What it does**

- Builds a concept-level WDH table from category-level WDH.
- Produces one WDH row per concept.

**Command**

```bash
python3 -m src.sem_cat.06_concepts_wdh
```

By default, this reads canonical paths from `src/sem_cat/sem_cat_paths.toml`. Override examples:

```bash
python3 -m src.sem_cat.06_concepts_wdh \
  --concepts data/sem_cat/concepts/concepts_with_english_experiment.csv \
  --out-file data/sem_cat/concepts/concepts_wdh_experiment.tsv
```

**Output**

```text
data/sem_cat/concepts/concepts_wdh.tsv
```

**Schema**

```text
category_id, pos, concept_id, concept_ru, concept_en, wdh
```

### Step 07 — Propagate WDH to meanings

```text
concept_id in meanings
        │
        ▼
 concept_wdh
        │
        ├─ optional: compare with gloss_wdh
        ▼
final wdh + conflict flags
```

**What it does**

- Joins concept-level WDH onto meanings that already have `concept_id`.
- Optionally compares concept WDH with gloss-based WDH from step 04.

**Commands**

```bash
# Safe baseline: propagate without gloss-based conflict detection
python3 -m src.sem_cat.07_propagate_wdh \
  --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
  --data-dir data/vepkar \
  --out-dir data/sem_cat/results

# Full mode: include gloss-based WDH comparison
python3 -m src.sem_cat.07_propagate_wdh \
  --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
  --data-dir data/vepkar \
  --domains-file data/sem_cat/04_glosses_wn_domains.csv \
  --out-dir data/sem_cat/results
```

**Important behavior**

- `--domains-file` is optional.
- If you pass it, it must point to a real output of step 04.
- Meaningful non-`factotum` gloss-domain evidence may override concept WDH.
- Fallback `factotum` should not override a more specific concept domain.

**Output**

```text
data/sem_cat/results/meanings_vep_concept_wdh.csv
data/sem_cat/results/meanings_olo_concept_wdh.csv
data/sem_cat/results/meanings_lud_concept_wdh.csv
data/sem_cat/results/meanings_krl_concept_wdh.csv
data/sem_cat/results/wdh_conflicts.csv
```

### Step 08 — Gap audit

```text
meanings + concepts + optional enriched files
                     │
                     ▼
         "What is missing, weak, or weird?"
                     │
                     ▼
                  audit_*.csv
```

**What it does**

- Finds meanings without concept coverage.
- Measures concept usage.
- Detects concept IDs outside the concept catalog.
- Builds clusters of similar `meaning_ru`.
- Optionally analyzes systematic WDH disagreement if enriched step-07 files are available.

**Commands**

```bash
# Basic audit
python3 -m src.sem_cat.08_gap_audit \
  --data-dir data/vepkar \
  --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
  --out-dir data/sem_cat/results

# Full audit with enriched step-07 outputs
python3 -m src.sem_cat.08_gap_audit \
  --data-dir data/vepkar \
  --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
  --enriched-dir data/sem_cat/results \
  --out-dir data/sem_cat/results
```

**Important behavior**

- If `--enriched-dir` is provided but no enriched files are found, the script should warn clearly and continue without WDH-disagreement analysis.
- That warning is useful, not rude.

**Output**

```text
data/sem_cat/results/audit_meanings_without_concept.csv
data/sem_cat/results/audit_concept_usage.csv
data/sem_cat/results/audit_concept_ids_outside_catalog.csv
data/sem_cat/results/audit_category_coverage.csv
data/sem_cat/results/audit_wdh_disagreement.csv
data/sem_cat/results/audit_meaning_ru_clusters.csv
```

---

## Heuristics documentation

For a focused description of the QA and comparison heuristics used by the current codebase, see:

- [`src/sem_cat/heuristics.md`](heuristics.md)

That file is the compact field guide to the project’s suspicious instincts.

---

## What to inspect after each step

This section is a **working checklist**, not a formal contract.
Treat it as practical triage guidance.

### After step 02
Start with:

- `qa_keep`
- `qa_score`
- `qa_flags`
- sidecar `.blanks.csv`

Useful questions:

- Which glosses fail QA outright?
- Which glosses are nonblank but suspicious?
- Which model behaves worst on abbreviations, names, particles, and short function words?
- Are repeated-token loops rare, systematic, or spectacular?

### After step 03
Start with:

- `03_translation_review_queue.csv`
- highest-risk rows
- disagreements between models
- adjudication fields in the gold template

Useful questions:

- Which glosses deserve expert review first?
- Which disagreements are semantically important, and which are merely stylistic?
- Which model is safest for boring lexicographic work?
- Which model becomes inventive at precisely the wrong moment?

### After steps 04–05
Start with:

- `lookup_status`
- `wn_synset`
- `wn_domain`
- coverage in `meanings_*_domains.csv`

Useful questions:

- How often do glosses land in `factotum`?
- Which POS classes underperform?
- Are short glosses and proper names dominating lookup failure?

### After steps 06–08
Start with:

- `concepts_wdh.tsv`
- `wdh_conflicts.csv` (from step 07)
- `audit_concept_ids_outside_catalog.csv`
- `audit_meanings_without_concept.csv`
- `audit_wdh_disagreement.csv` (from step 08)
- `audit_meaning_ru_clusters.csv`

Useful questions:

- Which concept IDs are used in meanings but absent from the concept catalog?
- Which concept-vs-gloss conflicts are systematic rather than random?
- Which frequent `meaning_ru` clusters should drive the next ontology-expansion pass?

---

## Minimal workflow from the repository root

```bash
source .venv/bin/activate

# 0. tests
pytest tests/sem_cat/ -q

# 1. smoke tests
python3 -m src.sem_cat.02_translate_glosses --model-key google --backend-info
python3 -m src.sem_cat.02_translate_glosses --model-key helsinki_opus_mt_ru_en --backend-info
python3 -m src.sem_cat.02_translate_glosses --model-key nllb_3_3b --backend-info

# 2. translations
python3 -m src.sem_cat.02_translate_glosses --model-key google
python3 -m src.sem_cat.02_translate_glosses --model-key helsinki_opus_mt_ru_en --device cuda
python3 -m src.sem_cat.02_translate_glosses --model-key nllb_3_3b --device cuda --round-trip

# 3. comparison
python3 -m src.sem_cat.03_compare_translations \
  --translations google=data/sem_cat/02_glosses_translated_google.csv \
  --translations helsinki_opus_mt_ru_en=data/sem_cat/02_glosses_translated_helsinki_opus_mt_ru_en.csv \
  --translations nllb_3_3b=data/sem_cat/02_glosses_translated_nllb_3_3b.csv

# 4. WordNet lookup
python3 -m src.sem_cat.04_wordnet_lookup \
  --translated-file data/sem_cat/03_translation_comparison_full.csv \
  --wn-domains-file data/sem_cat/00_wn-domains-3.2-20070223

# 5. assign domains
python3 -m src.sem_cat.05_assign_domains \
  --data-dir data/vepkar \
  --domains-file data/sem_cat/04_glosses_wn_domains.csv \
  --out-dir data/sem_cat/results

# 6. concept-level WDH
python3 -m src.sem_cat.06_concepts_wdh

# 7. propagate WDH
python3 -m src.sem_cat.07_propagate_wdh \
  --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
  --data-dir data/vepkar \
  --domains-file data/sem_cat/04_glosses_wn_domains.csv \
  --out-dir data/sem_cat/results

# 8. gap audit
python3 -m src.sem_cat.08_gap_audit \
  --data-dir data/vepkar \
  --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
  --enriched-dir data/sem_cat/results \
  --out-dir data/sem_cat/results
```

---

## Tests

```bash
pytest tests/sem_cat/ -q
pytest tests/sem_cat/translators/test_part1.py -q
pytest tests/sem_cat/test_part2.py -q
pytest tests/sem_cat/test_part3.py -q
```

The focused tests are designed to stay offline and fast.
That is a feature, not a lack of ambition.

---

## Operational notes

- Use the canonical registry key `helsinki_opus_mt_ru_en` in CLI commands, but do not confuse it with the actual model name `Helsinki-NLP/opus-mt-ru-en`.
- Do not improvise aliases in `03_compare_translations`.
- Step 02 cache behavior is incremental by design.
- Step 07 can run without `--domains-file`; use that mode if step 04 output is not ready yet.
- Step 08 is a major driver for iterative ontology expansion.
- The most valuable review targets are usually:
  - high-risk comparison rows,
  - nonblank outputs with `qa_keep=False`,
  - `factotum`-heavy WordNet outputs,
  - concept IDs outside the catalog,
  - systematic WDH conflicts,
  - frequent `meaning_ru` clusters with weak coverage.
