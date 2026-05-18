# sem_cat · Semantic Categorization for VepKar

```text
   ┌───────────────────────────────┐        🌐
   │  📖 lemma: liib              │    .- - - - - - - - - -.
   │  🧷 pos:   NOUN              │   (  🍞 food  ·  culture  )
   │  🔤 gloss: хлеб              │    `- - - - - - - - - -'
   └───────────────────────────────┘            🎯
       VepKar dictionary entry           WordNet Domain "cloud"
```

This sub-package implements a **semantic categorization pipeline** for the VepKar dictionary:

- input: VepKar **per-meaning** exports (4 languages / varieties),
- processing: normalize Russian glosses, translate to English, map to WordNet synsets,
- output: assign **WordNet Domains** semantic labels to each meaning.

The goal is to make a reusable, semi-automatic module that can later be integrated back into VepKar.

---

## Data flow overview

```text
data/vepkar/meanings_*.csv
         │
         ▼
   [gloss normalization]
         │
         ▼
 unique Russian glosses (≈ 42,962)
         │
         ▼
   [translation backend]
   (Google, MarianMT, NLLB, ...)
         │
         ▼
  English glosses (cache)
         │
         ▼
 [multi-model comparison]  ← optional, for quality assessment
         │
         ▼
 [WordNet synset lookup + wn-domains]
         │
         ▼
  (gloss_ru, wn_synset, wn_domain)
         │
         ▼
 [merge back into meanings_*.csv]
         │
         ▼
 data/sem_cat/results/meanings_{lang}_domains.csv

Concept-aware path (parallel, does not replace gloss pipeline):

data/sem_cat/concept_categories/concept_categories_wdh.tsv  ─┐
data/sem_cat/concepts/concepts_with_english_417.csv         ─┤
         │                                                   ▼
         │                                          [concepts_wdh]
         │                                                   │
         ▼                                                   ▼
data/vepkar/meanings_*.csv (with concept_id) ─────► [propagate WDH to meanings]
         │                                                   │
         │                                                   ▼
         │                                    meanings_{lang}_concept_wdh.csv
         │                                    wdh_conflicts.csv
         │
         ▼
   [gap audit]
         │
         ▼
   audit_*.csv reports
```

Key design constraints:

- **Cache everything**: translation and synset/domain lookup are computed once per unique gloss.
- **POS-aware** lookup: where possible, disambiguate using the VepKar POS tag.
- **Graceful fallbacks**: if no synset/domain is found, assign a neutral domain (e.g. `factotum`).

---

## Code structure

```text
src/sem_cat/
├── README.md                     # This file (technical overview)
├── __init__.py                   # Package marker
├── 01_meanings_examples_counter.ipynb
│                                 # Exploration of meanings & examples
├── 02_translate_glosses.py       # Step 1: RU→EN gloss translation + QA flags
├── 03_compare_translations.py    # Step 2: N-model comparison, risk scoring
├── 04_wordnet_lookup.py          # Step 3: EN→WordNet synset→WN domain
├── 05_assign_domains.py          # Step 4: merge domains into meanings
├── 06_concepts_wdh.py            # Step 5: build concept-level WDH table
├── 07_propagate_wdh.py           # Step 6: propagate concept WDH to meanings
├── 08_gap_audit.py               # Step 7: gap audit for concept coverage
├── 06_concepts_wdh.py            # Step 5: build concept-level WDH table
├── 07_propagate_wdh.py           # Step 6: propagate concept WDH to meanings
├── 08_gap_audit.py               # Step 7: gap audit for concept coverage
├── utils/
│   ├── __init__.py
│   ├── gloss_normalizer.py       # Parentheses & ';'-based gloss processing
│   ├── vepkar_loader.py          # Load and merge meanings_*.csv
│   ├── wn_domains.py             # Load wn-domains mapping, synset key helper
│   ├── concept_wdh.py            # Build concept-level WDH from categories
│   ├── meaning_propagation.py    # Propagate concept WDH to meanings
│   └── gap_audit.py              # Gap analysis for concept coverage
├── translators/
│   ├── __init__.py
│   ├── base.py                   # Abstract Translator class + error types
│   ├── google_translator.py      # Google Translate backend (deep_translator)
│   ├── hf_seq2seq_translator.py  # Generic HuggingFace seq2seq translator
│   ├── marian_translator.py      # MarianMT (thin wrapper around HFSeq2Seq)
│   ├── nllb_translator.py        # NLLB (facebook/nllb-200) translator
│   ├── model_registry.py         # ModelSpec definitions + legacy resolver
│   ├── factory.py                # build_translator() / build_reverse_translator()
│   └── generation_presets.py     # Generation parameter presets
├── compare/
│   ├── __init__.py
│   ├── loading.py                # Load and merge multiple model CSVs
│   ├── normalization.py          # Normalize outputs for comparison
│   ├── consensus.py              # Cluster near-identical outputs
│   ├── complexity.py             # Compute gloss complexity scores
│   ├── risk.py                   # Compute total risk scores
│   ├── proposal.py               # Select proposed translations
│   ├── output_tables.py          # Build comparison/review/gold DataFrames
│   └── data_structures.py        # ModelOutput, ComparisonResult, etc.
├── qa/
│   ├── __init__.py
│   ├── translation_qa.py         # QA analysis (keep/score/flags)
│   └── translation_flags.py      # Pattern-based flag detectors
├── io/
│   ├── __init__.py
│   ├── translation_cache.py      # Cache loading and validation
│   └── translation_rows.py       # Canonical output row builder
└── pipeline/
    ├── __init__.py
    └── translation_input.py      # Gloss metadata and input preparation
```

---

## Input formats

### `data/vepkar/meanings_*.csv`

Expected columns (all strings):

| Column | Description |
|--------|-------------|
| `id` | Row id inside this export file |
| `lemma_id` | Id of the lemma in VepKar |
| `meaning_id` | Id of the meaning in VepKar |
| `meaning_num` | Number of the meaning within the lemma (1, 2, 3…) |
| `lemma` | Lemma form in Veps or Karelian |
| `lang` | Language code: `vep`, `olo`, `lud`, `krl` |
| `pos` | POS tag (UPOS+custom, e.g. `NOUN`, `VERB`, `PROPN`) |
| `meaning_ru` | Short Russian gloss (1–3 words; may contain `;` and `(...)`) |

### `data/sem_cat/00_wn-domains-3.2-20070223`

WordNet Domains 3.2 mapping file. Format:

```text
00001740-n    factotum
00001930-n    cognition
```

Left column: 8-digit offset + POS letter (`n`, `v`, `a`, `r`).
Right column: one of 164 fine-grained domain labels.

Loaded via `utils.wn_domains.load_wn_domains()` → `{"00001740-n": ["factotum"], ...}`

---

## Step 02 — Translate glosses (`02_translate_glosses.py`)

**Purpose:** load VepKar meanings, extract unique Russian glosses, translate to English, save cache with QA metadata.

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/vepkar/` | Path to directory with `meanings_*.csv` |
| `--out-dir` | `data/sem_cat/` | Output directory (if `--out-file` not given) |
| `--out-file` | auto | Full output CSV path |
| `--model-key` | resolved from `--backend` | Translation model key (preferred) |
| `--backend` | `marian` | legacy: `marian`, `google`, or `nllb` |
| `--nllb-model` | `facebook/nllb-200-distilled-1.3B` | legacy: NLLB model name |
| `--device` | `cpu` | `cpu` or `cuda` for local models |
| `--batch-size` | `64` | Batch size for translation |
| `--round-trip` | off | Back-translate EN→RU for QA |
| `--offset` | `0` | Skip first N glosses after cache filter |
| `--limit` | all | Process at most N glosses |
| `--shuffle` | off | Shuffle glosses before offset/limit |
| `--seed` | `42` | Random seed for shuffle |
| `--gloss-filter` | none | Substring filter on `gloss_ru` |
| `--translation-input-mode` | `raw` | `raw`, `pos`, or `pos_meaning` |
| `--debug-sample` | `0` | Print raw output for first N items |
| `--retry` | backend-specific | Number of retry attempts |
| `--retry-delay` | backend-specific | Extra sleep (s) between retries |
| `--google-retries` | `2` | legacy alias for `--retry` |
| `--google-retry-delay` | `1.0` | legacy alias for `--retry-delay` |
| `--backend-info` | off | Test single translation and exit |

### Usage examples

#### 1. Sanity check before a long run

```bash
python3 -m src.sem_cat.02_translate_glosses --model-key google --backend-info
```

#### 2. Quick smoke test on a slow laptop (50 glosses, ~3 min)

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --model-key helsinki_opus_mt_ru_en --device cpu --limit 50
```

#### 3. Full NLLB run with round-trip QA (GPU recommended ☕)

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --model-key nllb_distilled_1_3b --device cuda --round-trip
```

#### 4. Full Google run with round-trip QA (~8–9 hours, grab coffee ☕☕)

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --model-key google --round-trip
```

### Output columns

See the root README.md for the full schema. Key columns:
- **Always:** `gloss_ru`, `gloss_en`, `qa_keep`, `qa_score`, `qa_flags`
- **With `--round-trip`:** `+ gloss_ru_back`, `roundtrip_distance`
- **With `--translation-input-mode pos/pos_meaning`:** `+ pos_hint`, `meaning_hint`, `source_count`

### QA flags reference

| Flag | Meaning |
|------|---------|
| `empty_translation` | Backend returned empty or None |
| `punctuation_only` | Result is only punctuation |
| `repeated_token_loop` | Obvious repetition garbage |
| `no_ascii_letters` | No Latin alphabet in result |
| `too_long_for_gloss` | Result >5× length of input |
| `multiword_for_singleword` | Single RU word → 3+ EN words |
| `roundtrip_far` | Back-translation edit distance > 0.5 |

> `qa_keep=False` does NOT blank out `gloss_en`. The raw translation is always saved so experts can see what went wrong.

### Incremental behavior

If the output CSV already exists with column `gloss_ru`, previously translated glosses are skipped automatically. Allows interrupted runs and subset experiments without redoing everything.

---

## Step 03 — Compare translations (`03_compare_translations.py`)

**Purpose:** merge N model outputs by `gloss_ru`, compute `risk_score` from QA signals and cross-model disagreement, produce sorted expert review queue and Gold Standard template.

```bash
# Compare all models
python3 -m src.sem_cat.03_compare_translations \
    --translations google=data/sem_cat/02_glosses_translated_google.csv \
    --translations helsinki_opus_mt_ru_en=data/sem_cat/02_glosses_translated_helsinki_opus_mt_ru_en.csv \
    --translations nllb_distilled_1_3b=data/sem_cat/02_glosses_translated_nllb_distilled_1_3b.csv

# Top-500 riskiest rows, single-word glosses first
python3 -m src.sem_cat.03_compare_translations \
    --translations google=data/sem_cat/02_glosses_translated_google.csv \
    --translations nllb_distilled_1_3b=data/sem_cat/02_glosses_translated_nllb_distilled_1_3b.csv \
    --top-k 500 --single-word-first
```

**Output:**

| File | Description |
|------|-------------|
| `03_translation_comparison_full.csv` | All glosses with risk scores |
| `03_translation_review_queue.csv` | Suspicious rows, sorted by risk ↓ |
| `03_translation_gold_template.csv` | Expert correction template |

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--translations` | required | `model_key=path.csv` (repeatable) |
| `--risk-threshold` | `0.35` | Min risk score for review queue |
| `--top-k` | all | Keep only top-N rows in review file |
| `--single-word-first` | off | Prioritize ambiguous single-word glosses |
| `--include-low-risk` | off | Include all rows in review file |
| `--verbose` | off | Print verbose output |

---

## Step 04 — WordNet lookup (`04_wordnet_lookup.py`)

**Purpose:** load translated glosses, attach POS, look up WordNet synsets, map to WordNet Domains.

```bash
python3 -m src.sem_cat.04_wordnet_lookup \
    --translated-file data/sem_cat/03_translation_comparison_full.csv \
    --wn-domains-file data/sem_cat/00_wn-domains-3.2-20070223
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--translated-file` | required | Input CSV from Step 02 or 03 |
| `--wn-domains-file` | required | Path to WordNet Domains mapping |
| `--out-file` | auto | Output CSV path |
| `--data-dir` | `data/vepkar/` | For POS derivation from meanings |
| `--pos-source` | `none` | `none`, `file`, or `meanings` |
| `--pos-file` | none | CSV with `gloss_ru,pos` columns |

### POS mapping to WordNet

| UPOS | WordNet POS |
|------|-------------|
| `NOUN` | `n` |
| `VERB` | `v` |
| `ADJ` | `a` |
| `ADV` | `r` |
| `NUM`, `PROPN` | (skipped → `factotum`) |

---

## Step 05 — Assign domains to meanings (`05_assign_domains.py`)

**Purpose:** merge WordNet Domain labels back into the four meanings files.

```bash
python3 -m src.sem_cat.05_assign_domains \
    --data-dir data/vepkar \
    --domains-file data/sem_cat/04_glosses_wn_domains.csv \
    --out-dir data/sem_cat/results
```

For each language `{vep, olo, lud, krl}` creates:

```text
data/sem_cat/results/meanings_{lang}_domains.csv
```

Columns: all original `meanings_{lang}.csv` columns + `wn_synset`, `wn_domain`.

---

## Full pipeline (from repo root)

```bash
source .venv/bin/activate

# 0. NLTK data (once)
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# 1. Translate — multiple models with round-trip QA ☕☕
python3 -m src.sem_cat.02_translate_glosses --model-key google --round-trip
python3 -m src.sem_cat.02_translate_glosses --model-key nllb_distilled_1_3b --device cuda --round-trip
python3 -m src.sem_cat.02_translate_glosses --model-key helsinki_opus_mt_ru_en --device cuda --round-trip

# 2. Compare models → expert review queue
python3 -m src.sem_cat.03_compare_translations \
    --translations google=data/sem_cat/02_glosses_translated_google.csv \
    --translations nllb_distilled_1_3b=data/sem_cat/02_glosses_translated_nllb_distilled_1_3b.csv \
    --translations helsinki_opus_mt_ru_en=data/sem_cat/02_glosses_translated_helsinki_opus_mt_ru_en.csv

# 3. WordNet lookup → domains
python3 -m src.sem_cat.04_wordnet_lookup \
    --translated-file data/sem_cat/03_translation_comparison_full.csv \
    --wn-domains-file data/sem_cat/00_wn-domains-3.2-20070223

# 4. Merge domains into meanings
python3 -m src.sem_cat.05_assign_domains \
    --domains-file data/sem_cat/04_glosses_wn_domains.csv \
    --out-dir data/sem_cat/results
```

Fast dev loop on a weak laptop:

```bash
# Translate 50 glosses with Marian (CPU, ~3 min per batch on old hardware)
python3 -m src.sem_cat.02_translate_glosses \
    --model-key helsinki_opus_mt_ru_en --device cpu --limit 50

# Run WordNet lookup on the partial result
python3 -m src.sem_cat.04_wordnet_lookup \
    --translated-file data/sem_cat/02_glosses_translated_helsinki_opus_mt_ru_en.csv \
    --wn-domains-file data/sem_cat/00_wn-domains-3.2-20070223 \
    --out-file data/sem_cat/04_glosses_wn_domains_dev.csv
```

---

## Step 06 — Build concept-level WDH (`06_concepts_wdh.py`)

**Purpose:** derive WDH labels for each concept by inheriting from its category.

```bash
python3 -m src.sem_cat.06_concepts_wdh \
    --cat-wdh data/sem_cat/concept_categories/concept_categories_wdh.tsv \
    --concepts data/sem_cat/concepts/concepts_with_english_417.csv \
    --out-file data/sem_cat/concepts/concepts_wdh.tsv
```

**Output:** `data/sem_cat/concepts/concepts_wdh.tsv`

Columns: `category_id`, `pos`, `concept_id`, `concept_ru`, `concept_en`,
`wdh`, `wdh_source`, `wdh_confidence`, `wdh_note`.

---

## Step 07 — Propagate WDH to meanings (`07_propagate_wdh.py`)

**Purpose:** propagate concept-level WDH to meanings that have a `concept_id`.
Optionally compares with gloss-based WDH from WordNet domain lookup and
records conflicts (gloss-based evidence wins).

```bash
python3 -m src.sem_cat.07_propagate_wdh \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --data-dir data/vepkar/ \
    --out-dir data/sem_cat/results/

# With gloss-based WDH for conflict detection:
python3 -m src.sem_cat.07_propagate_wdh \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --data-dir data/vepkar/ \
    --domains-file data/sem_cat/04_glosses_wn_domains.csv \
    --out-dir data/sem_cat/results/
```

**Output:**
- `data/sem_cat/results/meanings_{lang}_concept_wdh.csv` — per-language enriched meanings
- `data/sem_cat/results/wdh_conflicts.csv` — rows where concept and gloss WDH disagree

New columns added to meanings: `concept_wdh`, `gloss_wdh`, `wdh`,
`wdh_source`, `wdh_conflict`, `wdh_conflict_note`.

---

## Step 08 — Gap audit (`08_gap_audit.py`)

**Purpose:** identify missing or weak concept coverage in VepKar meanings.
Primary grouping axis: `meaning_ru`.

```bash
python3 -m src.sem_cat.08_gap_audit \
    --data-dir data/vepkar/ \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --out-dir data/sem_cat/results/

# With enriched meanings for WDH disagreement analysis:
python3 -m src.sem_cat.08_gap_audit \
    --data-dir data/vepkar/ \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --enriched-dir data/sem_cat/results/ \
    --out-dir data/sem_cat/results/
```

**Output reports:**

| File | Description |
|------|-------------|
| `audit_meanings_without_concept.csv` | Russian glosses without concept assignment, sorted by frequency |
| `audit_concept_usage.csv` | How many meanings use each concept_id |
| `audit_category_coverage.csv` | Per-category concept assignment statistics |
| `audit_wdh_disagreement.csv` | Systematic WDH conflicts between concept and gloss sources |
| `audit_meaning_ru_clusters.csv` | Groups of meanings sharing the same normalized Russian gloss |

---

## Concept-aware pipeline (from repo root)

After running steps 02–05, add concept-level WDH and gap analysis:

```bash
# 5. Build concept-level WDH
python3 -m src.sem_cat.06_concepts_wdh \
    --cat-wdh data/sem_cat/concept_categories/concept_categories_wdh.tsv \
    --concepts data/sem_cat/concepts/concepts_with_english_417.csv

# 6. Propagate WDH to meanings (with conflict detection)
python3 -m src.sem_cat.07_propagate_wdh \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --data-dir data/vepkar/ \
    --domains-file data/sem_cat/04_glosses_wn_domains.csv \
    --out-dir data/sem_cat/results/

# 7. Gap audit
python3 -m src.sem_cat.08_gap_audit \
    --data-dir data/vepkar/ \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --enriched-dir data/sem_cat/results/ \
    --out-dir data/sem_cat/results/
```

---

## Step 06 — Build concept-level WDH (`06_concepts_wdh.py`)

**Purpose:** derive WDH labels for each concept by inheriting from its category.

```bash
python3 -m src.sem_cat.06_concepts_wdh \
    --cat-wdh data/sem_cat/concept_categories/concept_categories_wdh.tsv \
    --concepts data/sem_cat/concepts/concepts_with_english_417.csv \
    --out-file data/sem_cat/concepts/concepts_wdh.tsv
```

**Output:** `data/sem_cat/concepts/concepts_wdh.tsv`

Columns: `category_id`, `pos`, `concept_id`, `concept_ru`, `concept_en`,
`wdh`, `wdh_source`, `wdh_confidence`, `wdh_note`.

---

## Step 07 — Propagate WDH to meanings (`07_propagate_wdh.py`)

**Purpose:** propagate concept-level WDH to meanings that have a `concept_id`.
Optionally compares with gloss-based WDH from WordNet domain lookup and
records conflicts (gloss-based evidence wins).

```bash
python3 -m src.sem_cat.07_propagate_wdh \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --data-dir data/vepkar/ \
    --out-dir data/sem_cat/results/

# With gloss-based WDH for conflict detection:
python3 -m src.sem_cat.07_propagate_wdh \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --data-dir data/vepkar/ \
    --domains-file data/sem_cat/04_glosses_wn_domains.csv \
    --out-dir data/sem_cat/results/
```

**Output:**
- `data/sem_cat/results/meanings_{lang}_concept_wdh.csv` — per-language enriched meanings
- `data/sem_cat/results/wdh_conflicts.csv` — rows where concept and gloss WDH disagree

New columns added to meanings: `concept_wdh`, `gloss_wdh`, `wdh`,
`wdh_source`, `wdh_conflict`, `wdh_conflict_note`.

---

## Step 08 — Gap audit (`08_gap_audit.py`)

**Purpose:** identify missing or weak concept coverage in VepKar meanings.
Primary grouping axis: `meaning_ru`.

```bash
python3 -m src.sem_cat.08_gap_audit \
    --data-dir data/vepkar/ \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --out-dir data/sem_cat/results/

# With enriched meanings for WDH disagreement analysis:
python3 -m src.sem_cat.08_gap_audit \
    --data-dir data/vepkar/ \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --enriched-dir data/sem_cat/results/ \
    --out-dir data/sem_cat/results/
```

**Output reports:**

| File | Description |
|------|-------------|
| `audit_meanings_without_concept.csv` | Russian glosses without concept assignment, sorted by frequency |
| `audit_concept_usage.csv` | How many meanings use each concept_id |
| `audit_category_coverage.csv` | Per-category concept assignment statistics |
| `audit_wdh_disagreement.csv` | Systematic WDH conflicts between concept and gloss sources |
| `audit_meaning_ru_clusters.csv` | Groups of meanings sharing the same normalized Russian gloss |

---

## Concept-aware pipeline (from repo root)

After running steps 02–05, add concept-level WDH and gap analysis:

```bash
# 5. Build concept-level WDH
python3 -m src.sem_cat.06_concepts_wdh \
    --cat-wdh data/sem_cat/concept_categories/concept_categories_wdh.tsv \
    --concepts data/sem_cat/concepts/concepts_with_english_417.csv

# 6. Propagate WDH to meanings (with conflict detection)
python3 -m src.sem_cat.07_propagate_wdh \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --data-dir data/vepkar/ \
    --domains-file data/sem_cat/04_glosses_wn_domains.csv \
    --out-dir data/sem_cat/results/

# 7. Gap audit
python3 -m src.sem_cat.08_gap_audit \
    --data-dir data/vepkar/ \
    --concepts-wdh data/sem_cat/concepts/concepts_wdh.tsv \
    --enriched-dir data/sem_cat/results/ \
    --out-dir data/sem_cat/results/
```

---

## Notes

- `02_translate_glosses.py` is incremental: existing `gloss_ru` values are skipped automatically on reruns.
- `04_wordnet_lookup.py` works best with POS information; use `--pos-source meanings` to derive POS automatically.
- Short and ambiguous glosses still need expert review even after QA flags and WordNet lookup.
- Best candidates for manual review: high `qa_score`, non-empty `qa_flags`, `lookup_status=not_found`, `wn_domain=factotum`.
- Concept-level WDH (steps 06–07) is a parallel path to the gloss-based pipeline. It does not replace gloss-based WDH; conflicts are recorded and gloss-based evidence takes priority.
- The gap audit (step 08) is designed for iterative ontology expansion. Use `audit_meanings_without_concept.csv` to find high-frequency glosses that need new concept assignments.
- Concept-level WDH (steps 06–07) is a parallel path to the gloss-based pipeline. It does not replace gloss-based WDH; conflicts are recorded and gloss-based evidence takes priority.
- The gap audit (step 08) is designed for iterative ontology expansion. Use `audit_meanings_without_concept.csv` to find high-frequency glosses that need new concept assignments.
