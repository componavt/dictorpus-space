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

## 🔗 Data flow overview

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
   (MarianMT, Google, ...)
         │
         ▼
  English glosses (cache)
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
```

Key design constraints:

- **Cache everything**: translation and synset/domain lookup are computed once per unique gloss.
- **POS-aware** lookup: where possible, disambiguate using the VepKar POS tag.
- **Graceful fallbacks**: if no synset/domain is found, assign a neutral domain (e.g. `factotum`).

---

## 📁 Code structure

```text
src/sem_cat/
├── README.md                     # This file (technical overview)
├── __init__.py                   # Package marker
├── 01_meanings_examples_counter.ipynb
│                                 # Exploration of meanings & examples
├── 02_translate_glosses.py       # Step 1: RU→EN gloss translation + QA flags
├── 03_compare_translations.py    # Step 2: compare Marian vs Google, risk scoring
├── 04_wordnet_lookup.py          # Step 3: EN→WordNet synset→WN domain
├── 05_assign_domains.py          # Step 4: merge domains into meanings
├── utils/
│   ├── __init__.py
│   ├── gloss_normalizer.py       # Parentheses & ';'-based gloss processing
│   ├── vepkar_loader.py          # Load and merge meanings_*.csv
│   └── wn_domains.py             # Load wn-domains mapping, synset key helper
└── translators/
    ├── __init__.py
    ├── base.py                   # Abstract Translator class
    ├── google_translator.py      # Google Translate backend (deep_translator)
    └── marian_translator.py      # Local MarianMT backend
```

---

## 🧩 Input formats

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

## 🧩 Input formats

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

## 🔧 Step 1 — Translate glosses (`02_translate_glosses.py`)

**Purpose:** load VepKar meanings, extract unique Russian glosses, translate to English, save cache with QA metadata.

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/vepkar/` | Path to directory with `meanings_*.csv` |
| `--out-dir` | `data/sem_cat/` | Output directory (if `--out-file` not given) |
| `--out-file` | auto | Full output CSV path |
| `--backend` | `marian` | `marian` (local) or `google` (API) |
| `--device` | `cpu` | `cpu` or `cuda` for MarianMT |
| `--batch-size` | `64` | Batch size for Marian translation |
| `--round-trip` | off | Back-translate EN→RU for QA |
| `--offset` | `0` | Skip first N glosses after cache filter |
| `--limit` | all | Process at most N glosses |
| `--shuffle` | off | Shuffle glosses before offset/limit |
| `--seed` | `42` | Random seed for shuffle |
| `--gloss-filter` | none | Substring filter on `gloss_ru` |
| `--translation-input-mode` | `raw` | `raw`, `pos`, or `pos_meaning` |
| `--debug-sample` | `0` | Print raw output for first N items |
| `--google-retries` | `2` | Retry count for empty Google responses |
| `--google-retry-delay` | `1.0` | Extra sleep (s) between retries |
| `--backend-info` | off | Test single translation and exit |

### Usage examples

#### 1. Sanity check before a long run

```bash
python3 -m src.sem_cat.02_translate_glosses --backend google --backend-info
```

#### 2. Quick smoke test on a slow laptop (50 glosses, ~3 min)

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian --device cpu --limit 50
```

#### 3. Debug raw Google responses (5 items, verbose)

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend google --limit 5 --debug-sample 5
```

#### 4. Full Marian run with round-trip QA (GPU recommended ☕)

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian --device cuda --round-trip \
    --out-file data/sem_cat/02_glosses_translated_marian_rt.csv
```

#### 5. Full Google run with round-trip QA (~8–9 hours, grab coffee ☕☕)

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend google --round-trip \
    --out-file data/sem_cat/02_glosses_translated_google_rt.csv
```

#### 6. Subset by offset (e.g. resume from row 500)

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian --offset 500 --limit 100
```

#### 7. Experimental context-aware input

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian --translation-input-mode pos_meaning --limit 200
```

### Output columns

- **Always:** `gloss_ru`, `gloss_en`, `qa_keep`, `qa_score`, `qa_flags`
- **With `--round-trip`:** `+ gloss_ru_back`, `roundtrip_distance`
- **With `--translation-input-mode pos/pos_meaning`:** `+ pos_hint`, `meaning_hint`, `source_count`

**Example output rows:**

```csv
gloss_ru,gloss_en,qa_keep,qa_score,qa_flags
помощь,help,True,0.0,
тоня,"Tonia, turn it on.",True,0.2,multiword_for_singleword
-же,,False,1.0,empty_translation
```

> 💡 Even when `qa_keep=False`, `gloss_en` is saved as-is so experts can
> diagnose what the backend produced.

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

> ⚠️ `qa_keep=False` does NOT blank out `gloss_en`. The raw translation is always saved so experts can see what went wrong.

### Incremental behavior

If the output CSV already exists with column `gloss_ru`, previously translated glosses are skipped automatically. Allows interrupted runs and subset experiments without redoing everything.

---

## 🔀 Step 2 — Compare translations (`03_compare_translations.py`)

**Purpose:** merge Marian and Google caches by `gloss_ru`, compute `risk_score` from QA signals and cross-backend disagreement, produce sorted expert review queue and Gold Standard template.

```bash
# Basic run (reads both _rt.csv files by default)
python3 -m src.sem_cat.03_compare_translations

# Top-500 riskiest rows, single-word glosses first
python3 -m src.sem_cat.03_compare_translations \
    --top-k 500 --single-word-first
```

**Input:** `02_glosses_translated_marian_rt.csv` + `02_glosses_translated_google_rt.csv`

**Output:**

| File | Description |
|------|-------------|
| `03_translation_comparison_full.csv` | All glosses with risk scores |
| `03_translation_review_queue.csv` | Suspicious rows, sorted by risk ↓ |
| `03_translation_gold_template.csv` | Expert correction template |

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--risk-threshold` | `0.35` | Min risk score for review queue |
| `--top-k` | all | Keep only top-N rows in review file |
| `--single-word-first` | off | Prioritize ambiguous single-word glosses |
| `--prefer-backend-strategy` | `heuristic` | `conservative` → more rows to manual review |

---

## 🧭 Step 3 — WordNet lookup (`04_wordnet_lookup.py`)

**Purpose:** load translated glosses, attach POS, look up WordNet synsets, map to WordNet Domains.

```bash
python3 -m src.sem_cat.04_wordnet_lookup \
    --translated-file data/sem_cat/03_translation_comparison_full.csv \
    --wn-domains-file data/sem_cat/00_wn-domains-3.2-20070223
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--translated-file` | required | Input CSV from Step 1 or 2 |
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

### Output columns

`gloss_ru`, `gloss_en`, `pos`, `wn_pos`, `wn_synset`, `synset_count`, `lookup_status`, `wn_domain`, `qa_skip_reason` (+ preserved QA columns if present).

---

## 🏷️ Step 4 — Assign domains to meanings (`05_assign_domains.py`)

**Purpose:** merge WordNet Domain labels back into the four meanings files.

```bash
python3 -m src.sem_cat.05_assign_domains \
    --data-dir data/vepkar \
    --domains-file data/sem_cat/04_glosses_wn_domains.csv \
    --out-dir data/sem_cat/results
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/vepkar/` | Path to VepKar meanings directory |
| `--domains-file` | `data/sem_cat/04_glosses_wn_domains.csv` | WN domains CSV from Step 3 |
| `--out-dir` | `data/sem_cat/results/` | Output directory for enriched CSVs |

For each language `{vep, olo, lud, krl}` creates:

```text
data/sem_cat/results/meanings_{lang}_domains.csv
```

Columns: all original `meanings_{lang}.csv` columns + `wn_synset`, `wn_domain`.

---

## 🔄 Full pipeline (from repo root)

```bash
source .venv/bin/activate

# 0. NLTK data (once)
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# 1. Translate — both backends with round-trip QA ☕☕
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian --device cuda --round-trip \
    --out-file data/sem_cat/02_glosses_translated_marian_rt.csv
python3 -m src.sem_cat.02_translate_glosses \
    --backend google --round-trip \
    --out-file data/sem_cat/02_glosses_translated_google_rt.csv

# 2. Compare backends → expert review queue
python3 -m src.sem_cat.03_compare_translations

# 3. WordNet lookup → domains
python3 -m src.sem_cat.04_wordnet_lookup \
    --translated-file data/sem_cat/03_translation_comparison_full.csv \
    --wn-domains-file data/sem_cat/00_wn-domains-3.2-20070223 \
    --out-file data/sem_cat/04_glosses_wn_domains.csv

# 4. Merge domains into meanings
python3 -m src.sem_cat.05_assign_domains \
    --domains-file data/sem_cat/04_glosses_wn_domains.csv \
    --out-dir data/sem_cat/results
```

Fast dev loop on a weak laptop 🐢:

```bash
# Translate 50 glosses with Marian (CPU, ~3 min per batch on old hardware)
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian --device cpu --limit 50

# Run WordNet lookup on the partial result
python3 -m src.sem_cat.04_wordnet_lookup \
    --translated-file data/sem_cat/02_glosses_translated_marian.csv \
    --wn-domains-file data/sem_cat/00_wn-domains-3.2-20070223 \
    --out-file data/sem_cat/04_glosses_wn_domains_dev.csv
```

---

## Notes

- `02_translate_glosses.py` is incremental: existing `gloss_ru` values are skipped automatically on reruns.
- `04_wordnet_lookup.py` works best with POS information; use `--pos-source meanings` to derive POS automatically.
- Short and ambiguous glosses still need expert review even after QA flags and WordNet lookup.
- Best candidates for manual review: high `qa_score`, non-empty `qa_flags`, `lookup_status=not_found`, `wn_domain=factotum`.
