# dictorpus-space

Exploring hidden gems in the VepKar linguistic corpus using Python & AI models.
Low-resource languages: Veps and Karelian (four varieties).

```text
      📁 ___________           📁 ___________           📁 ___________
        /         /|          /         /|          /         /|
       /  data  / |   -->   /  src   /  |   -->   / results/  |
      /________/  |        /________/   |        /________/   |
      |        |  |        |        |   |        |        |   |
      | VepKar | /         |  sem_  |  /         |  CSVs  |  /
      |  CSVs  |/          |  cat   | /          | + stats| /
      ---------            ---------            ----------
        input                code                 output
```

## What is this repository?

This repo hosts small, focused experiments around the **VepKar** corpus and dictionaries:
- analysis of lexical and morphological distributions,
- semi-automatic tools for lexicography,
- semantic categorization experiments.

The main active sub-project is:

- `src/sem_cat/` — experiments on assigning **WordNet Domains** semantic labels
  to VepKar dictionary *meanings* (per-sense, not per-lemma).

---

## Project structure

```text
dictorpus-space/
├── data/
│   ├── vepkar/           # Raw exports from VepKar (meanings_*.csv, examples_*.csv)
│   └── sem_cat/          # Derived data for semantic categorization
│       └── wn-domains-3.2-20070223   # WordNet Domains mapping file
├── src/
│   ├── notebooks/        # One-off analysis notebooks
│   └── sem_cat/          # Semantic categorization pipeline
│       ├── translators/  # Translation backends (registry + factory)
│       │   ├── base.py            # Abstract Translator + error types
│       │   ├── google_translator.py
│       │   ├── hf_seq2seq_translator.py
│       │   ├── marian_translator.py
│       │   ├── nllb_translator.py
│       │   ├── model_registry.py  # ModelSpec definitions
│       │   └── factory.py         # build_translator() / build_reverse_translator()
│       ├── compare/      # Multi-model comparison pipeline
│       ├── qa/           # Translation QA flag detectors
│       ├── io/           # Cache loading, row building
│       ├── pipeline/     # Translation input preparation
│       ├── utils/        # Loaders, gloss normalization, wn-domains helpers
│       ├── 01_meanings_examples_counter.ipynb
│       ├── 02_translate_glosses.py         # RU → EN gloss translation
│       ├── 03_compare_translations.py      # N-model comparison + review queue
│       ├── 04_wordnet_lookup.py            # EN gloss → WN synset → WN domain
│       └── 05_assign_domains.py            # Merge domains back to meanings
├── tests/
│   └── sem_cat/          # Unit tests for the pipeline
├── README.md             # This file
└── requirements.txt
```

---

## Translator architecture

The translation layer uses a **registry + factory** pattern:

- **Model registry** ([`model_registry.py`](src/sem_cat/translators/model_registry.py))
  defines `ModelSpec` dataclasses for each supported model.
  Six models are registered:
  - `google` — Google Translate via `deep_translator`
  - `helsinki_opus_mt_ru_en` — MarianMT (Helsinki-NLP/opus-mt-ru-en)
  - `nllb_distilled_1_3b` — NLLB distilled 1.3B
  - `nllb_1_3b` — NLLB 1.3B
  - `nllb_3_3b` — NLLB 3.3B
  - `wmt19_ru_en` — Facebook WMT19 ru-en

- **Factory** ([`factory.py`](src/sem_cat/translators/factory.py))
  builds translator instances from `ModelSpec`.
  All models are first-class citizens; Google is not a special case.

- **Import safety**: No translator module fails to import due to missing
  optional dependencies. `deep_translator`, `torch`, and `transformers`
  are loaded lazily at instantiation time. If a dependency is missing,
  a clear `BackendUnavailableError` is raised.

- **Common contract**: Every translator implements `translate(text) -> str | None`
  and `translate_batch(texts) -> list[str | None]`. Failed translations
  return `None`, never empty string.

---

## How to set up the environment (Linux / WSL)

Use a local virtual environment. From the repository root:

```bash
# 1. Clone the repository (if not already)
git clone https://github.com/componavt/dictorpus-space.git
cd dictorpus-space

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install -U pip
pip install -r requirements.txt
```

---

## NLTK data (WordNet resources)

The semantic pipeline uses **NLTK WordNet** and **Open Multilingual Wordnet** data.
Once per environment, download the required corpora:

```bash
source .venv/bin/activate  # if not already active
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Running scripts

### Smoke test

```bash
# Verify a single model is available
python3 -m src.sem_cat.02_translate_glosses --model-key google --backend-info
python3 -m src.sem_cat.02_translate_glosses --model-key helsinki_opus_mt_ru_en --backend-info
```

### Step 02 — Translate glosses

```bash
# Translate with a specific model
python3 -m src.sem_cat.02_translate_glosses --model-key google
python3 -m src.sem_cat.02_translate_glosses --model-key nllb_distilled_1_3b --device cuda

# Legacy compatibility (still works)
python3 -m src.sem_cat.02_translate_glosses --backend marian
python3 -m src.sem_cat.02_translate_glosses --backend google
python3 -m src.sem_cat.02_translate_glosses --backend nllb --nllb-model facebook/nllb-200-distilled-1.3B
```

### Step 03 — Multi-model comparison

```bash
# Compare all models
python3 -m src.sem_cat.03_compare_translations \
    --translations google=data/sem_cat/02_glosses_translated_google.csv \
    --translations helsinki_opus_mt_ru_en=data/sem_cat/02_glosses_translated_helsinki_opus_mt_ru_en.csv \
    --translations nllb_distilled_1_3b=data/sem_cat/02_glosses_translated_nllb_distilled_1_3b.csv \
    --translations nllb_1_3b=data/sem_cat/02_glosses_translated_nllb_1_3b.csv \
    --translations nllb_3_3b=data/sem_cat/02_glosses_translated_nllb_3_3b.csv \
    --translations wmt19_ru_en=data/sem_cat/02_glosses_translated_wmt19_ru_en.csv

# Compare a subset
python3 -m src.sem_cat.03_compare_translations \
    --translations google=data/sem_cat/02_glosses_translated_google.csv \
    --translations nllb_distilled_1_3b=data/sem_cat/02_glosses_translated_nllb_distilled_1_3b.csv
```

### Steps 04–05 — WordNet lookup and domain assignment

```bash
python3 -m src.sem_cat.04_wordnet_lookup \
    --translated-file data/sem_cat/03_translation_comparison_full.csv \
    --wn-domains-file data/sem_cat/00_wn-domains-3.2-20070223

python3 -m src.sem_cat.05_assign_domains \
    --data-dir data/vepkar \
    --domains-file data/sem_cat/04_glosses_wn_domains.csv \
    --out-dir data/sem_cat/results
```

---

## Output schemas

### Step 02 output (`02_glosses_translated_{model_key}.csv`)

| Column | Description |
|--------|-------------|
| `gloss_ru` | Original Russian gloss |
| `gloss_en` | Translated English gloss |
| `qa_keep` | Whether the translation passes QA |
| `qa_score` | QA penalty score (0.0 = clean) |
| `qa_flags` | Semicolon-separated QA flags |
| `qa_version` | QA rule version |
| `model_key` | Registry key of the model used |
| `model_name` | Human-readable model name |
| `backend_family` | Backend family (google, hf_seq2seq, nllb) |
| `translation_input_mode` | Input mode (raw, pos, pos_meaning) |
| `input_text_used` | Actual text sent to translator |
| `pos_hint` | Dominant POS tag (if available) |
| `meaning_hint` | Full meaning context (if available) |
| `source_count` | Number of meanings using this gloss |
| `gloss_ru_back` | Back-translated Russian (if --round-trip) |
| `roundtrip_distance` | Edit distance for round-trip |
| `is_singleword_ru` | Whether gloss_ru is a single word |
| `input_token_count` | Token count of input |
| `output_token_count` | Token count of output |

### Step 03 output

| File | Description |
|------|-------------|
| `03_translation_comparison_full.csv` | All glosses with risk scores |
| `03_translation_review_queue.csv` | Suspicious rows, sorted by risk |
| `03_translation_gold_template.csv` | Expert correction template |

---

## Running tests

```bash
# All tests
pytest tests/sem_cat/ -q

# Translator tests only (offline, no model downloads)
pytest tests/sem_cat/translators/test_part1.py -q

# QA and I/O tests
pytest tests/sem_cat/test_part2.py -q

# Comparison pipeline tests
pytest tests/sem_cat/test_part3.py -q
```

---

## License & usage

This is a research-oriented repository.
Check the [LICENSE](LICENSE) file for the legal details.

The VepKar data themselves have their own licenses and must be cited appropriately
if you use them in derived work.
