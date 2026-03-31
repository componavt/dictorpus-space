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
├── 02_translate_glosses.py       # Step 1: RU → EN gloss translation
├── 03_wordnet_lookup.py          # Step 2: EN → WordNet synset → WN domain
├── 04_assign_domains.py          # Step 3: merge domains into meanings
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

- `id` — row id inside this export file,
- `lemma_id` — id of the lemma in VepKar,
- `meaning_id` — id of the meaning in VepKar,
- `meaning_num` — number of the meaning within the lemma (1, 2, 3…),
- `lemma` — lemma form in Veps or Karelian,
- `lang` — language code: `vep`, `olo`, `lud`, `krl`,
- `pos` — part-of-speech tag (UPOS+custom, e.g. `NOUN`, `VERB`, `PROPN`),
- `meaning_ru` — short Russian gloss (usually 1–3 words; may contain `;` and `(...)`).

### `data/sem_cat/wn-domains-3.2-20070223`

Text file (WordNet Domains 3.2):

```text
00001740-n    factotum
00001930-n    cognition
...
```

- left: 8-digit offset plus POS character (`n`, `v`, `a`, `r`),
- right: domain label (164 fine-grained domains).

Loaded via `utils.wn_domains.load_wn_domains()` into:

```python
{"00001740-n": ["factotum"], ...}
```

---

## 🔧 Step 1 — Translate glosses (`02_translate_glosses.py`)

Script: `src/sem_cat/02_translate_glosses.py`  

Purpose: translate all **unique primary** Russian glosses to English and store them in a cache.

Primary gloss is defined as:

- take the first part before `;`,
- remove all parentheticals `(…)`,
- lowercase and strip whitespace.

### CLI usage

From the repository root:

```bash
source .venv/bin/activate
python3 -m src.sem_cat.02_translate_glosses \
    --data-dir data/vepkar \
    --out-file data/sem_cat/glosses_translated.csv \
    --backend marian \
    --batch-size 64
```

Arguments:

- `--data-dir` — path to `meanings_*.csv` (default: `../../data/vepkar` from script dir),
- `--out-file` — translation cache CSV (default: `../../data/sem_cat/glosses_translated.csv`),
- `--backend` — `"marian"` or `"google"` (default: `"marian"`),
- `--batch-size` — batch size for MarianMT (default: `64`).

Cache format:

```text
gloss_ru,gloss_en,backend
"помощь","help","marian"
...
```

If the file already exists, previously translated `gloss_ru` values are skipped.

---

## 🧭 Step 2 — WordNet lookup (`03_wordnet_lookup.py`)

Script: `src/sem_cat/03_wordnet_lookup.py`  

Purpose: for each `(gloss_en, pos)` pair, find the best WordNet synset and map it to a WordNet Domain.

### POS mapping

```python
POS_MAP = {
    "NOUN": "n",
    "VERB": "v",
    "ADJ":  "a",
    "ADV":  "r",
}
```

All other POS are treated as **non-WordNet** and default to `"factotum"`.

### CLI usage

```bash
source .venv/bin/activate
python3 -m src.sem_cat.03_wordnet_lookup \
    --translated-file data/sem_cat/glosses_translated.csv \
    --wn-domains-file data/sem_cat/wn-domains-3.2-20070223 \
    --out-file data/sem_cat/glosses_wn_domains.csv
```

Output CSV:

```text
gloss_ru,gloss_en,backend,wn_synset,wn_domain
"помощь","help","marian","aid.n.02","act"
"бежать","run","marian","run.v.01","motion"
...
```

---

## 🏷️ Step 3 — Assign domains to meanings (`04_assign_domains.py`)

Script: `src/sem_cat/04_assign_domains.py`  

Purpose: merge WordNet Domain labels back into the four meanings files, one enriched file per language.

### CLI usage

```bash
source .venv/bin/activate
python3 -m src.sem_cat.04_assign_domains \
    --data-dir data/vepkar \
    --domains-file data/sem_cat/glosses_wn_domains.csv \
    --out-dir data/sem_cat/results
```

For each language `lang in {vep, olo, lud, krl}` it creates:

```text
data/sem_cat/results/meanings_{lang}_domains.csv
```

Columns:

- all original `meanings_{lang}.csv` columns,
- plus `wn_synset` and `wn_domain`.

Coverage statistics (per language) are printed to stdout.

---

## 🔄 Reproducible pipeline summary

From repo root:

```bash
source .venv/bin/activate

# 0. Ensure NLTK data
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# 1. Translate glosses (cached)
python3 -m src.sem_cat.02_translate_glosses --backend marian

# 2. WordNet lookup → domains
python3 -m src.sem_cat.03_wordnet_lookup

# 3. Merge domains into meanings
python3 -m src.sem_cat.04_assign_domains
```

Intermediate artifacts:

- `data/sem_cat/glosses_translated.csv`
- `data/sem_cat/glosses_wn_domains.csv`
- `data/sem_cat/results/meanings_{lang}_domains.csv`

These files can be versioned and analyzed independently.
