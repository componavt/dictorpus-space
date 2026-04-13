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

Purpose:
- load VepKar meanings,
- extract unique primary Russian glosses,
- translate them into English,
- save a translation cache CSV,
- attach QA metadata for later review.

Run from repository root:

```bash
source .venv/bin/activate
python3 -m src.sem_cat.02_translate_glosses --backend marian
```

Typical custom output file:

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian \
    --device cuda \
    --out-file data/sem_cat/glosses_translated_marian_2026.csv
```

Useful development run on a slow laptop:

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian \
    --device cpu \
    --limit 50
```

Focused subset run:

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian \
    --offset 500 \
    --limit 100
```

Round-trip QA run:

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian \
    --device cuda \
    --round-trip \
    --out-file data/sem_cat/glosses_translated_marian_rt.csv
```

Context-aware experimental run:

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian \
    --translation-input-mode pos_meaning \
    --limit 200
```

#### Input

By default the script reads VepKar meanings from:

```text
data/vepkar/meanings_vep.csv
data/vepkar/meanings_olo.csv
data/vepkar/meanings_lud.csv
data/vepkar/meanings_krl.csv
```

Expected important columns in these files:

- `lemma`
- `lang`
- `pos`
- `meaning_ru`

#### Main arguments

- `--data-dir`  
  Path to the directory with `meanings_*.csv`.

- `--out-dir`  
  Output directory used when `--out-file` is not given.

- `--out-file`  
  Full output path for the translation cache CSV.

- `--backend {marian,google}`  
  Translation backend.  
  `marian` = local HuggingFace MarianMT.  
  `google` = Google Translate via Python backend.

- `--device {cpu,cuda}`  
  Device for Marian translation.

- `--batch-size`  
  Batch size for Marian translation.

- `--round-trip`  
  After RU → EN translation, also run EN → RU back-translation and save QA fields.

- `--offset`  
  Skip the first N glosses after cache filtering.

- `--limit`  
  Process at most N glosses after cache filtering and offset.

- `--shuffle`  
  Shuffle remaining glosses before applying offset and limit.

- `--seed`  
  Random seed used with `--shuffle`.

- `--gloss-filter`  
  Translate only glosses containing the given substring.

- `--translation-input-mode {raw,pos,pos_meaning}`  
  Controls what is sent to the translator:
  - `raw`: only `gloss_ru`
  - `pos`: `POS | gloss_ru`
  - `pos_meaning`: `POS | gloss_ru | meaning_hint`

#### Output file

Default output file if `--out-file` is not specified:

```text
data/sem_cat/glosses_translated_{backend}.csv
```

Typical output columns:

- always:
  - `gloss_ru`
  - `gloss_en`
  - `qa_keep`
  - `qa_score`
  - `qa_flags`

- with `--round-trip`:
  - `gloss_ru_back`
  - `roundtrip_distance`

- with `--translation-input-mode pos` or `pos_meaning`:
  - `pos_hint`
  - `meaning_hint`
  - `source_count`

Example:

```csv
gloss_ru,gloss_en,qa_keep,qa_score,qa_flags
помощь,help,True,0.0,
тоня,"Tonia, turn it on.",True,0.2,multiword_for_singleword
..., ,False,1.0,empty_translation
```

Meaning of QA fields:

- `qa_keep`  
  `True` if the translation is kept in `gloss_en`.  
  `False` only for clearly unusable output.

- `qa_score`  
  Heuristic suspicion score from `0.0` to `1.0`.

- `qa_flags`  
  Semicolon-separated QA flags, for example:
  - `empty_translation`
  - `punctuation_only`
  - `repeated_token_loop`
  - `too_long_for_gloss`
  - `multiword_for_singleword`
  - `no_ascii_letters`
  - `roundtrip_far`

#### Incremental behavior

If the output CSV already exists and contains column `gloss_ru`, previously translated glosses are skipped automatically.

This allows:
- interrupted runs,
- continuing on another computer,
- small subset experiments without redoing the whole dataset.


---

## 🧭 Step 2 — WordNet lookup (`03_wordnet_lookup.py`)

Script: `src/sem_cat/03_wordnet_lookup.py`  

Purpose:
- load translated glosses from `02_translate_glosses.py`,
- attach POS information,
- look up WordNet synsets for English glosses,
- map synsets to WordNet Domains,
- save an enriched CSV for further analysis.

Run from repository root:

```bash
source .venv/bin/activate
python3 -m src.sem_cat.03_wordnet_lookup \
    --translated-file data/sem_cat/glosses_translated_marian_2026.csv \
    --wn-domains-file data/wn-domains-3.2-20070223
```

#### Input

Main input:
- translated CSV from `02_translate_glosses.py`

Example:

```csv
gloss_ru,gloss_en,qa_keep,qa_score,qa_flags
помощь,help,True,0.0,
бежать,run,True,0.0,
тоня,"Tonia, turn it on.",True,0.2,multiword_for_singleword
```

WordNet Domains file:
- `data/wn-domains-3.2-20070223`

Format example:

```text
00001740-n    factotum
00001930-n    cognition
```

#### Main arguments

- `--translated-file`  
  Input CSV produced by `02_translate_glosses.py`.

- `--wn-domains-file`  
  Path to the WordNet Domains mapping file.

- `--out-file`  
  Output CSV path.  
  Default: `data/sem_cat/glosses_wn_domains.csv`

- `--data-dir`  
  Path to `data/vepkar/`, used when POS is derived from meanings files.

- `--pos-source {none,file,meanings}`  
  Source of POS values:
  - `none` = do not use POS
  - `file` = read POS from `--pos-file`
  - `meanings` = derive dominant POS from VepKar meanings

- `--pos-file`  
  Optional CSV with columns `gloss_ru,pos` when `--pos-source file` is used.

#### POS mapping to WordNet

Current mapping:

```python
POS_MAP = {
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
    "NUM": None,
    "PROPN": None,
}
```

`NUM` and `PROPN` are not looked up in WordNet and fall back to `factotum`.

#### Output file

Default output file:

```text
data/sem_cat/glosses_wn_domains.csv
```

Typical output columns:

- `gloss_ru`
- `gloss_en`
- `pos`
- `wn_pos`
- `wn_synset`
- `synset_count`
- `lookup_status`
- `wn_domain`
- `qa_skip_reason`

If input QA columns exist, they are preserved:
- `qa_keep`
- `qa_score`
- `qa_flags`

Example:

```csv
gloss_ru,gloss_en,pos,wn_pos,wn_synset,synset_count,lookup_status,wn_domain,qa_skip_reason,qa_keep,qa_score,qa_flags
помощь,help,NOUN,n,aid.n.02,14,found,act,,True,0.0,
бежать,run,VERB,v,run.v.01,57,found,motion,,True,0.0,
тоня,"Tonia, turn it on.",NOUN,n,,0,not_found,factotum,,True,0.2,multiword_for_singleword
, ,NOUN,n,,0,skipped_empty,factotum,empty_gloss_en,False,1.0,empty_translation
```

Meaning of main fields:

- `wn_synset`  
  Best synset selected by NLTK WordNet.

- `synset_count`  
  Number of synsets returned by lookup.

- `lookup_status`  
  Lookup result status, for example:
  - `found`
  - `not_found`
  - `skipped_empty`
  - `skipped_pos`

- `wn_domain`  
  Domain label from wn-domains, or `factotum` as fallback.

- `qa_skip_reason`  
  Additional explanation related to QA or empty input.

#### Optimization

The script caches lookup results by `(gloss_en, wn_pos)`, so repeated identical lookups are not recomputed.

---

## Recommended run order

Full basic pipeline:

```bash
source .venv/bin/activate

python3 -m src.sem_cat.02_translate_glosses \
    --backend marian \
    --device cuda \
    --out-file data/sem_cat/glosses_translated_marian_2026.csv

python3 -m src.sem_cat.03_wordnet_lookup \
    --translated-file data/sem_cat/glosses_translated_marian_2026.csv \
    --wn-domains-file data/wn-domains-3.2-20070223 \
    --out-file data/sem_cat/glosses_wn_domains_2026.csv
```

Fast development loop on a weak laptop:

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend marian \
    --device cpu \
    --limit 50

python3 -m src.sem_cat.03_wordnet_lookup \
    --translated-file data/sem_cat/glosses_translated_marian.csv \
    --wn-domains-file data/wn-domains-3.2-20070223
```

Comparison run with Google:

```bash
python3 -m src.sem_cat.02_translate_glosses \
    --backend google \
    --out-file data/sem_cat/glosses_translated_google.csv
```

---

## Notes

- `02_translate_glosses.py` is incremental: existing `gloss_ru` values are skipped.
- `03_wordnet_lookup.py` works best with POS information.
- Short and ambiguous glosses still need expert review even after QA flags and WordNet lookup.
- The most useful rows for manual review are usually:
  - high `qa_score`
  - non-empty `qa_flags`
  - `lookup_status=not_found`
  - `wn_domain=factotum`

---

## Files produced by the pipeline

Most important intermediate files:

- `data/sem_cat/glosses_translated_marian.csv`
- `data/sem_cat/glosses_translated_google.csv`
- `data/sem_cat/glosses_wn_domains.csv`

These files are suitable for:
- manual expert review,
- backend comparison,
- WordNet coverage analysis,
- later merge into enriched meanings files.

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
