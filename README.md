# dictorpus-space

🔍 Exploring hidden gems in the VepKar linguistic corpus using Python & AI models. 🤖📊  
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

## 🧠 What is this repository?

This repo hosts small, focused experiments around the **VepKar** corpus and dictionaries:
- analysis of lexical and morphological distributions,
- semi-automatic tools for lexicography,
- semantic categorization experiments.

The main active sub-project right now is:

- `src/sem_cat/` — experiments on assigning **WordNet Domains** semantic labels  
  to VepKar dictionary *meanings* (per-sense, not per-lemma).

---

## 📁 Project structure

Current layout (simplified):

```text
dictorpus-space/
├── data/
│   ├── vepkar/           # Raw exports from VepKar (meanings_*.csv, examples_*.csv)
│   └── sem_cat/          # Derived data for semantic categorization
│       └── wn-domains-3.2-20070223   # WordNet Domains mapping file
├── src/
│   ├── notebooks/        # One-off analysis notebooks (not directly used in pipeline)
│   └── sem_cat/          # Semantic categorization pipeline (code)
│       ├── utils/        # Loaders, gloss normalization, wn-domains helpers
│       ├── translators/  # Translation backends (Google, MarianMT, etc.)
│       ├── 01_meanings_examples_counter.ipynb  # Data exploration notebook
│       ├── 02_translate_glosses.py             # RU → EN gloss translation
│       ├── 03_wordnet_lookup.py                # EN gloss → WN synset → WN domain
│       └── 04_assign_domains.py                # Merge domains back to meanings
├── README.md             # This file (high-level overview)
└── LICENSE
```

For a more technical description of the semantic categorization pipeline,  
see [`src/sem_cat/README.md`](src/sem_cat/README.md).

---

## 🛠️ How to set up the environment (Linux / WSL)

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

## 📚 NLTK data (WordNet resources)

The semantic pipeline uses **NLTK WordNet** and **Open Multilingual Wordnet** data.  
Once per environment, download the required corpora:

```bash
source .venv/bin/activate  # if not already active
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

You can run this from any directory, as long as the correct virtual environment is active.

---

## ▶️ Running scripts

From the repository root and with the virtualenv active:

```bash
# 1. Explore raw meanings & examples (in notebook, optional)
#    Open src/sem_cat/01_meanings_examples_counter.ipynb in Jupyter or Colab.

# 2. Translate unique Russian glosses to English
python3 -m src.sem_cat.02_translate_glosses --backend marian

# 3. Map translated glosses to WordNet synsets and WordNet Domains
python3 -m src.sem_cat.03_wordnet_lookup

# 4. Assign domains back to VepKar meanings (one enriched file per language)
python3 -m src.sem_cat.04_assign_domains
```

Each script has its own `--help` with additional options (paths, batch size, etc.):

```bash
python3 -m src.sem_cat.02_translate_glosses --help
```

---

## 🧪 Notebooks

Some exploratory work lives in:

- `src/notebooks/` — various earlier experiments (e.g. Ludic verb stems),
- `src/sem_cat/01_meanings_examples_counter.ipynb` —  
  statistics on meanings & examples before semantic categorization:
  - counts per language,
  - POS distribution,
  - gloss length, multi-part glosses, parentheticals,
  - top Russian words in glosses,
  - coverage of examples per meaning.

These notebooks are not part of the production pipeline,  
but they document the reasoning behind the design choices.

---

## 💡 License & usage

This is a research-oriented repository.  
Check the [LICENSE](LICENSE) file for the legal details.

The VepKar data themselves have their own licenses and must be cited appropriately  
if you use them in derived work.
