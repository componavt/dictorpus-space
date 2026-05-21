# dictorpus-space

Small research experiments around the **VepKar** corpus and dictionaries:
Veps + Karelian, low-resource languages, messy data, occasional joy.

```text
data/  -->  src/  -->  results/
  |         |          |
  |         |          └─ CSVs, audits, review queues
  |         └─ scripts and experiments
  └─ VepKar exports and derived files
```

## What is this repository?

This repository hosts compact research tooling for:
- corpus and dictionary analysis,
- lexicographic utilities,
- semantic categorization experiments.

## Main active sub-project

The main active sub-project is:

- [`src/sem_cat/`](src/sem_cat/README.md) — a semantic categorization pipeline
  that assigns **WordNet Domains** labels to VepKar dictionary **meanings**
  (per-sense, not per-lemma).

In short, it does this:

```text
Russian glosses
      ↓
translation
      ↓
WordNet lookup
      ↓
domain assignment
      ↓
concept-level WDH
      ↓
gap audit
```

If you want the real technical documentation, commands, outputs, caveats,
and the concept-aware pipeline, go here:

- **[`src/sem_cat/README.md`](src/sem_cat/README.md)**

## Repository sketch

```text
dictorpus-space/
├── data/         # raw exports + derived experiment data
├── src/          # active code
│   └── sem_cat/  # main active semantic-labeling pipeline
├── tests/        # unit tests
└── README.md     # this file
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pytest tests/sem_cat/ -q
```

## Status

This repo is research-oriented and evolves iteratively.
The documentation in `src/sem_cat/README.md` is the canonical guide
for the active semantic-labeling workflow.