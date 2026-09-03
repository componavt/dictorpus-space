#!/bin/sh
set -eu

# Run from the repository root
#  --exclude-pattern "__init__.py" \

mkdir -p out_gitingest

gitingest . \
  --include-pattern "README.md" \
  --include-pattern "requirements.txt" \
  --include-pattern "requirements.lock" \
  --include-pattern "pytest.ini" \
  --include-pattern "src/sem_cat/*.py" \
  --include-pattern "src/sem_cat/**/*.py" \
  --include-pattern "src/sem_cat/*.md" \
  --include-pattern "src/sem_cat/**/*.md" \
  --include-pattern "src/sem_cat/*.toml" \
  --include-pattern "src/sem_cat/**/*.toml" \
  --include-pattern "tests/sem_cat/*.py" \
  --include-pattern "tests/sem_cat/**/*.py" \
  --include-pattern "data/sem_cat/concepts/get_concepts_with_english.py" \
  --include-pattern "data/sem_cat/concepts/split_concepts_for_translation.py" \
  --exclude-pattern "src/notebooks/*" \
  --exclude-pattern "*/__pycache__/*" \
  --exclude-pattern "*.pyc" \
  --exclude-pattern "*.ipynb" \
  --exclude-pattern "out_gitingest/*" \
  --output out_gitingest/semcat_46.md
