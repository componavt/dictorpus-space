#!/bin/sh
# run from repository root folder
gitingest src \
  --include-pattern "*.py" \
  --include-pattern "*.md" \
  --include-pattern "*.toml" \
  --exclude-pattern "src/notebooks" \
  --exclude-pattern "*/__pycache__/*" \
  --exclude-pattern "*.ipynb" \
  --output out/ai_concat/digest_git_ingest_27.txt
