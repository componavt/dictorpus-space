#!/bin/sh
# run from repository root folder
gitingest src \
  --include-pattern "*.py" \
  --include-pattern "*.md" \
  --exclude-pattern "src/notebooks" \
  --exclude-pattern "*/__pycache__/*" \
  --exclude-pattern "*.ipynb" \
  --output out/ai_concat/digest_git_ingest_16_after_kodik2.txt
