#!/bin/sh
# run from repository root folder
gitingest src \
  --include-pattern "*.py" \
  --include-pattern "*.md" \
  --include-pattern "*.toml" \
  --exclude-pattern "src/notebooks" \
  --exclude-pattern "*/__pycache__/*" \
  --exclude-pattern "*.ipynb" \
  --output out_gitingest/semcat_37_step_one_01.txt
