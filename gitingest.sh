#!/bin/sh
# run from repository root folder
gitingest src \
  --include-pattern "*.py" \
  --include-pattern "*.md" \
  --include-pattern "*.toml" \
  --exclude-pattern "src/notebooks" \
  --exclude-pattern "*/__pycache__/*" \
  --exclude-pattern "*.ipynb" \
  --output out_gitingest/semcat_38_step_one_02_task_pos.txt
