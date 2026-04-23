#!/bin/sh
# run from repository root folder
# Запускать из корня репозитория
files-to-prompt src \
  -e py \
  -e md \
  --ignore "__pycache__" \
  --ignore "notebooks" \
  --ignore "*.ipynb" \
  --markdown \
  -o out/ai_concat/files-to-prompt_01.md
