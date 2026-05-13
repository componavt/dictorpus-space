#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

CHUNK_SIZE = 200

REQUIRED_COLUMNS = {
    "category_id",
    "concept_id",
    "concept_ru",
    "concept_en",
    "definition_ru",
    "definition_en",
}

OPTIONAL_POS_COLUMNS = {"pos", "POS"}


def is_blank(value: str | None) -> bool:
    return value is None or value.strip() == ""


def chunked(rows: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def normalize_fieldnames(fieldnames: list[str]) -> list[str]:
    return [name.strip() for name in fieldnames]


def resolve_pos_column(fieldnames: list[str]) -> str | None:
    for name in fieldnames:
        if name in OPTIONAL_POS_COLUMNS:
            return name
    return None


def build_output_path(base_output: Path, index: int) -> Path:
    suffix = base_output.suffix if base_output.suffix else ".csv"
    stem = base_output.stem if base_output.stem else base_output.name
    parent = base_output.parent if str(base_output.parent) else Path(".")
    return parent / f"{stem}{index}{suffix}"


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: python3 split_concepts_for_translation.py INPUT_TSV OUTPUT_BASE.csv",
            file=sys.stderr,
        )
        return 1

    input_path = Path(sys.argv[1])
    output_base = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            print("Input file has no header.", file=sys.stderr)
            return 1

        original_fieldnames = normalize_fieldnames(reader.fieldnames)
        reader.fieldnames = original_fieldnames

        missing_columns = REQUIRED_COLUMNS - set(original_fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            print(f"Missing required columns: {missing}", file=sys.stderr)
            return 1

        pos_column = resolve_pos_column(original_fieldnames)
        output_fieldnames = [
            "category_id",
            pos_column if pos_column else "pos",
            "concept_id",
            "concept_ru",
            "concept_en",
            "definition_ru",
            "definition_en",
        ]

        filtered_rows: list[dict[str, str]] = []
        for row in reader:
            normalized_row = {
                key.strip(): (value if value is not None else "")
                for key, value in row.items()
            }
            if is_blank(normalized_row.get("concept_en")) and is_blank(normalized_row.get("definition_en")):
                filtered_rows.append(normalized_row)

    if not filtered_rows:
        print("No rows with both concept_en and definition_en missing were found.")
        return 0

    output_base.parent.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    for index, rows_chunk in enumerate(chunked(filtered_rows, CHUNK_SIZE), start=1):
        output_path = build_output_path(output_base, index)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows_chunk)
        created_files.append(output_path)

    print(f"Selected rows: {len(filtered_rows)}")
    print(f"Created files: {len(created_files)}")
    for path in created_files:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
