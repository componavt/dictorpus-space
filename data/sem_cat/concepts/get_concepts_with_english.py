#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

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


def normalize_fieldnames(fieldnames: list[str]) -> list[str]:
    return [name.strip() for name in fieldnames]


def resolve_pos_column(fieldnames: list[str]) -> str | None:
    for name in fieldnames:
        if name in OPTIONAL_POS_COLUMNS:
            return name
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python3 split_concepts_for_translation.py INPUT_TSV",
            file=sys.stderr,
        )
        return 1

    input_path = Path(sys.argv[1])

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
        if pos_column is None:
            print("Missing required column: pos or POS", file=sys.stderr)
            return 1

        output_fieldnames = [
            "category_id",
            pos_column,
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

            if (
                not is_blank(normalized_row.get("category_id"))
                and not is_blank(normalized_row.get(pos_column))
                and not is_blank(normalized_row.get("concept_id"))
                and not is_blank(normalized_row.get("concept_ru"))
                and not is_blank(normalized_row.get("concept_en"))
                and not is_blank(normalized_row.get("definition_ru"))
                and not is_blank(normalized_row.get("definition_en"))
            ):
                filtered_rows.append(normalized_row)

    record_count = len(filtered_rows)
    output_path = input_path.parent / f"concepts_with_english_{record_count}.csv"

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Selected rows: {record_count}")
    print(f"Created file: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
