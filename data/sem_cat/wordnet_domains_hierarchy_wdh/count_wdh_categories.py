#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: python3 count_wdh_categories.py INPUT_FILE OUTPUT_FILE",
            file=sys.stderr,
        )
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    counter = Counter()

    with input_path.open("r", encoding="utf-8", newline="") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(
                    f"Skipping malformed line {line_number}: {line}",
                    file=sys.stderr,
                )
                continue

            _, categories_field = parts
            categories = categories_field.strip().split()

            for category in categories:
                if category:
                    counter[category] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["wn_domain_category", "count"])

        for category, count in sorted(counter.items()):
            writer.writerow([category, count])

    print(f"Unique categories: {len(counter)}")
    print(f"Created file: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
