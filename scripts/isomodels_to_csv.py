#!/usr/bin/env python3
import csv
import re
import sys


def split_fields(line):
    # Split on tabs or 2+ spaces to handle aligned columns.
    parts = [p.strip() for p in re.split(r"\t+|\s{2,}", line.rstrip())]
    return [p for p in parts if p]


def main(argv):
    if len(argv) != 3:
        print("Usage: isomodels_to_csv.py <input> <output_csv>", file=sys.stderr)
        return 2

    input_path, output_path = argv[1], argv[2]

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = [line.rstrip("\n") for line in infile if line.strip()]

    if not lines:
        print("Input file is empty.", file=sys.stderr)
        return 2

    header = split_fields(lines[0])
    rows = []
    for line in lines[1:]:
        fields = split_fields(line)
        if not fields:
            continue
        if len(fields) < len(header):
            fields += [""] * (len(header) - len(fields))
        rows.append([v.strip() for v in fields[: len(header)]])

    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([h.strip() for h in header])
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
