#!/usr/bin/env python3
import csv
import re
import sys


def split_fields(line):
    # Split on tabs or 2+ spaces to handle aligned columns.
    parts = [p.strip() for p in re.split(r"\t+|\s{2,}", line.rstrip())]
    return parts


def main(argv):
    if len(argv) != 3:
        print("Usage: test_to_csv.py <input> <output>", file=sys.stderr)
        return 2

    input_path, output_path = argv[1], argv[2]

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = [line.rstrip("\n") for line in infile]

    if len(lines) < 3:
        print("Input file does not look like expected format.", file=sys.stderr)
        return 2

    header = split_fields(lines[1])

    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)

        for line in lines[3:]:
            if not line.strip():
                continue
            row = split_fields(line)
            if len(row) < len(header):
                row += [""] * (len(header) - len(row))
            writer.writerow(row[: len(header)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
