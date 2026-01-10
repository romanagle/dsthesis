#!/usr/bin/env python3
import csv
import re
import sys


def normalize_isomodels_name(name):
    # Strip trailing ".trnaX" if present to match test.csv Name field.
    return re.sub(r"\.trna\d+$", "", name)


def dedupe_headers(headers):
    counts = {}
    out = []
    for h in headers:
        n = counts.get(h, 0)
        counts[h] = n + 1
        if n == 0:
            out.append(h)
        else:
            out.append(f"{h}_{n+1}")
    return out


def main(argv):
    if len(argv) != 4:
        print(
            "Usage: combine_isomodels_test.py <isomodels_csv> <test_csv> <output_csv>",
            file=sys.stderr,
        )
        return 2

    iso_path, test_path, out_path = argv[1], argv[2], argv[3]

    with open(iso_path, newline="", encoding="utf-8") as f:
        iso_reader = csv.reader(f)
        iso_header = [h.strip() for h in next(iso_reader)]
        iso_rows = {}
        for row in iso_reader:
            if not row:
                continue
            row = [v.strip() for v in row]
            iso_rows[normalize_isomodels_name(row[0])] = row

    with open(test_path, newline="", encoding="utf-8") as f:
        test_reader = csv.reader(f)
        test_header = [h.strip() for h in next(test_reader)]
        test_header = dedupe_headers(test_header)
        test_rows = {}
        for row in test_reader:
            if not row:
                continue
            row = [v.strip() for v in row]
            test_rows[row[0]] = row

    out_header = iso_header + test_header[1:]

    written = 0
    missing = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        for key, iso_row in iso_rows.items():
            test_row = test_rows.get(key)
            if not test_row:
                missing += 1
                continue
            writer.writerow(iso_row + test_row[1:])
            written += 1

    print(f"Wrote {written} rows to {out_path}", file=sys.stderr)
    if missing:
        print(f"Skipped {missing} isomodels rows without test match", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
