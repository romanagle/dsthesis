#!/usr/bin/env python3
import csv
import re
import sys


def split_fields(line):
    # Split on tabs or 2+ spaces to handle aligned columns.
    parts = [p.strip() for p in re.split(r"\t+|\s{2,}", line.rstrip())]
    return parts


def load_isomodels(path):
    with open(path, "r", encoding="utf-8") as infile:
        lines = [line.rstrip("\n") for line in infile if line.strip()]

    if not lines:
        raise ValueError("isomodels file is empty")

    header = split_fields(lines[0])
    rows = {}
    for line in lines[1:]:
        fields = split_fields(line)
        if not fields:
            continue
        if len(fields) < len(header):
            fields += [""] * (len(header) - len(fields))
        rows[fields[0]] = fields[: len(header)]
    return header, rows


def fasta_records(path):
    with open(path, "r", encoding="utf-8") as infile:
        header = None
        seq_lines = []
        for line in infile:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:].strip()
                seq_lines = []
            elif not line.startswith("#"):
                seq_lines.append(line.strip())
        if header is not None:
            yield header, "".join(seq_lines)


def primary_sequence(seq):
    # Keep only the tRNA gene sequence in uppercase.
    return "".join(ch for ch in seq if ch.isupper())


def key_from_header(header):
    return header.split("|", 1)[0]


def key_from_isomodels_id(isomodels_id):
    return isomodels_id.split("|", 1)[0]


def main(argv):
    if len(argv) != 4:
        print(
            "Usage: collate_isomodels_first100.py <isomodels> <fasta> <output_csv>",
            file=sys.stderr,
        )
        return 2

    isomodels_path, fasta_path, output_path = argv[1], argv[2], argv[3]
    iso_header, iso_rows = load_isomodels(isomodels_path)

    # Map isomodels by sequence id (first token before '|').
    iso_by_key = {}
    for iso_id, row in iso_rows.items():
        iso_by_key[key_from_isomodels_id(iso_id)] = row

    output_header = ["sequence_name", "primary_sequence"] + iso_header[1:]
    written = 0
    missing = 0

    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(output_header)

        for header, seq in fasta_records(fasta_path):
            key = key_from_header(header)
            row = iso_by_key.get(key)
            if not row:
                missing += 1
                continue
            writer.writerow([row[0], primary_sequence(seq)] + row[1:])
            written += 1

    print(f"Wrote {written} rows to {output_path}", file=sys.stderr)
    if missing:
        print(f"Skipped {missing} FASTA records without isomodels match", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
