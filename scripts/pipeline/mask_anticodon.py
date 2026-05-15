#!/usr/bin/env python3
"""
Add anticodon-masked sequence column to a Sprinzl-annotated tRNAscan CSV.

Uses the anticodon_start column written by add_sprinzl.py, which records
the exact 0-indexed character position of Sprinzl position 34 in the primary
sequence. Replaces positions 34-36 with NNN.

Sequences where anticodon_start is absent (sprinzl_ok=0) are left unmasked.

Usage
-----
    python scripts/mask_anticodon.py
    python scripts/mask_anticodon.py --input  trnascan_GTDB_r226_v2_sprinzl.csv \\
                                     --output trnascan_GTDB_r226_v2_sprinzl.csv
"""

import csv
import os
import sys
import argparse
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_sprinzl.csv")
    parser.add_argument("--output", default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_sprinzl.csv")
    args = parser.parse_args()

    tmp_path = args.output + ".tmp"
    n_masked = n_skipped = n_total = 0

    with open(args.input,  newline="", encoding="utf-8") as fin, \
         open(tmp_path,    "w", newline="", encoding="utf-8") as fout:

        reader     = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ["anticodon_masked_sequence"]
        writer     = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            n_total += 1
            seq   = row["primary_sequence"]
            start = row.get("anticodon_start", "").strip()

            if start and start.lstrip("-").isdigit():
                s = int(start)
                row["anticodon_masked_sequence"] = seq[:s] + "NNN" + seq[s + 3:]
                n_masked += 1
            else:
                row["anticodon_masked_sequence"] = ""
                n_skipped += 1

            row.pop(None, None)
            writer.writerow(row)

            if n_total % 500_000 == 0:
                print(f"  {n_total:,} rows written …", flush=True)

    os.replace(tmp_path, args.output)
    print(f"Done. {n_total:,} rows written to {args.output}")
    print(f"  Masked:  {n_masked:,}")
    print(f"  Skipped (no Sprinzl mapping): {n_skipped:,}")


if __name__ == "__main__":
    main()
