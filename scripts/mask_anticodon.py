#!/usr/bin/env python3
"""
Add anticodon-masked sequence column to tRNA CSV.

For each row:
  1. Find the anticodon loop in the secondary structure (a run of dots
     flanked by paired stems).
  2. Search for the Codon string within that loop to locate Sprinzl
     positions 34-36 (works for canonical 7-nt loops and non-canonical
     extended loops such as the 9-nt loop in row 77).
  3. Mask those 3 positions with 'NNN' and write to a new column
     `anticodon_masked_sequence`.
  4. Also record `anticodon_start` and `anticodon_end` (0-indexed,
     half-open) for downstream use.

Output: results/runs/isomodels_first100_anticodon_masked.csv
"""

import re
import sys
import pandas as pd


def find_anticodon_in_ss(seq: str, codon: str, ss: str):
    """
    Return (start, end) of the anticodon in seq (0-indexed, half-open).
    Strategy: for every run of dots in ss, look for the codon inside
    that loop (with at least 1 flanking nt on each side). The first
    match is the anticodon loop.

    Returns None if no loop contains the codon.
    """
    codon = codon.upper().replace("U", "T")
    seq_norm = seq.upper().replace("U", "T")

    for m in re.finditer(r"\.+", ss):
        ls, le = m.start(), m.end()
        loop_len = le - ls
        if loop_len < 5:          # too short to contain a 3-mer with flanking
            continue
        # Search for codon within the loop, keeping ≥1 nt on each side
        search_start = ls + 1
        search_end   = le - 3     # last valid start so codon ends before le
        for pos in range(search_start, search_end + 1):
            if seq_norm[pos:pos + 3] == codon:
                return pos, pos + 3

    return None


def mask_anticodon(seq: str, start: int, end: int, mask: str = "NNN") -> str:
    return seq[:start] + mask + seq[end:]


def main():
    in_path  = "results/runs/isomodels_first100_with_test.csv"
    out_path = "results/runs/isomodels_first100_anticodon_masked.csv"

    df = pd.read_csv(in_path)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    starts, ends, masked_seqs = [], [], []
    failures = []

    for i, row in df.iterrows():
        seq   = row["primary_sequence"]
        codon = row["Codon"]
        ss    = row["secondary_structure"]

        result = find_anticodon_in_ss(seq, codon, ss)
        if result is None:
            failures.append(i)
            starts.append(None)
            ends.append(None)
            masked_seqs.append(seq)   # leave sequence unmasked on failure
        else:
            s, e = result
            starts.append(s)
            ends.append(e)
            masked_seqs.append(mask_anticodon(seq, s, e))

    df["anticodon_start"]           = starts
    df["anticodon_end"]             = ends
    df["anticodon_masked_sequence"] = masked_seqs

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    if failures:
        print(f"WARNING: could not locate anticodon for {len(failures)} rows: {failures}",
              file=sys.stderr)
    else:
        print("Anticodon located successfully for all rows.")


if __name__ == "__main__":
    main()
