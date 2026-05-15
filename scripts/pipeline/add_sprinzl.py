#!/usr/bin/env python3
"""
add_sprinzl.py
--------------
Map each tRNA secondary structure to Sprinzl position labels and
extract nucleotides at each Sprinzl position.

Handles:
  - 3-bp or 4-bp D stem
  - D loop 6-13 nt (standard 6-11 plus extended 12-13 using positions 20c/20d)
  - Type I variable loop (3-5 nt, all dots)
  - Type II variable loop (stem-loop structure for Leu/Ser), up to 47z (31 nt total)
  - Mismatches within stems (any char in stem positions)

Output columns added to the input CSV:
  sprinzl_ok       : 1 if mapping succeeded, 0 otherwise
  sprinzl_type     : 'type_I', 'type_II', or 'fail'
  pos_1 .. pos_73  : nucleotide at each Sprinzl position ('' if absent)
  pos_17a, pos_20a, pos_20b, pos_20c, pos_20d : variable D loop positions
  pos_47a .. pos_47z : variable type II variable loop positions

Usage
-----
python scripts/add_sprinzl.py \\
    --input  trnascan_GTDB_r226_v2_final_matched.csv \\
    --output trnascan_GTDB_r226_v2_sprinzl.csv
"""

import argparse
import csv
import re
import sys
import os
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


# ── Sprinzl position tables ────────────────────────────────────────────────────

# D loop Sprinzl labels by loop length (dots between D stem 5' end and D stem 3')
D_LOOP_LABELS = {
    6:  ['14','15','18','19','20','21'],
    7:  ['14','15','16','18','19','20','21'],
    8:  ['14','15','16','17','18','19','20','21'],
    9:  ['14','15','16','17','17a','18','19','20','21'],
    10: ['14','15','16','17','17a','18','19','20','20a','21'],
    11: ['14','15','16','17','17a','18','19','20','20a','20b','21'],
    12: ['14','15','16','17','17a','18','19','20','20a','20b','20c','21'],
    13: ['14','15','16','17','17a','18','19','20','20a','20b','20c','20d','21'],
}

# D stem 5' labels: 4-bp uses 10-13, 3-bp uses 11-13 (position 10 absent)
D5_LABELS = {4: ['10','11','12','13'], 3: ['11','12','13']}
# D stem 3' labels: 4-bp uses 22-25, 3-bp uses 23-25 (position 22 absent)
D3_LABELS = {4: ['22','23','24','25'], 3: ['23','24','25']}

# Variable loop type I labels by length
VAR_I_LABELS = {
    3: ['44','45','46'],
    4: ['44','45','46','47'],
    5: ['44','45','46','47','48'],
}

# All possible Sprinzl positions (for output column names)
CANONICAL_POS = (
    [str(i) for i in range(1, 44)] +
    ['44','45','46','47',
     '47a','47b','47c','47d','47e','47f','47g','47h','47i','47j',
     '47k','47l','47m','47n','47o','47p','47q','47r','47s','47t',
     '47u','47v','47w','47x','47y','47z',
     '48'] +
    [str(i) for i in range(49, 74)] +
    ['17a','20a','20b','20c','20d']
)

# ── Parser ─────────────────────────────────────────────────────────────────────

def sprinzl_map(ss):
    """
    Map a tRNAscan-SE secondary structure string to Sprinzl position labels.

    Returns:
        (labels, trna_type) where labels is a list of Sprinzl labels (or None
        for unassigned positions) and trna_type is 'type_I' or 'type_II'.
        Returns (None, 'fail') if mapping fails.
    """
    n = len(ss)
    labels = [None] * n

    # ── 1. Discriminator: last character ──────────────────────────────────────
    if n < 25:
        return None, 'fail'
    labels[n - 1] = '73'

    # ── 2. Work backwards from end to find T loop ─────────────────────────────
    # Find the rightmost run of exactly 7 dots (T loop = Sprinzl 54-60)
    # excluding the discriminator (last char)
    t_loop_pos = None
    for i in range(n - 8 - 5 - 1, 0, -1):   # leave room for T stem 3' + acc 3'
        if ss[i:i+7] == '.......':
            t_loop_pos = i
            break
    if t_loop_pos is None:
        return None, 'fail'

    for i in range(7):
        labels[t_loop_pos + i] = str(54 + i)     # 54-60

    # ── 3. T stem 5' (5 chars before T loop) → Sprinzl 49-53 ─────────────────
    t5_start = t_loop_pos - 5
    if t5_start < 0:
        return None, 'fail'
    for i in range(5):
        labels[t5_start + i] = str(49 + i)

    # ── 4. T stem 3' (5 chars after T loop) + acc stem 3' (7 chars) + disc ──
    after_t_loop = t_loop_pos + 7
    remaining = n - 1 - after_t_loop   # chars between T loop end and discriminator
    if remaining < 12:
        return None, 'fail'
    # T stem 3': positions after T loop = Sprinzl 61-65
    for i in range(5):
        labels[after_t_loop + i] = str(61 + i)
    # Acc stem 3': next 7 chars = Sprinzl 66-72
    for i in range(7):
        labels[after_t_loop + 5 + i] = str(66 + i)
    # Anything between (after_t_loop + 12) and (n - 1) is unexpected
    if after_t_loop + 12 != n - 1:
        return None, 'fail'

    # ── 5. Acc stem 5' (first 7 chars) → Sprinzl 1-7 ─────────────────────────
    if n < 9:
        return None, 'fail'
    for i in range(7):
        labels[i] = str(i + 1)

    # ── 6. Linker 8-9 (chars 7-8) ─────────────────────────────────────────────
    labels[7] = '8'
    labels[8] = '9'

    # ── 7. Middle section: positions 9 to t5_start - 1 ────────────────────────
    # Contains: D_stem5' D_loop D_stem3' linker26 AC_stem5' AC_loop AC_stem3' var_loop
    mid = ss[9:t5_start]
    mid_offset = 9

    # ── 7a. D stem 5': 3 or 4 chars ──────────────────────────────────────────
    d5_len = 4
    if len(mid) > 3 and mid[3] == '.':
        d5_len = 3  # only 3-char D stem before D loop starts
    # Confirm the character at d5_len is a dot (start of D loop)
    if len(mid) <= d5_len or mid[d5_len] not in '.':
        # Revert: try 4-bp
        d5_len = 4
    d5_labels = D5_LABELS.get(d5_len)
    if d5_labels is None:
        return None, 'fail'
    for i, sp in enumerate(d5_labels):
        labels[mid_offset + i] = sp
    pos = mid_offset + d5_len

    # ── 7b. D loop: dots until D stem 3' ──────────────────────────────────────
    # D stem 3' is d5_len consecutive non-dot characters after the D loop
    # Find the run of d5_len '<' (or any non-dot) after the dot block
    d_loop_start = pos
    # Move pos forward through dots
    while pos < mid_offset + len(mid) and ss[pos] == '.':
        pos += 1
    d_loop_end = pos
    d_loop_len = d_loop_end - d_loop_start

    if d_loop_len not in D_LOOP_LABELS:
        return None, 'fail'
    for i, sp in enumerate(D_LOOP_LABELS[d_loop_len]):
        labels[d_loop_start + i] = sp

    # ── 7c. D stem 3': d5_len chars ────────────────────────────────────────────
    d3_labels = D3_LABELS[d5_len]
    for i, sp in enumerate(d3_labels):
        labels[pos + i] = sp
    pos += d5_len

    # ── 7d. Linker 26 ──────────────────────────────────────────────────────────
    if pos >= n or ss[pos] != '.':
        return None, 'fail'
    labels[pos] = '26'
    pos += 1

    # ── 7e. Anticodon stem 5' (5 chars) → Sprinzl 27-31 ──────────────────────
    if pos + 5 > n:
        return None, 'fail'
    for i in range(5):
        labels[pos + i] = str(27 + i)
    pos += 5

    # ── 7f. Anticodon loop (must be exactly 7 dots) → Sprinzl 32-38 ──────────
    if ss[pos:pos + 7] != '.......':
        return None, 'fail'
    for i in range(7):
        labels[pos + i] = str(32 + i)
    pos += 7

    # ── 7g. Anticodon stem 3' (5 chars) → Sprinzl 39-43 ──────────────────────
    for i in range(5):
        labels[pos + i] = str(39 + i)
    pos += 5

    # ── 7h. Variable loop: pos to t5_start ────────────────────────────────────
    var_start = pos
    var_end   = t5_start
    var_region = ss[var_start:var_end]
    var_len = len(var_region)

    if var_len == 0:
        return None, 'fail'

    if all(c == '.' for c in var_region):
        # Type I
        if var_len not in VAR_I_LABELS:
            return None, 'fail'
        for i, sp in enumerate(VAR_I_LABELS[var_len]):
            labels[var_start + i] = sp
        trna_type = 'type_I'

    else:
        # Type II: variable arm forms a stem-loop
        # Sprinzl convention: 44, 45, 46, 47, [47a..47p], 48
        # We assign leading dots as 44, 45, 46, then stem-loop positions as
        # 47, 47a, 47b, ..., back to 48 at the end.
        # Simple assignment: fill from 44, use 47a-47p for the stem-loop interior
        sp_labels = ['44','45','46','47']
        extra_letters = 'abcdefghijklmnopqrstuvwxyz'
        extra_count = var_len - 5   # 44,45,46,47,...,48 = 5 for type I; extras go in between
        if extra_count < 0:
            return None, 'fail'
        for i in range(extra_count):
            if i < len(extra_letters):
                sp_labels.append(f'47{extra_letters[i]}')
            else:
                return None, 'fail'
        sp_labels.append('48')
        if len(sp_labels) != var_len:
            return None, 'fail'
        for i, sp in enumerate(sp_labels):
            labels[var_start + i] = sp
        trna_type = 'type_II'

    # Sanity check: all positions assigned
    if any(l is None for l in labels):
        return None, 'fail'

    return labels, trna_type


def extract_nucleotides(seq, labels):
    """
    Given a sequence and its Sprinzl labels, return a dict
    mapping Sprinzl position → nucleotide (uppercase).
    """
    result = {}
    for nt, sp in zip(seq, labels):
        if sp is not None:
            result[sp] = nt.upper()
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--input',  required=True, help='Matched trnascan CSV')
parser.add_argument('--output', required=True, help='Output CSV with Sprinzl columns')
args = parser.parse_args()

# ── Process ────────────────────────────────────────────────────────────────────

out_cols_extra = ['sprinzl_ok', 'sprinzl_type', 'anticodon_start'] + [f'pos_{p}' for p in CANONICAL_POS]

n_ok = n_fail = n_type2 = 0

with open(args.input) as fin, open(args.output, 'w', newline='') as fout:
    reader = csv.reader(fin)
    in_header = next(reader)
    writer = csv.writer(fout)
    writer.writerow(in_header + out_cols_extra)

    seq_idx = in_header.index('primary_sequence')
    ss_idx  = in_header.index('secondary_structure')

    for i, row in enumerate(reader):
        if row[0] == 'tRNAscanID':
            writer.writerow(row + [''] * len(out_cols_extra))
            continue

        seq = row[seq_idx]
        ss  = row[ss_idx]

        labels, trna_type = sprinzl_map(ss)

        if labels is None:
            row += ['0', 'fail', ''] + [''] * len(CANONICAL_POS)
            n_fail += 1
        else:
            anticodon_start = labels.index('34')
            nts = extract_nucleotides(seq, labels)
            sprinzl_cols = [nts.get(p, '') for p in CANONICAL_POS]
            row += ['1', trna_type, str(anticodon_start)] + sprinzl_cols
            n_ok += 1
            if trna_type == 'type_II':
                n_type2 += 1

        writer.writerow(row)

        if (i + 1) % 500_000 == 0:
            print(f'{i+1:,} rows processed  ok={n_ok:,} fail={n_fail:,}', flush=True)

total = n_ok + n_fail
print(f'\nDone.')
print(f'  Mapped:  {n_ok:,} ({100*n_ok/total:.1f}%)  '
      f'[type_I={n_ok-n_type2:,}, type_II={n_type2:,}]')
print(f'  Failed:  {n_fail:,} ({100*n_fail/total:.1f}%)')
