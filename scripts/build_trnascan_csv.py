#!/usr/bin/env python3
"""
build_trnascan_csv.py — parallel tRNAscan-SE combiner with per-family output

Expects a results directory containing per-genome file pairs produced by
tRNAscan-SE (one pair per GTDB genome):
    <genome_id>.isomodels.out   secondary-score / isomodels output  (-s flag)
    <genome_id>.trnascan.out    main scan output                     (-o flag)

A single struct file (written with -f, contains all genomes concatenated) is
loaded once and shared across workers via fork copy-on-write.

Parallelism:
  Each worker reads one genome's pair of files, joins with the struct lookup,
  and returns its rows.  The parent routes rows into per-phylum (or per-genome,
  per-species) CSV files as results arrive.

Usage:
    build_trnascan_csv.py <results_dir> <output_dir>
                          [--isomodels-suffix SUFFIX]
                          [--test-suffix SUFFIX]
                          [--struct-suffix SUFFIX]
                          [--workers N]
                          [--group-by {phylum,genome,species}]

Output:
    <output_dir>/
        index.csv          — group name, filename, row count
        <group>.csv        — one CSV per taxonomic group
"""

import argparse
import csv
import multiprocessing as mp
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Module-level globals — set in the parent, inherited by forked workers
# via copy-on-write.  Workers receive only small task args through the queue.
# ---------------------------------------------------------------------------
_out_header: list = []
_group_by: str = "phylum"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _split_fields(line: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\t+|\s{2,}", line.rstrip())]
    return [p for p in parts if p]


def parse_isomodels(path: str) -> tuple[list, dict]:
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        return [], {}
    header = _split_fields(lines[0])
    rows: dict = {}
    for line in lines[1:]:
        fields = _split_fields(line)
        if not fields:
            continue
        if len(fields) < len(header):
            fields += [""] * (len(header) - len(fields))
        fields = [v.strip() for v in fields[: len(header)]]
        key = fields[0]  # full tRNAscanID, e.g. NZ_CAXTDJ010000001.1.trna1
        rows[key] = fields
    return header, rows


def parse_test_out(path: str) -> tuple[list, dict]:
    """Parse tRNAscan-SE .out (3-line header).

    Duplicate column renames:
        second Begin -> intron_begin
        second End   -> intron_end
        second Score -> conf_score
    Drops 'Type' (redundant with Anticodon_predicted_isotype).
    """
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    if len(lines) < 3:
        return [], {}

    raw_header = _split_fields(lines[1])
    rename_second = {"Begin": "intron_begin", "End": "intron_end", "Score": "conf_score"}
    seen: dict = {}
    deduped: list = []
    for col in raw_header:
        n = seen.get(col, 0)
        seen[col] = n + 1
        if n == 0:
            deduped.append(col)
        elif n == 1 and col in rename_second:
            deduped.append(rename_second[col])
        else:
            deduped.append(f"{col}_{n + 1}")

    keep = [i for i, h in enumerate(deduped) if h != "Type"]
    header = [deduped[i] for i in keep]

    rows: dict = {}
    for line in lines[3:]:
        if not line.strip():
            continue
        fields = _split_fields(line)
        if not fields:
            continue
        if len(fields) < len(deduped):
            fields += [""] * (len(deduped) - len(fields))
        fields = [fields[i] for i in keep]
        # key = Name + ".trna" + tRNA# to match the isomodels tRNAscanID
        key = fields[0].strip() + ".trna" + fields[1].strip()
        rows[key] = fields
    return header, rows  # header[0] = "Name", skipped in output


def parse_struct(path: str) -> dict:
    """Returns {key: (primary_seq, dot_bracket)}.
    Primary sequence = full Seq: line uppercased (preserves intron/variable positions
    so that len(primary_seq) == len(dot_bracket) and positions align).
    """
    structs: dict = {}
    current_key = None
    current_seq = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            first = line.split()[0] if line.split() else ""
            if "Length:" in line:
                # ID line: "NZ_CAXTDJ010000001.1.trna1 (begin-end)  Length: N bp"
                current_key = first
                current_seq = None
            elif line.startswith("Seq:") and current_key:
                raw = line.split("Seq:", 1)[1].strip()
                current_seq = raw.upper()
            elif line.startswith("Str:") and current_key:
                structs[current_key] = (current_seq or "", line.split("Str:", 1)[1].strip())
                current_key = None
                current_seq = None
    return structs


# ---------------------------------------------------------------------------
# Worker — one call per genome
# ---------------------------------------------------------------------------

def _process_genome(args: tuple) -> list:
    """Read one genome's three files and return a list of (group_key, output_row) tuples."""
    iso_path, test_path, struct_path = args

    try:
        _, iso_rows = parse_isomodels(iso_path)
        _, test_rows = parse_test_out(test_path)
        structs = parse_struct(struct_path) if struct_path else {}
    except Exception as e:
        print(f"  WARNING: skipping {iso_path}: {e}", file=sys.stderr)
        return []

    # GenomeID = filename stem (strip the isomodels suffix)
    genome_id = os.path.basename(iso_path)
    for suffix in (".isomodels.out", ".isomodels", ".out"):
        if genome_id.endswith(suffix):
            genome_id = genome_id[: -len(suffix)]
            break

    result = []
    for key, iso_row in iso_rows.items():
        test_row = test_rows.get(key)
        if not test_row:
            continue

        trnascan_id = iso_row[0]                         # e.g. NZ_CAXTDJ010000001.1.trna1
        contig_id = re.sub(r"\.trna\d+$", "", trnascan_id)  # e.g. NZ_CAXTDJ010000001.1

        struct_entry = structs.get(key)
        primary_seq, struct = struct_entry if struct_entry else ("", "")

        row = (
            [trnascan_id, contig_id, genome_id, primary_seq]
            + [v.strip() for v in iso_row[1:]]
            + [v.strip() for v in test_row[1:]]
            + [struct]
        )
        result.append((genome_id, row))

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name.strip()) or "unknown"


def _detect_headers(iso_path: str, test_path: str) -> tuple[list, list]:
    iso_header, _ = parse_isomodels(iso_path)
    test_header, _ = parse_test_out(test_path)
    return iso_header, test_header


def _scan_pairs(
    results_dir: str, iso_suffix: str, test_suffix: str, struct_suffix: str
) -> list[tuple[str, str, str | None]]:
    """Find all (iso_path, test_path, struct_path_or_None) triples in results_dir."""
    pairs = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(iso_suffix):
            continue
        stem = fname[: -len(iso_suffix)]
        test_path = os.path.join(results_dir, stem + test_suffix)
        if not os.path.isfile(test_path):
            print(f"  WARNING: no {test_suffix} for {fname}, skipping", file=sys.stderr)
            continue
        struct_path = os.path.join(results_dir, stem + struct_suffix)
        pairs.append((
            os.path.join(results_dir, fname),
            test_path,
            struct_path if os.path.isfile(struct_path) else None,
        ))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list) -> int:
    parser = argparse.ArgumentParser(
        description="Parallel tRNAscan-SE combiner: per-genome files → per-family CSVs"
    )
    parser.add_argument("results_dir", help="Directory with per-genome tRNAscan-SE output files")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--isomodels-suffix", default=".isomodels.out",
        help="Suffix identifying isomodels files (default: .isomodels.out)",
    )
    parser.add_argument(
        "--test-suffix", default=".trnascan.out",
        help="Suffix identifying main scan output files (default: .trnascan.out)",
    )
    parser.add_argument(
        "--struct-suffix", default=".struct.out",
        help="Suffix identifying secondary structure files (default: .struct.out)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes (default: all CPUs)",
    )
    parser.add_argument(
        "--group-by", choices=["phylum", "genome", "species"], default="phylum",
        help="Taxonomic level used to partition output files (default: phylum)",
    )
    args = parser.parse_args(argv[1:])
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: find genome file pairs
    # ------------------------------------------------------------------
    print("Phase 1: scanning for genome file pairs...", file=sys.stderr)
    pairs = _scan_pairs(args.results_dir, args.isomodels_suffix, args.test_suffix, args.struct_suffix)
    if not pairs:
        print("No file pairs found. Check --isomodels-suffix and --test-suffix.", file=sys.stderr)
        return 1
    print(f"  {len(pairs):,} genomes", file=sys.stderr)

    # ------------------------------------------------------------------
    # Phase 2: detect output header from first genome
    # ------------------------------------------------------------------
    print("Phase 2: detecting headers...", file=sys.stderr)

    global _out_header, _group_by
    _group_by = args.group_by

    iso_header, test_header = _detect_headers(*pairs[0][:2])
    _out_header = (
        ["tRNAscanID", "ContigID", "GenomeID", "primary_sequence"]
        + iso_header[1:]    # Anticodon_predicted_isotype, Ala, Arg, ...
        + test_header[1:]   # tRNA #, Begin, End, Codon, intron_begin, ...
        + ["secondary_structure"]
    )

    # ------------------------------------------------------------------
    # Phase 3: process genomes in parallel, route rows into per-group CSVs
    # ------------------------------------------------------------------
    n_workers = args.workers or mp.cpu_count()
    print(f"Phase 3: processing {len(pairs):,} genomes with {n_workers} workers...", file=sys.stderr)

    group_counts: dict = defaultdict(int)
    written_headers: set = set()   # groups whose CSV already has a header row
    row_buffer: dict = defaultdict(list)  # group_key -> [rows] (flushed periodically)
    FLUSH_ROWS = 50_000  # flush after this many buffered rows total

    written = missing_struct = 0

    def flush_buffer() -> None:
        for gk, rows in row_buffer.items():
            fpath = os.path.join(args.output_dir, _safe_filename(gk) + ".csv")
            mode = "w" if gk not in written_headers else "a"
            with open(fpath, mode, newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if gk not in written_headers:
                    w.writerow(_out_header)
                    written_headers.add(gk)
                w.writerows(rows)
        row_buffer.clear()

    buffered = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        for rows in pool.imap_unordered(_process_genome, pairs, chunksize=4):
            for group_key, row in rows:
                row_buffer[group_key].append(row)
                group_counts[group_key] += 1
                written += 1
                buffered += 1
                if not row[3]:   # primary_seq column
                    missing_struct += 1
            if buffered >= FLUSH_ROWS:
                flush_buffer()
                buffered = 0

    flush_buffer()  # write any remaining rows

    # Write index
    index_path = os.path.join(args.output_dir, "index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "file", "rows"])
        w.writeheader()
        w.writerows(
            sorted(
                [{"group": g, "file": _safe_filename(g) + ".csv", "rows": c}
                 for g, c in group_counts.items()],
                key=lambda r: r["group"],
            )
        )

    print(
        f"\nDone. {written:,} rows across {len(group_counts)} groups → {args.output_dir}",
        file=sys.stderr,
    )
    print(f"Index: {index_path}", file=sys.stderr)
    if missing_struct:
        print(f"  {missing_struct:,} rows missing sequence/struct (not in struct file)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
