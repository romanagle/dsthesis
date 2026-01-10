#!/usr/bin/env python3
import csv
import sys


def parse_struct(path):
    structs = {}
    current_id = None
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("C") and " " in line:
                current_id = line.split(" ", 1)[0]
                current_id = current_id.split("|", 1)[0]
                continue
            if line.startswith("Str:") and current_id:
                structs[current_id] = line.split("Str:", 1)[1].strip()
                current_id = None
    return structs


def main(argv):
    if len(argv) != 4:
        print(
            "Usage: add_secondary_structure.py <struct_file> <input_csv> <output_csv>",
            file=sys.stderr,
        )
        return 2

    struct_path, input_csv, output_csv = argv[1], argv[2], argv[3]
    struct_map = parse_struct(struct_path)

    with open(input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header_raw = next(reader)
        header = [h.strip() for h in header_raw]
        if "SequenceID" not in header:
            print("SequenceID column not found in CSV.", file=sys.stderr)
            return 2
        id_idx = header.index("SequenceID")
        out_header = header + ["secondary_structure"]
        rows = [out_header]
        for row in reader:
            if not row:
                continue
            row = [v.strip() for v in row]
            seq_id = row[id_idx]
            rows.append(row + [struct_map.get(seq_id, "")])

    with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
