#!/usr/bin/env python3
import csv
import sys


def split_sequence_name(name):
    parts = name.split("|")
    if len(parts) < 4:
        parts += [""] * (4 - len(parts))
    parts = parts[:4]
    # Strip any suffix after a dot in Species (e.g., ".trna1").
    if parts[3]:
        parts[3] = parts[3].split(".", 1)[0]
    return parts


def main(argv):
    if len(argv) != 3:
        print("Usage: split_sequence_name.py <input_csv> <output_csv>", file=sys.stderr)
        return 2

    input_path, output_path = argv[1], argv[2]

    with open(input_path, newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = [h.strip() for h in next(reader)]

        if "sequence_name" in header:
            name_col = "sequence_name"
        elif "Name" in header:
            name_col = "Name"
        elif "tRNAscanID" in header:
            name_col = "tRNAscanID"
        elif "Species" in header:
            name_col = None
        else:
            print(
                "No sequence name column found (expected 'sequence_name', 'Name', or 'tRNAscanID').",
                file=sys.stderr,
            )
            return 2

        if name_col is None:
            species_idx = header.index("Species")
            rows_out = [header]
            for row in reader:
                if not row:
                    continue
                row = [v.strip() for v in row]
                if row[species_idx]:
                    row[species_idx] = row[species_idx].split(".", 1)[0]
                rows_out.append(row)
        else:
            name_idx = header.index(name_col)
            new_header = (
                header[:name_idx]
                + ["SequenceID", "GenomeID", "Phylum/Class", "Species"]
                + header[name_idx + 1 :]
            )

            rows_out = [new_header]
            for row in reader:
                if not row:
                    continue
                row = [v.strip() for v in row]
                name_value = row[name_idx]
                parts = split_sequence_name(name_value)
                new_row = row[:name_idx] + parts + row[name_idx + 1 :]
                rows_out.append(new_row)

    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
