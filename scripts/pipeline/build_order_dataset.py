#!/usr/bin/env python3
"""
Build per-order coevolution datasets.

For each order with >= MIN_GENOMES and all 20 canonical aaRS covered,
outputs paired FASTA files:
  order_datasets/{order}/{AA}_aaRS.faa   - protein sequences
  order_datasets/{order}/{AA}_tRNA.faa   - cognate tRNA sequences

Headers include genome accession so sequences can be paired downstream.
"""

import csv
import os
import re
from collections import defaultdict
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


# ── Config ────────────────────────────────────────────────────────────────────
HITS_CSV       = f"{DATA_ROOT}/tigrfam_hits.csv"
TRNASCAN_CSV   = f"{DATA_ROOT}/trnascan_GTDB_r226_v2_final.csv"
COMBINED_FAA   = f"{DATA_ROOT}/gtdb_all_tagged.faa"
OUT_DIR        = f"{DATA_ROOT}/order_datasets"
MIN_GENOMES    = 10   # minimum genomes per taxonomic group
TAX_LEVELS     = ["domain", "phylum", "class", "order"]

# aaRS protein_name → tRNA isotype(s)
AARS_TO_ISOTYPE = {
    "AlaRS":         ["Ala"],
    "ArgRS":         ["Arg"],
    "AsnRS":         ["Asn"],
    "AspRS":         ["Asp"],
    "CysRS":         ["Cys"],
    "GlnRS":         ["Gln"],
    "GluRS":         ["Glu"],
    "GlyRS_alpha":   ["Gly"],
    "GlyRS_beta":    ["Gly"],
    "GlyRS_dimeric": ["Gly"],
    "HisRS":         ["His"],
    "IleRS":         ["Ile", "Ile2"],
    "LeuRS_fam1":    ["Leu"],
    "LeuRS_fam2":    ["Leu"],
    "LysRS_fam1":    ["Lys"],
    "LysRS_fam2":    ["Lys"],
    "MetRS":         ["Met", "fMet"],
    "PheRS_alpha":   ["Phe"],
    "PheRS_beta_fam1": ["Phe"],
    "PheRS_beta_fam2": ["Phe"],
    "ProRS_fam1":    ["Pro"],
    "ProRS_fam2":    ["Pro"],
    "SerRS":         ["Ser"],
    "ThrRS":         ["Thr"],
    "TrpRS":         ["Trp"],
    "TyrRS":         ["Tyr"],
    "ValRS":         ["Val"],
}

# Canonical AA set (20); each must be covered per order
AA_GROUPS = {
    "Ala": ["AlaRS"],
    "Arg": ["ArgRS"],
    "Asn": ["AsnRS"],
    "Asp": ["AspRS"],
    "Cys": ["CysRS"],
    "Gln": ["GlnRS"],
    "Glu": ["GluRS"],
    "Gly": ["GlyRS_alpha", "GlyRS_beta", "GlyRS_dimeric"],
    "His": ["HisRS"],
    "Ile": ["IleRS"],
    "Leu": ["LeuRS_fam1", "LeuRS_fam2"],
    "Lys": ["LysRS_fam1", "LysRS_fam2"],
    "Met": ["MetRS"],
    "Phe": ["PheRS_alpha", "PheRS_beta_fam1", "PheRS_beta_fam2"],
    "Pro": ["ProRS_fam1", "ProRS_fam2"],
    "Ser": ["SerRS"],
    "Thr": ["ThrRS"],
    "Trp": ["TrpRS"],
    "Tyr": ["TyrRS"],
    "Val": ["ValRS"],
}
ALL_AA = set(AA_GROUPS.keys())
NAME_TO_AA = {n: aa for aa, names in AA_GROUPS.items() for n in names}


def normalize_genome_id(gid):
    gid = gid.replace("_genomic", "")
    if gid.startswith("GCF_"):
        return "RS_" + gid
    if gid.startswith("GCA_"):
        return "GB_" + gid
    return gid


def safe_dirname(order):
    return re.sub(r"[^\w\-]", "_", order)


# ── 1. Load hits: best aaRS hit per (genome, protein_name) by e-value ─────────
print("Loading aaRS hits …")

# genome_info[genome] = {order, domain, phylum, ...}
genome_info  = {}
# hits[genome][protein_name] = (protein_accession, evalue)
best_hit = defaultdict(dict)

with open(HITS_CSV) as f:
    for row in csv.DictReader(f):
        pname  = row["protein_name"]
        genome = row["genome_accession"]
        order  = row["order"]
        if not order or pname not in AARS_TO_ISOTYPE:
            continue
        ev = float(row["evalue"])
        prev = best_hit[genome].get(pname)
        if prev is None or ev < prev[1]:
            best_hit[genome][pname] = (row["protein_accession"], ev)
        if genome not in genome_info:
            genome_info[genome] = {k: row[k] for k in
                ("order", "domain", "phylum", "class", "family", "genus", "species")}

print(f"  {len(best_hit):,} genomes with aaRS hits")


# ── 2. Load tRNAscan: tRNAs per (genome, isotype) ─────────────────────────────
print("Loading tRNAscan data …")

# trna_seqs[genome][isotype] = [sequence, ...]
trna_seqs = defaultdict(lambda: defaultdict(list))

with open(TRNASCAN_CSV) as f:
    for row in csv.DictReader(f):
        isotype = row["Anticodon_predicted_isotype"]
        seq     = row["primary_sequence"].upper()
        genome  = normalize_genome_id(row["GenomeID"])
        if isotype in {"Undet", "Sup", "Anticodon_predicted_isotype"} or not seq:
            continue
        trna_seqs[genome][isotype].append(seq)

print(f"  {len(trna_seqs):,} genomes with tRNA data")


# ── 3. Identify qualifying groups at each taxonomic level ─────────────────────
print("Identifying qualifying taxonomic groups …")

trna_genome_set = set(trna_seqs.keys())

# level_genomes[level][group] = set of genomes
# level_aa[level][group]      = set of AAs covered
level_genomes = {lvl: defaultdict(set) for lvl in TAX_LEVELS}
level_aa      = {lvl: defaultdict(set) for lvl in TAX_LEVELS}

for genome, hits in best_hit.items():
    if genome not in trna_genome_set:
        continue
    info = genome_info[genome]
    for pname in hits:
        aa = NAME_TO_AA.get(pname)
        if aa:
            for lvl in TAX_LEVELS:
                group = info.get(lvl, "")
                if group:
                    level_aa[lvl][group].add(aa)
                    level_genomes[lvl][group].add(genome)

qualifying = {}   # level -> set of group names
for lvl in TAX_LEVELS:
    qualifying[lvl] = {
        g for g in level_genomes[lvl]
        if level_aa[lvl][g] >= ALL_AA and len(level_genomes[lvl][g]) >= MIN_GENOMES
    }
    print(f"  {lvl:8s}: {len(qualifying[lvl])} groups with all 20 aaRS and >= {MIN_GENOMES} genomes")


# ── 4. Build index of protein sequences needed ────────────────────────────────
print("Indexing needed protein sequences …")

all_needed_genomes = set()
for lvl in TAX_LEVELS:
    for group in qualifying[lvl]:
        all_needed_genomes |= level_genomes[lvl][group]

need_proteins = {}  # faa_key -> (genome, protein_name)
for genome in all_needed_genomes:
    faa_genome = genome + "_protein"
    for pname, (pacc, _) in best_hit[genome].items():
        if NAME_TO_AA.get(pname) in ALL_AA:
            key = f"{faa_genome}||{pacc}"
            need_proteins[key] = (genome, pname)

print(f"  {len(need_proteins):,} protein sequences to extract")


# ── 5. Stream combined FAA and extract needed sequences ───────────────────────
print(f"Streaming {COMBINED_FAA} …")

extracted = {}  # (genome, pname) -> sequence
current_key = None
current_seq = []

with open(COMBINED_FAA) as f:
    for line in f:
        if line.startswith(">"):
            if current_key and current_key in need_proteins:
                k = need_proteins[current_key]
                extracted[k] = "".join(current_seq)
            header = line[1:].split()[0]
            current_key = header
            current_seq = []
        else:
            current_seq.append(line.strip())
    # final record
    if current_key and current_key in need_proteins:
        k = need_proteins[current_key]
        extracted[k] = "".join(current_seq)

print(f"  Extracted {len(extracted):,} sequences")


# ── 6. Write output FASTAs ─────────────────────────────────────────────────────
print("Writing output FASTAs …")

os.makedirs(OUT_DIR, exist_ok=True)

n_written = 0
for lvl in TAX_LEVELS:
    lvl_dir = os.path.join(OUT_DIR, lvl)
    os.makedirs(lvl_dir, exist_ok=True)

    for group in sorted(qualifying[lvl]):
        group_dir = os.path.join(lvl_dir, safe_dirname(group))
        os.makedirs(group_dir, exist_ok=True)

        genomes = level_genomes[lvl][group]

        for aa, pnames in AA_GROUPS.items():
            aaRS_records = []
            trna_records = []

            for genome in sorted(genomes):
                # aaRS: pick best protein_name for this AA in this genome
                best_pname, best_seq = None, None
                best_ev = float("inf")
                for pname in pnames:
                    if pname in best_hit[genome]:
                        pacc, ev = best_hit[genome][pname]
                        if ev < best_ev:
                            seq = extracted.get((genome, pname))
                            if seq:
                                best_ev    = ev
                                best_pname = pname
                                best_seq   = seq
                if best_seq:
                    aaRS_records.append(f">{genome}|{best_pname}\n{best_seq}\n")

                # tRNAs: all cognate copies in this genome
                isotypes = AARS_TO_ISOTYPE.get(best_pname or pnames[0], [aa])
                seen_seqs = set()
                for iso in isotypes:
                    for i, seq in enumerate(trna_seqs.get(genome, {}).get(iso, [])):
                        if seq not in seen_seqs:
                            seen_seqs.add(seq)
                            trna_records.append(f">{genome}|{iso}|copy{i+1}\n{seq}\n")

            if aaRS_records:
                with open(os.path.join(group_dir, f"{aa}_aaRS.faa"), "w") as f:
                    f.writelines(aaRS_records)
            if trna_records:
                with open(os.path.join(group_dir, f"{aa}_tRNA.fna"), "w") as f:
                    f.writelines(trna_records)

        n_written += 1
        if n_written % 50 == 0:
            print(f"  … {n_written} groups written")

print(f"\nDone. {n_written} total groups written to {OUT_DIR}/")
