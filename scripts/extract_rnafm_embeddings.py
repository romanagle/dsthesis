#!/usr/bin/env python3
"""
Extract per-sequence RNA-FM embeddings for tRNA sequences.

Mean-pools the final-layer token embeddings over sequence positions
(excluding <cls> and <eos>) to produce one 640-dim vector per tRNA.

Usage
-----
Unmasked sequences:
    python scripts/extract_rnafm_embeddings.py /path/to/RNA-FM_pretrained.pth

Anticodon-masked sequences:
    python scripts/extract_rnafm_embeddings.py /path/to/RNA-FM_pretrained.pth --masked

Outputs (suffix _masked for masked runs)
-------
results/embeddings/rnafm_embeddings[_masked].npy
results/embeddings/rnafm_labels[_masked].npy
results/embeddings/rnafm_ids[_masked].npy
"""

import sys
import os
import argparse
import numpy as np
import torch
import fm

CSV_PATH = "results/runs/isomodels_first100_with_test.csv"
OUT_DIR  = "results/embeddings"
LAYER    = 12


def read_fasta(path):
    seqs = []
    with open(path) as f:
        header, buf = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    seqs.append((header[1:], "".join(buf)))
                header, buf = line, []
            else:
                buf.append(line)
        if header is not None:
            seqs.append((header[1:], "".join(buf)))
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default=None,
                        help="Path to local RNA-FM_pretrained.pth (auto-downloads if omitted)")
    parser.add_argument("--masked", action="store_true",
                        help="Use anticodon-masked sequences")
    args = parser.parse_args()

    fasta_path = "data/tRNA_first100_masked.fasta" if args.masked else "data/tRNA_first100.fasta"
    suffix     = "_masked" if args.masked else ""

    print(f"Loading RNA-FM model …", flush=True)
    model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"  Model loaded  |  sequences: {'masked' if args.masked else 'unmasked'}")

    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["SequenceID"]                  = df["SequenceID"].str.strip()
    df["Anticodon_predicted_isotype"] = df["Anticodon_predicted_isotype"].str.strip()
    label_map = dict(zip(df["SequenceID"], df["Anticodon_predicted_isotype"]))

    data = read_fasta(fasta_path)
    print(f"  {len(data)} sequences read from {fasta_path}")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_embeddings, all_labels, all_ids = [], [], []

    for i, (seq_id, seq) in enumerate(data):
        _, _, tokens = batch_converter([(seq_id, seq)])

        with torch.no_grad():
            out = model(tokens, repr_layers=[LAYER], return_contacts=False)

        token_embs = out["representations"][LAYER][0]
        mean_emb   = token_embs[1:-1].mean(dim=0)

        all_embeddings.append(mean_emb.numpy())
        all_labels.append(label_map.get(seq_id, ""))
        all_ids.append(seq_id)

        if (i + 1) % 10 == 0 or (i + 1) == len(data):
            print(f"  [{i+1}/{len(data)}] processed", flush=True)

    emb_array = np.array(all_embeddings, dtype=np.float32)
    lbl_array = np.array(all_labels)
    id_array  = np.array(all_ids)

    np.save(os.path.join(OUT_DIR, f"rnafm_embeddings{suffix}.npy"), emb_array)
    np.save(os.path.join(OUT_DIR, f"rnafm_labels{suffix}.npy"),     lbl_array)
    np.save(os.path.join(OUT_DIR, f"rnafm_ids{suffix}.npy"),        id_array)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  rnafm_embeddings{suffix}.npy  shape={emb_array.shape}")


if __name__ == "__main__":
    main()
