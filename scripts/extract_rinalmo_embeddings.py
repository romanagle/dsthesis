#!/usr/bin/env python3
"""
Extract per-sequence RiNALMo embeddings for tRNA sequences.

Mean-pools the final transformer layer over sequence positions
(excluding <cls> and <eos> tokens) to produce one vector per tRNA.

Usage
-----
On GPU (recommended, uses giga model with flash attention):
    python scripts/extract_rinalmo_embeddings.py
    python scripts/extract_rinalmo_embeddings.py --masked

On CPU (uses micro model, no flash attention):
    python scripts/extract_rinalmo_embeddings.py --cpu
    python scripts/extract_rinalmo_embeddings.py --cpu --masked

Outputs (suffix _masked for masked runs)
-------
results/embeddings/rinalmo_embeddings[_masked].npy
results/embeddings/rinalmo_labels[_masked].npy
results/embeddings/rinalmo_ids[_masked].npy
"""

import sys
import os
import argparse
import numpy as np
import torch

FASTA_PATH = "data/tRNA_first100.fasta"
CSV_PATH   = "results/runs/isomodels_first100_with_test.csv"
OUT_DIR    = "results/embeddings"


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


def load_model(use_cpu: bool):
    from rinalmo.pretrained import get_pretrained_model
    from rinalmo.config import model_config

    if use_cpu:
        # Micro model + standard (non-flash) attention for CPU inference
        model_name = "micro-v1"
        lm_config  = "micro"

        config = model_config(lm_config)
        config.model.transformer.use_flash_attn = False   # CPU path

        from rinalmo.data.alphabet import Alphabet
        from rinalmo.model.model import RiNALMo

        weights_path = os.path.expanduser(f"~/.cache/rinalmo_pretrained/{model_name}.pt")
        assert os.path.exists(weights_path), (
            f"Micro model not found at {weights_path}.\n"
            "Download with:\n"
            "  curl -L https://zenodo.org/records/15043668/files/rinalmo_micro_pretrained.pt "
            f"-o {weights_path}"
        )
        model = RiNALMo(config)
        alphabet = Alphabet(**config['alphabet'])
        # strict=False: RotaryPositionEmbedding.inv_freq is computed, not in saved state dict
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
        device = torch.device("cpu")
    else:
        # Giga model on CUDA with flash attention
        model_name = "giga-v1"
        lm_config  = "giga"
        model, alphabet = get_pretrained_model(model_name=model_name, lm_config=lm_config)
        device = torch.device("cuda")

    model = model.to(device)
    model.eval()
    return model, alphabet, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true",
                        help="Run on CPU with micro model (no flash attention)")
    parser.add_argument("--masked", action="store_true",
                        help="Use anticodon-masked sequences")
    args = parser.parse_args()

    fasta_path = "data/tRNA_first100_masked.fasta" if args.masked else "data/tRNA_first100.fasta"
    suffix     = "_masked" if args.masked else ""

    use_cpu = args.cpu or not torch.cuda.is_available()
    if not args.cpu and not torch.cuda.is_available():
        print("No CUDA device found — falling back to CPU / micro model.", flush=True)

    print(f"Loading RiNALMo ({'micro/CPU' if use_cpu else 'giga/GPU'}) …", flush=True)
    model, alphabet, device = load_model(use_cpu)
    print(f"  Model loaded on {device}  |  sequences: {'masked' if args.masked else 'unmasked'}", flush=True)

    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["SequenceID"]                  = df["SequenceID"].str.strip()
    df["Anticodon_predicted_isotype"] = df["Anticodon_predicted_isotype"].str.strip()
    label_map = dict(zip(df["SequenceID"], df["Anticodon_predicted_isotype"]))

    data = read_fasta(fasta_path)
    print(f"  {len(data)} sequences read from {fasta_path}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_embeddings, all_labels, all_ids = [], [], []

    for i, (seq_id, seq) in enumerate(data):
        tokens = torch.tensor(
            alphabet.batch_tokenize([seq]),
            dtype=torch.int64,
            device=device,
        )

        with torch.no_grad():
            if not use_cpu:
                # giga model expects fp16 for flash attention
                with torch.cuda.amp.autocast():
                    out = model(tokens)
            else:
                out = model(tokens)

        # representation shape: (1, seq_len+2, embed_dim)
        rep = out["representation"][0]      # (seq_len+2, embed_dim)
        mean_emb = rep[1:-1].mean(dim=0)    # mean over sequence positions

        all_embeddings.append(mean_emb.float().cpu().numpy())
        all_labels.append(label_map.get(seq_id, ""))
        all_ids.append(seq_id)

        if (i + 1) % 10 == 0 or (i + 1) == len(data):
            print(f"  [{i+1}/{len(data)}] processed", flush=True)

    emb_array = np.array(all_embeddings, dtype=np.float32)
    lbl_array = np.array(all_labels)
    id_array  = np.array(all_ids)

    np.save(os.path.join(OUT_DIR, f"rinalmo_embeddings{suffix}.npy"), emb_array)
    np.save(os.path.join(OUT_DIR, f"rinalmo_labels{suffix}.npy"),     lbl_array)
    np.save(os.path.join(OUT_DIR, f"rinalmo_ids{suffix}.npy"),        id_array)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  rinalmo_embeddings{suffix}.npy  shape={emb_array.shape}  dtype={emb_array.dtype}")
    print(f"  rinalmo_labels{suffix}.npy      shape={lbl_array.shape}")
    print(f"  rinalmo_ids{suffix}.npy         shape={id_array.shape}")


if __name__ == "__main__":
    main()
