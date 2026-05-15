#!/usr/bin/env python3
"""
predict_isotype.py
------------------
Run the full tRNA isotype classification pipeline on a single sequence.

Steps
-----
  1. Mask anticodon (position ~33) and variable loop (type-II tRNAs)
  2. Extract per-position RiNALMo embeddings
  3. Build normalised adjacency matrix from secondary structure
  4. Encode phylogenetic context (OHE, domain → family; optional)
  5. Run GNN classifier → softmax probabilities over 19 isotypes

Usage
-----
  # Minimal – sequence + secondary structure only (no taxonomy):
  python scripts/predict_isotype.py \\
      --sequence  GCGGATTTAGCTCAGTGGTAGAGCGCTTGCTTGCAAGCAAGAGGCCCGGTTCAATCCGGGTAATCTGCA \\
      --secondary-structure ">>>>>..<<<<.....>>>>.......<<<<>>>>>.....<<<<<<<....." \\
      --codon GCC

  # With taxonomy (improves borderline predictions):
  python scripts/predict_isotype.py \\
      --sequence  GCGGATTTAGCTCAGT... \\
      --secondary-structure ">>>>>..." \\
      --codon GCC \\
      --domain Bacteria --phylum Proteobacteria \\
      --class Alphaproteobacteria --order Rhizobiales --family Rhizobiaceae

  # Using a different trained model:
  python scripts/predict_isotype.py \\
      --sequence  ... \\
      --secondary-structure "..." \\
      --model-dir gnn_pelagibacterales

Phylo input
-----------
  The model was trained with OHE phylo features spanning domain → family
  (~8 k dimensions).  Providing taxonomy is optional; if omitted (or if the
  taxon is absent from the training vocabulary) those dimensions are zeroed and
  the prediction relies solely on sequence/structure context.

  The encoder is loaded from {model_dir}/phylo_encoder.pkl (saved by
  train_gnn.py ≥ this version).  If the pickle is missing, it is rebuilt from
  --csv and cached for future calls.  If --csv is also absent, phylo features
  are set to zero with a warning.

Outputs
-------
  Sorted table of isotype probabilities printed to stdout.
  Use --json to emit machine-readable JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Predict tRNA isotype from sequence.")
parser.add_argument("--sequence",            required=True,
                    help="tRNA sequence (raw nucleotides, RNA or DNA alphabet)")
parser.add_argument("--secondary-structure", required=True, dest="secondary_structure",
                    help="Dot-bracket secondary structure (same length as sequence). "
                         "Use > for 5' base-pair partner, < for 3' partner.")
parser.add_argument("--codon",               default=None,
                    help="Anticodon sequence (e.g. GCC).  Used to mask the anticodon "
                         "loop before embedding.  If omitted, positions 33-35 are masked.")
parser.add_argument("--domain",  default=None, help="Taxonomic domain  (e.g. Bacteria)")
parser.add_argument("--phylum",  default=None, help="Taxonomic phylum")
parser.add_argument("--class",   default=None, dest="taxclass",
                    help="Taxonomic class")
parser.add_argument("--order",   default=None, help="Taxonomic order")
parser.add_argument("--family",  default=None, help="Taxonomic family")
parser.add_argument("--model-dir", default="gnn_final",
                    help="Directory containing gnn_model.pt, gnn_results.json, "
                         "and (optionally) phylo_encoder.pkl")
parser.add_argument("--csv",     default="data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv",
                    help="Training CSV used to rebuild phylo encoder if pickle is absent")
parser.add_argument("--json",    action="store_true",
                    help="Emit results as JSON to stdout instead of a table")
parser.add_argument("--no-phylo", action="store_true",
                    help="Force phylo features to zero even if taxonomy is provided")
args = parser.parse_args()

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANTICODON_POS = 33          # 0-indexed position of anticodon first base
PHYLO_COLS    = ["domain", "phylum", "class", "order", "family"]
VAR_LOOP_THRESHOLD = 5      # loops longer than this → type-II masking

# ── Structural utilities (mirror of train_gnn.py) ─────────────────────────────

def parse_pairs(ss: str) -> dict[int, int]:
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == ">":
            stack.append(i)
        elif c == "<":
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j
    return pairs


def detect_variable_loop(ss: str) -> tuple[int, int]:
    """Return (start_idx, length) of the variable loop region."""
    pairs = parse_pairs(ss)
    five_prime = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in five_prime:
        if not cur:
            cur = [p]
        elif p == cur[-1] + 1 and pairs[p] == pairs[cur[-1]] - 1:
            cur.append(p)
        else:
            stems.append(cur)
            cur = [p]
    if cur:
        stems.append(cur)
    if len(stems) < 4:
        return 0, 0
    ac3_end  = sorted(pairs[p] for p in stems[2])[-1]
    t5_start = stems[-1][0]
    return ac3_end + 1, t5_start - ac3_end - 1


def build_adj_matrix(ss: str, seq_len: int, max_len: int) -> np.ndarray:
    """Normalised D^-1/2 A D^-1/2 adjacency with backbone + base-pair edges."""
    A = np.eye(max_len, dtype=np.float32)
    for i in range(min(seq_len - 1, max_len - 1)):
        A[i, i + 1] = A[i + 1, i] = 1.0
    for i, j in parse_pairs(ss).items():
        if i < max_len and j < max_len:
            A[i, j] = A[j, i] = 1.0
    # Zero-out variable-loop region for type-II tRNAs
    vl_start, vl_len = detect_variable_loop(ss)
    if vl_len > VAR_LOOP_THRESHOLD:
        vl_end = min(vl_start + vl_len, max_len)
        A[vl_start:vl_end, :] = 0.0
        A[:, vl_start:vl_end] = 0.0
    deg = A.sum(axis=1)
    d   = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (A * d[:, None]) * d[None, :]


def mask_anticodon(seq: str, codon: str | None) -> str:
    """Replace the anticodon triplet with NNN.

    If *codon* is given, find the occurrence closest to position 33.
    Otherwise mask positions 33-35 directly.
    """
    seq_up = seq.upper().replace("U", "T")
    if codon is not None:
        codon_norm = codon.upper().replace("U", "T")
        matches, start = [], 0
        while True:
            pos = seq_up.find(codon_norm, start)
            if pos == -1:
                break
            matches.append(pos)
            start = pos + 1
        if matches:
            best = min(matches, key=lambda p: abs(p - ANTICODON_POS))
            return seq_up[:best] + "NNN" + seq_up[best + 3:]
    # Fallback: mask canonical anticodon positions 33-35 (0-indexed)
    s = ANTICODON_POS
    return seq_up[:s] + "NNN" + seq_up[s + 3:]


# ── GNN architecture (must exactly match train_gnn.py) ────────────────────────

class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(torch.bmm(adj, self.linear(x))))


class GNNClassifier(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, n_layers: int,
                 dropout: float, n_classes: int,
                 phylo_dim: int = 0, var_loop_dim: int = 1):
        super().__init__()
        dims = [emb_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            DenseGCNLayer(dims[i], dims[i + 1], dropout)
            for i in range(n_layers)
        ])
        self.phylo_dim    = phylo_dim
        self.var_loop_dim = var_loop_dim
        self.head = nn.Linear(hidden_dim + phylo_dim + var_loop_dim, n_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                phylo: torch.Tensor | None = None,
                var_loop: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        pooled = x.mean(dim=1)
        parts  = [pooled]
        if self.phylo_dim > 0 and phylo is not None:
            parts.append(phylo)
        if self.var_loop_dim > 0 and var_loop is not None:
            parts.append(var_loop)
        return self.head(torch.cat(parts, dim=1))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_dir: str) -> tuple[GNNClassifier, list[str], int]:
    """Load checkpoint and return (model, classes, max_len).

    Hyperparameters are inferred from the state-dict shapes so the script
    works with any trained checkpoint, not just gnn_final.
    """
    results_path = os.path.join(model_dir, "gnn_results.json")
    model_path   = os.path.join(model_dir, "gnn_model.pt")

    if not os.path.exists(model_path):
        sys.exit(f"ERROR: model checkpoint not found: {model_path}")

    with open(results_path) as f:
        cfg = json.load(f)

    classes   = cfg["classes"]
    n_classes = len(classes)

    state = torch.load(model_path, weights_only=True, map_location="cpu")

    # Infer architecture from weight shapes
    emb_dim    = state["layers.0.linear.weight"].shape[1]   # 1280
    hidden_dim = state["layers.0.linear.weight"].shape[0]   # 256
    n_layers   = sum(1 for k in state if k.startswith("layers."))
    head_in    = state["head.weight"].shape[1]               # 256 + phylo_dim + 1
    phylo_dim  = head_in - hidden_dim - 1                    # var_loop_dim always 1
    var_loop_dim = 1

    # max_len: prefer gnn_results.json, fall back to NPZ scalar
    max_len = cfg.get("max_len", None)
    if max_len is None:
        npz_path = os.path.join(model_dir, "perpos_embeddings.npz")
        if os.path.exists(npz_path):
            npz     = np.load(npz_path, allow_pickle=True)
            max_len = int(npz["max_len"])
        else:
            print("WARNING: max_len not found in gnn_results.json or NPZ; "
                  "will equal sequence length.", file=sys.stderr)
            max_len = None   # resolved later from sequence

    model = GNNClassifier(
        emb_dim, hidden_dim, n_layers, dropout=0.0,
        n_classes=n_classes, phylo_dim=phylo_dim, var_loop_dim=var_loop_dim,
    )
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    print(f"Loaded model: {n_layers}-layer GCN  emb={emb_dim}  hidden={hidden_dim}  "
          f"phylo_dim={phylo_dim}  n_classes={n_classes}  max_len={max_len}",
          file=sys.stderr)
    return model, classes, max_len


# ── Phylo encoder ─────────────────────────────────────────────────────────────

def load_phylo_encoder(model_dir: str, csv_path: str | None, expected_dim: int):
    """Return a fitted sklearn OneHotEncoder, or None if unavailable.

    Load order:
      1. {model_dir}/phylo_encoder.pkl  (fast, exact match to training run)
      2. Re-fit from csv_path and cache to pickle  (slow first run)
      3. None (phylo features will be zeroed)
    """
    from sklearn.preprocessing import OneHotEncoder

    pkl_path = os.path.join(model_dir, "phylo_encoder.pkl")

    # 1. Cached encoder
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            enc = pickle.load(f)
        # Quick dim check
        test_row = [["Unknown"] * len(PHYLO_COLS)]
        dim = enc.transform(test_row).shape[1]
        if dim != expected_dim:
            print(f"WARNING: phylo_encoder.pkl produces dim={dim} but model expects "
                  f"{expected_dim}.  Phylo features will be truncated/padded.",
                  file=sys.stderr)
        return enc

    # 2. Re-fit from CSV
    if csv_path and os.path.exists(csv_path):
        print(f"Fitting phylo encoder from {csv_path} (cached to {pkl_path}) …",
              file=sys.stderr)
        import pandas as pd
        df = pd.read_csv(csv_path, usecols=PHYLO_COLS, low_memory=False)
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                            dtype=np.float32)
        enc.fit(df[PHYLO_COLS].fillna("Unknown"))
        with open(pkl_path, "wb") as f:
            pickle.dump(enc, f)
        dim = enc.transform([["Unknown"] * len(PHYLO_COLS)]).shape[1]
        if dim != expected_dim:
            print(f"WARNING: refitted encoder dim={dim} ≠ model expected {expected_dim}.  "
                  f"Phylo features will be truncated/padded.", file=sys.stderr)
        return enc

    print("WARNING: No phylo encoder found and --csv not provided.  "
          "Phylo features set to zero — prediction accuracy may be reduced "
          "for borderline isotypes.", file=sys.stderr)
    return None


def encode_phylo(enc, taxonomy: dict[str, str], expected_dim: int) -> np.ndarray:
    """Transform one taxonomy row → float32 vector of exactly *expected_dim*."""
    row = [[
        taxonomy.get("domain")  or "Unknown",
        taxonomy.get("phylum")  or "Unknown",
        taxonomy.get("class")   or "Unknown",
        taxonomy.get("order")   or "Unknown",
        taxonomy.get("family")  or "Unknown",
    ]]
    vec = enc.transform(row).flatten().astype(np.float32)
    # Adjust dimension if encoder/model were fit on slightly different vocabs
    if vec.shape[0] < expected_dim:
        vec = np.pad(vec, (0, expected_dim - vec.shape[0]))
    elif vec.shape[0] > expected_dim:
        vec = vec[:expected_dim]
    return vec


# ── RiNALMo embedding ─────────────────────────────────────────────────────────

def embed_sequence(seq: str, max_len: int) -> np.ndarray:
    """Return per-position RiNALMo embeddings, zero-padded to (max_len, 1280)."""
    print("Loading RiNALMo giga-v1 …", file=sys.stderr)
    from rinalmo.pretrained import get_pretrained_model
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

    rinalmo, alphabet = get_pretrained_model(model_name="giga-v1", lm_config="giga")
    rinalmo.eval().to(DEVICE)
    for m in rinalmo.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    emb_dim = rinalmo.embedding.weight.shape[1]
    tokens  = torch.tensor(
        alphabet.batch_tokenize([seq]), dtype=torch.int64, device=DEVICE
    )
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = rinalmo(tokens)
    # out["representation"]: (1, seq_len+2, emb_dim) — includes BOS/EOS tokens
    rep = out["representation"].float().cpu().numpy()[0, 1:len(seq) + 1, :]

    emb = np.zeros((max_len, emb_dim), dtype=np.float32)
    n   = min(len(seq), max_len)
    emb[:n, :] = rep[:n, :]
    return emb


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    seq = args.sequence.strip().upper().replace("U", "T")
    ss  = args.secondary_structure.strip()

    if len(seq) != len(ss):
        sys.exit(f"ERROR: sequence length ({len(seq)}) ≠ secondary structure "
                 f"length ({len(ss)})")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model, classes, max_len = load_model(args.model_dir)
    if max_len is None:
        max_len = len(seq)

    # Infer phylo_dim from model head
    head_in   = model.head.weight.shape[1]
    phylo_dim = model.phylo_dim

    # ── 2. Preprocess sequence ────────────────────────────────────────────────
    # Anticodon masking
    seq_masked = mask_anticodon(seq, args.codon)

    # Variable loop detection + masking (type-II tRNAs)
    vl_start, vl_len = detect_variable_loop(ss)
    var_loop_length  = min(vl_len, 127)
    if vl_len > VAR_LOOP_THRESHOLD:
        seq_masked = seq_masked[:vl_start] + "N" * vl_len + seq_masked[vl_start + vl_len:]
        print(f"Type-II tRNA detected: variable loop {vl_len} nt at positions "
              f"{vl_start}–{vl_start + vl_len - 1} masked.", file=sys.stderr)

    print(f"Sequence (masked): {seq_masked}", file=sys.stderr)

    # ── 3. Build adjacency matrix ─────────────────────────────────────────────
    adj = build_adj_matrix(ss, len(seq), max_len)  # (max_len, max_len)

    # ── 4. Embed ──────────────────────────────────────────────────────────────
    if not args.no_phylo and not args.json:
        print("Embedding sequence …", file=sys.stderr)
    emb = embed_sequence(seq_masked, max_len)       # (max_len, 1280)

    # ── 5. Phylo OHE ─────────────────────────────────────────────────────────
    phylo_vec = np.zeros(phylo_dim, dtype=np.float32)

    taxonomy = {
        "domain": args.domain,
        "phylum": args.phylum,
        "class":  args.taxclass,
        "order":  args.order,
        "family": args.family,
    }
    has_taxonomy = any(v is not None for v in taxonomy.values())

    if phylo_dim > 0 and has_taxonomy and not args.no_phylo:
        enc = load_phylo_encoder(
            args.model_dir,
            args.csv if os.path.exists(args.csv) else None,
            phylo_dim,
        )
        if enc is not None:
            phylo_vec = encode_phylo(enc, taxonomy, phylo_dim)
            n_active  = int((phylo_vec > 0).sum())
            print(f"Phylo OHE: {n_active} active dimensions  "
                  f"(domain={args.domain}, phylum={args.phylum}, "
                  f"class={args.taxclass}, order={args.order}, family={args.family})",
                  file=sys.stderr)
    elif phylo_dim > 0 and not has_taxonomy and not args.no_phylo:
        print("No taxonomy provided — phylo features zeroed.", file=sys.stderr)

    # ── 6. Forward pass ───────────────────────────────────────────────────────
    x_t   = torch.from_numpy(emb).unsqueeze(0).to(DEVICE)        # (1, max_len, 1280)
    adj_t = torch.from_numpy(adj).unsqueeze(0).to(DEVICE)        # (1, max_len, max_len)
    p_t   = (torch.from_numpy(phylo_vec).unsqueeze(0).to(DEVICE)
             if phylo_dim > 0 else None)                          # (1, phylo_dim) | None
    vl_t  = torch.tensor([[var_loop_length / 25.0]],
                          dtype=torch.float32, device=DEVICE)     # (1, 1)

    with torch.no_grad():
        logits = model(x_t, adj_t, p_t, vl_t)                    # (1, n_classes)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]       # (n_classes,)

    # ── 7. Output ─────────────────────────────────────────────────────────────
    order = np.argsort(probs)[::-1]
    top   = classes[order[0]]

    if args.json:
        result = {
            "top_prediction": top,
            "confidence":     float(probs[order[0]]),
            "probabilities":  {cls: float(probs[i])
                               for i, cls in enumerate(classes)},
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'─'*40}")
        print(f"  Top prediction : {top}  ({probs[order[0]]:.1%})")
        print(f"{'─'*40}")
        print(f"  {'Isotype':<10}  {'Probability':>11}  {'Bar'}")
        print(f"  {'─'*7:<10}  {'─'*11:>11}  {'─'*20}")
        for i in order:
            bar = "█" * int(probs[i] * 40)
            print(f"  {classes[i]:<10}  {probs[i]:>10.4f}   {bar}")
        print(f"{'─'*40}\n")


if __name__ == "__main__":
    main()
