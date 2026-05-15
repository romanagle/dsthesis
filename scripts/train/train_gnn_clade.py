#!/usr/bin/env python3
"""
train_gnn_clade.py
------------------
Train a fresh GNN isotype classifier on sequences from a single taxonomic
clade, optionally augmented with parent-class sequences at lower weight.
Uses pre-computed RiNALMo embeddings — no RiNALMo rerun needed.

Why this is useful
------------------
The global GNN learns consensus rules (e.g. C73 = His for bacteria broadly).
A clade-trained GNN learns what discriminates isotypes *within* the clade,
so ISM on that model reveals clade-specific identity elements (e.g. A73 = His
in Pelagibacterales) that ISM on the global model cannot recover.

Parent-class augmentation
-------------------------
With --parent-col / --parent-weight, sequences from the parent taxonomic
level (e.g. class=Alphaproteobacteria) are added to training at a lower loss
weight.  This regularises the small clade dataset without letting the global
consensus swamp the clade-specific signal.

Recommended parent_weight ≈ n_clade / n_parent_sample, so both groups
contribute roughly equally to the total loss.  The default auto-computes this.

Usage
-----
    # Clade-only (44 sequences — may overfit)
    python scripts/train_gnn_clade.py \\
        --clade Pelagibacterales --clade-col order \\
        --emb-dir gnn_final --outdir gnn_pelagibacterales

    # With Alphaproteobacteria background (recommended)
    python scripts/train_gnn_clade.py \\
        --clade Pelagibacterales --clade-col order \\
        --parent-col class \\
        --emb-dir gnn_final --outdir gnn_pelagibacterales

    # Then run ISM:
    python scripts/compute_ism_clade.py \\
        --emb-dir gnn_final \\
        --model-path gnn_pelagibacterales/gnn_model.pt \\
        --clades Pelagibacterales --clade-col order \\
        --isotypes His --use-all
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir",         default="embeddings/gnn")
parser.add_argument("--csv",             default="data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir",          required=True)
parser.add_argument("--clade",           required=True,
                    help="Target clade value (e.g. Pelagibacterales)")
parser.add_argument("--clade-col",       default="order")
# Parent-class augmentation
parser.add_argument("--parent-col",      default=None,
                    help="CSV column for the parent taxonomic level "
                         "(e.g. 'class' when --clade-col is 'order'). "
                         "Sequences from the parent class are added to training "
                         "at --parent-weight.")
parser.add_argument("--parent-weight",   type=float, default=None,
                    help="Loss weight for parent-class sequences (default: "
                         "auto = n_clade / n_parent_sample, so both groups "
                         "contribute equally to total loss).")
parser.add_argument("--parent-max-seqs", type=int, default=5000,
                    help="Max parent-class sequences to include (random sample). "
                         "Keeps memory and runtime manageable. Default: 5000.")
# Architecture / training
parser.add_argument("--n-layers",        type=int,   default=3)
parser.add_argument("--hidden-dim",      type=int,   default=256)
parser.add_argument("--dropout",         type=float, default=0.3)
parser.add_argument("--lr",              type=float, default=1e-3)
parser.add_argument("--epochs",          type=int,   default=300)
parser.add_argument("--patience",        type=int,   default=25)
parser.add_argument("--batch-size",      type=int,   default=32)
parser.add_argument("--val-size",        type=float, default=0.2)
parser.add_argument("--seed",            type=int,   default=42)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ── Graph utilities ───────────────────────────────────────────────────────────
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == '>': stack.append(i)
        elif c == '<':
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


VAR_LOOP_THRESHOLD = 5


def detect_variable_loop(ss: str) -> tuple[int, int]:
    pairs = parse_pairs(ss)
    five_prime = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in five_prime:
        if not cur:
            cur = [p]
        elif p == cur[-1] + 1 and pairs[p] == pairs[cur[-1]] - 1:
            cur.append(p)
        else:
            stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4:
        return 0, 0
    ac3_end  = sorted(pairs[p] for p in stems[2])[-1]
    t5_start = stems[-1][0]
    return ac3_end + 1, t5_start - ac3_end - 1


def build_adj_matrix(ss, seq_len, max_len):
    A = np.eye(max_len, dtype=np.float32)
    for i in range(min(seq_len - 1, max_len - 1)):
        A[i, i + 1] = A[i + 1, i] = 1.0
    for i, j in parse_pairs(ss).items():
        if i < max_len and j < max_len:
            A[i, j] = A[j, i] = 1.0
    vl_start, vl_len = detect_variable_loop(ss)
    if vl_len > VAR_LOOP_THRESHOLD:
        vl_end = min(vl_start + vl_len, max_len)
        A[vl_start:vl_end, :] = 0.0
        A[:, vl_start:vl_end] = 0.0
    deg = A.sum(axis=1)
    d   = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (A * d[:, None]) * d[None, :]


# ── GNN architecture ─────────────────────────────────────────────────────────
class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        return self.dropout(F.relu(torch.bmm(adj, self.linear(x))))


class GNNClassifier(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_layers, dropout, n_classes,
                 phylo_dim=0, var_loop_dim=1):
        super().__init__()
        dims = [emb_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            DenseGCNLayer(dims[i], dims[i + 1], dropout)
            for i in range(n_layers)
        ])
        self.phylo_dim    = phylo_dim
        self.var_loop_dim = var_loop_dim
        self.head = nn.Linear(hidden_dim + phylo_dim + var_loop_dim, n_classes)

    def forward(self, x, adj, phylo=None, var_loop=None):
        for layer in self.layers:
            x = layer(x, adj)
        pooled = x.mean(dim=1)
        parts = [pooled]
        if self.phylo_dim > 0 and phylo is not None:
            parts.append(phylo)
        if self.var_loop_dim > 0 and var_loop is not None:
            parts.append(var_loop)
        return self.head(torch.cat(parts, dim=1))


# ── Load NPZ metadata ─────────────────────────────────────────────────────────
npz_path  = os.path.join(args.emb_dir, "perpos_embeddings.npz")
mmap_path = os.path.join(args.emb_dir, "emb_mmap.npy")
for p in [npz_path, mmap_path]:
    if not os.path.exists(p):
        sys.exit(f"ERROR: {p} not found")

print("Loading metadata …", flush=True)
data      = np.load(npz_path, allow_pickle=True)
seq_lens  = data["seq_lens"]
labels    = data["labels"].astype(np.int64)
classes   = data["classes"].astype(str)
seqs      = data["sequences"]
ss_arr    = data["secondary_structures"]
max_len   = int(data["max_len"])
n_classes = len(classes)


phylo_ohe = (data["phylo_ohe"].astype(np.float32) if "phylo_ohe" in data else None)
phylo_dim = int(phylo_ohe.shape[1]) if phylo_ohe is not None else 0

var_loop_lengths = (data["var_loop_lengths"].astype(np.float32) / 25.0
                    if "var_loop_lengths" in data else None)
var_loop_dim = 1 if var_loop_lengths is not None else 0

print(f"  N={len(labels):,}  max_len={max_len}  n_classes={n_classes}", flush=True)

# ── Build masked_seq → clade / parent lookup ──────────────────────────────────
print("Building NNN-start map …", flush=True)
ss_to_nnn_start: dict[str, int] = {}
for s, ss in zip(seqs, ss_arr):
    s_str = str(s)
    if "NNN" in s_str:
        ss_key = str(ss)
        if ss_key not in ss_to_nnn_start:
            ss_to_nnn_start[ss_key] = s_str.index("NNN")

csv_cols = ["primary_sequence", "secondary_structure", args.clade_col]
if args.parent_col and args.parent_col not in csv_cols:
    csv_cols.append(args.parent_col)

print(f"Loading CSV ({', '.join(csv_cols[2:])}) …", flush=True)
csv_df = pd.read_csv(args.csv, usecols=csv_cols, low_memory=False).dropna(subset=csv_cols)

masked_to_clade:  dict[str, str] = {}
masked_to_parent: dict[str, str] = {}
for ss, grp in csv_df.groupby("secondary_structure"):
    nnn_start = ss_to_nnn_start.get(str(ss))
    if nnn_start is None:
        continue
    nnn_end = nnn_start + 3
    for _, row in grp.iterrows():
        seq    = row["primary_sequence"].upper()
        masked = seq[:nnn_start] + "NNN" + seq[nnn_end:]
        masked_to_clade[masked] = row[args.clade_col]
        if args.parent_col:
            masked_to_parent[masked] = row[args.parent_col]

all_clades  = np.array([masked_to_clade.get(str(s).upper(), "")  for s in seqs])
all_parents = (np.array([masked_to_parent.get(str(s).upper(), "") for s in seqs])
               if args.parent_col else None)

# ── Select clade sequences ─────────────────────────────────────────────────────
clade_idx = np.where(all_clades == args.clade)[0]
if len(clade_idx) == 0:
    sys.exit(f"ERROR: no sequences for {args.clade_col}={args.clade}")

clade_labels = labels[clade_idx]
present_isos, counts = np.unique(clade_labels, return_counts=True)
print(f"\nClade '{args.clade}': {len(clade_idx):,} sequences", flush=True)
for iso_idx, cnt in zip(present_isos, counts):
    print(f"  {classes[iso_idx]:10s}: {cnt}", flush=True)

# ── Select parent-class sequences (excluding the clade itself) ────────────────
parent_idx    = np.array([], dtype=int)
parent_weight = 0.0

if args.parent_col and all_parents is not None:
    # Find the parent value for this clade (majority vote)
    clade_parents = all_parents[clade_idx]
    clade_parents = clade_parents[clade_parents != ""]
    if len(clade_parents) == 0:
        print("WARNING: could not determine parent clade — skipping augmentation",
              flush=True)
    else:
        vals, cnts = np.unique(clade_parents, return_counts=True)
        parent_val = vals[np.argmax(cnts)]
        print(f"\nParent {args.parent_col}: {parent_val}", flush=True)

        # All parent-class sequences NOT in the target clade
        parent_mask = (all_parents == parent_val) & (all_clades != args.clade)
        all_parent_idx = np.where(parent_mask)[0]
        n_parent_avail = len(all_parent_idx)
        print(f"  Parent seqs (excl. clade): {n_parent_avail:,}", flush=True)

        # Cap at --parent-max-seqs
        n_parent = min(args.parent_max_seqs, n_parent_avail)
        parent_idx = rng.choice(all_parent_idx, n_parent, replace=False)
        print(f"  Using {n_parent:,} parent seqs (--parent-max-seqs={args.parent_max_seqs})",
              flush=True)

        # Auto-compute weight so both groups contribute ~equally to total loss
        if args.parent_weight is not None:
            parent_weight = args.parent_weight
        else:
            parent_weight = len(clade_idx) / n_parent
        print(f"  Parent loss weight: {parent_weight:.4f}  "
              f"(effective: {len(clade_idx):.0f} vs "
              f"{n_parent * parent_weight:.0f} total weight)",
              flush=True)

# ── Build combined dataset ────────────────────────────────────────────────────
# Clade sequences come first (indices 0..n_clade-1),
# parent sequences after (indices n_clade..n_clade+n_parent-1).
# Validation is drawn from clade sequences only.
combined_idx    = np.concatenate([clade_idx, parent_idx]).astype(int)
n_clade         = len(clade_idx)
n_combined      = len(combined_idx)
combined_labels = labels[combined_idx]

# Sample weights: 1.0 for clade, parent_weight for parent
sample_weights          = np.ones(n_combined, dtype=np.float32)
sample_weights[n_clade:] = parent_weight

print(f"\nTotal training pool: {n_combined:,} sequences", flush=True)

# ── Build adjacency matrices ──────────────────────────────────────────────────
print("Building adjacency matrices …", flush=True)
combined_ss   = ss_arr[combined_idx]
combined_lens = seq_lens[combined_idx]
adj_all = np.stack([
    build_adj_matrix(str(combined_ss[i]), int(combined_lens[i]), max_len)
    for i in range(n_combined)
])
print(f"  Done: {adj_all.shape}", flush=True)

# ── Load embeddings ───────────────────────────────────────────────────────────
print("Loading embeddings (mmap) …", flush=True)
emb_mmap = np.load(mmap_path, mmap_mode='r')
emb_dim  = emb_mmap.shape[2]
emb_all  = np.array(emb_mmap[combined_idx])
print(f"  emb_all: {emb_all.shape}", flush=True)

phylo_all = phylo_ohe[combined_idx]       if phylo_ohe        is not None else None
vl_all    = var_loop_lengths[combined_idx] if var_loop_lengths is not None else None
if vl_all is not None:
    vl_all = vl_all.reshape(-1, 1)

# ── Train / val split (clade sequences only for val) ─────────────────────────
clade_local_idx = np.arange(n_clade)   # indices into combined arrays

splittable = np.isin(
    combined_labels[:n_clade],
    present_isos[counts >= 2]
)
if splittable.sum() < 4:
    train_clade = clade_local_idx
    val_local   = np.array([], dtype=int)
    print("WARNING: too few clade sequences for val split — training without val",
          flush=True)
else:
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_size, random_state=args.seed
    )
    split_idx = clade_local_idx[splittable]
    tr_rel, va_rel = next(sss.split(split_idx, combined_labels[split_idx]))
    train_clade = np.concatenate([split_idx[tr_rel],
                                   clade_local_idx[~splittable]])
    val_local   = split_idx[va_rel]

# Parent sequences always go into train only
parent_local_idx = np.arange(n_clade, n_combined)
train_local      = np.concatenate([train_clade, parent_local_idx])

print(f"Train: {len(train_local):,} "
      f"({len(train_clade):,} clade + {len(parent_local_idx):,} parent)  "
      f"Val: {len(val_local):,} (clade only)",
      flush=True)


# ── Training helpers ──────────────────────────────────────────────────────────
def make_batch(idx):
    x  = torch.from_numpy(emb_all[idx]).float()
    a  = torch.from_numpy(adj_all[idx]).float()
    y  = torch.from_numpy(combined_labels[idx]).long()
    w  = torch.from_numpy(sample_weights[idx]).float()
    p  = (torch.from_numpy(phylo_all[idx]).float() if phylo_all is not None else None)
    vl = (torch.from_numpy(vl_all[idx]).float()    if vl_all    is not None else None)
    return x, a, y, w, p, vl


def run_epoch(idx, train=True):
    model.train(train)
    order = (rng.permutation(len(idx)) if train else np.arange(len(idx)))
    total_loss, total_correct, total_n = 0.0, 0, 0

    for start in range(0, len(idx), args.batch_size):
        batch_local = order[start:start + args.batch_size]
        batch_idx   = idx[batch_local]
        x, a, y, w, p, vl = make_batch(batch_idx)
        x, a, y = x.to(DEVICE), a.to(DEVICE), y.to(DEVICE)
        w  = w.to(DEVICE)
        p  = p.to(DEVICE)  if p  is not None else None
        vl = vl.to(DEVICE) if vl is not None else None

        with torch.set_grad_enabled(train):
            logits = model(x, a, p, vl)
            # Per-sample weighted loss
            loss_per = F.cross_entropy(logits, y, reduction='none')
            loss     = (loss_per * w).sum() / w.sum()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss    += loss.item() * len(y)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n       += len(y)

    return total_loss / total_n, total_correct / total_n


# ── Model ─────────────────────────────────────────────────────────────────────
model = GNNClassifier(
    emb_dim, args.hidden_dim, args.n_layers, args.dropout,
    n_classes, phylo_dim, var_loop_dim,
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5, min_lr=1e-5
)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {n_params:,} parameters", flush=True)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_loss = float("inf")
best_state    = None
no_improve    = 0

print(f"Training for up to {args.epochs} epochs (patience={args.patience}) …\n",
      flush=True)

for epoch in range(1, args.epochs + 1):
    tr_loss, tr_acc = run_epoch(train_local, train=True)

    if len(val_local) > 0:
        with torch.no_grad():
            va_loss, va_acc = run_epoch(val_local, train=False)
        scheduler.step(va_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
                  f"val loss={va_loss:.4f} acc={va_acc:.3f}",
                  flush=True)

        if va_loss < best_val_loss - 1e-5:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(best val loss={best_val_loss:.4f})", flush=True)
                break
    else:
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train loss={tr_loss:.4f} acc={tr_acc:.3f}",
                  flush=True)
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ── Save ──────────────────────────────────────────────────────────────────────
if best_state:
    model.load_state_dict(best_state)

model_out = os.path.join(args.outdir, "gnn_model.pt")
torch.save(model.state_dict(), model_out)
print(f"\nSaved → {model_out}", flush=True)

if len(val_local) > 0:
    with torch.no_grad():
        _, va_acc = run_epoch(val_local, train=False)
    print(f"Final val accuracy (clade only): {va_acc:.3f}", flush=True)

print(f"\nRun ISM with:")
print(f"  python scripts/compute_ism_clade.py \\")
print(f"    --emb-dir {args.emb_dir} \\")
print(f"    --model-path {model_out} \\")
print(f"    --clades {args.clade} --clade-col {args.clade_col} \\")
print(f"    --isotypes His --use-all")
