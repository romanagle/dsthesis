#!/usr/bin/env python3
"""
eval_film_per_domain.py
-----------------------
Add per_domain_accuracy to an existing gnn_results.json produced by
train_gnn.py --film.  Loads the saved checkpoint, runs inference on the
same stratified test split, and patches the JSON in-place.

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_film_per_domain.py \\
        --results-dir gnn_film \\
        --npz         embeddings/gnn/perpos_embeddings.npz \\
        --phylo-encoder gnn_final/phylo_encoder.pkl
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir",   default="gnn_film")
parser.add_argument("--npz",           default="embeddings/gnn/perpos_embeddings.npz")
parser.add_argument("--phylo-encoder", default=None)
parser.add_argument("--batch-size",    type=int, default=256)
parser.add_argument("--test-size",     type=float, default=0.2)
parser.add_argument("--seed",          type=int, default=42)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ── Load existing results to get architecture params ──────────────────────────
results_path = os.path.join(args.results_dir, "gnn_results.json")
with open(results_path) as f:
    results = json.load(f)

if "per_domain_accuracy" in results:
    print("per_domain_accuracy already present — nothing to do.")
    for dom, acc in results["per_domain_accuracy"].items():
        print(f"  {dom}: {acc:.4f}")
    raise SystemExit(0)

film_cfg  = results.get("film", {})
N_ORDERS  = film_cfg.get("n_orders", 0)
FILM_DIM  = film_cfg.get("emb_dim", 32)
hp        = results.get("hyperparams", {})
N_LAYERS  = hp.get("n_layers", 3)
HIDDEN    = hp.get("hidden_dim", 256)
DROPOUT   = hp.get("dropout", 0.3)
N_CLASSES = results["n_classes"]

# ── Load NPZ metadata ─────────────────────────────────────────────────────────
print(f"Loading {args.npz} …", flush=True)
data      = np.load(args.npz, allow_pickle=True)
seq_lens  = data["seq_lens"]
labels    = data["labels"].astype(np.int64)
ss_arr    = data["secondary_structures"]
max_len   = int(data["max_len"])
domains   = (data["domains"].astype(str) if "domains" in data
             else np.full(len(labels), "Unknown", dtype=object))
unique_domains = sorted(set(domains))

phylo_ohe = data["phylo_ohe"].astype(np.float32) if "phylo_ohe" in data else None
PHYLO_DIM = int(phylo_ohe.shape[1]) if phylo_ohe is not None else 0

var_loop_lengths = (data["var_loop_lengths"].astype(np.float32) / 25.0
                    if "var_loop_lengths" in data else None)
VAR_LOOP_DIM = 1 if var_loop_lengths is not None else 0

# ── Derive order labels from phylo_ohe ────────────────────────────────────────
order_labels = None
if N_ORDERS > 0 and phylo_ohe is not None:
    import pickle as _pkl, glob as _glob
    _enc_path = args.phylo_encoder
    if _enc_path is None:
        _candidates = (
            [os.path.join(args.results_dir, "phylo_encoder.pkl")] +
            _glob.glob(os.path.join(os.path.dirname(args.npz),
                                    "..", "*", "phylo_encoder.pkl"))
        )
        _enc_path = next((p for p in _candidates if os.path.exists(p)), None)
    if _enc_path:
        with open(_enc_path, "rb") as _f:
            _enc = _pkl.load(_f)
        _n_cats      = [len(c) for c in _enc.categories_]
        _order_start = sum(_n_cats[:3])
        _order_end   = _order_start + _n_cats[3]
        order_labels = phylo_ohe[:, _order_start:_order_end].argmax(axis=1).astype(np.int64)
        print(f"  Order labels loaded ({_n_cats[3]} orders)", flush=True)
    else:
        print("  Warning: phylo_encoder.pkl not found — order_idx will be None", flush=True)

# ── Load mmap embeddings ──────────────────────────────────────────────────────
mmap_path = os.path.join(os.path.dirname(args.npz), "emb_mmap.npy")
print(f"Memory-mapping {mmap_path} …", flush=True)
emb = np.load(mmap_path, mmap_mode="r")
EMB_DIM = emb.shape[2]

# ── Reconstruct model ─────────────────────────────────────────────────────────
class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        return self.dropout(F.relu(torch.bmm(adj, self.linear(x))))


class FiLMBottleneck(nn.Module):
    def __init__(self, hidden_dim, n_orders, order_emb_dim=32):
        super().__init__()
        self.order_emb = nn.Embedding(n_orders, order_emb_dim)
        self.cond_net  = nn.Linear(order_emb_dim, 2 * hidden_dim)

    def forward(self, h, order_idx):
        cond        = self.order_emb(order_idx)
        gamma, beta = self.cond_net(cond).chunk(2, dim=-1)
        if h.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta  = beta.unsqueeze(1)
        return (1 + gamma) * h + beta


class GNNClassifier(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_layers, dropout, n_classes,
                 phylo_dim=0, var_loop_dim=1, film_n_orders=0, film_emb_dim=32):
        super().__init__()
        dims = [emb_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            DenseGCNLayer(dims[i], dims[i + 1], dropout)
            for i in range(n_layers)
        ])
        self.phylo_dim    = phylo_dim
        self.var_loop_dim = var_loop_dim
        self.film = (FiLMBottleneck(hidden_dim, film_n_orders, film_emb_dim)
                     if film_n_orders > 0 else None)
        self.head = nn.Linear(hidden_dim + phylo_dim + var_loop_dim, n_classes)

    def forward(self, x, adj, phylo=None, var_loop=None, order_idx=None):
        for layer in self.layers:
            x = layer(x, adj)
        if self.film is not None and order_idx is not None:
            x = self.film(x, order_idx)
        pooled = x.mean(dim=1)
        parts  = [pooled]
        if self.phylo_dim > 0 and phylo is not None:
            parts.append(phylo)
        if self.var_loop_dim > 0 and var_loop is not None:
            parts.append(var_loop)
        return self.head(torch.cat(parts, dim=1))


model = GNNClassifier(EMB_DIM, HIDDEN, N_LAYERS, DROPOUT, N_CLASSES,
                      PHYLO_DIM, VAR_LOOP_DIM,
                      film_n_orders=N_ORDERS, film_emb_dim=FILM_DIM).to(DEVICE)
ckpt_path = os.path.join(args.results_dir, "gnn_model.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.0
print(f"Loaded checkpoint: {ckpt_path}", flush=True)

# ── Reproduce the same test split ────────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size,
                              random_state=args.seed)
_, test_idx = next(sss.split(labels, labels))


def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == ">": stack.append(i)
        elif c == "<":
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def detect_variable_loop(ss):
    pairs = parse_pairs(ss)
    five_prime = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in five_prime:
        if not cur: cur = [p]
        elif p == cur[-1] + 1 and pairs[p] == pairs[cur[-1]] - 1: cur.append(p)
        else: stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4: return 0, 0
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
    if vl_len > 5:
        vl_end = min(vl_start + vl_len, max_len)
        A[vl_start:vl_end, :] = 0.0
        A[:, vl_start:vl_end] = 0.0
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (A * d[:, None]) * d[None, :]


class TestDataset(Dataset):
    def __init__(self, indices):
        self.idx   = indices
        self.ss    = ss_arr[indices]
        self.sl    = seq_lens[indices]
        self.y     = labels[indices]
        self.phylo = phylo_ohe[indices] if phylo_ohe is not None else None
        self.vl    = var_loop_lengths[indices] if var_loop_lengths is not None else None
        self.order = order_labels[indices] if order_labels is not None else None

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        x   = emb[self.idx[i]].astype(np.float32)
        adj = build_adj_matrix(str(self.ss[i]), int(self.sl[i]), max_len)
        phylo = self.phylo[i] if self.phylo is not None else None
        vl    = np.array([self.vl[i]], dtype=np.float32) if self.vl is not None else None
        order = int(self.order[i]) if self.order is not None else None
        return x, adj, phylo, vl, order, int(self.y[i])


def collate_fn(batch):
    xs, adjs, ps, vls, orders, ys = zip(*batch)
    out = [torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(adjs))]
    out.append(torch.from_numpy(np.stack(ps)) if ps[0] is not None else None)
    out.append(torch.from_numpy(np.stack(vls)) if vls[0] is not None else None)
    out.append(torch.tensor(orders, dtype=torch.long) if orders[0] is not None else None)
    out.append(torch.tensor(ys))
    return tuple(out)


def _to(t, device): return t.to(device) if t is not None else None


test_ds     = TestDataset(test_idx)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, collate_fn=collate_fn)

all_preds = []
with torch.no_grad():
    for xb, ab, pb, vlb, ob, yb in test_loader:
        logits = model(xb.to(DEVICE), ab.to(DEVICE),
                       _to(pb, DEVICE), _to(vlb, DEVICE),
                       _to(ob, DEVICE)).cpu()
        all_preds.append(logits.argmax(1).numpy())
preds = np.concatenate(all_preds)

overall_acc = accuracy_score(labels[test_idx], preds)
print(f"\nOverall accuracy: {overall_acc:.4f}", flush=True)

per_domain_acc = {}
test_domains = domains[test_idx]
for dom in unique_domains:
    mask = (test_domains == dom)
    if mask.sum() > 0:
        per_domain_acc[dom] = round(float(accuracy_score(labels[test_idx][mask],
                                                          preds[mask])), 4)
        print(f"  {dom}: {per_domain_acc[dom]:.4f}", flush=True)

# ── Patch results JSON in-place ───────────────────────────────────────────────
results["per_domain_accuracy"] = per_domain_acc
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nPatched → {results_path}")
