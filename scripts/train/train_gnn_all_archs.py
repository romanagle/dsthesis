#!/usr/bin/env python3
"""
train_gnn_all_archs.py
----------------------
Trains any of 10 architectures as a 23-class combined-domain tRNA classifier.
Domain-specific IE definitions for 5 isotypes: Ala, Trp, Gly, Gln, Pro.
ISM: direct one-hot substitution for raw models, RiNALMo re-embedding for emb models.

Architectures
-------------
  raw_shallow   : one-hot → mean pool → MLP(256) → 23
  raw_gnn       : one-hot → GNN (backbone only) → pool → 23
  raw_gnn_ss    : one-hot → GNN (backbone + SS) → pool → 23
  raw_deep      : one-hot → mean pool → MLP(256,256,256) → 23
  emb_shallow   : RiNALMo → mean pool → MLP(256) → 23
  emb_se        : RiNALMo → SE attention → pool → 23
  emb_gnn_no_ss : RiNALMo → GNN (backbone only) → pool → 23
  emb_gnn       : RiNALMo → GNN (backbone + SS) → pool → 23
  emb_transformer: RiNALMo → Transformer → pool → 23
  emb_deep      : RiNALMo → mean pool → MLP(256,256,256) → 23

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python scripts/train/train_gnn_all_archs.py \\
        --arch raw_shallow \\
        --ism-isotypes Ala Trp Gly Gln Pro \\
        --outdir arch_ablation_5iso/raw_shallow
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

ARCH_CHOICES = [
    "raw_shallow", "raw_gnn", "raw_gnn_ss", "raw_deep",
    "emb_shallow", "emb_se", "emb_gnn_no_ss", "emb_gnn", "emb_transformer", "emb_deep",
]
IS_RAW = {a: a.startswith("raw") for a in ARCH_CHOICES}
USE_SS = {a: a in ("raw_gnn_ss", "emb_gnn") for a in ARCH_CHOICES}
IS_GNN = {a: "gnn" in a for a in ARCH_CHOICES}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--arch",           choices=ARCH_CHOICES, required=True)
parser.add_argument("--npz",            default="embeddings/gnn/perpos_embeddings.npz")
parser.add_argument("--csv",            default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir",         default=None)
parser.add_argument("--ism-isotypes",   nargs="+", default=["Ala", "Trp", "Gly", "Gln", "Pro"])
parser.add_argument("--ism-n-seqs",     type=int, default=100)
parser.add_argument("--ism-batch-size", type=int, default=64)
parser.add_argument("--batch-size",     type=int, default=256)
parser.add_argument("--test-size",      type=float, default=0.2)
parser.add_argument("--seed",           type=int, default=42)
args = parser.parse_args()

outdir = args.outdir or f"arch_ablation_5iso/{args.arch}"
os.makedirs(outdir, exist_ok=True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Arch: {args.arch}  Device: {DEVICE}", flush=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_LAYERS_GNN = 3
HIDDEN_DIM   = 256
DROPOUT      = 0.3
WD           = 6e-5
EPOCHS       = 20
# Transformer needs a lower LR; everything else uses the GNN-tuned 7e-3
LR = 1e-4 if args.arch == "emb_transformer" else 7e-3

NUCS        = ["A", "C", "G", "U"]
NUC_IDX_MAP = {n: i for i, n in enumerate(NUCS)}
ISM_NUCS    = ["A", "C", "G", "T"]

# ── Domain-specific IE definitions (acceptor branch only, anticodon excluded) ─
IE_DEFS = {
    "Ala": {
        # G3·U70 wobble pair is the primary Ala IE; A73 is secondary
        "Bacteria": [
            ("3",  "G", "primary"),
            ("70", "U", "primary"),
            ("73", "A", "secondary"),
            ("2",  "G", "secondary"),
            ("71", "C", "secondary"),
            ("4",  "G", "weak"),
            ("69", "C", "weak"),
        ],
        "Archaea": [  # … in known_IEs → same as bacteria
            ("3",  "G", "primary"),
            ("70", "U", "primary"),
            ("73", "A", "secondary"),
            ("2",  "G", "secondary"),
            ("71", "C", "secondary"),
            ("4",  "G", "weak"),
            ("69", "C", "weak"),
        ],
    },
    "Trp": {
        # Bac: G73, A1–U72, G2–C71, G3–C70, G4–C69, G5–C68
        "Bacteria": [
            ("73", "G", "primary"),
            ("1",  "A", "primary"),
            ("72", "U", "primary"),
            ("2",  "G", "secondary"),
            ("71", "C", "secondary"),
            ("3",  "G", "secondary"),
            ("70", "C", "secondary"),
            ("4",  "G", "weak"),
            ("69", "C", "weak"),
            ("5",  "G", "weak"),
            ("68", "C", "weak"),
        ],
        # Arc: A73, G1–C72, G2–C71
        "Archaea": [
            ("73", "A", "primary"),
            ("1",  "G", "primary"),
            ("72", "C", "primary"),
            ("2",  "G", "secondary"),
            ("71", "C", "secondary"),
        ],
    },
    "Gly": {
        # Bac: U73, G1–C72, C2–G71, G3–C70
        "Bacteria": [
            ("73", "U", "primary"),
            ("1",  "G", "primary"),
            ("72", "C", "primary"),
            ("2",  "C", "secondary"),
            ("71", "G", "secondary"),
            ("3",  "G", "secondary"),
            ("70", "C", "secondary"),
        ],
        # Arc: A73, C2–G71, G3–C70
        "Archaea": [
            ("73", "A", "primary"),
            ("2",  "C", "secondary"),
            ("71", "G", "secondary"),
            ("3",  "G", "secondary"),
            ("70", "C", "secondary"),
        ],
    },
    "Gln": {
        # Bac: G73, U1–A72, G2–C71, G3–C70
        "Bacteria": [
            ("73", "G", "primary"),
            ("1",  "U", "primary"),
            ("72", "A", "primary"),
            ("2",  "G", "secondary"),
            ("71", "C", "secondary"),
            ("3",  "G", "secondary"),
            ("70", "C", "secondary"),
        ],
        "Archaea": [  # … in known_IEs → same as bacteria
            ("73", "G", "primary"),
            ("1",  "U", "primary"),
            ("72", "A", "primary"),
            ("2",  "G", "secondary"),
            ("71", "C", "secondary"),
            ("3",  "G", "secondary"),
            ("70", "C", "secondary"),
        ],
    },
    "Pro": {
        # Bac: A73, G72 (minimal acceptor-branch IEs; core IEs excluded)
        "Bacteria": [
            ("73", "A", "primary"),
            ("72", "G", "secondary"),
        ],
        # Arc: A73, G1–C72, G2–C71, G3–C70
        "Archaea": [
            ("73", "A", "primary"),
            ("1",  "G", "primary"),
            ("72", "C", "primary"),
            ("2",  "G", "secondary"),
            ("71", "C", "secondary"),
            ("3",  "G", "secondary"),
            ("70", "C", "secondary"),
        ],
    },
}
TIER_WEIGHTS = {"primary": 3, "secondary": 2, "weak": 1}

# ── Sprinzl / adjacency utilities ────────────────────────────────────────────
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == ">": stack.append(i)
        elif c == "<":
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss):
    pairs = parse_pairs(ss)
    n = len(ss)
    five_prime = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in five_prime:
        if not cur: cur = [p]
        elif p == cur[-1] + 1 and pairs[p] == pairs[cur[-1]] - 1: cur.append(p)
        else: stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4: return {}
    sp = {}
    acc5 = stems[0]; acc3 = sorted(pairs[p] for p in acc5)
    for i, p in enumerate(acc5): sp[p] = str(i + 1)
    for i, p in enumerate(acc3): sp[p] = str(66 + i)
    disc = sorted(p for p in range(acc3[-1] + 1, n) if p not in pairs)
    for i, p in enumerate(disc): sp[p] = str(73 + i)
    d5 = stems[1]
    for i, p in enumerate([p for p in range(acc5[-1] + 1, d5[0]) if p not in pairs]):
        sp[p] = str(8 + i)
    d3 = sorted(pairs[p] for p in d5); nd = len(d5)
    for i, p in enumerate(d5): sp[p] = str(10 + i)
    for i, p in enumerate(d3): sp[p] = str(25 - nd + 1 + i)
    d_loop = [p for p in range(d5[-1] + 1, d3[0]) if p not in pairs]
    for i, p in enumerate(d_loop):
        labs = ["14", "15", "16", "17", "17a", "18", "19", "20", "20a", "20b", "21"]
        if i < len(labs): sp[p] = labs[i]
    ac5 = stems[2]
    lk2 = [p for p in range(d3[-1] + 1, ac5[0]) if p not in pairs]
    if lk2: sp[lk2[-1]] = "26"
    ac3 = sorted(pairs[p] for p in ac5)
    for i, p in enumerate(ac5): sp[p] = str(27 + i)
    for i, p in enumerate(ac3): sp[p] = str(39 + i)
    for i, p in enumerate([p for p in range(ac5[-1] + 1, ac3[0]) if p not in pairs]):
        sp[p] = str(32 + i)
    t5 = stems[-1]; t3 = sorted(pairs[p] for p in t5)
    for i, p in enumerate(t5): sp[p] = str(49 + i)
    for i, p in enumerate(t3): sp[p] = str(61 + i)
    for i, p in enumerate([p for p in range(t5[-1] + 1, t3[0]) if p not in pairs]):
        sp[p] = str(54 + i)
    var = [p for p in range(ac3[-1] + 1, t5[0])]
    if len(stems) == 5:
        for i, p in enumerate(var): sp[p] = f"e{i + 1}"
    else:
        for i, p in enumerate([p for p in var if p not in pairs]): sp[p] = str(44 + i)
    return sp


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


def build_adj_matrix(ss, seq_len, max_len, use_ss=True):
    A = np.eye(max_len, dtype=np.float32)
    for i in range(min(seq_len - 1, max_len - 1)):
        A[i, i + 1] = A[i + 1, i] = 1.0
    if use_ss:
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


def seq_to_onehot(seq, max_len):
    x = np.zeros((max_len, 4), dtype=np.float32)
    for i, c in enumerate(seq[:max_len]):
        c = c.upper()
        if c == "T": c = "U"
        idx = NUC_IDX_MAP.get(c)
        if idx is not None: x[i, idx] = 1.0
    return x


def get_sprinzl_map(iso, domain, csv_path):
    df = pd.read_csv(
        csv_path,
        usecols=["primary_sequence", "secondary_structure",
                 "Anticodon_predicted_isotype", "domain"],
        low_memory=False,
    )
    df = df[(df["Anticodon_predicted_isotype"] == iso) &
            (df["domain"] == domain)].dropna(
        subset=["primary_sequence", "secondary_structure"])
    if df.empty: return {}, {}
    modal_len = df["primary_sequence"].str.len().mode()[0]
    modal_ss  = (df[df["primary_sequence"].str.len() == modal_len]
                 ["secondary_structure"].mode()[0])
    raw_to_sp = build_sprinzl_map(modal_ss)
    return raw_to_sp, {v: k for k, v in raw_to_sp.items()}


# ── Load NPZ ──────────────────────────────────────────────────────────────────
print(f"Loading {args.npz} …", flush=True)
data      = np.load(args.npz, allow_pickle=True)
seq_lens  = data["seq_lens"]
labels    = data["labels"].astype(np.int64)
classes   = data["classes"].astype(str)
seqs      = data["sequences"]
ss_arr    = data["secondary_structures"]
max_len   = int(data["max_len"])
N_CLASSES = len(classes)
domains        = (data["domains"].astype(str) if "domains" in data
                  else np.full(len(labels), "Unknown", dtype=object))
unique_domains = sorted(set(domains))

phylo_ohe = data["phylo_ohe"].astype(np.float32) if "phylo_ohe" in data else None
PHYLO_DIM = int(phylo_ohe.shape[1]) if phylo_ohe is not None else 0

var_loop_lengths = (data["var_loop_lengths"].astype(np.float32) / 25.0
                    if "var_loop_lengths" in data else None)
VAR_LOOP_DIM = 1 if var_loop_lengths is not None else 0

# Load RiNALMo mmap only for emb architectures
emb_mmap = None
EMB_DIM = 4  # default for raw
if not IS_RAW[args.arch]:
    mmap_path = os.path.join(os.path.dirname(args.npz), "emb_mmap.npy")
    print(f"Memory-mapping {mmap_path} …", flush=True)
    emb_mmap = np.load(mmap_path, mmap_mode="r")
    EMB_DIM = emb_mmap.shape[2]

print(f"  N={len(labels):,}  max_len={max_len}  EMB_DIM={EMB_DIM}  N_CLASSES={N_CLASSES}",
      flush=True)
print(f"  Domains: { {d: int((domains == d).sum()) for d in unique_domains} }", flush=True)

sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
train_idx, test_idx = next(sss.split(labels, labels))
print(f"  Train={len(train_idx):,}  Test={len(test_idx):,}", flush=True)

# ── Model definitions ─────────────────────────────────────────────────────────

class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        return self.dropout(F.relu(torch.bmm(adj, self.linear(x))))


class GNNClassifier(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_layers, dropout, n_classes,
                 phylo_dim=0, var_loop_dim=0):
        super().__init__()
        dims = [emb_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            DenseGCNLayer(dims[i], dims[i + 1], dropout) for i in range(n_layers)
        ])
        self.phylo_dim    = phylo_dim
        self.var_loop_dim = var_loop_dim
        self.head = nn.Linear(hidden_dim + phylo_dim + var_loop_dim, n_classes)

    def forward(self, x, adj, phylo=None, var_loop=None, **_):
        for layer in self.layers:
            x = layer(x, adj)
        pooled = x.mean(dim=1)
        parts  = [pooled]
        if self.phylo_dim > 0 and phylo is not None: parts.append(phylo)
        if self.var_loop_dim > 0 and var_loop is not None: parts.append(var_loop)
        return self.head(torch.cat(parts, dim=1))


class MLPClassifier(nn.Module):
    def __init__(self, emb_dim, hidden_dims, dropout, n_classes, phylo_dim=0, var_loop_dim=0):
        super().__init__()
        self.phylo_dim    = phylo_dim
        self.var_loop_dim = var_loop_dim
        in_dim = emb_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head  = nn.Linear(in_dim + phylo_dim + var_loop_dim, n_classes)

    def forward(self, x, adj=None, phylo=None, var_loop=None, **_):
        pooled = self.trunk(x.mean(dim=1))
        parts  = [pooled]
        if self.phylo_dim > 0 and phylo is not None: parts.append(phylo)
        if self.var_loop_dim > 0 and var_loop is not None: parts.append(var_loop)
        return self.head(torch.cat(parts, dim=1))


class SEClassifier(nn.Module):
    """Squeeze-and-Excite attention over positions then pool → head."""
    def __init__(self, emb_dim, hidden_dim, dropout, n_classes,
                 phylo_dim=0, var_loop_dim=0, max_len=194):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden_dim)
        ratio = max(1, max_len // 16)
        self.se_fc1   = nn.Linear(max_len, ratio)
        self.se_fc2   = nn.Linear(ratio, max_len)
        self.dropout  = nn.Dropout(dropout)
        self.phylo_dim    = phylo_dim
        self.var_loop_dim = var_loop_dim
        self.head = nn.Linear(hidden_dim + phylo_dim + var_loop_dim, n_classes)

    def forward(self, x, adj=None, phylo=None, var_loop=None, **_):
        h    = F.relu(self.proj(x))                               # (B, L, hidden)
        se_w = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(
            h.mean(dim=-1)))))                                     # (B, L)
        pooled = self.dropout((h * se_w.unsqueeze(-1)).mean(dim=1))  # (B, hidden)
        parts = [pooled]
        if self.phylo_dim > 0 and phylo is not None: parts.append(phylo)
        if self.var_loop_dim > 0 and var_loop is not None: parts.append(var_loop)
        return self.head(torch.cat(parts, dim=1))


class TransformerClassifier(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout, n_classes,
                 phylo_dim=0, var_loop_dim=0, n_heads=8, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True,
        )
        self.transformer  = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout      = nn.Dropout(dropout)
        self.phylo_dim    = phylo_dim
        self.var_loop_dim = var_loop_dim
        self.head = nn.Linear(hidden_dim + phylo_dim + var_loop_dim, n_classes)

    def forward(self, x, adj=None, phylo=None, var_loop=None, **_):
        pooled = self.dropout(self.transformer(self.proj(x)).mean(dim=1))
        parts  = [pooled]
        if self.phylo_dim > 0 and phylo is not None: parts.append(phylo)
        if self.var_loop_dim > 0 and var_loop is not None: parts.append(var_loop)
        return self.head(torch.cat(parts, dim=1))


def build_model(arch):
    if arch == "raw_shallow":
        return MLPClassifier(4, [HIDDEN_DIM], DROPOUT, N_CLASSES, PHYLO_DIM, VAR_LOOP_DIM)
    elif arch in ("raw_gnn", "raw_gnn_ss"):
        return GNNClassifier(4, HIDDEN_DIM, N_LAYERS_GNN, DROPOUT, N_CLASSES, PHYLO_DIM, VAR_LOOP_DIM)
    elif arch == "raw_deep":
        return MLPClassifier(4, [HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM], DROPOUT, N_CLASSES, PHYLO_DIM, VAR_LOOP_DIM)
    elif arch == "emb_shallow":
        return MLPClassifier(EMB_DIM, [HIDDEN_DIM], DROPOUT, N_CLASSES, PHYLO_DIM, VAR_LOOP_DIM)
    elif arch == "emb_se":
        return SEClassifier(EMB_DIM, HIDDEN_DIM, DROPOUT, N_CLASSES, PHYLO_DIM, VAR_LOOP_DIM, max_len)
    elif arch in ("emb_gnn_no_ss", "emb_gnn"):
        return GNNClassifier(EMB_DIM, HIDDEN_DIM, N_LAYERS_GNN, DROPOUT, N_CLASSES, PHYLO_DIM, VAR_LOOP_DIM)
    elif arch == "emb_transformer":
        return TransformerClassifier(EMB_DIM, HIDDEN_DIM, DROPOUT, N_CLASSES, PHYLO_DIM, VAR_LOOP_DIM)
    elif arch == "emb_deep":
        return MLPClassifier(EMB_DIM, [HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM], DROPOUT, N_CLASSES, PHYLO_DIM, VAR_LOOP_DIM)
    raise ValueError(f"Unknown arch: {arch}")


model = build_model(args.arch).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel params: {n_params:,}", flush=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
_use_ss = USE_SS[args.arch]
_is_raw = IS_RAW[args.arch]
_is_gnn = IS_GNN[args.arch]


class tRNADataset(Dataset):
    def __init__(self, indices):
        self.idx   = indices
        self.seqs  = seqs[indices]
        self.ss    = ss_arr[indices]
        self.sl    = seq_lens[indices]
        self.y     = labels[indices]
        self.phylo = phylo_ohe[indices] if phylo_ohe is not None else None
        self.vl    = var_loop_lengths[indices] if var_loop_lengths is not None else None

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        if _is_raw:
            x = seq_to_onehot(str(self.seqs[i]), max_len)
        else:
            x = emb_mmap[self.idx[i]].astype(np.float32)
        adj = (build_adj_matrix(str(self.ss[i]), int(self.sl[i]), max_len, _use_ss)
               if _is_gnn else np.zeros((1,), dtype=np.float32))
        phylo = self.phylo[i] if self.phylo is not None else None
        vl    = np.array([self.vl[i]], dtype=np.float32) if self.vl is not None else None
        return x, adj, phylo, vl, int(self.y[i])


def collate_fn(batch):
    xs, adjs, ps, vls, ys = zip(*batch)
    out = [torch.from_numpy(np.stack(xs))]
    out.append(torch.from_numpy(np.stack(adjs)) if _is_gnn else None)
    out.append(torch.from_numpy(np.stack(ps)) if ps[0] is not None else None)
    out.append(torch.from_numpy(np.stack(vls)) if vls[0] is not None else None)
    out.append(torch.tensor(ys))
    return tuple(out)


def _to(t, device): return t.to(device) if t is not None else None


# ── Training ──────────────────────────────────────────────────────────────────
# num_workers=0 for emb models: mmap opened in the parent cannot safely be
# accessed from forked DataLoader workers and causes segfaults.
_num_workers = 0 if not _is_raw else 4
train_ds  = tRNADataset(train_idx)
loader    = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=_num_workers, collate_fn=collate_fn,
                       pin_memory=(DEVICE.type == "cuda"))
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
criterion = nn.CrossEntropyLoss()

print(f"\n── Training {args.arch} for {EPOCHS} epochs ──", flush=True)
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, ab, pb, vlb, yb in loader:
        xb = xb.to(DEVICE); ab = _to(ab, DEVICE); pb = _to(pb, DEVICE)
        vlb = _to(vlb, DEVICE); yb = yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb, ab, pb, vlb), yb)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * len(yb)
    if epoch % 5 == 0 or epoch == EPOCHS:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss / len(train_ds):.4f}", flush=True)

# ── Accuracy (overall + per domain) ──────────────────────────────────────────
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout): m.p = 0.0

test_ds     = tRNADataset(test_idx)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=_num_workers, collate_fn=collate_fn)
all_preds = []
with torch.no_grad():
    for xb, ab, pb, vlb, yb in test_loader:
        logits = model(xb.to(DEVICE), _to(ab, DEVICE), _to(pb, DEVICE),
                       _to(vlb, DEVICE)).cpu()
        all_preds.append(logits.argmax(1).numpy())
preds = np.concatenate(all_preds)
acc   = accuracy_score(labels[test_idx], preds)
print(f"\nTest accuracy: {acc:.4f}", flush=True)

per_domain_acc = {}
test_domains   = domains[test_idx]
for dom in unique_domains:
    mask = (test_domains == dom)
    if mask.sum() > 0:
        per_domain_acc[dom] = round(float(accuracy_score(labels[test_idx][mask],
                                                          preds[mask])), 4)
        print(f"  {dom}: {per_domain_acc[dom]:.4f}", flush=True)

torch.save(model.state_dict(), os.path.join(outdir, "arch_model.pt"))

# ── ISM ───────────────────────────────────────────────────────────────────────
ism_isos     = [iso for iso in args.ism_isotypes if iso in classes]
test_labels  = labels[test_idx]
ie_scores    = {}

# Lazy-load RiNALMo for emb models
_rinalmo = _alphabet = None

def _load_rinalmo():
    global _rinalmo, _alphabet
    if _rinalmo is None:
        print("Loading RiNALMo for ISM …", flush=True)
        from rinalmo.pretrained import get_pretrained_model
        _rinalmo, _alphabet = get_pretrained_model(model_name="giga-v1", lm_config="giga")
        _rinalmo.eval().to(DEVICE)
        for m in _rinalmo.modules():
            if isinstance(m, nn.Dropout): m.p = 0.0
    return _rinalmo, _alphabet


def _embed_rinalmo(seqs_list):
    rinalmo, alphabet = _load_rinalmo()
    toks = torch.tensor(alphabet.batch_tokenize(seqs_list), dtype=torch.int64, device=DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = rinalmo(toks)
    emb_out = out["representation"].float().cpu().numpy()
    pad = np.zeros((len(seqs_list), max_len, EMB_DIM), dtype=np.float32)
    for bi, s in enumerate(seqs_list):
        Ls = min(len(s), max_len)
        pad[bi, :Ls] = emb_out[bi, 1:Ls + 1]
    return pad


def _classify(x_np, adj_np, phylo_np=None, vl_np=None):
    with torch.no_grad():
        x  = torch.from_numpy(x_np).to(DEVICE)
        a  = _to(torch.from_numpy(adj_np) if adj_np is not None else None, DEVICE)
        p  = _to(torch.from_numpy(phylo_np) if phylo_np is not None else None, DEVICE)
        vl = _to(torch.from_numpy(vl_np) if vl_np is not None else None, DEVICE)
        return model(x, a, p, vl).cpu().numpy()


for iso in ism_isos:
    iso_idx = int(np.where(classes == iso)[0][0])

    for dom in unique_domains:
        mask    = (test_labels == iso_idx) & (test_domains == dom)
        n_avail = mask.sum()
        if n_avail == 0:
            print(f"\n[{iso} / {dom}] No test sequences — skipping", flush=True)
            continue

        # Check if IE_DEFS has entries for this (iso, domain)
        if iso not in IE_DEFS or dom not in IE_DEFS[iso]:
            print(f"\n[{iso} / {dom}] No IE definitions — skipping", flush=True)
            continue

        global_idx = test_idx[mask]
        n_ism      = min(args.ism_n_seqs, n_avail)
        rng        = np.random.default_rng(args.seed)
        sel        = rng.choice(n_avail, n_ism, replace=False)

        ism_seqs  = seqs[global_idx][sel]
        ism_lens  = seq_lens[global_idx][sel]
        ism_ss    = ss_arr[global_idx][sel]
        ism_phylo = (phylo_ohe[global_idx][sel] if phylo_ohe is not None else None)
        ism_vl    = (var_loop_lengths[global_idx][sel] if var_loop_lengths is not None else None)

        if _is_gnn:
            ism_adj = np.stack([
                build_adj_matrix(str(ism_ss[i]), int(ism_lens[i]), max_len, _use_ss)
                for i in range(n_ism)
            ])
        else:
            ism_adj = None

        print(f"\n[{iso} / {dom}]  ISM: {n_ism} seqs …", flush=True)
        n_classes = len(classes)
        ism_sum   = np.zeros((max_len, 4, n_classes), dtype=np.float64)

        for si in range(n_ism):
            seq = str(ism_seqs[si])
            L   = min(int(ism_lens[si]), max_len)

            phylo_1 = (np.tile(ism_phylo[si], (1, 1)).astype(np.float32)
                       if ism_phylo is not None else None)
            vl_1    = (np.array([[ism_vl[si]]], dtype=np.float32)
                       if ism_vl is not None else None)
            adj_1   = ism_adj[si:si + 1] if _is_gnn else None

            if _is_raw:
                x0 = seq_to_onehot(seq, max_len)
            else:
                x0 = _embed_rinalmo([seq])[0]
            logit_orig = _classify(x0[None], adj_1, phylo_1, vl_1)[0]

            mutant_x, positions, nuc_idxs = [], [], []
            for pos in range(L):
                for ni, nuc in enumerate(ISM_NUCS):
                    if _is_raw:
                        xm = x0.copy()
                        xm[pos, :] = 0.0
                        nuc_u = nuc if nuc != "T" else "U"
                        xm[pos, NUC_IDX_MAP.get(nuc_u, 0)] = 1.0
                        mutant_x.append(xm)
                    else:
                        mutant_x.append(seq[:pos] + nuc + seq[pos + 1:])
                    positions.append(pos)
                    nuc_idxs.append(ni)

            ism_seq = np.zeros((max_len, 4, n_classes), dtype=np.float64)
            for start in range(0, len(positions), args.ism_batch_size):
                end = start + args.ism_batch_size
                B   = len(positions[start:end])
                if _is_raw:
                    xb_mut = np.stack(mutant_x[start:end])
                else:
                    xb_mut = _embed_rinalmo(mutant_x[start:end])
                adj_b   = (np.tile(ism_adj[si], (B, 1, 1)) if _is_gnn else None)
                phylo_b = (np.tile(ism_phylo[si], (B, 1)).astype(np.float32)
                           if ism_phylo is not None else None)
                vl_b    = (np.tile(vl_1, (B, 1)) if vl_1 is not None else None)
                logits  = _classify(xb_mut, adj_b, phylo_b, vl_b)
                for bi in range(B):
                    ism_seq[positions[start + bi], nuc_idxs[start + bi], :] = (
                        logits[bi] - logit_orig
                    )

            ism_sum += ism_seq
            if (si + 1) % 10 == 0 or si + 1 == n_ism:
                print(f"  {si + 1}/{n_ism} done", flush=True)

        mean_sal_all = ism_sum / n_ism
        target_sal   = mean_sal_all[:, :, iso_idx]
        other_mean   = (mean_sal_all.sum(axis=2) - target_sal) / (n_classes - 1)
        mean_sal     = (target_sal - other_mean).astype(np.float32)
        mean_sal    -= mean_sal.mean(axis=1, keepdims=True)

        sal_path = os.path.join(outdir, f"ism_saliency_{iso}_{dom}.npy")
        np.save(sal_path, mean_sal)
        print(f"  Saved → {sal_path}", flush=True)

        # IE score with domain-specific definitions
        _, sp_to_raw = get_sprinzl_map(iso, dom, args.csv)
        global_max   = float(np.max(np.maximum(mean_sal, 0)))
        ws = wt = 0.0
        if global_max > 0:
            for sp_label, exp_nuc, tier in IE_DEFS[iso][dom]:
                raw = sp_to_raw.get(sp_label)
                w   = TIER_WEIGHTS[tier]
                if raw is None or raw >= mean_sal.shape[0]: continue
                comp = max(float(mean_sal[raw, NUC_IDX_MAP[exp_nuc]]), 0.0) / global_max
                ws  += w * comp; wt += w
        ie_score = (ws / wt * 100) if wt > 0 else 0.0
        ie_scores[f"{iso}_{dom}"] = round(float(ie_score), 2)
        print(f"  IE score ({dom}): {ie_score:.1f}/100", flush=True)

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "model":               args.arch,
    "classes":             list(classes),
    "n_classes":           N_CLASSES,
    "max_len":             max_len,
    "domains":             unique_domains,
    "n_train":             len(train_idx),
    "n_test":              len(test_idx),
    "n_params":            n_params,
    "n_ism_per_group":     args.ism_n_seqs,
    "accuracy":            round(float(acc), 4),
    "per_domain_accuracy": per_domain_acc,
    "ie_scores":           ie_scores,
    "hyperparams": {
        "n_layers": N_LAYERS_GNN if _is_gnn else len([HIDDEN_DIM]),
        "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
        "lr": LR, "weight_decay": WD, "epochs": EPOCHS,
    },
}
json_path = os.path.join(outdir, "arch_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {json_path}")
print("Done.")
