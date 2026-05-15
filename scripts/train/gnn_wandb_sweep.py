#!/usr/bin/env python3
"""
gnn_wandb_sweep.py
------------------
Weights & Biases Bayesian sweep over GNN hyperparameters.
Optimisation target: ISM-based IE interpretability score.

Sweep parameters
----------------
  n_layers    : number of GCN message-passing layers   [1, 2, 3]
  hidden_dim  : width of each GCN layer                [64, 128, 256, 512]
  dropout     : dropout after each GCN layer           [0.0, 0.1, 0.2, 0.3]
  lr          : Adam learning rate                     log-uniform [1e-4, 1e-2]
  weight_decay: Adam weight decay                      log-uniform [1e-6, 1e-2]
  epochs      : training epochs                        [20, 30, 50]

Data (NPZ + adj matrices + Sprinzl map) are loaded once at module level
and shared across all trials in the same agent process.
RiNALMo is loaded lazily on first ISM call and cached for subsequent trials.

Usage
-----
  # Step 1 — create sweep, get sweep_id printed to stdout
  conda run -n myenv python scripts/gnn_wandb_sweep.py --create-sweep

  # Step 2 — launch one agent (runs --count trials then exits)
  conda run -n myenv python scripts/gnn_wandb_sweep.py --agent <sweep_id> --count 40

  # Step 2 alt — launch multiple agents in parallel (separate terminals)
  conda run -n myenv python scripts/gnn_wandb_sweep.py --agent <sweep_id> --count 20 &
  conda run -n myenv python scripts/gnn_wandb_sweep.py --agent <sweep_id> --count 20 &

  # Shortcut — create sweep AND run a single agent (useful for quick tests)
  conda run -n myenv python scripts/gnn_wandb_sweep.py --run --count 5 --ism-n-seqs 20 --epochs-override 5
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import wandb

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir",    default="embeddings/gnn",
                    help="Directory containing perpos_embeddings.npz")
parser.add_argument("--csv",        default="data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--project",    default="tRNA-GNN-sweep",
                    help="W&B project name")
parser.add_argument("--entity",     default=None,
                    help="W&B entity (team/username) — uses default if omitted")

# Sweep control
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--create-sweep", action="store_true",
                   help="Register the sweep and print its ID, then exit")
group.add_argument("--agent",        metavar="SWEEP_ID",
                   help="Run as agent against an existing sweep ID")
group.add_argument("--run",          action="store_true",
                   help="Create sweep + immediately run --count trials (convenience shortcut)")

parser.add_argument("--count",          type=int, default=40,
                    help="Number of trials per agent")
parser.add_argument("--ism-n-seqs",     type=int, default=100,
                    help="Target sequences for ISM attribution (fixed across all trials)")
parser.add_argument("--ism-batch-size", type=int, default=64)
parser.add_argument("--test-size",      type=float, default=0.2)
parser.add_argument("--batch-size",     type=int, default=256)
parser.add_argument("--seed",           type=int, default=42)
parser.add_argument("--epochs-override", type=int, default=None,
                    help="Override the epochs value from wandb config (useful for quick tests)")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
print(f"Device: {DEVICE}", flush=True)

# ── Sweep configuration ───────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "ie_score", "goal": "maximize"},
    "parameters": {
        "n_layers":     {"values": [1, 2, 3]},
        "hidden_dim":   {"values": [64, 128, 256, 512]},
        "dropout":      {"values": [0.0, 0.1, 0.2, 0.3]},
        "lr":           {"min": 1e-4, "max": 1e-2,
                         "distribution": "log_uniform_values"},
        "weight_decay": {"min": 1e-6, "max": 1e-2,
                         "distribution": "log_uniform_values"},
        "epochs":       {"values": [20, 30, 50]},
    },
}

# ── Identity element definitions ──────────────────────────────────────────────
IE_DEFS = {
    "Ala": [
        ("3",  "G", "primary"), ("70", "U", "primary"),
        ("73", "A", "secondary"), ("1",  "G", "secondary"),
        ("72", "C", "secondary"), ("2",  "C", "weak"), ("71", "G", "weak"),
    ],
    "Gly": [
        ("73", "A", "primary"), ("3",  "G", "primary"), ("70", "C", "primary"),
        ("2",  "C", "secondary"), ("71", "G", "secondary"),
    ],
}
TIER_WEIGHTS = {"primary": 3, "secondary": 2, "weak": 1}
NUCS         = ["A", "C", "G", "U"]
NUC_IDX_MAP  = {n: i for i, n in enumerate(NUCS)}

# ── Sprinzl map utilities ─────────────────────────────────────────────────────
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == '>': stack.append(i)
        elif c == '<':
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs

def build_sprinzl_map(ss):
    pairs = parse_pairs(ss)
    n     = len(ss)
    five_prime = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in five_prime:
        if not cur: cur = [p]
        elif p == cur[-1]+1 and pairs[p] == pairs[cur[-1]]-1: cur.append(p)
        else: stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4: return {}
    sp = {}
    acc5 = stems[0]; acc3 = sorted(pairs[p] for p in acc5)
    for i, p in enumerate(acc5): sp[p] = str(i+1)
    for i, p in enumerate(acc3): sp[p] = str(66+i)
    disc = sorted(p for p in range(acc3[-1]+1, n) if p not in pairs)
    for i, p in enumerate(disc): sp[p] = str(73+i)
    d5 = stems[1]
    for i, p in enumerate([p for p in range(acc5[-1]+1, d5[0]) if p not in pairs]):
        sp[p] = str(8+i)
    d3 = sorted(pairs[p] for p in d5); nd = len(d5)
    for i, p in enumerate(d5): sp[p] = str(10+i)
    for i, p in enumerate(d3): sp[p] = str(25-nd+1+i)
    d_loop = [p for p in range(d5[-1]+1, d3[0]) if p not in pairs]
    for i, p in enumerate(d_loop):
        if i < len(["14","15","16","17","17a","18","19","20","20a","20b","21"]):
            sp[p] = ["14","15","16","17","17a","18","19","20","20a","20b","21"][i]
    ac5 = stems[2]
    lk2 = [p for p in range(d3[-1]+1, ac5[0]) if p not in pairs]
    if lk2: sp[lk2[-1]] = "26"
    ac3 = sorted(pairs[p] for p in ac5)
    for i, p in enumerate(ac5): sp[p] = str(27+i)
    for i, p in enumerate(ac3): sp[p] = str(39+i)
    for i, p in enumerate([p for p in range(ac5[-1]+1, ac3[0]) if p not in pairs]):
        sp[p] = str(32+i)
    t5 = stems[-1]; t3 = sorted(pairs[p] for p in t5)
    for i, p in enumerate(t5): sp[p] = str(49+i)
    for i, p in enumerate(t3): sp[p] = str(61+i)
    for i, p in enumerate([p for p in range(t5[-1]+1, t3[0]) if p not in pairs]):
        sp[p] = str(54+i)
    var = [p for p in range(ac3[-1]+1, t5[0])]
    if len(stems) == 5:
        for i, p in enumerate(var): sp[p] = f"e{i+1}"
    else:
        for i, p in enumerate([p for p in var if p not in pairs]): sp[p] = str(44+i)
    return sp

def build_adj_matrix(ss, seq_len, max_len):
    A = np.eye(max_len, dtype=np.float32)
    for i in range(min(seq_len - 1, max_len - 1)):
        A[i, i+1] = A[i+1, i] = 1.0
    for i, j in parse_pairs(ss).items():
        if i < max_len and j < max_len:
            A[i, j] = A[j, i] = 1.0
    deg = A.sum(axis=1)
    d   = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (A * d[:, None]) * d[None, :]

# ── Load data (once, shared across all trials) ────────────────────────────────
emb_path = os.path.join(args.emb_dir, "perpos_embeddings.npz")
print(f"Loading {emb_path} …", flush=True)
data     = np.load(emb_path, allow_pickle=True)
emb_f32  = data["emb"].astype(np.float32)       # (N, max_len, 1280)
seq_lens = data["seq_lens"]
labels   = data["labels"].astype(np.int64)
seqs     = data["sequences"]
secondary_structures = data["secondary_structures"]
max_len  = int(data["max_len"])
EMB_DIM  = emb_f32.shape[2]
N        = len(labels)

ISO    = str(data["iso"])    if "iso"    in data else None
domain = str(data["domain"]) if "domain" in data else "Unknown"
if ISO is None or ISO not in IE_DEFS:
    sys.exit(f"Isotype '{ISO}' not in IE_DEFS. Available: {list(IE_DEFS.keys())}")
IES = IE_DEFS[ISO]
print(f"  N={N}  max_len={max_len}  ISO={ISO}  domain={domain}", flush=True)

# Sprinzl map
print("Building Sprinzl map …", flush=True)
df_ss    = pd.read_csv(args.csv,
                       usecols=["primary_sequence", "secondary_structure",
                                "Anticodon_predicted_isotype", "domain"],
                       low_memory=False)
df_iso   = df_ss[(df_ss["Anticodon_predicted_isotype"] == ISO) &
                 (df_ss["domain"] == domain)].dropna(
    subset=["primary_sequence", "secondary_structure"])
modal_len = df_iso["primary_sequence"].str.len().mode()[0]
modal_ss  = (df_iso[df_iso["primary_sequence"].str.len() == modal_len]
             ["secondary_structure"].mode()[0])
raw_to_sp = build_sprinzl_map(modal_ss)
sp_to_raw = {v: k for k, v in raw_to_sp.items()}
del df_ss, df_iso
print(f"  Sprinzl positions mapped: {len(raw_to_sp)}", flush=True)

# Adjacency matrices
print("Precomputing adjacency matrices …", flush=True)
adj_all = np.stack([
    build_adj_matrix(str(ss), int(sl), max_len)
    for ss, sl in zip(secondary_structures, seq_lens)
])
print(f"  adj_all: {adj_all.shape}", flush=True)

# Fixed train/test split (identical across all trials)
sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size,
                              random_state=args.seed)
train_idx, test_idx = next(sss.split(labels, labels))
target_test_mask = labels[test_idx] == 1
target_test_idx  = test_idx[target_test_mask]
print(f"  Train={len(train_idx)}  Test={len(test_idx)}  "
      f"Target test seqs={target_test_mask.sum()}", flush=True)

# Fixed ISM subsample indices (same sequences evaluated every trial)
rng         = np.random.default_rng(args.seed)
n_ism       = min(args.ism_n_seqs, int(target_test_mask.sum()))
ism_sub_idx = rng.choice(len(target_test_idx), n_ism, replace=False)
ism_seqs    = seqs[target_test_idx][ism_sub_idx]
ism_lens    = seq_lens[target_test_idx][ism_sub_idx]
ism_adj     = adj_all[target_test_idx][ism_sub_idx]
print(f"  ISM sequences (fixed): {n_ism}", flush=True)

# ── IE score ──────────────────────────────────────────────────────────────────
def compute_ie_score(sal):
    global_max = float(np.max(np.maximum(sal, 0)))
    if global_max == 0:
        return 0.0
    ws = wt = 0.0
    for sp_label, exp_nuc, tier in IES:
        raw = sp_to_raw.get(sp_label)
        w   = TIER_WEIGHTS[tier]
        if raw is None or raw >= sal.shape[0]:
            continue
        comp = max(float(sal[raw, NUC_IDX_MAP[exp_nuc]]), 0.0) / global_max
        ws  += w * comp
        wt  += w
    return (ws / wt * 100) if wt > 0 else 0.0

# ── Lazy RiNALMo loader ───────────────────────────────────────────────────────
ISM_NUCS  = ["A", "C", "G", "T"]
_rinalmo  = None
_alphabet = None

def _load_rinalmo():
    global _rinalmo, _alphabet
    if _rinalmo is None:
        print("Loading RiNALMo for ISM …", flush=True)
        from rinalmo.pretrained import get_pretrained_model
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

        _rinalmo, _alphabet = get_pretrained_model(model_name="giga-v1",
                                                    lm_config="giga")
        _rinalmo.eval().to(DEVICE)
        for m in _rinalmo.modules():
            if isinstance(m, nn.Dropout): m.p = 0.0
    return _rinalmo, _alphabet

# ── ISM attribution ───────────────────────────────────────────────────────────
def compute_ism_mean(sequences_sub, seq_lens_sub, adj_sub, classifier):
    rinalmo, alphabet = _load_rinalmo()

    def _embed(seqs_list):
        toks = torch.tensor(
            alphabet.batch_tokenize(seqs_list), dtype=torch.int64, device=DEVICE
        )
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = rinalmo(toks)
        emb = out["representation"].float().cpu().numpy()
        pad = np.zeros((len(seqs_list), max_len, EMB_DIM), dtype=np.float32)
        for bi, s in enumerate(seqs_list):
            Ls = min(len(s), max_len)
            pad[bi, :Ls] = emb[bi, 1:Ls + 1]
        return pad

    def _classify(emb_pad, adj_np):
        with torch.no_grad():
            x = torch.from_numpy(emb_pad).to(DEVICE)
            a = torch.from_numpy(adj_np).to(DEVICE)
            return classifier(x, a)[:, 1].cpu().numpy()

    ism_sum = np.zeros((max_len, 4), dtype=np.float64)
    n = len(sequences_sub)

    for si in range(n):
        seq = sequences_sub[si]
        L   = min(int(seq_lens_sub[si]), max_len)

        logit_orig = _classify(_embed([seq]), adj_sub[si:si+1])[0]

        mutants, positions, nuc_idxs = [], [], []
        for p in range(L):
            for ni, nuc in enumerate(ISM_NUCS):
                mutants.append(seq[:p] + nuc + seq[p + 1:])
                positions.append(p)
                nuc_idxs.append(ni)

        ism_seq = np.zeros((max_len, 4), dtype=np.float64)
        for start in range(0, len(mutants), args.ism_batch_size):
            batch  = mutants[start:start + args.ism_batch_size]
            B      = len(batch)
            emb_b  = _embed(batch)
            adj_b  = np.tile(adj_sub[si], (B, 1, 1))
            logits = _classify(emb_b, adj_b)
            for bi in range(B):
                ism_seq[positions[start + bi], nuc_idxs[start + bi]] = (
                    logits[bi] - logit_orig
                )

        ism_sum += ism_seq
        if (si + 1) % 10 == 0 or si + 1 == n:
            print(f"    ISM {si+1}/{n}", flush=True)

    result = (ism_sum / n).astype(np.float32)
    result -= result.mean(axis=1, keepdims=True)   # row-centre
    return result

# ── GNN model ─────────────────────────────────────────────────────────────────
class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        return self.dropout(F.relu(torch.bmm(adj, self.linear(x))))


class GNNClassifier(nn.Module):
    def __init__(self, emb_dim, hidden_dim=256, n_layers=2, dropout=0.0):
        super().__init__()
        dims = [emb_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            DenseGCNLayer(dims[i], dims[i + 1], dropout)
            for i in range(n_layers)
        ])
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return self.head(x.mean(dim=1))

# ── Single trial ──────────────────────────────────────────────────────────────
def train():
    run = wandb.init()
    cfg = run.config

    epochs     = args.epochs_override if args.epochs_override else cfg.epochs
    hidden_dim = cfg.hidden_dim
    n_layers   = cfg.n_layers
    dropout    = cfg.dropout
    lr         = cfg.lr
    wd         = cfg.weight_decay

    print(f"\n[Trial] layers={n_layers} hidden={hidden_dim} "
          f"dropout={dropout} lr={lr:.2e} wd={wd:.2e} epochs={epochs}",
          flush=True)

    model = GNNClassifier(EMB_DIM, hidden_dim, n_layers, dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"n_params": n_params}, step=0)

    X_tr = torch.from_numpy(emb_f32[train_idx])
    A_tr = torch.from_numpy(adj_all[train_idx])
    y_tr = torch.from_numpy(labels[train_idx])
    X_te = torch.from_numpy(emb_f32[test_idx])
    A_te = torch.from_numpy(adj_all[test_idx])
    y_te = torch.from_numpy(labels[test_idx])

    loader    = DataLoader(TensorDataset(X_tr, A_tr, y_tr),
                           batch_size=args.batch_size, shuffle=True,
                           pin_memory=(DEVICE.type == "cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, ab, yb in loader:
            xb, ab, yb = xb.to(DEVICE), ab.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb, ab), yb)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * len(yb)
        avg_loss = total_loss / len(y_tr)
        wandb.log({"train_loss": avg_loss, "epoch": epoch}, step=epoch)

    # Accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X_te.to(DEVICE), A_te.to(DEVICE)).cpu()
    acc = accuracy_score(labels[test_idx], logits.argmax(1).numpy())

    # ISM IE score
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.p = 0.0

    print(f"  Computing ISM ({n_ism} seqs) …", flush=True)
    mean_sal  = compute_ism_mean(ism_seqs, ism_lens, ism_adj, model)
    ie_score  = compute_ie_score(mean_sal)

    print(f"  accuracy={acc:.4f}  ie_score={ie_score:.1f}", flush=True)
    wandb.log({"accuracy": acc, "ie_score": ie_score})
    run.finish()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    kwargs = dict(project=args.project)
    if args.entity:
        kwargs["entity"] = args.entity

    if args.create_sweep:
        sweep_id = wandb.sweep(SWEEP_CONFIG, **kwargs)
        print(f"\nSweep created: {sweep_id}")
        print(f"Run agent with:")
        print(f"  conda run -n myenv python scripts/gnn_wandb_sweep.py "
              f"--agent {sweep_id} --count {args.count}")

    elif args.agent:
        wandb.agent(args.agent, function=train, count=args.count, **kwargs)

    elif args.run:
        sweep_id = wandb.sweep(SWEEP_CONFIG, **kwargs)
        print(f"Sweep created: {sweep_id}", flush=True)
        wandb.agent(sweep_id, function=train, count=args.count, **kwargs)
