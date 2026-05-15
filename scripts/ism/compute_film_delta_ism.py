#!/usr/bin/env python3
"""
compute_film_delta_ism.py
--------------------------
For each (isotype × order) pair, compute the FiLM-delta ISM:

    delta[pos, nuc] = ISM(seq, order=order_idx) − ISM(seq, order=None)

Using the SAME sequences for both passes isolates the FiLM contribution
from backbone sequence effects. Both the per-order FiLM-on saliency and
the pure delta are saved.

Usage
-----
    python scripts/ism/compute_film_delta_ism.py
    python scripts/ism/compute_film_delta_ism.py \
        --isotypes His Ala Gly \
        --min-seqs 5 \
        --n-seqs   20 \
        --model    gnn_film/gnn_model.pt \
        --outdir   gnn_film/film_delta_ism
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir",       default=os.path.join(DATA_ROOT, "embeddings/gnn"))
parser.add_argument("--model",         default=os.path.join(DATA_ROOT, "gnn_film/gnn_model.pt"))
parser.add_argument("--phylo-encoder", default=os.path.join(DATA_ROOT, "gnn_final/phylo_encoder.pkl"))
parser.add_argument("--outdir",        default=os.path.join(DATA_ROOT, "gnn_film/film_delta_ism"))
parser.add_argument("--isotypes",      nargs="+", default=None,
                    help="Isotypes to run (default: all in NPZ)")
parser.add_argument("--orders",        nargs="+", default=None,
                    help="Specific orders to run (default: all with >= --min-seqs)")
parser.add_argument("--min-seqs",      type=int, default=5,
                    help="Minimum sequences per (iso, order) to include (default: 5)")
parser.add_argument("--n-seqs",        type=int, default=20,
                    help="Sequences per (iso, order) for ISM (default: 20)")
parser.add_argument("--all-seqs",      action="store_true",
                    help="Use all sequences (not just test set) for ISM. Safe for FiLM delta "
                         "because the delta measures model conditioning, not sequence identity.")
parser.add_argument("--batch-size",    type=int, default=64)
parser.add_argument("--test-size",     type=float, default=0.2)
parser.add_argument("--seed",          type=int, default=42)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

ISM_NUCS   = ["A", "C", "G", "T"]
N_LAYERS   = 3
HIDDEN_DIM = 256
DROPOUT    = 0.0   # eval mode

# ── Load phylo encoder ────────────────────────────────────────────────────────
with open(args.phylo_encoder, "rb") as f:
    enc = pickle.load(f)
cats          = enc.categories_
n_cats        = [len(c) for c in cats]
order_start   = sum(n_cats[:3])
order_end     = order_start + n_cats[3]
order_classes = cats[3]
order_name_to_idx = {name: i for i, name in enumerate(order_classes)}
N_ORDERS = len(order_classes)

# ── Load embeddings ───────────────────────────────────────────────────────────
npz_path  = os.path.join(args.emb_dir, "perpos_embeddings.npz")
mmap_path = os.path.join(args.emb_dir, "emb_mmap.npy")

print(f"Loading metadata …", flush=True)
data         = np.load(npz_path, allow_pickle=True)
seq_lens     = data["seq_lens"]
labels       = data["labels"].astype(np.int64)
classes      = data["classes"].astype(str)
seqs         = data["sequences"].astype(str)
ss_arr       = data["secondary_structures"].astype(str)
phylo_ohe    = data["phylo_ohe"].astype(np.float32)
var_loop_lengths = data["var_loop_lengths"] if "var_loop_lengths" in data else None
max_len      = int(data["max_len"])
EMB_DIM      = int(np.load(mmap_path, mmap_mode="r").shape[2])

order_labels = phylo_ohe[:, order_start:order_end].argmax(axis=1).astype(np.int64)

PHYLO_DIM    = phylo_ohe.shape[1]
VAR_LOOP_DIM = 1 if var_loop_lengths is not None else 0
N_CLASSES    = len(classes)

print(f"  N={len(labels):,}  max_len={max_len}  EMB_DIM={EMB_DIM}  "
      f"N_CLASSES={N_CLASSES}  N_ORDERS={N_ORDERS}", flush=True)

seqs_mmap = np.load(mmap_path, mmap_mode="r")

# ── Sequence pool: test set only (default) or all sequences (--all-seqs) ─────
if args.all_seqs:
    pool_idx    = np.arange(len(labels))
else:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    _, pool_idx = next(sss.split(labels, labels))
test_labels  = labels[pool_idx]
test_orders  = order_labels[pool_idx]

# ── GNN architecture (must match train_gnn.py) ───────────────────────────────
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == ">": stack.append(i)
        elif c == "<":
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def detect_variable_loop(ss):
    pairs = parse_pairs(ss)
    fps   = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in fps:
        if not cur: cur = [p]
        elif p == cur[-1]+1 and pairs[p] == pairs[cur[-1]]-1: cur.append(p)
        else: stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4: return 0, 0
    ac3_end  = sorted(pairs[p] for p in stems[2])[-1]
    t5_start = stems[-1][0]
    return ac3_end + 1, t5_start - ac3_end - 1


VAR_LOOP_THRESHOLD = 5


def build_adj_matrix(ss, seq_len, max_len):
    A = np.eye(max_len, dtype=np.float32)
    for i in range(min(seq_len - 1, max_len - 1)):
        A[i, i+1] = A[i+1, i] = 1.0
    for i, j in parse_pairs(ss).items():
        if i < max_len and j < max_len:
            A[i, j] = A[j, i] = 1.0
    vl_start, vl_len = detect_variable_loop(ss)
    if vl_len > VAR_LOOP_THRESHOLD:
        vl_end = min(vl_start + vl_len, max_len)
        A[vl_start:vl_end, :] = 0.0
        A[:, vl_start:vl_end] = 0.0
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (A * d[:, None]) * d[None, :]


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
        nn.init.zeros_(self.cond_net.weight)
        nn.init.zeros_(self.cond_net.bias)

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
            DenseGCNLayer(dims[i], dims[i+1], dropout) for i in range(n_layers)
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


model = GNNClassifier(EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, N_CLASSES,
                      PHYLO_DIM, VAR_LOOP_DIM,
                      film_n_orders=N_ORDERS, film_emb_dim=32).to(DEVICE)
model.load_state_dict(torch.load(args.model, map_location=DEVICE))
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout): m.p = 0.0
print(f"Loaded model from {args.model}", flush=True)

# ── Load RiNALMo ──────────────────────────────────────────────────────────────
print("Loading RiNALMo …", flush=True)
from rinalmo.pretrained import get_pretrained_model
rinalmo, alphabet = get_pretrained_model(model_name="giga-v1", lm_config="giga")
rinalmo.eval().to(DEVICE)
for m in rinalmo.modules():
    if isinstance(m, nn.Dropout): m.p = 0.0


def _embed(seqs_list):
    toks = torch.tensor(alphabet.batch_tokenize(seqs_list),
                        dtype=torch.int64, device=DEVICE)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = rinalmo(toks)
    emb = out["representation"].float().cpu().numpy()
    pad = np.zeros((len(seqs_list), max_len, EMB_DIM), dtype=np.float32)
    for bi, s in enumerate(seqs_list):
        Ls = min(len(s), max_len)
        pad[bi, :Ls] = emb[bi, 1:Ls+1]
    return pad


def _to(t, device):
    return t.to(device) if t is not None else None


def _classify(emb_pad, adj_np, phylo_np, vl_np, order_np):
    """Two forward passes: FiLM on (order_np given) and FiLM off (order=None)."""
    with torch.no_grad():
        x  = torch.from_numpy(emb_pad).to(DEVICE)
        a  = torch.from_numpy(adj_np).to(DEVICE)
        p  = _to(torch.from_numpy(phylo_np) if phylo_np is not None else None, DEVICE)
        vl = _to(torch.from_numpy(vl_np)    if vl_np    is not None else None, DEVICE)
        oi = _to(torch.tensor(order_np, dtype=torch.long)
                 if order_np is not None else None, DEVICE)
        logits_on  = model(x, a, p, vl, oi).cpu().numpy()
        logits_off = model(x, a, p, vl, None).cpu().numpy()
    return logits_on, logits_off


def _normalize_sal(ism_sum_all, iso_idx, n_ism):
    """Apply the same (target − other_mean) normalization as train_gnn.py."""
    mean_all   = ism_sum_all / n_ism              # (max_len, 4, n_classes)
    target     = mean_all[:, :, iso_idx]
    other_mean = (mean_all.sum(axis=2) - target) / (N_CLASSES - 1)
    sal        = (target - other_mean).astype(np.float32)
    sal       -= sal.mean(axis=1, keepdims=True)
    return sal


# ── Determine which (iso, order) pairs to run ─────────────────────────────────
ism_isos = args.isotypes or list(classes)
ism_isos = [iso for iso in ism_isos if iso in classes]

rng = np.random.default_rng(args.seed)

pairs_run = 0
for iso in ism_isos:
    iso_idx = int(np.where(classes == iso)[0][0])

    iso_mask      = test_labels == iso_idx
    orders_in_iso = test_orders[iso_mask]
    unique_ords   = np.unique(orders_in_iso)

    for ord_int in unique_ords:
        ord_name = order_classes[ord_int]
        if args.orders and ord_name not in args.orders:
            continue

        mask   = iso_mask & (test_orders == ord_int)
        n_avail = mask.sum()
        if n_avail < args.min_seqs:
            continue

        out_on    = os.path.join(args.outdir, f"film_ism_on_{iso}_{ord_name}.npy")
        out_delta = os.path.join(args.outdir, f"film_ism_delta_{iso}_{ord_name}.npy")
        if os.path.exists(out_on) and os.path.exists(out_delta):
            print(f"[{iso} / {ord_name}] already done — skipping", flush=True)
            continue

        global_idx = pool_idx[mask]
        n_ism      = min(args.n_seqs, n_avail)
        sel        = rng.choice(n_avail, n_ism, replace=False)

        ism_seqs  = seqs[global_idx][sel]
        ism_lens  = seq_lens[global_idx][sel]
        ism_ss    = ss_arr[global_idx][sel]
        ism_adj   = np.stack([
            build_adj_matrix(str(ism_ss[i]), int(ism_lens[i]), max_len)
            for i in range(n_ism)
        ])
        ism_phylo = phylo_ohe[global_idx][sel]
        ism_vl    = (var_loop_lengths[global_idx][sel]
                     if var_loop_lengths is not None else None)
        order_fixed = np.full(1, ord_int, dtype=np.int64)

        print(f"\n[{iso} / {ord_name}]  n={n_ism} seqs …", flush=True)

        sum_on  = np.zeros((max_len, 4, N_CLASSES), dtype=np.float64)
        sum_off = np.zeros((max_len, 4, N_CLASSES), dtype=np.float64)

        for si in range(n_ism):
            seq     = ism_seqs[si]
            L       = min(int(ism_lens[si]), max_len)
            phylo_1 = np.tile(ism_phylo[si], (1, 1)).astype(np.float32)
            vl_1    = (np.array([[ism_vl[si]]], dtype=np.float32)
                       if ism_vl is not None else None)

            lo_on, lo_off = _classify(
                _embed([seq]), ism_adj[si:si+1], phylo_1, vl_1, order_fixed)
            logit_orig_on  = lo_on[0]
            logit_orig_off = lo_off[0]

            mutants, positions, nuc_idxs = [], [], []
            for pos in range(L):
                for ni, nuc in enumerate(ISM_NUCS):
                    mutants.append(seq[:pos] + nuc + seq[pos+1:])
                    positions.append(pos)
                    nuc_idxs.append(ni)

            ism_on_seq  = np.zeros((max_len, 4, N_CLASSES), dtype=np.float64)
            ism_off_seq = np.zeros((max_len, 4, N_CLASSES), dtype=np.float64)

            for start in range(0, len(mutants), args.batch_size):
                batch   = mutants[start:start + args.batch_size]
                B       = len(batch)
                emb_b   = _embed(batch)
                adj_b   = np.tile(ism_adj[si], (B, 1, 1))
                phylo_b = np.tile(ism_phylo[si], (B, 1)).astype(np.float32)
                vl_b    = (np.tile(vl_1, (B, 1)) if vl_1 is not None else None)
                order_b = np.tile(order_fixed, B)

                lg_on, lg_off = _classify(emb_b, adj_b, phylo_b, vl_b, order_b)

                for bi in range(B):
                    p_i, n_i = positions[start+bi], nuc_idxs[start+bi]
                    ism_on_seq[p_i,  n_i, :] = lg_on[bi]  - logit_orig_on
                    ism_off_seq[p_i, n_i, :] = lg_off[bi] - logit_orig_off

            sum_on  += ism_on_seq
            sum_off += ism_off_seq

            if (si+1) % 5 == 0 or si+1 == n_ism:
                print(f"  {si+1}/{n_ism}", flush=True)

        sal_on    = _normalize_sal(sum_on,  iso_idx, n_ism)
        sal_delta = _normalize_sal(sum_on - sum_off, iso_idx, n_ism)

        np.save(out_on,    sal_on)
        np.save(out_delta, sal_delta)
        print(f"  Saved → {out_on}", flush=True)
        print(f"  Saved → {out_delta}", flush=True)
        pairs_run += 1

print(f"\nDone. Ran {pairs_run} (isotype × order) pairs.", flush=True)
