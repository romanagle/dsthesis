#!/usr/bin/env python3
"""
eval_film_ism_supplement.py
---------------------------
Adds ISM scores for Trp, Gln, Pro (× 2 domains) to an existing FiLM model's
gnn_results.json.  Loads the saved checkpoint, runs ISM on the same stratified
test split, computes domain-specific IE scores, and patches the JSON in-place.

Supports both:
  • Emb GCN -SS +FiLM  (gnn_film/)   — uses RiNALMo re-embedding for ISM
  • Raw GCN +SS +FiLM  (gnn_raw_film/) — uses direct one-hot substitution

Usage
-----
    # For the emb FiLM model
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_film_ism_supplement.py \\
        --results-dir  gnn_film \\
        --npz          embeddings/gnn/perpos_embeddings.npz \\
        --phylo-encoder gnn_final/phylo_encoder.pkl \\
        --isotypes     Trp Gln Pro

    # For the raw FiLM model
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_film_ism_supplement.py \\
        --results-dir  gnn_raw_film \\
        --npz          embeddings/gnn/perpos_embeddings.npz \\
        --phylo-encoder gnn_final/phylo_encoder.pkl \\
        --isotypes     Trp Gln Pro \\
        --raw
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

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir",   required=True)
parser.add_argument("--npz",           default="embeddings/gnn/perpos_embeddings.npz")
parser.add_argument("--csv",           default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--phylo-encoder", default=None)
parser.add_argument("--isotypes",      nargs="+", default=["Trp", "Gln", "Pro"])
parser.add_argument("--ism-n-seqs",     type=int,   default=100)
parser.add_argument("--ism-batch-size", type=int,   default=64)
parser.add_argument("--batch-size",     type=int,   default=256)
parser.add_argument("--test-size",      type=float, default=0.2)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--raw",            action="store_true",
                    help="Model uses raw one-hot features (no RiNALMo needed for ISM)")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ── IE definitions (acceptor branch only, domain-specific) ────────────────────
IE_DEFS = {
    "Ala": {
        "Bacteria": [
            ("3",  "G", "primary"), ("70", "U", "primary"),
            ("73", "A", "secondary"), ("2",  "G", "secondary"), ("71", "C", "secondary"),
            ("4",  "G", "weak"), ("69", "C", "weak"),
        ],
        "Archaea": [
            ("3",  "G", "primary"), ("70", "U", "primary"),
            ("73", "A", "secondary"), ("2",  "G", "secondary"), ("71", "C", "secondary"),
            ("4",  "G", "weak"), ("69", "C", "weak"),
        ],
    },
    "Trp": {
        "Bacteria": [
            ("73", "G", "primary"), ("1",  "A", "primary"), ("72", "U", "primary"),
            ("2",  "G", "secondary"), ("71", "C", "secondary"),
            ("3",  "G", "secondary"), ("70", "C", "secondary"),
            ("4",  "G", "weak"), ("69", "C", "weak"), ("5",  "G", "weak"), ("68", "C", "weak"),
        ],
        "Archaea": [
            ("73", "A", "primary"), ("1",  "G", "primary"), ("72", "C", "primary"),
            ("2",  "G", "secondary"), ("71", "C", "secondary"),
        ],
    },
    "Gly": {
        "Bacteria": [
            ("73", "U", "primary"), ("1",  "G", "primary"), ("72", "C", "primary"),
            ("2",  "C", "secondary"), ("71", "G", "secondary"),
            ("3",  "G", "secondary"), ("70", "C", "secondary"),
        ],
        "Archaea": [
            ("73", "A", "primary"),
            ("2",  "C", "secondary"), ("71", "G", "secondary"),
            ("3",  "G", "secondary"), ("70", "C", "secondary"),
        ],
    },
    "Gln": {
        "Bacteria": [
            ("73", "G", "primary"), ("1",  "U", "primary"), ("72", "A", "primary"),
            ("2",  "G", "secondary"), ("71", "C", "secondary"),
            ("3",  "G", "secondary"), ("70", "C", "secondary"),
        ],
        "Archaea": [
            ("73", "G", "primary"), ("1",  "U", "primary"), ("72", "A", "primary"),
            ("2",  "G", "secondary"), ("71", "C", "secondary"),
            ("3",  "G", "secondary"), ("70", "C", "secondary"),
        ],
    },
    "Pro": {
        "Bacteria": [
            ("73", "A", "primary"), ("72", "G", "secondary"),
        ],
        "Archaea": [
            ("73", "A", "primary"), ("1",  "G", "primary"), ("72", "C", "primary"),
            ("2",  "G", "secondary"), ("71", "C", "secondary"),
            ("3",  "G", "secondary"), ("70", "C", "secondary"),
        ],
    },
}
TIER_WEIGHTS = {"primary": 3, "secondary": 2, "weak": 1}
NUCS         = ["A", "C", "G", "U"]
NUC_IDX_MAP  = {n: i for i, n in enumerate(NUCS)}
ISM_NUCS     = ["A", "C", "G", "T"]

# ── Load existing results JSON ────────────────────────────────────────────────
results_path = os.path.join(args.results_dir, "gnn_results.json")
with open(results_path) as f:
    results = json.load(f)

existing_ie = results.get("ie_scores", {})
need_isos   = [iso for iso in args.isotypes
               if not any(k.startswith(f"{iso}_") for k in existing_ie)]
if not need_isos:
    print("All requested isotypes already present in ie_scores — nothing to do.")
    for k, v in existing_ie.items():
        print(f"  {k}: {v:.4f}")
    raise SystemExit(0)

print(f"Will compute ISM for: {need_isos}", flush=True)

film_cfg  = results.get("film", {})
N_ORDERS  = film_cfg.get("n_orders", 0)
FILM_DIM  = film_cfg.get("emb_dim", 32)
hp        = results.get("hyperparams", {})
N_LAYERS  = hp.get("n_layers", 3)
HIDDEN    = hp.get("hidden_dim", 256)
DROPOUT   = hp.get("dropout", 0.3)
N_CLASSES = results["n_classes"]

# ── Load NPZ ──────────────────────────────────────────────────────────────────
print(f"Loading {args.npz} …", flush=True)
data      = np.load(args.npz, allow_pickle=True)
seq_lens  = data["seq_lens"]
labels    = data["labels"].astype(np.int64)
classes   = data["classes"].astype(str)
seqs      = data["sequences"]
ss_arr    = data["secondary_structures"]
max_len   = int(data["max_len"])
domains        = (data["domains"].astype(str) if "domains" in data
                  else np.full(len(labels), "Unknown", dtype=object))
unique_domains = sorted(set(domains))

phylo_ohe = data["phylo_ohe"].astype(np.float32) if "phylo_ohe" in data else None
PHYLO_DIM = int(phylo_ohe.shape[1]) if phylo_ohe is not None else 0

var_loop_lengths = (data["var_loop_lengths"].astype(np.float32) / 25.0
                    if "var_loop_lengths" in data else None)
VAR_LOOP_DIM = 1 if var_loop_lengths is not None else 0

# Derive order labels for FiLM
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
        order_labels = (phylo_ohe[:, _order_start:_order_end]
                        .argmax(axis=1).astype(np.int64))
        print(f"  Order labels loaded ({_n_cats[3]} orders)", flush=True)
    else:
        print("  Warning: phylo_encoder.pkl not found — FiLM will use order_idx=None",
              flush=True)

# Load mmap (only needed for emb model)
EMB_DIM = 4 if args.raw else None
emb = None
if not args.raw:
    mmap_path = os.path.join(os.path.dirname(args.npz), "emb_mmap.npy")
    print(f"Memory-mapping {mmap_path} …", flush=True)
    emb     = np.load(mmap_path, mmap_mode="r")
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
            DenseGCNLayer(dims[i], dims[i + 1], dropout) for i in range(n_layers)
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
        if self.phylo_dim > 0 and phylo is not None: parts.append(phylo)
        if self.var_loop_dim > 0 and var_loop is not None: parts.append(var_loop)
        return self.head(torch.cat(parts, dim=1))


model = GNNClassifier(EMB_DIM, HIDDEN, N_LAYERS, DROPOUT, N_CLASSES,
                      PHYLO_DIM, VAR_LOOP_DIM,
                      film_n_orders=N_ORDERS, film_emb_dim=FILM_DIM).to(DEVICE)
ckpt_path = os.path.join(args.results_dir, "gnn_model.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout): m.p = 0.0
print(f"Loaded checkpoint: {ckpt_path}", flush=True)

# ── Reproduce the same test split ─────────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
_, test_idx = next(sss.split(labels, labels))
test_labels  = labels[test_idx]
test_domains = domains[test_idx]

# ── Sprinzl utilities ─────────────────────────────────────────────────────────
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


# Lazy RiNALMo loader
_rinalmo = _alphabet = None

def _load_rinalmo():
    global _rinalmo, _alphabet
    if _rinalmo is None:
        print("Loading RiNALMo …", flush=True)
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


def _classify(x_np, adj_np, phylo_np=None, vl_np=None, order_np=None):
    def _t(a): return torch.from_numpy(a).to(DEVICE) if a is not None else None
    with torch.no_grad():
        return model(
            _t(x_np), _t(adj_np), _t(phylo_np), _t(vl_np),
            torch.tensor(order_np, dtype=torch.long).to(DEVICE)
            if order_np is not None else None,
        ).cpu().numpy()


# ── Run ISM for missing isotypes ──────────────────────────────────────────────
new_ie_scores = dict(existing_ie)
n_classes = N_CLASSES

for iso in need_isos:
    if iso not in classes:
        print(f"[{iso}] not in model classes — skipping", flush=True)
        continue
    iso_idx = int(np.where(classes == iso)[0][0])

    for dom in unique_domains:
        mask    = (test_labels == iso_idx) & (test_domains == dom)
        n_avail = mask.sum()
        if n_avail == 0:
            print(f"\n[{iso} / {dom}] No test sequences — skipping", flush=True)
            continue
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
        ism_adj   = np.stack([build_adj_matrix(str(ism_ss[i]), int(ism_lens[i]), max_len)
                               for i in range(n_ism)])
        ism_phylo = (phylo_ohe[global_idx][sel] if phylo_ohe is not None else None)
        ism_vl    = (var_loop_lengths[global_idx][sel] if var_loop_lengths is not None else None)
        ism_order = (order_labels[global_idx][sel] if order_labels is not None else None)

        print(f"\n[{iso} / {dom}]  ISM: {n_ism} seqs {'(one-hot)' if args.raw else '(RiNALMo)'} …",
              flush=True)
        ism_sum = np.zeros((max_len, 4, n_classes), dtype=np.float64)

        for si in range(n_ism):
            seq = str(ism_seqs[si])
            L   = min(int(ism_lens[si]), max_len)

            phylo_1 = (np.tile(ism_phylo[si], (1, 1)).astype(np.float32)
                       if ism_phylo is not None else None)
            vl_1    = (np.array([[ism_vl[si]]], dtype=np.float32)
                       if ism_vl is not None else None)
            order_1 = (np.array([ism_order[si]], dtype=np.int64)
                       if ism_order is not None else None)

            if args.raw:
                x0 = seq_to_onehot(seq, max_len)
            else:
                x0 = _embed_rinalmo([seq])[0]
            logit_orig = _classify(x0[None], ism_adj[si:si + 1],
                                   phylo_1, vl_1, order_1)[0]

            mutant_x, positions, nuc_idxs = [], [], []
            for pos in range(L):
                for ni, nuc in enumerate(ISM_NUCS):
                    if args.raw:
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
                end  = start + args.ism_batch_size
                B    = len(positions[start:end])
                if args.raw:
                    xb_mut = np.stack(mutant_x[start:end])
                else:
                    xb_mut = _embed_rinalmo(mutant_x[start:end])
                adj_b    = np.tile(ism_adj[si], (B, 1, 1))
                phylo_b  = (np.tile(ism_phylo[si], (B, 1)).astype(np.float32)
                            if ism_phylo is not None else None)
                vl_b     = (np.tile(vl_1, (B, 1)) if vl_1 is not None else None)
                order_b  = (np.tile(order_1, B)   if order_1 is not None else None)
                logits   = _classify(xb_mut, adj_b, phylo_b, vl_b, order_b)
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

        sal_path = os.path.join(args.results_dir, f"gnn_ism_saliency_{iso}_{dom}.npy")
        np.save(sal_path, mean_sal)
        print(f"  Saved → {sal_path}", flush=True)

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
        new_ie_scores[f"{iso}_{dom}"] = round(float(ie_score), 2)
        print(f"  IE score ({dom}): {ie_score:.1f}/100", flush=True)

# ── Patch results JSON in-place ───────────────────────────────────────────────
results["ie_scores"] = new_ie_scores
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nPatched → {results_path}")
print("All IE scores:", new_ie_scores)
