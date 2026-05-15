#!/usr/bin/env python3
"""
compute_pairwise_ism.py
-----------------------
Pairwise saturated mutagenesis restricted to Watson-Crick stem positions.

For every stem pair (i,j) from the secondary structure, all 16 double-mutant
sequences are embedded and classified.  Results are accumulated per Sprinzl
pair label, giving a canonical coordinate system across sequences of varying
length within a group.

Outputs per (iso, clade):
  pairwise_ism_{iso}_{clade}.npz
      sal_raw     float32 (n_pairs, 4, 4, n_classes)  mean Δlogit per double-mutant
      sal_disc    float32 (n_pairs, 4, 4)              differential saliency (target −
                                                        mean-other), centred over all 16
                                                        base-pair combinations per pair
      sprinzl_5   str (n_pairs,)   Sprinzl label of 5′ position
      sprinzl_3   str (n_pairs,)   Sprinzl label of 3′ position
      raw_pos_5   int (n_pairs,)   raw position of 5′ partner (modal SS)
      raw_pos_3   int (n_pairs,)   raw position of 3′ partner (modal SS)
      classes     str (n_classes,)
      iso_idx     int32

  gnn_ism_saliency_{iso}_{clade}.npy
      float32 (max_len, 4)  per-position marginal saliency compatible with
      plot_ism_logos.py.  Pass --no-cross-iso-norm to that script since these
      files already carry differential normalisation.

Usage:
    python dsthesis/scripts/analysis/compute_pairwise_ism.py \\
        --clades Bacteria Archaea --clade-col domain
    python dsthesis/scripts/analysis/compute_pairwise_ism.py \\
        --clades Pseudomonadales --isotypes Ala Gly
    python dsthesis/scripts/analysis/compute_pairwise_ism.py \\
        --clades Pseudomonadales --skip-existing --ism-n-seqs 50
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir",        default="embeddings/gnn")
parser.add_argument("--model-path",     default="gnn_final/gnn_model.pt",
                    help="Path to gnn_model.pt (default: gnn_final/gnn_model.pt)")
parser.add_argument("--csv",            default="data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir",         default=None,
                    help="Output directory (default: {emb-dir}/pairwise_ism)")
parser.add_argument("--clade-col",      default="order",
                    help="CSV column to filter by (default: order; use 'domain' for Bacteria/Archaea)")
parser.add_argument("--clades",         nargs="+", required=True,
                    help="Clade values to compute pairwise ISM for")
parser.add_argument("--isotypes",       nargs="+", default=None,
                    help="Isotypes to compute (default: all)")
parser.add_argument("--ism-n-seqs",     type=int, default=100)
parser.add_argument("--ism-batch-size", type=int, default=64)
parser.add_argument("--skip-existing",  action="store_true")
parser.add_argument("--use-all",        action="store_true",
                    help="Use all sequences (not just test split)")
parser.add_argument("--test-size",      type=float, default=0.2)
parser.add_argument("--seed",           type=int,   default=42)
args = parser.parse_args()

outdir = args.outdir or os.path.join(args.emb_dir, "pairwise_ism")
os.makedirs(outdir, exist_ok=True)

# ── Hyperparameters (must match train_gnn.py) ─────────────────────────────────
N_LAYERS   = 3
HIDDEN_DIM = 256
ISM_NUCS   = ["A", "C", "G", "T"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


# ── Secondary structure utilities ─────────────────────────────────────────────
def parse_pairs(ss: str) -> dict[int, int]:
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == '>': stack.append(i)
        elif c == '<':
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss: str) -> dict[int, str]:
    """Map raw sequence positions to canonical Sprinzl position labels."""
    pairs = parse_pairs(ss)
    n = len(ss)
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
        return {}
    sp = {}
    acc5 = stems[0]
    acc3 = sorted(pairs[p] for p in acc5)
    overhang = [p for p in range(acc5[0]) if p not in pairs]
    for idx, p in enumerate(reversed(overhang)):
        sp[p] = str(-(idx + 1))
    for idx, p in enumerate(acc5): sp[p] = str(idx + 1)
    for idx, p in enumerate(acc3): sp[p] = str(66 + idx)
    disc = sorted(p for p in range(acc3[-1] + 1, n) if p not in pairs)
    for idx, p in enumerate(disc): sp[p] = str(73 + idx)
    d5 = stems[1]
    for idx, p in enumerate([p for p in range(acc5[-1] + 1, d5[0]) if p not in pairs]):
        sp[p] = str(8 + idx)
    d3 = sorted(pairs[p] for p in d5)
    nd = len(d5)
    for idx, p in enumerate(d5): sp[p] = str(10 + idx)
    for idx, p in enumerate(d3): sp[p] = str(25 - nd + 1 + idx)
    d_loop = [p for p in range(d5[-1] + 1, d3[0]) if p not in pairs]
    for idx, p in enumerate(d_loop):
        labs = ["14", "15", "16", "17", "17a", "18", "19", "20", "20a", "20b", "21"]
        if idx < len(labs): sp[p] = labs[idx]
    ac5 = stems[2]
    lk2 = [p for p in range(d3[-1] + 1, ac5[0]) if p not in pairs]
    if lk2: sp[lk2[-1]] = "26"
    ac3 = sorted(pairs[p] for p in ac5)
    for idx, p in enumerate(ac5): sp[p] = str(27 + idx)
    for idx, p in enumerate(ac3): sp[p] = str(39 + idx)
    for idx, p in enumerate([p for p in range(ac5[-1] + 1, ac3[0]) if p not in pairs]):
        sp[p] = str(32 + idx)
    t5 = stems[-1]
    t3 = sorted(pairs[p] for p in t5)
    for idx, p in enumerate(t5): sp[p] = str(49 + idx)
    for idx, p in enumerate(t3): sp[p] = str(61 + idx)
    for idx, p in enumerate([p for p in range(t5[-1] + 1, t3[0]) if p not in pairs]):
        sp[p] = str(54 + idx)
    var = [p for p in range(ac3[-1] + 1, t5[0])]
    if len(stems) == 5:
        for idx, p in enumerate(var): sp[p] = f"e{idx + 1}"
    else:
        for idx, p in enumerate([p for p in var if p not in pairs]): sp[p] = str(44 + idx)
    return sp


VAR_LOOP_THRESHOLD = 5


def detect_variable_loop(ss: str) -> tuple[int, int]:
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


def build_adj_matrix(ss: str, seq_len: int, max_len: int) -> np.ndarray:
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
    d = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (A * d[:, None]) * d[None, :]


# ── GNN architecture (must match train_gnn.py) ────────────────────────────────
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
            DenseGCNLayer(dims[i], dims[i + 1], dropout) for i in range(n_layers)
        ])
        self.phylo_dim    = phylo_dim
        self.var_loop_dim = var_loop_dim
        self.head = nn.Linear(hidden_dim + phylo_dim + var_loop_dim, n_classes)

    def forward(self, x, adj, phylo=None, var_loop=None):
        for layer in self.layers:
            x = layer(x, adj)
        pooled = x.mean(dim=1)
        parts = [pooled]
        if self.phylo_dim > 0 and phylo is not None: parts.append(phylo)
        if self.var_loop_dim > 0 and var_loop is not None: parts.append(var_loop)
        return self.head(torch.cat(parts, dim=1))


# ── Load NPZ metadata ─────────────────────────────────────────────────────────
npz_path   = os.path.join(args.emb_dir, "perpos_embeddings.npz")
mmap_path  = os.path.join(args.emb_dir, "emb_mmap.npy")
model_path = args.model_path

for path in [npz_path, mmap_path, model_path]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: required file not found: {path}")

print(f"Loading metadata from {npz_path} …", flush=True)
data     = np.load(npz_path, allow_pickle=True)
seq_lens = data["seq_lens"]
labels   = data["labels"].astype(np.int64)
classes  = data["classes"].astype(str)

seqs      = data["sequences"]
ss_arr    = data["secondary_structures"]
max_len   = int(data["max_len"])
n_classes = len(classes)   # use the original class count to match the checkpoint

phylo_ohe = (data["phylo_ohe"].astype(np.float32) if "phylo_ohe" in data else None)
phylo_dim = int(phylo_ohe.shape[1]) if phylo_ohe is not None else 0
var_loop_lengths = (data["var_loop_lengths"].astype(np.float32) / 25.0
                    if "var_loop_lengths" in data else None)
var_loop_dim = 1 if var_loop_lengths is not None else 0

print(f"  N={len(labels):,}  max_len={max_len}  n_classes={n_classes}", flush=True)

# ── Test split ────────────────────────────────────────────────────────────────
if args.use_all:
    test_idx = np.arange(len(labels))
    print(f"  Using all {len(test_idx):,} sequences (--use-all)", flush=True)
else:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    _, test_idx = next(sss.split(labels, labels))
    print(f"  Test set: {len(test_idx):,} sequences", flush=True)
test_labels = labels[test_idx]

# ── Build clade lookup (anticodon-masked sequences → clade label) ─────────────
print("Building clade lookup …", flush=True)
ss_to_nnn_start: dict[str, int] = {}
for s, ss in zip(seqs, ss_arr):
    s_str = str(s)
    if "NNN" in s_str:
        ss_key = str(ss)
        if ss_key not in ss_to_nnn_start:
            ss_to_nnn_start[ss_key] = s_str.index("NNN")

csv_df = pd.read_csv(
    args.csv,
    usecols=["primary_sequence", "secondary_structure", args.clade_col],
    low_memory=False,
).dropna(subset=["primary_sequence", "secondary_structure", args.clade_col])

masked_to_clade: dict[str, str] = {}
for ss_str, grp in csv_df.groupby("secondary_structure"):
    nnn_start = ss_to_nnn_start.get(str(ss_str))
    if nnn_start is None:
        continue
    for _, row in grp.iterrows():
        seq_raw = row["primary_sequence"].upper()
        masked  = seq_raw[:nnn_start] + "NNN" + seq_raw[nnn_start + 3:]
        masked_to_clade[masked] = row[args.clade_col]

test_seqs_raw = seqs[test_idx]
test_clades   = np.array([masked_to_clade.get(str(s).upper(), "") for s in test_seqs_raw])

for clade in args.clades:
    print(f"  {clade}: {(test_clades == clade).sum()} test sequences", flush=True)

# ── Load model ────────────────────────────────────────────────────────────────
emb     = np.load(mmap_path, mmap_mode='r')
emb_dim = emb.shape[2]

model = GNNClassifier(emb_dim, HIDDEN_DIM, N_LAYERS, 0.0,
                      n_classes, phylo_dim, var_loop_dim).to(DEVICE)
state = torch.load(model_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Model loaded from {model_path}", flush=True)

print("Loading RiNALMo …", flush=True)
from rinalmo.pretrained import get_pretrained_model
rinalmo, alphabet = get_pretrained_model(model_name="giga-v1", lm_config="giga")
rinalmo.eval().to(DEVICE)
for m in rinalmo.modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.0
print("RiNALMo ready.", flush=True)


# ── Inference helpers ─────────────────────────────────────────────────────────
def _embed(seqs_list: list[str]) -> np.ndarray:
    toks = torch.tensor(
        alphabet.batch_tokenize(seqs_list), dtype=torch.int64, device=DEVICE
    )
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = rinalmo(toks)
    emb_out = out["representation"].float().cpu().numpy()
    pad = np.zeros((len(seqs_list), max_len, emb_dim), dtype=np.float32)
    for bi, s in enumerate(seqs_list):
        Ls = min(len(s), max_len)
        pad[bi, :Ls] = emb_out[bi, 1:Ls + 1]
    return pad


def _classify(emb_pad, adj_np, phylo_np=None, vl_np=None) -> np.ndarray:
    with torch.no_grad():
        x  = torch.from_numpy(emb_pad).to(DEVICE)
        a  = torch.from_numpy(adj_np).to(DEVICE)
        p  = torch.from_numpy(phylo_np).to(DEVICE) if phylo_np is not None else None
        vl = torch.from_numpy(vl_np).to(DEVICE)    if vl_np    is not None else None
        return model(x, a, p, vl).cpu().numpy()


# ── Sprinzl sort key (for output ordering) ───────────────────────────────────
def _sp_sort_key(label: str) -> tuple:
    if label.startswith("-"):
        try: return (-1, -int(label[1:]))
        except ValueError: return (-1, 0)
    if label.startswith("e"):
        try: return (100, int(label[1:]))
        except ValueError: return (100, 0)
    base = label.rstrip("ab")
    try: return (int(base), 0 if not label[len(base):] else ord(label[len(base)]))
    except ValueError: return (50, 0)


# ── Core pairwise ISM ─────────────────────────────────────────────────────────
def compute_pairwise_ism(
    group_mask: np.ndarray,
    group_name: str,
    iso: str,
    rng_: np.random.Generator,
) -> tuple[dict | None, dict | None]:
    """
    Run pairwise ISM at stem positions for sequences in group_mask labelled as iso.

    Returns:
      pair_sal_raw : dict (sp_i, sp_j) → float32 (4, 4, n_classes)  mean Δlogit
      pair_raw_pos : dict (sp_i, sp_j) → (raw_i, raw_j)  from first matching sequence
    Both are None if no sequences are available.
    """
    if iso not in classes:
        print(f"  [{iso}/{group_name}] isotype not in model classes — skipping",
              flush=True)
        return None, None

    iso_idx = int(np.where(classes == iso)[0][0])
    mask    = group_mask & (test_labels == iso_idx)
    n_avail = int(mask.sum())
    if n_avail == 0:
        print(f"  [{iso}/{group_name}] no test sequences — skipping", flush=True)
        return None, None

    global_idx = test_idx[mask]
    n_ism      = min(args.ism_n_seqs, n_avail)
    sel        = rng_.choice(n_avail, n_ism, replace=False)

    ism_seqs  = seqs[global_idx][sel]
    ism_lens  = seq_lens[global_idx][sel]
    ism_ss    = ss_arr[global_idx][sel]
    ism_adj   = np.stack([
        build_adj_matrix(str(ism_ss[i]), int(ism_lens[i]), max_len)
        for i in range(n_ism)
    ])
    ism_phylo = (phylo_ohe[global_idx][sel] if phylo_ohe is not None else None)
    ism_vl    = (var_loop_lengths[global_idx][sel] if var_loop_lengths is not None else None)

    print(f"\n  [{iso}/{group_name}]  pairwise ISM: {n_ism}/{n_avail} seqs …",
          flush=True)

    # Accumulate by Sprinzl pair key for canonical cross-sequence aggregation
    pair_sum: dict[tuple[str, str], np.ndarray]    = defaultdict(
        lambda: np.zeros((4, 4, n_classes), dtype=np.float64)
    )
    pair_count: dict[tuple[str, str], int]          = defaultdict(int)
    pair_raw_pos: dict[tuple[str, str], tuple[int, int]] = {}

    for si in range(n_ism):
        seq = str(ism_seqs[si])
        L   = min(int(ism_lens[si]), max_len)
        ss  = str(ism_ss[si])

        sp_map     = build_sprinzl_map(ss)   # raw_pos → Sprinzl label
        pairs_dict = parse_pairs(ss)

        # Stem pairs: i < j, both within sequence length, both with Sprinzl labels
        stem_pairs = [
            (i, pairs_dict[i])
            for i in pairs_dict
            if i < pairs_dict[i]
            and i < L and pairs_dict[i] < L
            and i < max_len and pairs_dict[i] < max_len
            and i in sp_map and pairs_dict[i] in sp_map
        ]

        if not stem_pairs:
            print(f"    seq {si}: no valid stem pairs — skipping", flush=True)
            continue

        phylo_1 = (np.tile(ism_phylo[si], (1, 1)).astype(np.float32)
                   if ism_phylo is not None else None)
        vl_1    = (np.array([[ism_vl[si]]], dtype=np.float32)
                   if ism_vl is not None else None)
        logit_orig = _classify(_embed([seq]), ism_adj[si:si + 1], phylo_1, vl_1)[0]

        # Build all 16 double-mutant sequences for every stem pair
        mutants: list[str]                          = []
        meta: list[tuple[tuple[str, str], int, int]] = []

        for (raw_i, raw_j) in stem_pairs:
            sp_key = (sp_map[raw_i], sp_map[raw_j])
            if sp_key not in pair_raw_pos:
                pair_raw_pos[sp_key] = (raw_i, raw_j)
            for ni in range(4):
                for nj in range(4):
                    # Substitute nucleotide ni at raw_i and nj at raw_j simultaneously
                    # (raw_i < raw_j is guaranteed by the stem_pairs construction)
                    mut = (seq[:raw_i] + ISM_NUCS[ni]
                           + seq[raw_i + 1:raw_j] + ISM_NUCS[nj]
                           + seq[raw_j + 1:])
                    mutants.append(mut)
                    meta.append((sp_key, ni, nj))

        # Classify in batches
        for start in range(0, len(mutants), args.ism_batch_size):
            batch  = mutants[start:start + args.ism_batch_size]
            B      = len(batch)
            emb_b  = _embed(batch)
            adj_b  = np.tile(ism_adj[si], (B, 1, 1))
            phylo_b = (np.tile(ism_phylo[si], (B, 1)).astype(np.float32)
                       if ism_phylo is not None else None)
            vl_b    = (np.tile([[ism_vl[si]]], (B, 1)).astype(np.float32)
                       if ism_vl is not None else None)
            logits  = _classify(emb_b, adj_b, phylo_b, vl_b)
            for bi in range(B):
                sp_key, ni, nj = meta[start + bi]
                pair_sum[sp_key][ni, nj] += logits[bi] - logit_orig

        for (raw_i, raw_j) in stem_pairs:
            pair_count[(sp_map[raw_i], sp_map[raw_j])] += 1

        if (si + 1) % 10 == 0 or si + 1 == n_ism:
            print(f"    {si + 1}/{n_ism} done", flush=True)

    # Average over sequences
    pair_sal_raw = {
        k: (v / pair_count[k]).astype(np.float32)
        for k, v in pair_sum.items()
        if pair_count[k] > 0
    }
    return pair_sal_raw, pair_raw_pos


# ── Save outputs ──────────────────────────────────────────────────────────────
def save_pairwise_outputs(
    pair_sal_raw: dict,
    pair_raw_pos: dict,
    iso: str,
    clade: str,
) -> None:
    iso_idx = int(np.where(classes == iso)[0][0])

    # Order pairs by ascending Sprinzl position of the 5′ partner
    sorted_keys = sorted(
        pair_sal_raw.keys(),
        key=lambda k: (_sp_sort_key(k[0]), _sp_sort_key(k[1]))
    )
    n_pairs = len(sorted_keys)
    if n_pairs == 0:
        print(f"  [{iso}/{clade}] no pairs accumulated — nothing to save", flush=True)
        return

    sal_raw = np.stack([pair_sal_raw[k] for k in sorted_keys])  # (n_pairs, 4, 4, n_classes)
    sp5_arr = np.array([k[0] for k in sorted_keys])
    sp3_arr = np.array([k[1] for k in sorted_keys])
    rp5_arr = np.array([pair_raw_pos[k][0] for k in sorted_keys])
    rp3_arr = np.array([pair_raw_pos[k][1] for k in sorted_keys])

    # Differential saliency: target logit column minus mean of all other isotypes,
    # then centre over the 16 base-pair combinations (analogous to row-centering
    # in single-position ISM).
    target_sal = sal_raw[:, :, :, iso_idx]                              # (n_pairs, 4, 4)
    other_mean = (sal_raw.sum(axis=3) - target_sal) / (n_classes - 1)  # (n_pairs, 4, 4)
    sal_disc   = (target_sal - other_mean).astype(np.float32)
    sal_disc  -= sal_disc.mean(axis=(1, 2), keepdims=True)              # centre per pair

    npz_path = os.path.join(outdir, f"pairwise_ism_{iso}_{clade}.npz")
    np.savez(
        npz_path,
        sal_raw   = sal_raw,
        sal_disc  = sal_disc,
        sprinzl_5 = sp5_arr,
        sprinzl_3 = sp3_arr,
        raw_pos_5 = rp5_arr,
        raw_pos_3 = rp3_arr,
        classes   = classes,
        iso_idx   = np.int32(iso_idx),
    )
    print(f"  Saved pairwise npz  → {npz_path}", flush=True)

    # ── Per-position marginal saliency for plot_ism_logos.py compatibility ────
    # For each stem position, marginalise over the partner nucleotide.
    # Position i (5′): sal[i, ni] = mean_{nj} sal_disc[pair, ni, nj]
    # Position j (3′): sal[j, nj] = mean_{ni} sal_disc[pair, ni, nj]
    sal_perpos = np.zeros((max_len, 4), dtype=np.float32)
    for k_idx, k in enumerate(sorted_keys):
        raw_i, raw_j = pair_raw_pos[k]
        if raw_i < max_len:
            sal_perpos[raw_i] = sal_disc[k_idx].mean(axis=1)   # mean over nj
        if raw_j < max_len:
            sal_perpos[raw_j] = sal_disc[k_idx].mean(axis=0)   # mean over ni
    sal_perpos -= sal_perpos.mean(axis=1, keepdims=True)        # row-centre

    npy_path = os.path.join(outdir, f"gnn_ism_saliency_{iso}_{clade}.npy")
    np.save(npy_path, sal_perpos)
    print(f"  Saved per-pos .npy  → {npy_path}", flush=True)


# ── Main loop ─────────────────────────────────────────────────────────────────
ism_isos = args.isotypes if args.isotypes else list(classes)
rng = np.random.default_rng(args.seed)

for clade in args.clades:
    clade_mask = (test_clades == clade)
    if clade_mask.sum() == 0:
        print(f"\n[{clade}] no test sequences found — skipping", flush=True)
        continue

    print(f"\n{'=' * 60}\n[{clade}] Computing pairwise ISM …", flush=True)

    for iso in ism_isos:
        npz_out = os.path.join(outdir, f"pairwise_ism_{iso}_{clade}.npz")
        if args.skip_existing and os.path.exists(npz_out):
            print(f"  [{iso}/{clade}] exists — skipping", flush=True)
            continue

        pair_sal_raw, pair_raw_pos = compute_pairwise_ism(
            clade_mask, clade, iso, rng
        )
        if pair_sal_raw is None:
            continue

        save_pairwise_outputs(pair_sal_raw, pair_raw_pos, iso, clade)

print("\nDone.")
print(f"Results in: {outdir}/")
print()
print("Visualise pairwise logos:")
print(f"  python dsthesis/scripts/figures/plot_pairwise_ism_logos.py --ism-dir {outdir}")
print()
print("Visualise standard per-position logos (uses the .npy marginal files):")
print(f"  python dsthesis/scripts/figures/plot_ism_logos.py \\")
print(f"    --emb-dir {args.emb_dir} --ism-dir {outdir} --no-cross-iso-norm")
