#!/usr/bin/env python3
"""
compute_ism_clade.py
--------------------
Compute GNN ISM saliency maps for specific taxonomic clades (e.g., orders,
families) using the pre-trained model in --emb-dir.

Works identically to compute_ism_all.py but filters ISM sequences by clade
rather than domain.  The existing pre-trained GNN is used as-is — no
retraining.  Saliency files are named:

    gnn_ism_saliency_{iso}_{clade}.npy

which drop straight into plot_ism_logos.py.

Clade membership is resolved by joining NPZ sequences back to the CSV on
primary_sequence → --clade-col.

Background subtraction modes
-----------------------------
--clade-background
    Background = within-clade ISM (all isotypes, same clade).
    Writes to {outdir}/clade_bg/.

--background-col CLASS_COL
    Background = parent-level ISM (e.g. class="Alphaproteobacteria" when
    clade-col="order" and clade="Pelagibacterales").  For each target clade
    the parent value is inferred automatically from the clade sequences.
    Writes to {outdir}/class_bg/.
    This mode supersedes --clade-background for suppressing globally-learned
    signals while preserving clade-specific identity elements.

Usage
-----
    # Compute ISM for two orders, all isotypes
    python scripts/compute_ism_clade.py \\
        --clades Pseudomonadales Streptomycetales \\
        --clade-col order

    # Specific isotypes only
    python scripts/compute_ism_clade.py \\
        --clades Pseudomonadales \\
        --clade-col order \\
        --isotypes Ala Gly His

    # Parent-class background subtraction
    python scripts/compute_ism_clade.py \\
        --clades Pelagibacterales \\
        --clade-col order \\
        --background-col class \\
        --use-all
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir",        default="embeddings/gnn",
                    help="Directory with perpos_embeddings.npz, emb_mmap.npy")
parser.add_argument("--model-path",     default="gnn_final/gnn_model.pt",
                    help="Path to gnn_model.pt (default: gnn_final/gnn_model.pt). "
                         "Use this to run ISM with a clade-trained model from train_gnn_clade.py.")
parser.add_argument("--csv",            default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir",         default=None,
                    help="Output directory (default: {emb-dir}/clade_ism)")
parser.add_argument("--clade-col",      default="order",
                    help="CSV column to use for clade (default: order)")
parser.add_argument("--background-col", default=None,
                    help="CSV column for the parent taxonomic level used as ISM "
                         "background (e.g. 'class' when --clade-col is 'order'). "
                         "The parent value is inferred automatically from the clade "
                         "sequences.  Saves background ISM to {outdir}/class_bg/ and "
                         "writes bg-subtracted saliency there.")
parser.add_argument("--clades",         nargs="+", required=True,
                    help="Clade values to compute ISM for")
parser.add_argument("--isotypes",       nargs="+", default=None,
                    help="Isotypes to compute ISM for (default: all)")
parser.add_argument("--ism-n-seqs",     type=int, default=100)
parser.add_argument("--ism-batch-size", type=int, default=64)
parser.add_argument("--skip-existing",  action="store_true")
parser.add_argument("--clade-background", action="store_true",
                    help="[legacy] Subtract a clade-only background to suppress "
                         "globally-learned signals.  Prefer --background-col instead.")
parser.add_argument("--use-all",        action="store_true",
                    help="Use all sequences (not just test split).")
parser.add_argument("--test-size",      type=float, default=0.2)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--culturable-only", action="store_true",
                    help="Restrict ISM sequences to cultured isolates "
                         "(ncbi_genome_category == 'none' in GTDB metadata).")
parser.add_argument("--bac-metadata",
                    default="/data/kate/group_II/bac120_metadata_r226.tsv",
                    help="GTDB bac120 metadata TSV (for --culturable-only)")
parser.add_argument("--arc-metadata",
                    default="/data/conner/GII/seq_2_taxonomy/GTDB_metadata/ar53_metadata.tsv",
                    help="GTDB ar53 metadata TSV (for --culturable-only)")
parser.add_argument("--ablate-phylo", action="store_true",
                    help="Zero out phylo_ohe during ISM forward passes. "
                         "Diff against normal run isolates phylo-conditional identity elements.")
args = parser.parse_args()

outdir = args.outdir or os.path.join(args.emb_dir, "clade_ism")
os.makedirs(outdir, exist_ok=True)

# ── Hyperparameters (must match train_gnn.py exactly) ─────────────────────────
N_LAYERS   = 3
HIDDEN_DIM = 256
ISM_NUCS   = ["A", "C", "G", "T"]

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


# ── GNN architecture (must match train_gnn.py exactly) ───────────────────────
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
npz_path   = os.path.join(args.emb_dir, "perpos_embeddings.npz")
mmap_path  = os.path.join(args.emb_dir, "emb_mmap.npy")
_global_model    = "gnn_final/gnn_model.pt"
_requested_model = args.model_path
if args.model_path and not os.path.exists(args.model_path):
    print(f"WARNING: --model-path {args.model_path} not found — "
          f"falling back to global model {_global_model}", flush=True)
    model_path = _global_model
else:
    model_path = _requested_model

for path in [npz_path, mmap_path, model_path]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: required file not found: {path}")

print(f"Loading metadata from {npz_path} …", flush=True)
data     = np.load(npz_path, allow_pickle=True)
seq_lens = data["seq_lens"]
labels   = data["labels"].astype(np.int64)
classes  = data["classes"].astype(str)

seqs     = data["sequences"]
ss_arr   = data["secondary_structures"]
max_len  = int(data["max_len"])
n_classes = len(classes)

phylo_ohe = (data["phylo_ohe"].astype(np.float32) if "phylo_ohe" in data else None)
phylo_dim = int(phylo_ohe.shape[1]) if phylo_ohe is not None else 0

var_loop_lengths = (data["var_loop_lengths"].astype(np.float32) / 25.0
                    if "var_loop_lengths" in data else None)
var_loop_dim = 1 if var_loop_lengths is not None else 0

print(f"  N={len(labels):,}  max_len={max_len}  n_classes={n_classes}", flush=True)
print(f"  Classes: {list(classes)}", flush=True)

# ── Select sequences to use for ISM ──────────────────────────────────────────
if args.use_all:
    test_idx    = np.arange(len(labels))
    print(f"  Using all {len(test_idx):,} sequences (--use-all)", flush=True)
else:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size,
                                  random_state=args.seed)
    _, test_idx = next(sss.split(labels, labels))
    print(f"  Test set: {len(test_idx):,} sequences", flush=True)
test_labels = labels[test_idx]

# ── Build (SS, masked_seq) → clade / parent lookup from CSV ──────────────────
# NPZ sequences have the anticodon replaced with NNN.  We recover the NNN
# position per unique secondary structure directly from the NPZ sequences
# (avoiding any Sprinzl arithmetic), then mask the matching CSV sequences at
# those same positions to build the lookup.

print("Building SS → NNN-start map from NPZ sequences …", flush=True)
ss_to_nnn_start: dict[str, int] = {}
for s, ss in zip(seqs, ss_arr):
    s_str = str(s)
    if "NNN" in s_str:
        ss_key = str(ss)
        if ss_key not in ss_to_nnn_start:
            ss_to_nnn_start[ss_key] = s_str.index("NNN")
print(f"  {len(ss_to_nnn_start)} unique secondary structures with NNN", flush=True)

# Columns to load from CSV
_csv_cols = ["primary_sequence", "secondary_structure", args.clade_col]
if args.background_col and args.background_col not in _csv_cols:
    _csv_cols.append(args.background_col)
if args.culturable_only:
    _csv_cols.append("GenomeID")

print(f"Loading CSV for clade lookup ({', '.join(_csv_cols[2:])}) …", flush=True)
csv_df = pd.read_csv(
    args.csv,
    usecols=_csv_cols,
    low_memory=False,
).dropna(subset=[c for c in _csv_cols if c != "GenomeID"])

if args.culturable_only:
    print("  Building culturable-genome filter …", flush=True)
    meta_frames = []
    for meta_path in [args.bac_metadata, args.arc_metadata]:
        if os.path.exists(meta_path):
            meta_frames.append(pd.read_csv(
                meta_path, sep="\t",
                usecols=["accession", "ncbi_genome_category"],
                low_memory=False,
            ))
    meta_df = pd.concat(meta_frames, ignore_index=True)
    meta_df["accession_norm"] = meta_df["accession"].str.replace(r"^RS_|^GB_", "", regex=True)
    culturable_accs = set(meta_df.loc[meta_df["ncbi_genome_category"] == "none", "accession_norm"])
    csv_df["accession_norm"] = csv_df["GenomeID"].str.replace(r"_genomic$", "", regex=True)
    before = len(csv_df)
    csv_df = csv_df[csv_df["accession_norm"].isin(culturable_accs)]
    print(f"  Culturable filter: {before:,} → {len(csv_df):,} rows", flush=True)

# For each CSV row whose SS is known, mask at the NNN position and add to dicts.
masked_to_clade:  dict[str, str] = {}
masked_to_parent: dict[str, str] = {}
n_matched = 0
for ss, grp in csv_df.groupby("secondary_structure"):
    nnn_start = ss_to_nnn_start.get(str(ss))
    if nnn_start is None:
        continue
    nnn_end = nnn_start + 3
    for _, row in grp.iterrows():
        seq = row["primary_sequence"].upper()
        masked = seq[:nnn_start] + "NNN" + seq[nnn_end:]
        masked_to_clade[masked] = row[args.clade_col]
        if args.background_col:
            masked_to_parent[masked] = row[args.background_col]
        n_matched += 1

print(f"  CSV rows: {len(csv_df):,}  clade-mapped: {n_matched:,}  "
      f"unique masked seqs: {len(masked_to_clade):,}", flush=True)

# Assign clade (and optionally parent) to every NPZ test sequence
test_seqs_raw  = seqs[test_idx]
test_clades    = np.array([masked_to_clade.get(str(s).upper(), "")
                           for s in test_seqs_raw])
test_parents   = (np.array([masked_to_parent.get(str(s).upper(), "")
                             for s in test_seqs_raw])
                  if args.background_col else None)

# Summary
for clade in args.clades:
    n = (test_clades == clade).sum()
    print(f"  {clade}: {n} test sequences", flush=True)

# ── Load model ────────────────────────────────────────────────────────────────
emb     = np.load(mmap_path, mmap_mode='r')
emb_dim = emb.shape[2]

model = GNNClassifier(emb_dim, HIDDEN_DIM, N_LAYERS, 0.0,
                      n_classes, phylo_dim, var_loop_dim).to(DEVICE)
state = torch.load(model_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Model loaded from {model_path}", flush=True)

# ── Load RiNALMo ──────────────────────────────────────────────────────────────
print("Loading RiNALMo …", flush=True)
from rinalmo.pretrained import get_pretrained_model
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

rinalmo, alphabet = get_pretrained_model(model_name="giga-v1", lm_config="giga")
rinalmo.eval().to(DEVICE)
for m in rinalmo.modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.0
print("RiNALMo ready.", flush=True)

# ── Embedding + classification helpers ───────────────────────────────────────
def _embed(seqs_list):
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


def _to(t, device):
    return t.to(device) if t is not None else None


def _classify(emb_pad, adj_np, phylo_np=None, vl_np=None):
    with torch.no_grad():
        x  = torch.from_numpy(emb_pad).to(DEVICE)
        a  = torch.from_numpy(adj_np).to(DEVICE)
        p  = _to(torch.from_numpy(phylo_np) if phylo_np is not None else None, DEVICE)
        vl = _to(torch.from_numpy(vl_np)    if vl_np    is not None else None, DEVICE)
        return model(x, a, p, vl).cpu().numpy()   # (B, n_classes)


# ── Core ISM computation ──────────────────────────────────────────────────────
def compute_raw_ism(
    group_mask: np.ndarray,
    group_name: str,
    iso: str,
    rng_: np.random.Generator,
    out_raw_path: str | None = None,
) -> np.ndarray | None:
    """
    Run ISM for sequences in ``group_mask`` labelled as ``iso``.

    Returns sal_stack of shape (n_ism, max_len, 4, n_classes) or None if no seqs.
    If ``out_raw_path`` is given, saves the full per-sequence stack there.
    """
    if iso not in classes:
        print(f"  [{iso}/{group_name}] isotype not in model classes — skipping",
              flush=True)
        return None

    iso_idx = int(np.where(classes == iso)[0][0])
    mask    = group_mask & (test_labels == iso_idx)
    n_avail = int(mask.sum())
    if n_avail == 0:
        print(f"  [{iso}/{group_name}] no test sequences — skipping", flush=True)
        return None

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
    ism_phylo = (phylo_ohe[global_idx][sel]
                 if phylo_ohe is not None else None)
    ism_vl    = (var_loop_lengths[global_idx][sel]
                 if var_loop_lengths is not None else None)

    print(f"\n  [{iso}/{group_name}]  ISM: {n_ism}/{n_avail} seqs …", flush=True)

    ism_list: list[np.ndarray] = []

    for si in range(n_ism):
        seq    = str(ism_seqs[si])
        L      = min(int(ism_lens[si]), max_len)
        phylo_1 = (np.tile(ism_phylo[si], (1, 1)).astype(np.float32)
                   if ism_phylo is not None else None)
        if args.ablate_phylo and phylo_1 is not None:
            phylo_1 = np.zeros_like(phylo_1)
        vl_1    = (np.array([[ism_vl[si]]], dtype=np.float32)
                   if ism_vl is not None else None)
        logit_orig = _classify(_embed([seq]), ism_adj[si:si + 1], phylo_1, vl_1)[0]

        mutants, positions, nuc_idxs = [], [], []
        for pos in range(L):
            for ni, nuc in enumerate(ISM_NUCS):
                mutants.append(seq[:pos] + nuc + seq[pos + 1:])
                positions.append(pos)
                nuc_idxs.append(ni)

        ism_seq = np.zeros((max_len, 4, n_classes), dtype=np.float64)
        for start in range(0, len(mutants), args.ism_batch_size):
            batch   = mutants[start:start + args.ism_batch_size]
            B       = len(batch)
            emb_b   = _embed(batch)
            adj_b   = np.tile(ism_adj[si], (B, 1, 1))
            phylo_b = (np.tile(ism_phylo[si], (B, 1)).astype(np.float32)
                       if ism_phylo is not None else None)
            if args.ablate_phylo and phylo_b is not None:
                phylo_b = np.zeros_like(phylo_b)
            vl_b    = (np.tile([[ism_vl[si]]], (B, 1)).astype(np.float32)
                       if ism_vl is not None else None)
            logits  = _classify(emb_b, adj_b, phylo_b, vl_b)
            for bi in range(B):
                ism_seq[positions[start + bi],
                        nuc_idxs[start + bi], :] = logits[bi] - logit_orig

        ism_list.append(ism_seq.astype(np.float32))
        if (si + 1) % 10 == 0 or si + 1 == n_ism:
            print(f"    {si + 1}/{n_ism} done", flush=True)

    sal_stack = np.stack(ism_list)  # (n_ism, max_len, 4, n_classes)

    if out_raw_path is not None:
        np.save(out_raw_path, sal_stack)
        print(f"  Saved raw ({n_ism} seqs) → {out_raw_path}", flush=True)

    return sal_stack


def save_differential_saliency(
    mean_sal_all: np.ndarray,
    iso: str,
    out_path: str,
) -> None:
    """Differential ISM: target logit minus mean-other, then row-centre."""
    iso_idx    = int(np.where(classes == iso)[0][0])
    target_sal = mean_sal_all[:, :, iso_idx]
    other_mean = (mean_sal_all.sum(axis=2) - target_sal) / (n_classes - 1)
    sal        = (target_sal - other_mean).astype(np.float32)
    sal       -= sal.mean(axis=1, keepdims=True)
    np.save(out_path, sal)
    print(f"  Saved saliency → {out_path}", flush=True)


def save_perseq_saliency(
    sal_stack: np.ndarray,   # (n_seqs, max_len, 4, n_classes)
    iso: str,
    perseq_path: str,
    sem_path: str,
) -> None:
    """
    Per-sequence differential saliency and SEM.

    perseq_path : (n_seqs, max_len, 4)  — per-sequence differential ISM.
                  Load multiple files and concatenate to aggregate across clades
                  (e.g. all orders in a family) without re-running the model.
    sem_path    : (max_len, 4)           — std(perseq, axis=0) / sqrt(n_seqs).
                  Larger for small clades; use as an uncertainty mask in logos.
    """
    iso_idx = int(np.where(classes == iso)[0][0])
    n_seqs  = sal_stack.shape[0]
    perseq  = np.empty((n_seqs, max_len, 4), dtype=np.float32)
    for si in range(n_seqs):
        target     = sal_stack[si, :, :, iso_idx]
        other_mean = (sal_stack[si].sum(axis=2) - target) / (n_classes - 1)
        diff       = (target - other_mean).astype(np.float32)
        diff      -= diff.mean(axis=1, keepdims=True)
        perseq[si] = diff
    np.save(perseq_path, perseq)
    sem = perseq.std(axis=0) / np.sqrt(n_seqs)
    np.save(sem_path, sem)
    print(f"  Saved perseq → {perseq_path}", flush=True)
    print(f"  Saved sem    → {sem_path}    (n={n_seqs})", flush=True)


# ── ISM loop ──────────────────────────────────────────────────────────────────
# ism_isos: isotypes to compute saliency output for (may be restricted by --isotypes)
# bg_isos:  always all isotypes — background must span the full isotype distribution
#           so that mean_over_iso'(bg[iso'][pos,nuc,His_idx]) is a true "average
#           tRNA" response, not a His-specific one.
ism_isos = args.isotypes if args.isotypes else list(classes)
bg_isos  = list(classes)
rng = np.random.default_rng(args.seed)

for clade in args.clades:
    clade_mask_full = (test_clades == clade)
    n_clade_total   = clade_mask_full.sum()
    if n_clade_total == 0:
        print(f"\n[{clade}] No test sequences found — skipping", flush=True)
        continue

    # Infer parent clade (if --background-col)
    parent_clade: str | None = None
    if args.background_col and test_parents is not None:
        vals = test_parents[clade_mask_full]
        vals = vals[vals != ""]
        if len(vals):
            parent_clade = Counter(vals).most_common(1)[0][0]
        print(f"\n[{clade}] Parent {args.background_col}: {parent_clade}", flush=True)

    # ── Step 1: ISM for target clade ─────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"[{clade}] Computing target ISM …", flush=True)

    clade_raws: dict[str, np.ndarray] = {}
    for iso in ism_isos:
        out_path = os.path.join(outdir, f"gnn_ism_saliency_{iso}_{clade}.npy")
        raw_path = os.path.join(outdir, f"gnn_ism_raw_{iso}_{clade}.npy")

        perseq_path = os.path.join(outdir, f"gnn_ism_perseq_{iso}_{clade}.npy")
        sem_path    = os.path.join(outdir, f"gnn_ism_sem_{iso}_{clade}.npy")

        # Skip logic
        needs_raw   = args.clade_background or (parent_clade is not None)
        sal_done    = os.path.exists(out_path)
        raw_done    = os.path.exists(raw_path)
        perseq_done = os.path.exists(perseq_path)
        if args.skip_existing:
            all_done = sal_done and perseq_done and (raw_done if needs_raw else True)
            if all_done:
                tag = "sal+perseq" + ("+raw" if needs_raw else "")
                print(f"  [{iso}/{clade}] exists ({tag}) — skipping", flush=True)
                if raw_done:
                    raw_data = np.load(raw_path)
                    clade_raws[iso] = raw_data.mean(axis=0) if raw_data.ndim == 4 else raw_data
                continue

        save_raw = raw_path if needs_raw else None
        result = compute_raw_ism(clade_mask_full, clade, iso, rng, save_raw)
        if result is None:
            continue

        mean_sal_all    = result.mean(axis=0)
        clade_raws[iso] = mean_sal_all
        save_differential_saliency(mean_sal_all, iso, out_path)
        save_perseq_saliency(result, iso, perseq_path, sem_path)

    # ── Step 2: ISM for parent class (background) ─────────────────────────────
    if parent_clade is not None:
        bg_dir = os.path.join(outdir, "class_bg")
        os.makedirs(bg_dir, exist_ok=True)

        parent_mask_full = (test_parents == parent_clade)
        n_parent_total   = int(parent_mask_full.sum())
        print(f"\n[{clade}] Computing background ISM for {parent_clade} "
              f"({n_parent_total:,} seqs) …", flush=True)

        bg_raws: dict[str, np.ndarray] = {}
        for iso in bg_isos:
            bg_raw_path = os.path.join(
                bg_dir, f"gnn_ism_raw_bg_{iso}_{parent_clade}.npy"
            )
            if args.skip_existing and os.path.exists(bg_raw_path):
                print(f"  [{iso}/{parent_clade}] bg raw exists — skipping", flush=True)
                raw_data = np.load(bg_raw_path)
                bg_raws[iso] = raw_data.mean(axis=0) if raw_data.ndim == 4 else raw_data
                continue

            result = compute_raw_ism(
                parent_mask_full, parent_clade, iso, rng, bg_raw_path
            )
            if result is not None:
                bg_raws[iso] = result.mean(axis=0)

        # ── Step 3: Background subtraction ───────────────────────────────────
        print(f"\n[{clade}] Background subtraction using {parent_clade} …",
              flush=True)

        for iso2 in ism_isos:
            if iso2 not in clade_raws:
                print(f"  [{iso2}/{clade}] no target raw — skipping", flush=True)
                continue
            if not bg_raws:
                print(f"  [{iso2}/{clade}] no background raws — skipping", flush=True)
                continue

            iso2_idx = int(np.where(classes == iso2)[0][0])

            # target_col: mean ISM change in iso2 logit for iso2-labeled clade seqs
            target_col = clade_raws[iso2][:, :, iso2_idx]              # (max_len, 4)

            # background: mean ISM change in iso2 logit across ALL parent-class
            # isotype sequences — captures what the model responds to globally
            bg_arrays = [bg_raws[iso3][:, :, iso2_idx] for iso3 in bg_raws]
            background = np.mean(bg_arrays, axis=0)                     # (max_len, 4)

            bg_sal  = (target_col - background).astype(np.float32)
            bg_sal -= bg_sal.mean(axis=1, keepdims=True)               # row-centre

            bg_sal_path = os.path.join(
                bg_dir, f"gnn_ism_saliency_{iso2}_{clade}.npy"
            )
            np.save(bg_sal_path, bg_sal)
            print(f"  [{iso2}/{clade}] bg-subtracted → {bg_sal_path}", flush=True)

    # ── Step 2 (legacy): Clade-background subtraction ────────────────────────
    elif args.clade_background:
        cb_dir = os.path.join(outdir, "clade_bg")
        os.makedirs(cb_dir, exist_ok=True)

        # Load any missing raw files
        for iso2 in ism_isos:
            if iso2 not in clade_raws:
                rp = os.path.join(outdir, f"gnn_ism_raw_{iso2}_{clade}.npy")
                if os.path.exists(rp):
                    raw_data = np.load(rp)
                    clade_raws[iso2] = raw_data.mean(axis=0) if raw_data.ndim == 4 else raw_data

        print(f"\n[{clade}] Clade-background subtraction: "
              f"{len(clade_raws)} isotypes loaded", flush=True)

        for iso2, arr in clade_raws.items():
            iso2_idx   = int(np.where(classes == iso2)[0][0])
            target_col = arr[:, :, iso2_idx]
            background = np.mean(
                [clade_raws[iso3][:, :, iso2_idx] for iso3 in clade_raws], axis=0
            )
            bg_sal  = (target_col - background).astype(np.float32)
            bg_sal -= bg_sal.mean(axis=1, keepdims=True)
            bg_path = os.path.join(cb_dir, f"gnn_ism_saliency_{iso2}_{clade}.npy")
            np.save(bg_path, bg_sal)
            print(f"  [{iso2}/{clade}] bg-subtracted → {bg_path}", flush=True)

    # Write sentinel so run_order_ism.py can detect completion without
    # knowing which isotypes had sequences for this clade.
    open(os.path.join(outdir, f"{clade}.done"), "w").close()

print("\nDone.")
print(f"Results in: {outdir}/")
print(f"\nPlot with:")
print(f"  python scripts/plot_ism_logos.py --emb-dir {args.emb_dir} --outdir {outdir}/logos")
if args.background_col:
    print(f"\nPlot background-subtracted saliency with:")
    print(f"  python scripts/plot_ism_logos.py \\")
    print(f"    --emb-dir {args.emb_dir} \\")
    print(f"    --ism-dir {outdir}/class_bg \\")
    print(f"    --outdir {outdir}/class_bg_logos \\")
    print(f"    --clade-col {args.clade_col}")
