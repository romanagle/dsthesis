#!/usr/bin/env python3
"""
compute_ism_all.py
------------------
Standalone ISM computation for all remaining tRNA isotypes using the
pre-trained GNN in gnn_final/.  Distributes work across all
available GPUs (one RiNALMo + GNN copy per GPU).

ISM delta calculation (identical to train_gnn.py):
  1.  For each (iso, dom) group, draw up to --ism-n-seqs test sequences.
  2.  Mutate every position to each of A/C/G/T; embed with RiNALMo; classify.
  3.  raw Δlogit = mutant_logit - wildtype_logit  (shape max_len × 4 × n_classes)
  4.  Average over sequences → mean_sal_all
  5.  Differential: target - mean(others), then row-centre (subtract mean over 4
      nucleotides at each position).

Ile2 → Ile:
  Labels are merged in each worker so the ISM for Ile is computed over its
  combined sequence pool.  After all workers finish, any pre-existing Ile2_*
  maps are averaged with the Ile maps and written back as the parent file.

Usage:
    python scripts/compute_ism_all.py
    python scripts/compute_ism_all.py --skip-existing
    python scripts/compute_ism_all.py --ism-n-seqs 50 --outdir gnn_final
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir",        default="embeddings/gnn",
                    help="Directory with perpos_embeddings.npz, emb_mmap.npy")
parser.add_argument("--model-path",     default="gnn_final/gnn_model.pt",
                    help="Path to gnn_model.pt (default: gnn_final/gnn_model.pt)")
parser.add_argument("--csv",            default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir",         default=None,
                    help="Output directory (default: same as --emb-dir)")
parser.add_argument("--ism-n-seqs",     type=int, default=100,
                    help="Sequences per (isotype × domain) for ISM (default: 100)")
parser.add_argument("--ism-batch-size", type=int, default=64)
parser.add_argument("--skip-existing",  action="store_true",
                    help="Skip (iso, dom) if the .npy file already exists")
parser.add_argument("--test-size",      type=float, default=0.2)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--already-done",   nargs="+",  default=["Ala", "Gly"],
                    help="Isotypes already computed; skip them (default: Ala Gly)")
parser.add_argument("--extra-skip",     nargs="+",  default=[],
                    help="Additional isotypes to skip (e.g. Ser Leu)")
args = parser.parse_args()

outdir = args.outdir or args.emb_dir
os.makedirs(outdir, exist_ok=True)

# ── Hyperparameters (must match train_gnn.py exactly) ─────────────────────────
N_LAYERS   = 3
HIDDEN_DIM = 256
DROPOUT    = 0.3          # dropout is zeroed in eval mode anyway
ISM_NUCS   = ["A", "C", "G", "T"]

# ── Graph / Sprinzl utilities (copied from train_gnn.py) ─────────────────────
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


# ── Per-GPU worker ────────────────────────────────────────────────────────────
def run_worker(rank: int, per_gpu_tasks: list[list[tuple]], cfg: dict):
    """Each GPU process handles its assigned (iso, dom) task pairs."""
    tasks = per_gpu_tasks[rank]
    if not tasks:
        return

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # ── Load metadata ─────────────────────────────────────────────────────────
    data      = np.load(cfg["npz_path"], allow_pickle=True)
    seq_lens  = data["seq_lens"]
    labels    = data["labels"].astype(np.int64)
    classes   = data["classes"].astype(str)

    seqs      = data["sequences"]
    ss_arr    = data["secondary_structures"]
    max_len   = int(data["max_len"])
    n_classes = len(classes)
    domains   = (data["domains"].astype(str) if "domains" in data
                 else np.full(len(labels), "Unknown", dtype=object))

    phylo_ohe = (data["phylo_ohe"].astype(np.float32)
                 if "phylo_ohe" in data else None)
    phylo_dim = int(phylo_ohe.shape[1]) if phylo_ohe is not None else 0

    var_loop_lengths = (data["var_loop_lengths"].astype(np.float32) / 25.0
                        if "var_loop_lengths" in data else None)
    var_loop_dim = 1 if var_loop_lengths is not None else 0

    # memory-mapped embeddings
    emb     = np.load(cfg["mmap_path"], mmap_mode='r')
    emb_dim = emb.shape[2]

    # Reproduce the same train/test split as train_gnn.py
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg["test_size"], random_state=cfg["seed"]
    )
    _, test_idx = next(sss.split(labels, labels))
    test_labels  = labels[test_idx]
    test_domains = domains[test_idx]

    # ── Load GNN model ────────────────────────────────────────────────────────
    model = GNNClassifier(
        emb_dim, HIDDEN_DIM, N_LAYERS, 0.0, n_classes, phylo_dim, var_loop_dim
    ).to(device)
    state = torch.load(cfg["model_path"], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # ── Load RiNALMo ──────────────────────────────────────────────────────────
    print(f"[GPU {rank}] Loading RiNALMo …", flush=True)
    from rinalmo.pretrained import get_pretrained_model
    DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

    rinalmo, alphabet = get_pretrained_model(
        model_name="giga-v1", lm_config="giga"
    )
    rinalmo.eval().to(device)
    for m in rinalmo.modules():
       if isinstance(m, nn.Dropout):
            m.p = 0.0
    print(f"[GPU {rank}] RiNALMo ready.  Tasks: {[f'{i}/{d}' for i,d in tasks]}",
          flush=True)

    # ── Embedding + classification helpers ───────────────────────────────────
    def _embed(seqs_list):
        toks = torch.tensor(
            alphabet.batch_tokenize(seqs_list), dtype=torch.int64, device=device
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

    def _classify(emb_pad, adj_np, phylo_np=None, vl_np=None):
        with torch.no_grad():
            x  = torch.from_numpy(emb_pad).to(device)
            a  = torch.from_numpy(adj_np).to(device)
            p  = (torch.from_numpy(phylo_np).to(device) if phylo_np is not None else None)
            vl = (torch.from_numpy(vl_np).to(device)    if vl_np    is not None else None)
            return model(x, a, p, vl).cpu().numpy()   # (B, n_classes)

    rng = np.random.default_rng(cfg["seed"])

    # ── ISM loop ──────────────────────────────────────────────────────────────
    for iso, dom in tasks:
        out_path = os.path.join(cfg["outdir"], f"gnn_ism_saliency_{iso}_{dom}.npy")
        if cfg["skip_existing"] and os.path.exists(out_path):
            print(f"[GPU {rank}] [{iso}/{dom}] already exists — skipping", flush=True)
            continue

        iso_idx = int(np.where(classes == iso)[0][0])
        mask    = (test_labels == iso_idx) & (test_domains == dom)
        n_avail = int(mask.sum())
        if n_avail == 0:
            print(f"[GPU {rank}] [{iso}/{dom}] no test sequences — skipping", flush=True)
            continue

        global_idx = test_idx[mask]
        n_ism      = min(cfg["ism_n_seqs"], n_avail)
        sel        = rng.choice(n_avail, n_ism, replace=False)

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

        print(f"\n[GPU {rank}] [{iso}/{dom}]  ISM: {n_ism} seqs …", flush=True)

        ism_sum = np.zeros((max_len, 4, n_classes), dtype=np.float64)

        for si in range(n_ism):
            seq    = ism_seqs[si]
            L      = min(int(ism_lens[si]), max_len)
            phylo_1 = (np.tile(ism_phylo[si], (1, 1)).astype(np.float32)
                       if ism_phylo is not None else None)
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
            for start in range(0, len(mutants), cfg["ism_batch_size"]):
                batch   = mutants[start:start + cfg["ism_batch_size"]]
                B       = len(batch)
                emb_b   = _embed(batch)
                adj_b   = np.tile(ism_adj[si], (B, 1, 1))
                phylo_b = (np.tile(ism_phylo[si], (B, 1)).astype(np.float32)
                           if ism_phylo is not None else None)
                vl_b    = (np.tile(vl_1, (B, 1))
                           if vl_1 is not None else None)
                logits  = _classify(emb_b, adj_b, phylo_b, vl_b)
                for bi in range(B):
                    ism_seq[positions[start + bi],
                            nuc_idxs[start + bi], :] = logits[bi] - logit_orig

            ism_sum += ism_seq
            if (si + 1) % 10 == 0 or si + 1 == n_ism:
                print(f"[GPU {rank}] [{iso}/{dom}]  {si + 1}/{n_ism}", flush=True)

        # ── ISM delta (identical to train_gnn.py) ────────────────────────────
        mean_sal_all = ism_sum / n_ism                           # (max_len, 4, n_classes)
        target_sal   = mean_sal_all[:, :, iso_idx]              # (max_len, 4)
        other_mean   = (mean_sal_all.sum(axis=2) - target_sal) / (n_classes - 1)
        mean_sal     = (target_sal - other_mean).astype(np.float32)
        mean_sal    -= mean_sal.mean(axis=1, keepdims=True)     # row-centre

        np.save(out_path, mean_sal)
        print(f"[GPU {rank}] [{iso}/{dom}]  Saved → {out_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    npz_path   = os.path.join(args.emb_dir, "perpos_embeddings.npz")
    mmap_path  = os.path.join(args.emb_dir, "emb_mmap.npy")
    model_path = args.model_path

    for path in [npz_path, mmap_path, model_path]:
        if not os.path.exists(path):
            sys.exit(f"ERROR: required file not found: {path}")

    # Discover classes and domains from the NPZ
    data          = np.load(npz_path, allow_pickle=True)
    _raw_classes  = data["classes"].astype(str)
    _ile2 = np.where(_raw_classes == "Ile2")[0]
    _ile  = np.where(_raw_classes == "Ile")[0]
    if len(_ile2) and len(_ile):
        _raw_classes = np.delete(_raw_classes, _ile2[0])
    classes       = list(_raw_classes)
    domains_arr   = (data["domains"].astype(str) if "domains" in data
                     else np.full(len(data["labels"]), "Unknown", dtype=object))
    unique_domains = sorted(set(domains_arr))

    # Determine which isotypes to compute
    skip_set = set(args.already_done) | set(args.extra_skip)
    todo_isos = [iso for iso in classes if iso not in skip_set]

    all_tasks = [(iso, dom) for iso in todo_isos for dom in unique_domains]
    print(f"Isotypes to compute : {todo_isos}")
    print(f"Domains             : {unique_domains}")
    print(f"Total tasks         : {len(all_tasks)}")

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        sys.exit("ERROR: no CUDA GPUs found.")
    print(f"Using {n_gpus} GPU(s)")

    # Round-robin distribution across GPUs
    per_gpu_tasks = [all_tasks[i::n_gpus] for i in range(n_gpus)]
    for gi, tasks in enumerate(per_gpu_tasks):
        print(f"  GPU {gi}: {[f'{i}/{d}' for i, d in tasks]}")

    cfg = dict(
        npz_path      = npz_path,
        mmap_path     = mmap_path,
        model_path    = model_path,
        outdir        = outdir,
        ism_n_seqs    = args.ism_n_seqs,
        ism_batch_size= args.ism_batch_size,
        skip_existing = args.skip_existing,
        test_size     = args.test_size,
        seed          = args.seed,
    )

    mp.set_start_method("spawn", force=True)
    mp.spawn(run_worker, args=(per_gpu_tasks, cfg), nprocs=n_gpus, join=True)

    # ── Combine Ile + Ile2 ────────────────────────────────────────────────────
    print("\n── Combining Ile + Ile2 ──────────────────────────────────────────────")
    for dom in unique_domains:
        ile_path  = os.path.join(outdir, f"gnn_ism_saliency_Ile_{dom}.npy")
        ile2_path = os.path.join(outdir, f"gnn_ism_saliency_Ile2_{dom}.npy")
        if not os.path.exists(ile_path):
            print(f"  [{dom}] Ile file missing — cannot combine")
            continue
        if not os.path.exists(ile2_path):
            print(f"  [{dom}] Ile2 file missing — cannot combine")
            continue
        ile  = np.load(ile_path)
        ile2 = np.load(ile2_path)
        combined = ((ile + ile2) / 2).astype(np.float32)
        np.save(ile_path, combined)
        print(f"  [{dom}] Ile (combined Ile+Ile2) saved → {ile_path}")

    print("\n── Plotting logos ────────────────────────────────────────────────────")
    logo_outdir = "figures/logos/ism"
    cmd = (
        f"python scripts/plot_ism_logos.py "
        f"--emb-dir {outdir} "
        f"--outdir {logo_outdir}"
    )
    print(f"Running: {cmd}")
    os.system(cmd)

    print("\nDone.")
