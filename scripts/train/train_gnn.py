#!/usr/bin/env python3
"""
train_gnn.py
------------
Multi-class GNN classifier for tRNA isotype classification.
Trained on all domains combined; ISM maps computed per (isotype × domain).

Architecture
------------
  - Node features : RiNALMo per-position embeddings (max_len × 1280)
  - Graph         : backbone bonds + base pairs (D⁻¹/²AD⁻¹/² normalised)
  - 3-layer dense GCN (hidden_dim=256, dropout=0.3) → mean pool
  - Concat phylo OHE (domain→family) → Linear(N_classes)

Hyperparameters (fixed)
-----------------------
  n_layers=3  hidden_dim=256  dropout=0.3
  lr=7e-3     weight_decay=6e-5  epochs=20

Outputs
-------
  {outdir}/gnn_model.pt
  {outdir}/gnn_ism_saliency_{iso}_{domain}.npy   per isotype × domain
  {outdir}/gnn_results.json

Usage
-----
    python scripts/train_gnn.py
    python scripts/train_gnn.py --emb-dir embeddings/gnn
    python scripts/train_gnn.py --ism-isotypes Ala Gly His
    python scripts/train_gnn.py --ism-n-seqs 100
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

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir",        default="embeddings/gnn")
parser.add_argument("--csv",            default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir",         default=None)
parser.add_argument("--ism-isotypes",   nargs="+", default=None,
                    help="Isotypes to compute ISM for (default: all classes in NPZ)")
parser.add_argument("--ism-orders",     nargs="+", default=None,
                    help="Orders to compute order-specific ISM for (e.g. Pelagibacterales)")
parser.add_argument("--ism-n-seqs",     type=int, default=100,
                    help="Sequences per (isotype × domain) for ISM (default: 100)")
parser.add_argument("--ism-batch-size", type=int, default=64)
parser.add_argument("--batch-size",     type=int, default=256)
parser.add_argument("--test-size",      type=float, default=0.2)
parser.add_argument("--seed",           type=int, default=42)
# FiLM options — omit --film to run the original pipeline unchanged
parser.add_argument("--film",           action="store_true",
                    help="Enable FiLM order-conditioning bottleneck (two-stage training)")
parser.add_argument("--film-lr",        type=float, default=1e-3,
                    help="Learning rate for FiLM fine-tuning stage (default: 1e-3)")
parser.add_argument("--film-epochs",    type=int, default=10,
                    help="Epochs for FiLM fine-tuning stage (default: 10)")
parser.add_argument("--film-emb-dim",   type=int, default=32,
                    help="Order embedding dimension inside FiLM (default: 32)")
parser.add_argument("--phylo-encoder",  default=None,
                    help="Path to phylo_encoder.pkl (default: auto-search near --emb-dir)")
args = parser.parse_args()

outdir = args.outdir or args.emb_dir
os.makedirs(outdir, exist_ok=True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_LAYERS   = 3
HIDDEN_DIM = 256
DROPOUT    = 0.3
LR         = 7e-3
WD         = 6e-5
EPOCHS     = 20

# ── Identity element definitions ─────────────────────────────────────────────
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
ISM_NUCS     = ["A", "C", "G", "T"]

# ── Sprinzl utilities ─────────────────────────────────────────────────────────
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
        labs = ["14","15","16","17","17a","18","19","20","20a","20b","21"]
        if i < len(labs): sp[p] = labs[i]
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
            stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4:
        return 0, 0
    ac3_end  = sorted(pairs[p] for p in stems[2])[-1]
    t5_start = stems[-1][0]
    return ac3_end + 1, t5_start - ac3_end - 1


VAR_LOOP_THRESHOLD = 5   # nt; loops longer than this are type-II (Ser/Leu)


def build_adj_matrix(ss, seq_len, max_len):
    A = np.eye(max_len, dtype=np.float32)
    for i in range(min(seq_len - 1, max_len - 1)):
        A[i, i+1] = A[i+1, i] = 1.0
    for i, j in parse_pairs(ss).items():
        if i < max_len and j < max_len:
            A[i, j] = A[j, i] = 1.0
    # Remove variable-loop edges for type-II tRNAs so the GNN does not
    # propagate messages through the masked variable arm.
    vl_start, vl_len = detect_variable_loop(ss)
    if vl_len > VAR_LOOP_THRESHOLD:
        vl_end = min(vl_start + vl_len, max_len)
        A[vl_start:vl_end, :] = 0.0
        A[:, vl_start:vl_end] = 0.0
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (A * d[:, None]) * d[None, :]


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
    if df.empty:
        return {}, {}
    modal_len = df["primary_sequence"].str.len().mode()[0]
    modal_ss  = (df[df["primary_sequence"].str.len() == modal_len]
                 ["secondary_structure"].mode()[0])
    raw_to_sp = build_sprinzl_map(modal_ss)
    return raw_to_sp, {v: k for k, v in raw_to_sp.items()}

# ── Load data ─────────────────────────────────────────────────────────────────
npz_path  = os.path.join(args.emb_dir, "perpos_embeddings.npz")
mmap_path = os.path.join(args.emb_dir, "emb_mmap.npy")

# Load small arrays from NPZ (these fit in RAM easily)
print(f"Loading metadata from {npz_path} …", flush=True)
data      = np.load(npz_path, allow_pickle=True)
seq_lens  = data["seq_lens"]
labels    = data["labels"].astype(np.int64)
classes   = data["classes"].astype(str)

seqs      = data["sequences"]
ss_arr    = data["secondary_structures"]
max_len   = int(data["max_len"])
N_CLASSES = len(classes)

domains = (data["domains"].astype(str) if "domains" in data
           else np.full(len(labels), "Unknown", dtype=object))
unique_domains = sorted(set(domains))

if "phylo_ohe" in data:
    phylo_ohe = data["phylo_ohe"].astype(np.float32)
    PHYLO_DIM = int(phylo_ohe.shape[1])
    print(f"  Phylo OHE: dim={PHYLO_DIM}", flush=True)
else:
    phylo_ohe = None
    PHYLO_DIM = 0
    print("  Warning: phylo_ohe not in NPZ — running without phylo features", flush=True)

if "var_loop_lengths" in data:
    var_loop_lengths = data["var_loop_lengths"].astype(np.float32) / 25.0  # normalise to ~[0,1]
    VAR_LOOP_DIM = 1
    n_type2 = int((data["var_loop_lengths"] > 5).sum())
    print(f"  var_loop_lengths: found ({n_type2:,} type-II sequences)", flush=True)
else:
    var_loop_lengths = None
    VAR_LOOP_DIM = 0
    print("  Warning: var_loop_lengths not in NPZ — Ser/Leu length feature disabled",
          flush=True)

if args.film and phylo_ohe is not None:
    import pickle as _pkl, glob as _glob
    # Locate phylo_encoder.pkl — prefer explicit arg, then search nearby dirs
    _enc_path = args.phylo_encoder
    if _enc_path is None:
        _candidates = (
            [os.path.join(args.emb_dir, "phylo_encoder.pkl")] +
            _glob.glob(os.path.join(os.path.dirname(args.emb_dir), "*", "phylo_encoder.pkl"))
        )
        _enc_path = next((p for p in _candidates if os.path.exists(p)), None)
    if _enc_path is None:
        print("  Warning: --film set but phylo_encoder.pkl not found — FiLM disabled",
              flush=True)
        order_labels = order_classes = None
        N_ORDERS = FILM_EMB_DIM = 0
    else:
        with open(_enc_path, "rb") as _f:
            _enc = _pkl.load(_f)
        _n_cats = [len(c) for c in _enc.categories_]
        _order_start = sum(_n_cats[:3])   # domain + phylum + class
        _order_end   = _order_start + _n_cats[3]
        order_classes = _enc.categories_[3]
        order_labels  = phylo_ohe[:, _order_start:_order_end].argmax(axis=1).astype(np.int64)
        N_ORDERS      = len(order_classes)
        FILM_EMB_DIM  = args.film_emb_dim
        print(f"  Order labels derived from phylo_ohe[{_order_start}:{_order_end}]: "
              f"{N_ORDERS} unique orders — FiLM enabled", flush=True)
elif args.film:
    print("  Warning: --film set but phylo_ohe unavailable — FiLM disabled", flush=True)
    order_labels = order_classes = None
    N_ORDERS = FILM_EMB_DIM = 0
else:
    order_labels = order_classes = None
    N_ORDERS     = 0
    FILM_EMB_DIM = 0

# Load emb via memory-map (does NOT read into RAM; pages loaded on demand)
if os.path.exists(mmap_path):
    print(f"Memory-mapping {mmap_path} …", flush=True)
    emb = np.load(mmap_path, mmap_mode='r')
else:
    raise FileNotFoundError(
        f"{mmap_path} not found.\n"
        f"Run first:  python scripts/npz_to_mmap.py --npz {npz_path}"
    )

N      = emb.shape[0]
EMB_DIM = emb.shape[2]
print(f"  N={N:,}  max_len={max_len}  EMB_DIM={EMB_DIM}  N_CLASSES={N_CLASSES}",
      flush=True)
print(f"  Classes: {list(classes)}", flush=True)
print(f"  Domains: { {d: int((domains==d).sum()) for d in unique_domains} }",
      flush=True)

sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size,
                              random_state=args.seed)
train_idx, test_idx = next(sss.split(labels, labels))
print(f"  Train={len(train_idx):,}  Test={len(test_idx):,}", flush=True)

# ── GNN architecture ──────────────────────────────────────────────────────────
class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        return self.dropout(F.relu(torch.bmm(adj, self.linear(x))))


class FiLMBottleneck(nn.Module):
    """FiLM conditioning: h → (1 + γ) * h + β, γ/β derived from a learned order embedding.
    Residual parameterization (zero-init cond_net) means FiLM is a no-op at the start of
    stage-2 fine-tuning, so it only learns what the frozen GNN backbone couldn't explain."""
    def __init__(self, hidden_dim, n_orders, order_emb_dim=32):
        super().__init__()
        self.order_emb = nn.Embedding(n_orders, order_emb_dim)
        self.cond_net  = nn.Linear(order_emb_dim, 2 * hidden_dim)
        nn.init.zeros_(self.cond_net.weight)
        nn.init.zeros_(self.cond_net.bias)

    def forward(self, h, order_idx):
        cond        = self.order_emb(order_idx)          # (B, order_emb_dim)
        gamma, beta = self.cond_net(cond).chunk(2, dim=-1)  # (B, hidden_dim)
        if h.dim() == 3:
            gamma = gamma.unsqueeze(1)                   # (B, 1, hidden_dim) → broadcast over L
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
            x = self.film(x, order_idx)                  # modulate per-position before pooling
        pooled = x.mean(dim=1)
        parts  = [pooled]
        if self.phylo_dim > 0 and phylo is not None:
            parts.append(phylo)
        if self.var_loop_dim > 0 and var_loop is not None:
            parts.append(var_loop)
        return self.head(torch.cat(parts, dim=1))


model = GNNClassifier(EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT,
                      N_CLASSES, PHYLO_DIM, VAR_LOOP_DIM,
                      film_n_orders=N_ORDERS,
                      film_emb_dim=FILM_EMB_DIM).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel params: {n_params:,}", flush=True)
print(f"Hyperparams: n_layers={N_LAYERS}  hidden={HIDDEN_DIM}  dropout={DROPOUT}  "
      f"lr={LR}  wd={WD}  epochs={EPOCHS}  n_classes={N_CLASSES}", flush=True)

# ── Dataset (adj computed on-the-fly; emb read from mmap per sample) ──────────
class tRNADataset(Dataset):
    def __init__(self, indices):
        self.idx      = indices
        self.ss       = ss_arr[indices]
        self.sl       = seq_lens[indices]
        self.y        = labels[indices]
        self.phylo    = phylo_ohe[indices] if phylo_ohe is not None else None
        self.vl       = var_loop_lengths[indices] if var_loop_lengths is not None else None
        self.order    = order_labels[indices] if order_labels is not None else None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x   = emb[self.idx[i]].astype(np.float32)          # (max_len, 1280)
        adj = build_adj_matrix(str(self.ss[i]), int(self.sl[i]), max_len)
        y   = int(self.y[i])
        phylo = self.phylo[i] if self.phylo is not None else None
        vl    = np.array([self.vl[i]], dtype=np.float32) if self.vl is not None else None
        order = int(self.order[i]) if self.order is not None else None
        return x, adj, phylo, vl, order, y


def collate_fn(batch):
    xs, adjs, ps, vls, orders, ys = zip(*batch)
    out = [torch.from_numpy(np.stack(xs)),
           torch.from_numpy(np.stack(adjs))]
    out.append(torch.from_numpy(np.stack(ps)) if ps[0] is not None else None)
    out.append(torch.from_numpy(np.stack(vls)) if vls[0] is not None else None)
    out.append(torch.tensor(orders, dtype=torch.long) if orders[0] is not None else None)
    out.append(torch.tensor(ys))
    return tuple(out)


# ── Training ──────────────────────────────────────────────────────────────────
train_ds  = tRNADataset(train_idx)
loader    = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=4, collate_fn=collate_fn,
                       pin_memory=(DEVICE.type == "cuda"))
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
criterion = nn.CrossEntropyLoss()

def _to(t, device):
    return t.to(device) if t is not None else None

# ── Stage 1: train full model (FiLM zero-init → no-op if present) ─────────────
print(f"\n── Stage 1: {EPOCHS} epochs (full model) ──", flush=True)
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, ab, pb, vlb, ob, yb in loader:
        xb  = xb.to(DEVICE)
        ab  = ab.to(DEVICE)
        pb  = _to(pb, DEVICE)
        vlb = _to(vlb, DEVICE)
        ob  = _to(ob, DEVICE)
        yb  = yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb, ab, pb, vlb, ob), yb)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * len(yb)
    if epoch % 5 == 0 or epoch == EPOCHS:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/len(train_ds):.4f}",
              flush=True)

# ── Stage 2: freeze GNN backbone, fine-tune FiLM + head ───────────────────────
if model.film is not None:
    print(f"\n── Stage 2: {args.film_epochs} epochs (FiLM + head, GNN frozen) ──",
          flush=True)
    for p in model.layers.parameters():
        p.requires_grad_(False)
    film_optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.film_lr, weight_decay=WD,
    )
    for epoch in range(1, args.film_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, ab, pb, vlb, ob, yb in loader:
            xb  = xb.to(DEVICE)
            ab  = ab.to(DEVICE)
            pb  = _to(pb, DEVICE)
            vlb = _to(vlb, DEVICE)
            ob  = _to(ob, DEVICE)
            yb  = yb.to(DEVICE)
            film_optimizer.zero_grad()
            loss = criterion(model(xb, ab, pb, vlb, ob), yb)
            loss.backward(); film_optimizer.step()
            total_loss += loss.item() * len(yb)
        if epoch % 5 == 0 or epoch == args.film_epochs:
            print(f"  Epoch {epoch:3d}/{args.film_epochs}  loss={total_loss/len(train_ds):.4f}",
                  flush=True)
    for p in model.layers.parameters():
        p.requires_grad_(True)

# ── Accuracy ──────────────────────────────────────────────────────────────────
model.eval()
test_ds     = tRNADataset(test_idx)
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
acc   = accuracy_score(labels[test_idx], preds)
print(f"\nTest accuracy: {acc:.4f}", flush=True)

torch.save(model.state_dict(), os.path.join(outdir, "gnn_model.pt"))

# ── ISM per (isotype × domain) ────────────────────────────────────────────────
ism_isos = args.ism_isotypes if args.ism_isotypes else list(classes)
# Only isotypes that are actually present in the test set
ism_isos = [iso for iso in ism_isos if iso in classes]

_rinalmo  = None
_alphabet = None

def _load_rinalmo():
    global _rinalmo, _alphabet
    if _rinalmo is None:
        print("Loading RiNALMo for ISM …", flush=True)
        from rinalmo.pretrained import get_pretrained_model
        _rinalmo, _alphabet = get_pretrained_model(model_name="giga-v1",
                                                    lm_config="giga")
        _rinalmo.eval().to(DEVICE)
        for m in _rinalmo.modules():
            if isinstance(m, nn.Dropout): m.p = 0.0
    return _rinalmo, _alphabet


model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout): m.p = 0.0

test_labels  = labels[test_idx]
test_domains = domains[test_idx]

ie_scores = {}   # {f"{iso}_{domain}": score}

for iso in ism_isos:
    iso_idx = int(np.where(classes == iso)[0][0])

    for dom in unique_domains:
        mask = (test_labels == iso_idx) & (test_domains == dom)
        n_avail = mask.sum()
        if n_avail == 0:
            print(f"\n[{iso} / {dom}] No test sequences — skipping", flush=True)
            continue

        global_idx = test_idx[mask]
        n_ism = min(args.ism_n_seqs, n_avail)
        rng   = np.random.default_rng(args.seed)
        sel   = rng.choice(n_avail, n_ism, replace=False)

        ism_seqs  = seqs[global_idx][sel]
        ism_lens  = seq_lens[global_idx][sel]
        ism_ss    = ss_arr[global_idx][sel]
        # compute adj on-the-fly to avoid holding adj_all in RAM
        ism_adj   = np.stack([
            build_adj_matrix(str(ism_ss[i]), int(ism_lens[i]), max_len)
            for i in range(n_ism)
        ])
        ism_phylo = (phylo_ohe[global_idx][sel]
                     if phylo_ohe is not None else None)
        ism_vl    = (var_loop_lengths[global_idx][sel]
                     if var_loop_lengths is not None else None)
        ism_order = (order_labels[global_idx][sel]
                     if order_labels is not None else None)

        print(f"\n[{iso} / {dom}]  ISM: {n_ism} seqs …", flush=True)

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

        def _classify(emb_pad, adj_np, phylo_np=None, vl_np=None, order_np=None):
            with torch.no_grad():
                x  = torch.from_numpy(emb_pad).to(DEVICE)
                a  = torch.from_numpy(adj_np).to(DEVICE)
                p  = _to(torch.from_numpy(phylo_np) if phylo_np is not None else None, DEVICE)
                vl = _to(torch.from_numpy(vl_np)    if vl_np    is not None else None, DEVICE)
                oi = _to(torch.tensor(order_np, dtype=torch.long)
                         if order_np is not None else None, DEVICE)
                return model(x, a, p, vl, oi).cpu().numpy()  # (B, n_classes)

        n_classes = len(classes)
        ism_sum = np.zeros((max_len, 4, n_classes), dtype=np.float64)

        for si in range(n_ism):
            seq = ism_seqs[si]
            L   = min(int(ism_lens[si]), max_len)

            phylo_1 = (np.tile(ism_phylo[si], (1, 1)).astype(np.float32)
                       if ism_phylo is not None else None)
            vl_1    = (np.array([[ism_vl[si]]], dtype=np.float32)
                       if ism_vl is not None else None)
            order_1 = (np.array([ism_order[si]], dtype=np.int64)
                       if ism_order is not None else None)
            logit_orig = _classify(_embed([seq]), ism_adj[si:si+1],
                                   phylo_1, vl_1, order_1)[0]

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
                phylo_b  = (np.tile(ism_phylo[si], (B, 1)).astype(np.float32)
                            if ism_phylo is not None else None)
                vl_b     = (np.tile(vl_1, (B, 1))
                            if vl_1 is not None else None)
                order_b  = (np.tile(order_1, B)
                            if order_1 is not None else None)
                logits   = _classify(emb_b, adj_b, phylo_b, vl_b, order_b)
                for bi in range(B):
                    ism_seq[positions[start + bi], nuc_idxs[start + bi], :] = (
                        logits[bi] - logit_orig  # (n_classes,)
                    )

            ism_sum += ism_seq
            if (si + 1) % 10 == 0 or si + 1 == n_ism:
                print(f"  {si + 1}/{n_ism} done", flush=True)

        mean_sal_all = ism_sum / n_ism  # (max_len, 4, n_classes)
        # Differential: how much more does this mutation affect the target
        # logit compared to all other isotype logits on average.
        # Universally conserved positions (U33, G18, G19, A37 ...) affect all
        # logits equally and therefore cancel out; isotype-specific identity
        # elements survive.
        target_sal = mean_sal_all[:, :, iso_idx]                          # (max_len, 4)
        other_mean = (mean_sal_all.sum(axis=2) - target_sal) / (n_classes - 1)
        mean_sal   = (target_sal - other_mean).astype(np.float32)
        mean_sal  -= mean_sal.mean(axis=1, keepdims=True)

        sal_path = os.path.join(outdir, f"gnn_ism_saliency_{iso}_{dom}.npy")
        np.save(sal_path, mean_sal)
        print(f"  Saved → {sal_path}", flush=True)

        # IE score
        if iso in IE_DEFS:
            _, sp_to_raw = get_sprinzl_map(iso, dom, args.csv)
            global_max   = float(np.max(np.maximum(mean_sal, 0)))
            ws = wt = 0.0
            if global_max > 0:
                for sp_label, exp_nuc, tier in IE_DEFS[iso]:
                    raw = sp_to_raw.get(sp_label)
                    w   = TIER_WEIGHTS[tier]
                    if raw is None or raw >= mean_sal.shape[0]: continue
                    comp = max(float(mean_sal[raw, NUC_IDX_MAP[exp_nuc]]), 0.0) / global_max
                    ws  += w * comp; wt += w
            ie_score = (ws / wt * 100) if wt > 0 else 0.0
            ie_scores[f"{iso}_{dom}"] = round(float(ie_score), 2)
            print(f"  IE score: {ie_score:.1f}/100", flush=True)

# ── ISM per (isotype × order) — FiLM-conditioned, order held fixed ────────────
if args.ism_orders and order_labels is not None:
    import pickle as _pkl
    with open(args.phylo_encoder, "rb") as _f:
        _enc = _pkl.load(_f)
    order_classes = _enc.categories_[3]
    order_name_to_idx = {name: i for i, name in enumerate(order_classes)}

    for order_name in args.ism_orders:
        if order_name not in order_name_to_idx:
            print(f"\n[order ISM] '{order_name}' not found in encoder — skipping", flush=True)
            continue
        order_int = order_name_to_idx[order_name]

        for iso in ism_isos:
            iso_idx = int(np.where(classes == iso)[0][0])
            mask    = (test_labels == iso_idx) & (order_labels[test_idx] == order_int)
            n_avail = mask.sum()
            if n_avail == 0:
                print(f"\n[{iso} / {order_name}] No test sequences — skipping", flush=True)
                continue

            global_idx = test_idx[mask]
            n_ism      = min(args.ism_n_seqs, n_avail)
            rng        = np.random.default_rng(args.seed)
            sel        = rng.choice(n_avail, n_ism, replace=False)

            ism_seqs  = seqs[global_idx][sel]
            ism_lens  = seq_lens[global_idx][sel]
            ism_ss    = ss_arr[global_idx][sel]
            ism_adj   = np.stack([
                build_adj_matrix(str(ism_ss[i]), int(ism_lens[i]), max_len)
                for i in range(n_ism)
            ])
            ism_phylo = phylo_ohe[global_idx][sel] if phylo_ohe is not None else None
            ism_vl    = var_loop_lengths[global_idx][sel] if var_loop_lengths is not None else None
            # Hold order fixed to this order for all ISM sequences
            order_fixed = np.full(1, order_int, dtype=np.int64)

            print(f"\n[{iso} / {order_name}]  ISM: {n_ism} seqs (order-fixed) …", flush=True)
            rinalmo, alphabet = _load_rinalmo()

            ism_sum = np.zeros((max_len, 4, len(classes)), dtype=np.float64)
            for si in range(n_ism):
                seq    = ism_seqs[si]
                L      = min(int(ism_lens[si]), max_len)
                phylo_1 = (np.tile(ism_phylo[si], (1, 1)).astype(np.float32)
                           if ism_phylo is not None else None)
                vl_1    = (np.array([[ism_vl[si]]], dtype=np.float32)
                           if ism_vl is not None else None)
                logit_orig = _classify(_embed([seq]), ism_adj[si:si+1],
                                       phylo_1, vl_1, order_fixed)[0]

                mutants, positions, nuc_idxs = [], [], []
                for pos in range(L):
                    for ni, nuc in enumerate(ISM_NUCS):
                        mutants.append(seq[:pos] + nuc + seq[pos + 1:])
                        positions.append(pos)
                        nuc_idxs.append(ni)

                ism_seq = np.zeros((max_len, 4, len(classes)), dtype=np.float64)
                for start in range(0, len(mutants), args.ism_batch_size):
                    batch  = mutants[start:start + args.ism_batch_size]
                    B      = len(batch)
                    emb_b  = _embed(batch)
                    adj_b  = np.tile(ism_adj[si], (B, 1, 1))
                    phylo_b = (np.tile(ism_phylo[si], (B, 1)).astype(np.float32)
                               if ism_phylo is not None else None)
                    vl_b    = (np.tile(vl_1, (B, 1)) if vl_1 is not None else None)
                    order_b = np.tile(order_fixed, B)
                    logits  = _classify(emb_b, adj_b, phylo_b, vl_b, order_b)
                    for bi in range(B):
                        ism_seq[positions[start + bi], nuc_idxs[start + bi], :] = (
                            logits[bi] - logit_orig
                        )
                ism_sum += ism_seq
                if (si + 1) % 10 == 0 or si + 1 == n_ism:
                    print(f"  {si + 1}/{n_ism} done", flush=True)

            mean_sal_all = ism_sum / n_ism
            target_sal   = mean_sal_all[:, :, iso_idx]
            other_mean   = (mean_sal_all.sum(axis=2) - target_sal) / (len(classes) - 1)
            mean_sal     = (target_sal - other_mean).astype(np.float32)
            mean_sal    -= mean_sal.mean(axis=1, keepdims=True)

            sal_path = os.path.join(outdir, f"gnn_ism_saliency_{iso}_{order_name}.npy")
            np.save(sal_path, mean_sal)
            print(f"  Saved → {sal_path}", flush=True)

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "classes":          list(classes),
    "n_classes":        N_CLASSES,
    "max_len":          max_len,
    "domains":          unique_domains,
    "n_train":          len(train_idx),
    "n_test":           len(test_idx),
    "n_params":         n_params,
    "n_ism_per_group":  args.ism_n_seqs,
    "accuracy":         round(float(acc), 4),
    "ie_scores":        ie_scores,
    "hyperparams": {
        "n_layers": N_LAYERS, "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
        "lr": LR, "weight_decay": WD, "epochs": EPOCHS,
    },
    "film": {
        "enabled":      model.film is not None,
        "n_orders":     N_ORDERS,
        "emb_dim":      FILM_EMB_DIM,
        "film_lr":      args.film_lr      if model.film is not None else None,
        "film_epochs":  args.film_epochs  if model.film is not None else None,
    },
}
with open(os.path.join(outdir, "gnn_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {outdir}/gnn_results.json")
print("Done.")
