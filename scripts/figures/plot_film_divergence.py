#!/usr/bin/env python3
"""
plot_film_divergence.py
-----------------------
Visualize FiLM-specific phylogenetic divergence across orders.

Inputs (from compute_film_delta_ism.py):
    gnn_film/film_delta_ism/film_ism_on_{iso}_{order}.npy    — FiLM-on saliency
    gnn_film/film_delta_ism/film_ism_delta_{iso}_{order}.npy — FiLM-delta saliency

Outputs:
    figures/film_divergence/{iso}_divergence_heatmap.{svg,png}
        Cross-order heatmap: rows=orders, cols=Sprinzl positions,
        colour=max |delta| per position (warm = phylo-divergent)

    figures/film_divergence/{iso}_divergence_bar.{svg,png}
        Per-Sprinzl-position variance across orders — a "divergence spectrum"

    figures/film_divergence/{iso}_{order}_delta_logo.{svg,png}
        FiLM-delta logo for individual orders

Usage
-----
    python scripts/figures/plot_film_divergence.py
    python scripts/figures/plot_film_divergence.py --isotypes His --top-orders 20
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import logomaker

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--delta-dir",  default=os.path.join(DATA_ROOT, "gnn_film/film_delta_ism"))
parser.add_argument("--csv",        default=os.path.join(DATA_ROOT, "data/trnascan_GTDB_r226_dedup_taxonomy.csv"))
parser.add_argument("--outdir",     default=os.path.join(DATA_ROOT, "figures/film_divergence"))
parser.add_argument("--isotypes",   nargs="+", default=None,
                    help="Isotypes to plot (default: all found in delta-dir)")
parser.add_argument("--top-orders", type=int, default=30,
                    help="Number of top-divergence orders to show in heatmap (default: 30)")
parser.add_argument("--pseudocount", type=float, default=0.5)
parser.add_argument("--rank-by", choices=["kl", "film"], default="kl",
                    help="Rank top orders by KL divergence from background (kl) or FiLM delta sum (film)")
parser.add_argument("--include-orders", nargs="+", default=[],
                    help="Orders to always include in the logo figure regardless of rank "
                         "(falls back to gnn_film/gnn_ism_saliency if no delta file)")
parser.add_argument("--bg-logo", default=None,
                    help="Domain name (e.g. Bacteria) to show as a reference logo at the top "
                         "of the figure, using gnn_film/gnn_ism_saliency_{iso}_{domain}.npy")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

NUCS     = ["A", "C", "G", "U"]
AC_EXCL  = {"34", "35", "36", "74", "75", "76"}
MIN_NORM = 1e-6
NUC_IDX  = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

# ── Load CSV for frequency logos ──────────────────────────────────────────────
print("Loading CSV for frequency logos …", flush=True)
df_csv = pd.read_csv(
    args.csv,
    usecols=["primary_sequence", "secondary_structure",
             "Anticodon_predicted_isotype", "order", "domain"],
    low_memory=False,
).dropna(subset=["primary_sequence", "secondary_structure",
                 "Anticodon_predicted_isotype", "order"])
df_csv["primary_sequence"] = (df_csv["primary_sequence"]
                               .str.upper().str.replace("T", "U", regex=False))
print(f"  {len(df_csv):,} sequences loaded", flush=True)


def _build_nuc_counts(df_sub: pd.DataFrame, sp_set: set) -> dict:
    counts: dict[str, np.ndarray] = {}
    for ss, grp in df_sub.groupby("secondary_structure"):
        sp_map = build_sprinzl_map(str(ss))
        if not sp_map:
            continue
        seqs = grp["primary_sequence"].tolist()
        for raw_pos, sp_lbl in sp_map.items():
            if sp_lbl not in sp_set:
                continue
            col = np.array(
                [NUC_IDX.get(s[raw_pos], -1) if raw_pos < len(s) else -1 for s in seqs],
                dtype=np.int8,
            )
            arr = counts.setdefault(sp_lbl, np.zeros(4, dtype=np.float64))
            for ni in range(4):
                arr[ni] += (col == ni).sum()
    return counts


def compute_order_kl_divergence(iso: str, order_name: str, sp_labels: list) -> float:
    """Sum of KL(order || domain-background) across Sprinzl positions."""
    sp_set = set(sp_labels) - AC_EXCL
    iso_df = df_csv[df_csv["Anticodon_predicted_isotype"] == iso]
    if iso_df.empty:
        return 0.0
    order_df = iso_df[iso_df["order"] == order_name]
    if order_df.empty:
        return 0.0
    dom   = order_df["domain"].mode()[0] if "domain" in order_df.columns else None
    bg_df = iso_df[iso_df["domain"] == dom] if dom is not None else iso_df

    cnt_ord = _build_nuc_counts(order_df, sp_set)
    cnt_bg  = _build_nuc_counts(bg_df,    sp_set)

    total_kl = 0.0
    for sp_lbl in sp_labels:
        if sp_lbl in AC_EXCL or sp_lbl not in cnt_bg:
            continue
        c_ord = cnt_ord.get(sp_lbl, np.zeros(4))
        c_bg  = cnt_bg[sp_lbl]
        p_ord = (c_ord + args.pseudocount) / (c_ord.sum() + 4 * args.pseudocount)
        p_bg  = (c_bg  + args.pseudocount) / (c_bg.sum()  + 4 * args.pseudocount)
        with np.errstate(divide="ignore", invalid="ignore"):
            total_kl += float(np.sum(np.where(p_ord > 0, p_ord * np.log2(p_ord / p_bg), 0.0)))
    return total_kl


def compute_order_ic_df(
    iso: str, order_name: str, sp_labels: list,
) -> pd.DataFrame:
    """IC bit-score logo using the order's modal SS Sprinzl map (consistent with delta logo)."""
    iso_df = df_csv[df_csv["Anticodon_predicted_isotype"] == iso]
    if iso_df.empty:
        return pd.DataFrame(np.zeros((len(sp_labels), 4)), columns=NUCS)
    order_df = iso_df[iso_df["order"] == order_name]
    if order_df.empty:
        return pd.DataFrame(np.zeros((len(sp_labels), 4)), columns=NUCS)

    # Use the same modal-SS Sprinzl map as the delta logo to avoid cross-length contamination
    sp_map = order_sp_maps.get(order_name) or get_order_sprinzl(iso, order_name)
    if not sp_map:
        return pd.DataFrame(np.zeros((len(sp_labels), 4)), columns=NUCS)
    sp_to_raw = {v: k for k, v in sp_map.items()}

    # Count nucleotides using the modal Sprinzl map's raw positions
    modal_len = order_df["primary_sequence"].str.len().mode()[0]
    modal_seqs = order_df[order_df["primary_sequence"].str.len() == modal_len][
        "primary_sequence"
    ].str.upper().str.replace("T", "U", regex=False).tolist()

    counts: dict[str, np.ndarray] = {}
    for sp_lbl, r in sp_to_raw.items():
        if sp_lbl in AC_EXCL:
            continue
        col = np.array(
            [NUC_IDX.get(s[r], -1) if r < len(s) else -1 for s in modal_seqs],
            dtype=np.int8,
        )
        arr = np.array([(col == i).sum() for i in range(4)], dtype=np.float64)
        if arr.sum() > 0:
            counts[sp_lbl] = arr

    rows = []
    for sp_lbl in sp_labels:
        if sp_lbl in AC_EXCL or sp_lbl not in counts:
            rows.append([0.0, 0.0, 0.0, 0.0])
            continue
        c = counts[sp_lbl]
        p = (c + args.pseudocount) / (c.sum() + 4 * args.pseudocount)
        with np.errstate(divide="ignore", invalid="ignore"):
            ic = 2.0 + float(np.sum(np.where(p > 0, p * np.log2(p), 0.0)))
        ic = max(0.0, ic)
        rows.append((p * ic).tolist())
    return pd.DataFrame(rows, columns=NUCS)


# ── Sprinzl utilities ────────────────────────────────────────────────────────
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == ">": stack.append(i)
        elif c == "<":
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss):
    pairs = parse_pairs(ss)
    n     = len(ss)
    fps   = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in fps:
        if not cur: cur = [p]
        elif p == cur[-1]+1 and pairs[p] == pairs[cur[-1]]-1: cur.append(p)
        else: stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4: return {}
    sp = {}
    acc5 = stems[0]; acc3 = sorted(pairs[p] for p in acc5)
    overhang = [p for p in range(acc5[0]) if p not in pairs]
    for i, p in enumerate(reversed(overhang)):
        sp[p] = str(-(i+1))
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


def get_modal_sprinzl(iso, csv_path):
    df = pd.read_csv(csv_path,
        usecols=["primary_sequence", "secondary_structure",
                 "Anticodon_predicted_isotype"],
        low_memory=False)
    sub = df[df["Anticodon_predicted_isotype"] == iso].dropna(
        subset=["primary_sequence", "secondary_structure"])
    if sub.empty: return {}, {}
    modal_len = sub["primary_sequence"].str.len().mode()[0]
    modal_ss  = (sub[sub["primary_sequence"].str.len() == modal_len]
                 ["secondary_structure"].mode()[0])
    raw_to_sp = build_sprinzl_map(modal_ss)
    return raw_to_sp, {v: k for k, v in raw_to_sp.items()}


_csv_order_cache: dict = {}
def get_order_sprinzl(iso, order_name):
    """Per-order Sprinzl map using the order's own modal secondary structure."""
    key = (iso, order_name)
    if key not in _csv_order_cache:
        df = pd.read_csv(args.csv,
            usecols=["primary_sequence", "secondary_structure",
                     "Anticodon_predicted_isotype", "order"],
            low_memory=False)
        sub = df[(df["Anticodon_predicted_isotype"] == iso) &
                 (df["order"] == order_name)].dropna(
            subset=["primary_sequence", "secondary_structure"])
        if sub.empty:
            _csv_order_cache[key] = {}
        else:
            modal_len = sub["primary_sequence"].str.len().mode()[0]
            modal_ss  = (sub[sub["primary_sequence"].str.len() == modal_len]
                         ["secondary_structure"].mode()[0])
            _csv_order_cache[key] = build_sprinzl_map(modal_ss)
    return _csv_order_cache[key]


def sp_sort_key(label):
    """Sort Sprinzl labels in tRNA order."""
    if label.startswith("-"):
        try: return (-1, -int(label[1:]))
        except ValueError: return (-1, 0)
    if label.startswith("e"):
        try: return (46, int(label[1:]))
        except ValueError: return (46, 0)
    base = label.rstrip("ab")
    try: return (int(base), 0 if not label[len(base):] else ord(label[len(base)]))
    except ValueError: return (50, 0)


# ── Discover available (iso, order) pairs ─────────────────────────────────────
iso_set = set()
for fn in os.listdir(args.delta_dir):
    m = re.match(r"film_ism_delta_(\w+)_(.+)\.npy$", fn)
    if m:
        iso_set.add(m.group(1))

isotypes = [i for i in (args.isotypes or sorted(iso_set)) if i in iso_set]
if not isotypes:
    print("No delta files found. Run compute_film_delta_ism.py first.")
    raise SystemExit(1)

print(f"Found isotypes: {isotypes}", flush=True)


# ── Per-isotype figures ───────────────────────────────────────────────────────
for iso in isotypes:
    # Collect all delta arrays for this iso
    delta_files = sorted(
        fn for fn in os.listdir(args.delta_dir)
        if re.match(rf"film_ism_delta_{iso}_(.+)\.npy$", fn)
    )
    if not delta_files:
        continue

    order_names  = []
    delta_arrays = []   # each: (max_len, 4)
    for fn in delta_files:
        order_name = re.match(rf"film_ism_delta_{iso}_(.+)\.npy$", fn).group(1)
        arr        = np.load(os.path.join(args.delta_dir, fn))
        order_names.append(order_name)
        delta_arrays.append(arr)

    # Build per-order Sprinzl maps and collect union of all Sprinzl positions.
    # This ensures each order's ISM is extracted at the raw position that
    # actually corresponds to each Sprinzl label for *that* order's tRNAs,
    # rather than using a global modal map that misaligns orders with
    # non-modal sequence lengths (e.g. Bathyarchaeales Ala disc at idx 74, not 71).
    print(f"[{iso}] building per-order Sprinzl maps …", flush=True)
    order_sp_maps: dict[str, dict] = {}
    for order_name in order_names:
        order_sp_maps[order_name] = get_order_sprinzl(iso, order_name)

    all_sp: set[str] = set()
    for sp_map in order_sp_maps.values():
        all_sp.update(sp for sp in sp_map.values() if sp not in AC_EXCL)
    sp_labels = sorted(all_sp, key=sp_sort_key)

    if not sp_labels:
        print(f"[{iso}] No Sprinzl positions — skipping", flush=True)
        continue

    # Per-order, per-position: max absolute delta, using per-order raw positions.
    heat = []
    for i, order_name in enumerate(order_names):
        sp_map   = order_sp_maps[order_name]
        sp_to_raw = {v: k for k, v in sp_map.items()}
        arr      = delta_arrays[i]
        row = []
        for sp_lbl in sp_labels:
            if sp_lbl in sp_to_raw:
                r = sp_to_raw[sp_lbl]
                row.append(float(np.abs(arr[r]).max()) if r < arr.shape[0] else 0.0)
            else:
                row.append(0.0)
        heat.append(row)
    heat = np.array(heat)   # (n_orders, n_positions)

    # Per-position divergence score = variance across orders
    divergence = heat.var(axis=0)

    # Sort orders by KL divergence (sequence divergence) or FiLM delta sum
    if args.rank_by == "kl":
        print(f"[{iso}] computing KL divergence scores …", flush=True)
        rank_scores = np.array([
            compute_order_kl_divergence(iso, name, sp_labels)
            for name in order_names
        ])
        rank_label = "KL divergence"
    else:
        rank_scores = heat.sum(axis=1)
        rank_label  = "FiLM delta sum"

    sorted_idx  = np.argsort(rank_scores)[::-1][:args.top_orders]
    heat_top    = heat[sorted_idx]
    names_top   = [order_names[i] for i in sorted_idx]

    print(f"\n[{iso}] {len(order_names)} orders, top {len(names_top)} by {rank_label} …",
          flush=True)

    # ── Figure 1: cross-order heatmap ─────────────────────────────────────────
    fig_h, ax_h = plt.subplots(figsize=(max(14, len(sp_labels) * 0.18),
                                        max(4, len(names_top) * 0.22)))
    im = ax_h.imshow(heat_top, aspect="auto", cmap="YlOrRd",
                     interpolation="nearest")
    ax_h.set_xticks(range(len(sp_labels)))
    ax_h.set_xticklabels(sp_labels, fontsize=5, rotation=90)
    ax_h.set_yticks(range(len(names_top)))
    ax_h.set_yticklabels(names_top, fontsize=6)
    ax_h.set_xlabel("Sprinzl position", fontsize=9)
    plt.colorbar(im, ax=ax_h, label="max |FiLM delta| over nucleotides", shrink=0.6)

    # shade anticodon region
    ac_x = [i for i, sp in enumerate(sp_labels) if sp in {"34","35","36"}]
    if ac_x:
        ax_h.axvspan(min(ac_x)-.5, max(ac_x)+.5, color="#aaaaaa", alpha=0.2, zorder=0)

    for ext in ["svg", "png"]:
        path = os.path.join(args.outdir, f"{iso}_divergence_heatmap.{ext}")
        fig_h.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig_h)
    print(f"  Saved {iso}_divergence_heatmap", flush=True)

    # ── Figure 2: per-position divergence bar ─────────────────────────────────
    fig_d, ax_d = plt.subplots(figsize=(max(12, len(sp_labels) * 0.18), 3))
    ax_d.bar(range(len(sp_labels)), divergence, color="#5B4A8A", width=0.8)
    ax_d.set_xticks(range(len(sp_labels)))
    ax_d.set_xticklabels(sp_labels, fontsize=5, rotation=90)
    ax_d.set_ylabel("Variance of |FiLM delta| across orders", fontsize=8)
    ax_d.set_xlabel("Sprinzl position", fontsize=9)
    ax_d.spines[["top", "right"]].set_visible(False)
    if ac_x:
        ax_d.axvspan(min(ac_x)-.5, max(ac_x)+.5, color="#aaaaaa", alpha=0.2, zorder=0)

    for ext in ["svg", "png"]:
        path = os.path.join(args.outdir, f"{iso}_divergence_bar.{ext}")
        fig_d.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig_d)
    print(f"  Saved {iso}_divergence_bar", flush=True)

    # ── Figure 3: delta logos + frequency logos for top-N most divergent orders ─
    # Build the ordered list: top-N by rank + any --include-orders not already present
    logo_order_names = list(names_top[:min(8, len(names_top))])
    logo_delta_map   = {order_names[i]: delta_arrays[i] for i in range(len(order_names))}

    for inc in args.include_orders:
        if inc not in logo_order_names:
            logo_order_names.append(inc)

    # For include-orders without a delta file, try loading regular FiLM saliency
    fallback_dir = os.path.dirname(args.delta_dir)  # gnn_film/
    for inc in args.include_orders:
        if inc not in logo_delta_map:
            fb = os.path.join(fallback_dir, f"gnn_ism_saliency_{iso}_{inc}.npy")
            if os.path.exists(fb):
                sal = np.load(fb)
                # Cross-isotype normalise: subtract mean across all isotypes for this order
                sibling_arrays = []
                for fn in os.listdir(fallback_dir):
                    if fn.startswith("gnn_ism_saliency_") and fn.endswith(f"_{inc}.npy"):
                        sibling_arrays.append(np.load(os.path.join(fallback_dir, fn))[:sal.shape[0]])
                if sibling_arrays:
                    cross_mean = np.mean(sibling_arrays, axis=0)
                    sal = sal - cross_mean
                logo_delta_map[inc] = sal
                if inc not in order_sp_maps:
                    order_sp_maps[inc] = get_order_sprinzl(iso, inc)
                print(f"  [{iso}/{inc}] using cross-iso-normed FiLM saliency (no delta file)", flush=True)
            else:
                logo_delta_map[inc] = np.zeros((1, 4))
                print(f"  [{iso}/{inc}] no delta or saliency file — blank delta logo", flush=True)

    # ── Optional domain-level background logo at the top ─────────────────────
    bg_sal_normed = None
    bg_sp_to_raw  = {}
    if args.bg_logo:
        bg_path = os.path.join(fallback_dir, f"gnn_ism_saliency_{iso}_{args.bg_logo}.npy")
        if os.path.exists(bg_path):
            bg_sal = np.load(bg_path)
            # Cross-iso normalise in raw-position space using domain-level saliency files
            sib_arrs = []
            for fn in os.listdir(fallback_dir):
                if fn.startswith("gnn_ism_saliency_") and fn.endswith(f"_{args.bg_logo}.npy"):
                    sib_arrs.append(np.load(os.path.join(fallback_dir, fn))[:bg_sal.shape[0]])
            if sib_arrs:
                bg_sal = bg_sal - np.mean(sib_arrs, axis=0)
            # Sprinzl map for domain background (filter df by domain column)
            bg_sub = df_csv[
                (df_csv["Anticodon_predicted_isotype"] == iso) &
                (df_csv["domain"] == args.bg_logo)
            ].dropna(subset=["primary_sequence", "secondary_structure"])
            if not bg_sub.empty:
                modal_len = bg_sub["primary_sequence"].str.len().mode()[0]
                modal_ss  = (bg_sub[bg_sub["primary_sequence"].str.len() == modal_len]
                             ["secondary_structure"].mode()[0])
                bg_sp_map = build_sprinzl_map(modal_ss)
                bg_sp_to_raw = {v: k for k, v in bg_sp_map.items()}
            bg_sal_normed = bg_sal
            print(f"  [{iso}] background logo: {args.bg_logo}", flush=True)
        else:
            print(f"  [{iso}] bg-logo file not found: {bg_path}", flush=True)

    n_bg = 1 if bg_sal_normed is not None else 0
    top_n_logos = len(logo_order_names)
    hr  = ([2] * n_bg) + [2, 1] * top_n_logos
    n_rows = n_bg + top_n_logos * 2
    fig_l, all_axes = plt.subplots(
        n_rows, 1,
        figsize=(max(14, len(sp_labels) * 0.18), n_bg * 2.5 + top_n_logos * 3.2),
        gridspec_kw={"height_ratios": hr},
    )
    if n_rows == 1:
        all_axes = [all_axes]
    fig_l.subplots_adjust(hspace=0.5)

    # ── Background logo panel ─────────────────────────────────────────────────
    if bg_sal_normed is not None:
        ax_bg = all_axes[0]
        rows = []
        for sp_lbl in sp_labels:
            if sp_lbl in bg_sp_to_raw:
                r = bg_sp_to_raw[sp_lbl]
                row = bg_sal_normed[r] if r < bg_sal_normed.shape[0] else np.zeros(4)
            else:
                row = np.zeros(4)
            best  = int(np.argmax(row))
            worst = int(np.argmin(row))
            r_out = [0.0, 0.0, 0.0, 0.0]
            if row[best]  > 0: r_out[best]  = float(row[best])
            if row[worst] < 0: r_out[worst] = float(row[worst])
            rows.append(r_out)
        bg_df = pd.DataFrame(rows, columns=NUCS)
        logomaker.Logo(bg_df, ax=ax_bg, color_scheme="classic",
                       font_name="DejaVu Sans", stack_order="big_on_top",
                       baseline_width=0.5, flip_below=True)
        ax_bg.set_xlim(-0.5, len(sp_labels) - 0.5)
        ax_bg.set_xticks(range(len(sp_labels)))
        ax_bg.set_xticklabels(sp_labels, fontsize=5, rotation=90)
        ax_bg.axhline(0, lw=0.5, color="black")
        ax_bg.set_ylabel("FiLM Δlogit", fontsize=7)
        ax_bg.text(0.01, 0.92, f"{iso} / {args.bg_logo} (background)",
                   transform=ax_bg.transAxes, fontsize=7, fontweight="bold",
                   va="top", color="#333333")
        ax_bg.spines[["top", "right"]].set_visible(False)
        if ac_x:
            ax_bg.axvspan(min(ac_x)-.5, max(ac_x)+.5, color="#e0e0e0", zorder=0)

    for ri, order_name in enumerate(logo_order_names):
        ax      = all_axes[n_bg + ri * 2]
        ax_freq = all_axes[n_bg + ri * 2 + 1]
        arr = logo_delta_map.get(order_name, np.zeros((1, 4)))

        # delta logo — use this order's own Sprinzl map for correct raw positions
        sp_to_raw_order = {v: k for k, v in order_sp_maps.get(order_name, {}).items()}
        rows = []
        for sp_lbl in sp_labels:
            if sp_lbl in sp_to_raw_order:
                r = sp_to_raw_order[sp_lbl]
                row = arr[r] if r < arr.shape[0] else np.zeros(4)
            else:
                row = np.zeros(4)
            best  = int(np.argmax(row))
            worst = int(np.argmin(row))
            r_out = [0.0, 0.0, 0.0, 0.0]
            if row[best]  > 0: r_out[best]  = float(row[best])
            if row[worst] < 0: r_out[worst] = float(row[worst])
            rows.append(r_out)
        logo_df = pd.DataFrame(rows, columns=NUCS)
        logomaker.Logo(logo_df, ax=ax, color_scheme="classic",
                       font_name="DejaVu Sans", stack_order="big_on_top",
                       baseline_width=0.5, flip_below=True)
        ax.set_xlim(-0.5, len(sp_labels) - 0.5)
        ax.set_xticks(range(len(sp_labels)))
        ax.set_xticklabels([], fontsize=0)
        ax.axhline(0, lw=0.5, color="black")
        is_included = order_name in args.include_orders and order_name not in names_top[:8]
        ylabel = "FiLM sal" if is_included else "FiLM Δlogit"
        ax.set_ylabel(ylabel, fontsize=7)
        label_color = "#cc6600" if is_included else "black"
        ax.text(0.01, 0.92, order_name, transform=ax.transAxes,
                fontsize=7, fontweight="bold", va="top", color=label_color)
        ax.spines[["top", "right"]].set_visible(False)
        if ac_x:
            ax.axvspan(min(ac_x)-.5, max(ac_x)+.5, color="#e0e0e0", zorder=0)

        # frequency logo (IC bit score, all positive, matches freq_logos style)
        freq_df = compute_order_ic_df(iso, order_name, sp_labels)
        logomaker.Logo(freq_df, ax=ax_freq, color_scheme="classic",
                       font_name="DejaVu Sans", stack_order="big_on_top",
                       baseline_width=0.5, flip_below=False)
        ax_freq.set_xlim(-0.5, len(sp_labels) - 0.5)
        ax_freq.set_xticks(range(len(sp_labels)))
        ax_freq.set_xticklabels(sp_labels, fontsize=5, rotation=90)
        ax_freq.axhline(0, lw=0.5, color="black")
        ax_freq.set_ylim(0, 2.0)
        ax_freq.set_ylabel("bits", fontsize=6)
        ax_freq.spines[["top", "right"]].set_visible(False)
        if ac_x:
            ax_freq.axvspan(min(ac_x)-.5, max(ac_x)+.5, color="#e0e0e0", zorder=0)

    for ext in ["svg", "png"]:
        path = os.path.join(args.outdir, f"{iso}_top_order_delta_logos.{ext}")
        fig_l.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig_l)
    print(f"  Saved {iso}_top_order_delta_logos", flush=True)

print("\nDone.", flush=True)
