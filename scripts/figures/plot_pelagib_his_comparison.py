#!/usr/bin/env python3
"""
plot_pelagib_his_comparison.py
------------------------------
Three-panel comparison for Pelagibacterales His tRNA:
  1. Normal ISM logo (GNN saliency, cross-iso normalized)
  2. Background-subtracted ISM logo (Pelagibacterales - Gammaproteobacteria class)
  3. Frequency log-odds logo (vs all Bacteria)

Saves: dsthesis/figures/ie_comparison_final/Pelagibacterales_His_comparison.{svg,png}
"""

from __future__ import annotations
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import logomaker

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")
sys.path.insert(0, os.path.join(DATA_ROOT, "scripts"))

ISO   = "His"
CLADE = "Pelagibacterales"
NUCS  = ["A", "C", "G", "U"]
AC_EXCL     = {"34", "35", "36", "74", "75", "76"}
AC_LOOP     = {"34", "35", "36"}
HIGHLIGHT   = {"73", "-1"}   # pos 73 (discriminator) and G-1
MIN_SAL     = 0.002
MIN_LOGODDS = 0.3
PSEUDOCOUNT = 0.5

import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--ism-dir", default=os.path.join(DATA_ROOT, "gnn_final/order_ism_culturable"))
_p.add_argument("--bg-dir",  default=os.path.join(DATA_ROOT, "gnn_final/order_ism_culturable/bac_bg"))
_p.add_argument("--outdir",  default=os.path.join(DATA_ROOT, "figures/ie_comparison_final"))
_args, _ = _p.parse_known_args()

ISM_DIR    = _args.ism_dir
BG_DIR     = _args.bg_dir
OUTDIR     = _args.outdir
NPZ_PATH   = os.path.join(DATA_ROOT, "embeddings/gnn/perpos_embeddings.npz")
CSV_PATH   = os.path.join(DATA_ROOT, "data/trnascan_GTDB_r226_dedup_taxonomy.csv")
os.makedirs(OUTDIR, exist_ok=True)

# ── Sprinzl helpers (copied from plot_ism_logos.py) ───────────────────────────
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == '>': stack.append(i)
        elif c == '<':
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss):
    pairs = parse_pairs(ss)
    n = len(ss)
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
    overhang = [p for p in range(acc5[0]) if p not in pairs]
    for i, p in enumerate(reversed(overhang)):
        sp[p] = str(-(i + 1))
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


def _is_integer_sp(label):
    try: int(label); return True
    except (ValueError, TypeError): return False


def _fill_canonical_5x(disp_pos, tick_labs):
    sp_int_map = {int(lbl): p for p, lbl in zip(disp_pos, tick_labs)
                  if _is_integer_sp(lbl) and int(lbl) > 0}
    if not sp_int_map: return disp_pos, tick_labs
    lo, hi = min(sp_int_map), max(sp_int_map)
    missing = [n for n in range(5, hi+1, 5) if lo <= n <= hi and n not in sp_int_map]
    if not missing: return disp_pos, tick_labs
    pairs = list(zip(disp_pos, tick_labs))
    for n in missing:
        prev_n = max((m for m in sp_int_map if m < n), default=None)
        next_n = min((m for m in sp_int_map if m > n), default=None)
        if prev_n is not None and next_n is not None:
            fake = (sp_int_map[prev_n] + sp_int_map[next_n]) / 2.0
        elif prev_n is not None: fake = float(sp_int_map[prev_n]) + 0.5
        else: fake = float(sp_int_map[next_n]) - 0.5
        pairs.append((fake, str(n)))
    pairs.sort(key=lambda x: x[0])
    return [p for p, _ in pairs], [lbl for _, lbl in pairs]


# ── Get Sprinzl map for His/Pelagibacterales ──────────────────────────────────
print("Building Sprinzl map for His/Pelagibacterales …")
df_sp = pd.read_csv(CSV_PATH,
    usecols=["primary_sequence", "secondary_structure",
             "Anticodon_predicted_isotype", "order"],
    low_memory=False)
sub_sp = df_sp[(df_sp["Anticodon_predicted_isotype"] == ISO) &
               (df_sp["order"] == CLADE)].dropna(
    subset=["primary_sequence", "secondary_structure"])
modal_len = sub_sp["primary_sequence"].str.len().mode()[0]
modal_ss  = sub_sp[sub_sp["primary_sequence"].str.len() == modal_len]["secondary_structure"].mode()[0]
raw_to_sp = build_sprinzl_map(modal_ss)
sp_to_raw = {v: k for k, v in raw_to_sp.items()}
print(f"  Modal SS length: {modal_len},  Sprinzl positions mapped: {len(raw_to_sp)}")

max_len = int(np.load(NPZ_PATH, allow_pickle=True)["max_len"])


# ── ISM logo helpers ──────────────────────────────────────────────────────────
def build_logo_df_ism(sal, display_positions, active_set, stacked=False):
    rows = []
    for p in display_positions:
        if not isinstance(p, (int, np.integer)) or p not in active_set:
            rows.append([0.0, 0.0, 0.0, 0.0])
        elif stacked:
            # Show all positive contributions above, all negative below
            row = sal[p]
            rows.append([float(v) for v in row])
        else:
            row = sal[p]
            best, worst = int(np.argmax(row)), int(np.argmin(row))
            r = [0.0, 0.0, 0.0, 0.0]
            if row[best]  > 0: r[best]  = float(row[best])
            if row[worst] < 0: r[worst] = float(row[worst])
            rows.append(r)
    return pd.DataFrame(rows, columns=NUCS)


def plot_ism_panel(ax, sal, title, subtract_mean_sal=None, stacked=False):
    if subtract_mean_sal is not None:
        sal = sal - subtract_mean_sal

    norms = np.linalg.norm(sal, axis=1)
    active = [p for p in range(len(norms))
              if norms[p] >= MIN_SAL and p in raw_to_sp and raw_to_sp[p] not in AC_EXCL]

    ac_raw = sorted(sp_to_raw[lbl] for lbl in AC_LOOP if lbl in sp_to_raw)
    disp   = sorted(set(active) | set(ac_raw))
    active_set = set(active)

    tick_labs = [raw_to_sp.get(p, str(p)) for p in disp]
    disp, tick_labs = _fill_canonical_5x(disp, tick_labs)

    logo_df = build_logo_df_ism(sal, disp, active_set, stacked=stacked)
    logomaker.Logo(logo_df, ax=ax, color_scheme="classic",
                   font_name="DejaVu Sans", stack_order="big_on_top",
                   baseline_width=0.5, flip_below=True)

    _style_ax(ax, disp, tick_labs, title)
    return disp, tick_labs


def _style_ax(ax, disp, tick_labs, title):
    ax.set_xticks(range(len(disp)))
    ax.set_xticklabels(tick_labs, fontsize=5.5, rotation=90)
    for lbl, xt in zip(tick_labs, ax.get_xticklabels()):
        if lbl in HIGHLIGHT:
            xt.set_color("red"); xt.set_fontweight("bold")
        elif lbl in AC_LOOP:
            xt.set_color("#888888")

    # anticodon shading
    ac_x = [i for i, lbl in enumerate(tick_labs) if lbl in AC_LOOP]
    if len(ac_x) >= 2:
        ax.axvspan(min(ac_x)-.5, max(ac_x)+.5, color="#e0e0e0", zorder=0)

    ax.axhline(0, lw=0.6, color="black")
    ax.set_xlabel("Sprinzl position", fontsize=8)
    ax.set_ylabel("Δlogit", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.01, 0.97, title, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top")


# ── Frequency log-odds panel ───────────────────────────────────────────────────
def plot_freq_panel(ax, title):
    df = pd.read_csv(CSV_PATH,
        usecols=["primary_sequence", "secondary_structure",
                 "Anticodon_predicted_isotype", "domain", "order"],
        low_memory=False).dropna(
        subset=["primary_sequence", "secondary_structure",
                "Anticodon_predicted_isotype", "domain"])
    df["primary_sequence"] = df["primary_sequence"].str.upper().str.replace("T", "U", regex=False)

    clade_df = df[(df["Anticodon_predicted_isotype"] == ISO) & (df["order"] == CLADE)]
    bac_df   = df[(df["Anticodon_predicted_isotype"] == ISO) & (df["domain"] == "Bacteria")]

    print(f"  Freq logo: {len(clade_df)} Pelagibacterales His  vs  {len(bac_df)} Bacteria His")

    # Build per-position counts using the modal Sprinzl map
    def count_nucs(sub_df):
        counts = {}  # sp_label → {A,C,G,U: count}
        for _, row in sub_df.iterrows():
            seq = row["primary_sequence"]
            ss  = row["secondary_structure"]
            sp  = build_sprinzl_map(ss)
            for raw_i, lbl in sp.items():
                if lbl in AC_EXCL: continue
                if raw_i >= len(seq): continue
                nuc = seq[raw_i]
                if nuc not in {"A", "C", "G", "U"}: continue
                counts.setdefault(lbl, {"A":0,"C":0,"G":0,"U":0})[nuc] += 1
        return counts

    fg = count_nucs(clade_df)
    bg = count_nucs(bac_df)

    all_lbl = sorted(set(fg) | set(bg), key=lambda x: sp_to_raw.get(x, 999))

    logodds = {}
    for lbl in all_lbl:
        if lbl in AC_EXCL: continue
        fg_c = fg.get(lbl, {})
        bg_c = bg.get(lbl, {})
        fg_tot = sum(fg_c.values()) + 4*PSEUDOCOUNT
        bg_tot = sum(bg_c.values()) + 4*PSEUDOCOUNT
        row = []
        for n in NUCS:
            p_fg = (fg_c.get(n, 0) + PSEUDOCOUNT) / fg_tot
            p_bg = (bg_c.get(n, 0) + PSEUDOCOUNT) / bg_tot
            row.append(np.log2(p_fg / p_bg))
        logodds[lbl] = row

    # Filter positions with signal, map to raw positions
    active_lbls = [lbl for lbl, row in logodds.items()
                   if max(abs(v) for v in row) >= MIN_LOGODDS
                   and lbl not in AC_EXCL and lbl in sp_to_raw]

    ac_raw = sorted(sp_to_raw[lbl] for lbl in AC_LOOP if lbl in sp_to_raw)
    all_raw = sorted(set(sp_to_raw[lbl] for lbl in active_lbls) | set(ac_raw))
    active_set = set(sp_to_raw[lbl] for lbl in active_lbls)

    tick_labs = [raw_to_sp.get(p, str(p)) for p in all_raw]
    all_raw_disp, tick_labs = _fill_canonical_5x(all_raw, tick_labs)

    rows = []
    for p in all_raw_disp:
        if not isinstance(p, (int, np.integer)) or p not in active_set:
            rows.append([0.0, 0.0, 0.0, 0.0])
        else:
            lbl = raw_to_sp.get(p)
            lo  = logodds.get(lbl, [0,0,0,0])
            best, worst = int(np.argmax(lo)), int(np.argmin(lo))
            r = [0.0, 0.0, 0.0, 0.0]
            if lo[best]  > 0: r[best]  = float(lo[best])
            if lo[worst] < 0: r[worst] = float(lo[worst])
            rows.append(r)

    logo_df = pd.DataFrame(rows, columns=NUCS)
    logomaker.Logo(logo_df, ax=ax, color_scheme="classic",
                   font_name="DejaVu Sans", stack_order="big_on_top",
                   baseline_width=0.5, flip_below=True)
    _style_ax(ax, all_raw_disp, tick_labs, title)
    ax.set_ylabel("log₂ odds", fontsize=8)


# ── Load saliency ─────────────────────────────────────────────────────────────
print("Loading saliency files …")
sal_normal = np.load(os.path.join(ISM_DIR, f"gnn_ism_saliency_{ISO}_{CLADE}.npy"))
sal_bg     = np.load(os.path.join(BG_DIR,  f"gnn_ism_saliency_{ISO}_{CLADE}.npy"))

# Cross-isotype mean for normal ISM (per-domain mean over all isotypes in ISM_DIR)
# Build from all saliency files in ISM_DIR whose suffix == CLADE
data_npz = np.load(NPZ_PATH, allow_pickle=True)
classes  = list(data_npz["classes"].astype(str))

all_sal = []
for iso2 in classes:
    p = os.path.join(ISM_DIR, f"gnn_ism_saliency_{iso2}_{CLADE}.npy")
    if os.path.exists(p):
        all_sal.append(np.load(p)[:max_len])
cross_mean = np.mean(all_sal, axis=0) if all_sal else None
print(f"  Cross-iso mean built from {len(all_sal)} isotypes for {CLADE}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
fig.subplots_adjust(hspace=0.6)

plot_ism_panel(axes[0], sal_normal[:max_len],
               f"ISM  —  {ISO} / {CLADE}",
               subtract_mean_sal=cross_mean)

plot_ism_panel(axes[1], sal_bg[:max_len],
               f"ISM (Pelagibacterales − Bacteria, cross-iso normed)  —  {ISO}",
               stacked=True)

plot_freq_panel(axes[2],
                f"Frequency log-odds  —  {ISO} / {CLADE}  vs  Bacteria")

for stem in [f"{CLADE}_{ISO}_comparison"]:
    for ext in ["svg", "png"]:
        path = os.path.join(OUTDIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

plt.close(fig)
print("Done.")
