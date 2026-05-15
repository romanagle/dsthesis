#!/usr/bin/env python3
"""
plot_ism_logos_highlight.py
---------------------------
Like plot_ism_logos.py, but draws a red box around every Sprinzl position
that appears in the known-identity-elements table (known_IEs.csv).

Usage
-----
    python scripts/plot_ism_logos_highlight.py \
        --emb-dir gnn_final \
        --known-ies data/known_IEs.csv \
        --outdir gnn_final/ism_logos_highlight
"""

from __future__ import annotations

import argparse
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logomaker

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir",  default="gnn_final",
                    help="Directory with perpos_embeddings.npz")
parser.add_argument("--ism-dir",  default=None,
                    help="Directory with gnn_ism_saliency_*.npy files "
                         "(default: same as --emb-dir)")
parser.add_argument("--csv",      default="data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--known-ies", default="data/known_IEs.csv",
                    help="CSV table of known identity elements (Giegé et al. style)")
parser.add_argument("--outdir",   default=None,
                    help="Output directory (default: {ism-dir}/ism_logos_highlight)")
parser.add_argument("--isotypes", nargs="+", default=None)
parser.add_argument("--domains",  nargs="+", default=None)
parser.add_argument("--min-sal",  type=float, default=0.002)
parser.add_argument("--top-n",    type=int,   default=None)
parser.add_argument("--ac-pos",   nargs="+",  default=["34", "35", "36", "74", "75", "76"])
parser.add_argument("--clade-col", default="domain")
parser.add_argument("--no-cross-iso-norm", action="store_true")
args = parser.parse_args()

ism_dir = args.ism_dir or args.emb_dir
outdir  = args.outdir or os.path.join(ism_dir, "ism_logos_highlight")
os.makedirs(outdir, exist_ok=True)

NUCS         = ["A", "C", "G", "U"]
AC_EXCL      = set(args.ac_pos)
AC_LOOP_LBLS = {"34", "35", "36"}

# ── Known identity elements ───────────────────────────────────────────────────
# Map CSV isotype names to the 3-letter codes used in saliency files
_ISO_NAME_MAP = {
    "Met f": "Met",
    "Ile2":  "Ile",
}

def _parse_ie_positions(text: str) -> set[str]:
    """
    Extract Sprinzl position labels from a free-text IE string.

    Handles:
      "A/G73, U73, G2–C71"  → {"73", "2", "71"}
      "G37c"                 → {"37", "37c"}   (c = footnote marker)
      "17a", "20a", "20b"    → kept as-is (valid Sprinzl suffixes)
      "…", "TId", "largeVRe" → ignored (no numeric content)
    """
    if not text or str(text).strip() in ("…", "...", "—", "", "nan"):
        return set()
    raw = re.findall(r"-?\d+[a-z]?", str(text))
    positions: set[str] = set()
    for tok in raw:
        positions.add(tok)
        # Always add the bare numeric form so footnote suffixes (37c→37) match
        numeric = re.match(r"-?\d+", tok).group()
        positions.add(numeric)
    return positions


def load_known_ies(path: str) -> dict[tuple[str, str], set[str]]:
    """
    Parse known_IEs.csv into {(isotype, domain): set_of_sprinzl_positions}.

    Domain values are normalised to "Bacteria" / "Archaea".
    """
    df = pd.read_csv(path, header=None)
    # Row 0 = column header row ("tRNA families", "Phylogenetic domain", …)
    # Row 1 = region sub-headers (Acceptor branch, Core region, Anticodon branch)
    # Rows 2+ = data
    result: dict[tuple[str, str], set[str]] = {}
    current_iso = None
    for _, row in df.iloc[2:].iterrows():
        vals = [str(v).strip() for v in row]
        iso_cell    = vals[0]
        domain_cell = vals[1]
        ie_cells    = vals[2:]

        # New isotype block if col 0 is non-empty
        if iso_cell and iso_cell.lower() != "nan":
            current_iso = _ISO_NAME_MAP.get(iso_cell, iso_cell)

        if current_iso is None:
            continue

        # Normalise domain
        dom_raw = domain_cell.strip("() ")
        if dom_raw in ("Bac",):
            domain = "Bacteria"
        elif dom_raw in ("Arc",):
            domain = "Archaea"
        else:
            continue

        positions: set[str] = set()
        for cell in ie_cells:
            positions |= _parse_ie_positions(cell)

        if positions:
            result[(current_iso, domain)] = positions

    return result


known_ies = load_known_ies(args.known_ies)
print(f"Loaded known IEs for {len(known_ies)} (isotype × domain) pairs:", flush=True)
for (iso, dom), pos in sorted(known_ies.items()):
    print(f"  {iso}/{dom}: {sorted(pos)}", flush=True)

# ── Sprinzl map ───────────────────────────────────────────────────────────────
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


_sprinzl_cache = {}

def get_sprinzl_map(iso, domain):
    key = (iso, domain)
    if key not in _sprinzl_cache:
        usecols = ["primary_sequence", "secondary_structure",
                   "Anticodon_predicted_isotype", args.clade_col]
        if args.clade_col != "domain":
            usecols.append("domain")
        df = pd.read_csv(args.csv, usecols=list(set(usecols)), low_memory=False)
        sub = df[(df["Anticodon_predicted_isotype"] == iso) &
                 (df[args.clade_col] == domain)].dropna(
            subset=["primary_sequence", "secondary_structure"])
        if sub.empty:
            _sprinzl_cache[key] = {}
        else:
            modal_len = sub["primary_sequence"].str.len().mode()[0]
            modal_ss  = (sub[sub["primary_sequence"].str.len() == modal_len]
                         ["secondary_structure"].mode()[0])
            _sprinzl_cache[key] = build_sprinzl_map(modal_ss)
    return _sprinzl_cache[key]

# ── Load NPZ metadata ─────────────────────────────────────────────────────────
npz_path = os.path.join(args.emb_dir, "perpos_embeddings.npz")
print(f"Loading metadata from {npz_path} …", flush=True)
data    = np.load(npz_path, allow_pickle=True)
max_len = int(data["max_len"])
classes = list(data["classes"].astype(str)) if "classes" in data else []
print(f"  max_len={max_len}  classes={classes}", flush=True)

# ── Discover saliency files ───────────────────────────────────────────────────
found = {}
for fname in os.listdir(ism_dir):
    if not (fname.startswith("gnn_ism_saliency_") and fname.endswith(".npy")):
        continue
    stem = fname[len("gnn_ism_saliency_"):-len(".npy")]
    matched = False
    for iso in (classes or [stem.split("_")[0]]):
        if stem.startswith(iso + "_"):
            dom = stem[len(iso) + 1:]
            found[(iso, dom)] = os.path.join(ism_dir, fname)
            matched = True
            break
    if not matched:
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            found[(parts[0], parts[1])] = os.path.join(ism_dir, fname)

if not found:
    raise SystemExit(
        f"No gnn_ism_saliency_{{iso}}_{{domain}}.npy files found in {ism_dir}."
    )

pairs = sorted(found.keys())
if args.isotypes:
    pairs = [(iso, dom) for iso, dom in pairs if iso in args.isotypes]
if args.domains:
    pairs = [(iso, dom) for iso, dom in pairs if dom in args.domains]

print(f"Plotting {len(pairs)} (isotype × domain) combinations …", flush=True)

# ── Cross-isotype mean ────────────────────────────────────────────────────────
_domain_mean_cache = {}

def get_domain_mean(domain, max_len):
    if domain not in _domain_mean_cache:
        arrays = []
        for (iso, dom), path in found.items():
            if dom == domain:
                arrays.append(np.load(path)[:max_len])
        _domain_mean_cache[domain] = np.mean(arrays, axis=0) if arrays else np.zeros((max_len, 4))
    return _domain_mean_cache[domain]


# ── Logo helpers ──────────────────────────────────────────────────────────────
def select_positions(sal, raw_to_sp):
    norms = np.linalg.norm(sal, axis=1)
    keep  = [p for p in range(len(norms))
             if norms[p] >= args.min_sal
             and p in raw_to_sp
             and raw_to_sp[p] not in AC_EXCL]
    if args.top_n and len(keep) > args.top_n:
        keep = sorted(keep, key=lambda p: norms[p], reverse=True)[:args.top_n]
        keep = sorted(keep)
    return keep


def build_logo_df(sal, display_positions, active_set):
    rows = []
    for p in display_positions:
        if p not in active_set:
            rows.append([0.0, 0.0, 0.0, 0.0])
        else:
            row = sal[p]
            best, worst = int(np.argmax(row)), int(np.argmin(row))
            logo_row = [0.0, 0.0, 0.0, 0.0]
            if row[best]  > 0: logo_row[best]  = float(row[best])
            if row[worst] < 0: logo_row[worst] = float(row[worst])
            rows.append(logo_row)
    return pd.DataFrame(rows, columns=NUCS)


def _shade_highlights(ax, sal, tick_labs, disp_pos, ac_set, known_pos):
    """
    Green: known IE from literature AND |max Δlogit| >= 2.
    Red:   known IE from literature but |max Δlogit| < 2.
    """
    HIGH_SAL_THRESH = 2.0
    for col_idx, (sp_lbl, disp_p) in enumerate(zip(tick_labs, disp_pos)):
        if sp_lbl not in known_pos or disp_p in ac_set:
            continue
        strong = disp_p < len(sal) and np.max(np.abs(sal[disp_p])) >= HIGH_SAL_THRESH
        ax.axvspan(col_idx - 0.48, col_idx + 0.48,
                   color="green" if strong else "red", alpha=0.15, zorder=2)


def plot_logo(ax, sal, iso, domain):
    raw_to_sp  = get_sprinzl_map(iso, domain)
    if not args.no_cross_iso_norm:
        domain_mean = get_domain_mean(domain, len(sal))
        sal = sal - domain_mean
    active_positions = select_positions(sal, raw_to_sp)
    if not active_positions:
        ax.text(0.5, 0.5, "no signal", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title(f"{iso} / {domain}", fontsize=8)
        return 0

    sp_to_raw   = {v: k for k, v in raw_to_sp.items()}
    ac_raw      = sorted(sp_to_raw[lbl] for lbl in AC_LOOP_LBLS if lbl in sp_to_raw)
    disp_pos    = sorted(set(active_positions) | set(ac_raw))
    active_set  = set(active_positions)
    ac_set      = set(ac_raw)

    logo_df   = build_logo_df(sal, disp_pos, active_set)
    tick_labs = [raw_to_sp.get(p, str(p)) for p in disp_pos]

    logomaker.Logo(logo_df, ax=ax, color_scheme="classic",
                   font_name="DejaVu Sans", stack_order="big_on_top",
                   baseline_width=0.5, flip_below=True)
    ax.set_xticks(range(len(tick_labs)))
    ax.set_xticklabels(tick_labs, fontsize=6, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.4, zorder=0)
    ax.set_xlim(-0.5, len(tick_labs) - 0.5)

    # Shade the anticodon gap
    ac_cols = [i for i, p in enumerate(disp_pos) if p in ac_set]
    if ac_cols:
        ax.axvspan(ac_cols[0] - 0.5, ac_cols[-1] + 0.5,
                   color="lightgrey", alpha=0.35, zorder=0)
        for lbl_obj, sp_lbl in zip(ax.get_xticklabels(), tick_labs):
            if sp_lbl in AC_LOOP_LBLS:
                lbl_obj.set_color("grey")

    # Green = high |Δlogit|, red = known IE (red drawn last to take priority)
    known_pos = known_ies.get((iso, domain), set())
    _shade_highlights(ax, sal, tick_labs, disp_pos, ac_set, known_pos)

    return len(active_positions)

# ── Individual logos ──────────────────────────────────────────────────────────
for iso, domain in pairs:
    sal = np.load(found[(iso, domain)])[:max_len]

    raw_to_sp = get_sprinzl_map(iso, domain)
    positions = select_positions(sal, raw_to_sp)
    if not positions:
        print(f"  [{iso}/{domain}] no positions above min-sal — skipping", flush=True)
        continue

    fig_w = max(8, 2 + len(positions) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))
    n_pos = plot_logo(ax, sal, iso, domain)
    ax.set_ylabel("mean ISM Δlogit", fontsize=9)
    ax.set_title(
        f"tRNA-{iso} ({domain})  |  GNN ISM identity logo\n"
        f"min |ISM| = {args.min_sal}  |  anticodon (34–36) masked  |  "
        f"cross-isotype normalized  |  Sprinzl positions  |  "
        f"green = |Δlogit| ≥ 2  |  red = known IE (literature)",
        fontsize=9,
    )
    fig.tight_layout()
    stem = f"ism_logo_{iso}_{domain}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{iso}/{domain}] {n_pos} positions → {stem}.{{pdf,png}}", flush=True)

# ── Per-isotype grid: domains as columns ─────────────────────────────────────
isos_present = sorted(set(iso for iso, _ in pairs))
doms_present = sorted(set(dom for _, dom in pairs))

for iso in isos_present:
    dom_cols = [dom for dom in doms_present if (iso, dom) in found]
    if not dom_cols:
        continue

    fig, axes = plt.subplots(1, len(dom_cols),
                             figsize=(max(10, len(dom_cols) * 7), 4),
                             squeeze=False)
    for col, dom in enumerate(dom_cols):
        ax  = axes[0][col]
        sal = np.load(found[(iso, dom)])[:max_len]
        plot_logo(ax, sal, iso, dom)
        ax.set_ylabel("Δlogit", fontsize=8)
        ax.set_title(f"{dom}", fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"tRNA-{iso}  —  GNN ISM identity logo by domain\n"
        f"min |ISM| = {args.min_sal}  |  anticodon masked  |  "
        f"cross-isotype normalized  |  red box = known IE",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    stem = f"ism_logo_{iso}_all_domains"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{iso}] domain grid → {stem}.{{pdf,png}}", flush=True)

# ── All-isotypes grid ─────────────────────────────────────────────────────────
if len(isos_present) > 1 and len(doms_present) >= 1:
    N_ROWS = len(isos_present)
    N_COLS = len(doms_present)
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(N_COLS * 7, N_ROWS * 3.2),
                             squeeze=False)

    for row, iso in enumerate(isos_present):
        for col, dom in enumerate(doms_present):
            ax = axes[row][col]
            if (iso, dom) in found:
                sal = np.load(found[(iso, dom)])[:max_len]
                plot_logo(ax, sal, iso, dom)
            else:
                ax.set_visible(False)
                continue
            ax.tick_params(axis="y", labelsize=6)
            if row == 0:
                ax.set_title(dom, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{iso}\nΔlogit", fontsize=8)

    fig.suptitle(
        f"GNN ISM identity logos — all isotypes × domains\n"
        f"min |ISM| = {args.min_sal}  |  anticodon masked  |  "
        f"cross-isotype normalized  |  red box = known IE (literature)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    stem = "ism_logo_all_isotypes_all_domains"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nFull grid → {stem}.{{pdf,png}}", flush=True)

    # Shared-scale version
    fig2, axes2 = plt.subplots(N_ROWS, N_COLS,
                               figsize=(N_COLS * 7, N_ROWS * 3.2),
                               squeeze=False)
    for row, iso in enumerate(isos_present):
        for col, dom in enumerate(doms_present):
            ax = axes2[row][col]
            if (iso, dom) in found:
                sal = np.load(found[(iso, dom)])[:max_len]
                plot_logo(ax, sal, iso, dom)
            else:
                ax.set_visible(False)
                continue
            ax.tick_params(axis="y", labelsize=6)
            if row == 0:
                ax.set_title(dom, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{iso}\nΔlogit", fontsize=8)

    all_ymins, all_ymaxs = [], []
    for ax in axes2.flat:
        if ax.get_visible() and (ax.lines or ax.patches):
            ylo, yhi = ax.get_ylim()
            all_ymins.append(ylo); all_ymaxs.append(yhi)
    if all_ymins:
        g_ymin, g_ymax = min(all_ymins), max(all_ymaxs)
        for ax in axes2.flat:
            if ax.get_visible():
                ax.set_ylim(g_ymin, g_ymax)

    fig2.suptitle(
        f"GNN ISM identity logos — all isotypes × domains  [shared y-axis scale]\n"
        f"min |ISM| = {args.min_sal}  |  anticodon masked  |  "
        f"cross-isotype normalized  |  red box = known IE (literature)",
        fontsize=12, y=1.01,
    )
    fig2.tight_layout()
    stem2 = "ism_logo_all_isotypes_all_domains_shared_scale"
    for ext in ("pdf", "png"):
        fig2.savefig(os.path.join(outdir, f"{stem2}.{ext}"),
                     bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"Shared-scale grid → {stem2}.{{pdf,png}}", flush=True)

# ── Structural-class grids ────────────────────────────────────────────────────
TRNA_CLASSES = {
    "1a": ["Arg", "Cys", "Ile", "Leu", "Met", "Val"],
    "1b": ["Glu", "Gln"],
    "1c": ["Trp", "Tyr"],
    "2a": ["Ala", "Gly", "His", "Pro", "Ser", "Thr"],
    "2b": ["Asp", "Asn", "Lys"],
    "2c": ["Phe"],
}

for cls_name, cls_isos in TRNA_CLASSES.items():
    cls_pairs = [(iso, dom) for iso, dom in pairs
                 if iso in cls_isos and (iso, dom) in found]
    if not cls_pairs:
        continue
    cls_isos_present = sorted(set(iso for iso, _ in cls_pairs),
                               key=lambda x: cls_isos.index(x) if x in cls_isos else 99)
    cls_doms_present = sorted(set(dom for _, dom in cls_pairs))

    N_ROWS = len(cls_isos_present)
    N_COLS = len(cls_doms_present)
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                              figsize=(N_COLS * 7, N_ROWS * 3.2),
                              squeeze=False)
    for row, iso in enumerate(cls_isos_present):
        for col, dom in enumerate(cls_doms_present):
            ax = axes[row][col]
            if (iso, dom) in found:
                sal = np.load(found[(iso, dom)])[:max_len]
                plot_logo(ax, sal, iso, dom)
            else:
                ax.set_visible(False)
                continue
            ax.tick_params(axis="y", labelsize=6)
            if row == 0:
                ax.set_title(dom, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{iso}\nΔlogit", fontsize=8)

    fig.suptitle(
        f"GNN ISM identity logos — Type {cls_name} tRNAs\n"
        f"min |ISM| = {args.min_sal}  |  anticodon masked  |  cross-isotype normalized  |  "
        f"green = |Δlogit| ≥ 2 & known IE  |  red = known IE (literature)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    stem = f"ism_logo_class_{cls_name}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [class {cls_name}] → {stem}.{{pdf,png}}", flush=True)

print(f"\nDone. Results in: {outdir}/")
