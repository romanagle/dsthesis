#!/usr/bin/env python3
"""
rank_ism_vs_known_IEs.py
--------------------------
Rank-based comparison of GNN ISM saliency scores against known tRNA identity
elements (IEs) from the literature (known_IEs.csv).

For each (isotype, domain):
  1. Rank Sprinzl positions by max|ISM saliency| descending.
  2. Check which known IEs appear in top-k.
  3. Compute recall@k and AUROC.

Outputs
-------
  {outdir}/{iso}_{domain}_ism_vs_IE.png   per-isotype saliency bar with IEs marked
  {outdir}/recall_heatmap.png             recall@k heatmap across isotypes × k
  {outdir}/recall_curves.png              recall@k curves (one line per isotype)
  {outdir}/rank_stats.csv                 summary stats table

Usage
-----
    python dsthesis/scripts/figures/rank_ism_vs_known_IEs.py
    python dsthesis/scripts/figures/rank_ism_vs_known_IEs.py --domains Bacteria
    python dsthesis/scripts/figures/rank_ism_vs_known_IEs.py --excl-anticodon
    python dsthesis/scripts/figures/rank_ism_vs_known_IEs.py --top-k 5 10 15 20
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir", default="gnn_final")
parser.add_argument("--csv", default="data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--ie-csv", default="data/known_IEs.csv")
parser.add_argument("--outdir", default=None)
parser.add_argument("--domains", nargs="+", default=None,
                    help="Domains to evaluate (default: Bacteria Archaea)")
parser.add_argument("--isotypes", nargs="+", default=None,
                    help="Isotypes to evaluate (default: all found)")
parser.add_argument("--top-k", nargs="+", type=int, default=[5, 10, 15, 20],
                    help="Recall@k thresholds")
parser.add_argument("--excl-anticodon", action="store_true",
                    help="Exclude anticodon positions (34/35/36) from ranking and IEs")
parser.add_argument("--min-sal", type=float, default=0.0,
                    help="Min max|saliency| to include a position in plots (default: 0.0)")
args = parser.parse_args()

outdir = args.outdir or os.path.join("results", "figures", "ism_logos", "ie_comparison")
os.makedirs(outdir, exist_ok=True)

NUCS = ["A", "C", "G", "U"]
AC_POSITIONS = {"34", "35", "36"}

# D-loop positions with valid a/b Sprinzl suffixes
VALID_AB_POS = {"17a", "20a", "20b"}

# ── Sprinzl map (copied from plot_pairwise_ism_comparison_logos.py) ───────────
def parse_pairs_ss(ss: str) -> dict[int, int]:
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == '>': stack.append(i)
        elif c == '<':
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss: str) -> dict[int, str]:
    pairs = parse_pairs_ss(ss)
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
    for i, p in enumerate(reversed(overhang)):
        sp[p] = str(-(i + 1))
    for i, p in enumerate(acc5): sp[p] = str(i + 1)
    for i, p in enumerate(acc3): sp[p] = str(66 + i)
    disc = sorted(p for p in range(acc3[-1] + 1, n) if p not in pairs)
    for i, p in enumerate(disc): sp[p] = str(73 + i)
    d5 = stems[1]
    for i, p in enumerate([p for p in range(acc5[-1] + 1, d5[0]) if p not in pairs]):
        sp[p] = str(8 + i)
    d3 = sorted(pairs[p] for p in d5)
    nd = len(d5)
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
    t5 = stems[-1]
    t3 = sorted(pairs[p] for p in t5)
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


_sp_cache: dict[tuple[str, str], dict[int, str]] = {}

def get_sprinzl_map(iso: str, domain: str) -> dict[int, str]:
    key = (iso, domain)
    if key not in _sp_cache:
        df = pd.read_csv(
            args.csv,
            usecols=["primary_sequence", "secondary_structure",
                     "Anticodon_predicted_isotype", "domain"],
            low_memory=False,
        )
        sub = df[
            (df["Anticodon_predicted_isotype"] == iso) &
            (df["domain"] == domain)
        ].dropna(subset=["primary_sequence", "secondary_structure"])
        if sub.empty:
            _sp_cache[key] = {}
        else:
            modal_len = sub["primary_sequence"].str.len().mode()[0]
            modal_ss  = (sub[sub["primary_sequence"].str.len() == modal_len]
                         ["secondary_structure"].mode()[0])
            _sp_cache[key] = build_sprinzl_map(modal_ss)
    return _sp_cache[key]


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


# ── Known IE parser ───────────────────────────────────────────────────────────
_DOMAIN_MAP = {"(Bac)": "Bacteria", "(Arc)": "Archaea"}
_ISO_MAP     = {"Met f": "Met"}

# TId = T-loop identity elements (T54, Ψ55, C56 in Sprinzl)
_TID_POSITIONS = {"54", "55", "56"}


def _normalize_iso(raw: str) -> str:
    raw = raw.strip()
    return _ISO_MAP.get(raw, raw)


def _extract_positions(cell: str) -> set[str]:
    """Extract Sprinzl position labels from one CSV cell."""
    if not isinstance(cell, str):
        return set()
    cell = cell.strip()
    if cell in ("…", "—", ""):
        return set()

    positions: set[str] = set()

    if "TId" in cell:
        positions.update(_TID_POSITIONS)

    # Remove non-positional keywords so their letters don't confuse the regex
    cleaned = re.sub(r'largeVR[a-z]?|TId|Δ|…|\.\.\.', ' ', cell)

    # Match: optional leading nuc letters, then position number + optional a/b suffix
    # Uses findall so base pairs (G2–C71) yield both numbers
    for m in re.finditer(r'(?:[ACGURYMSWacgurymswACGU/]*)(-?\d+[ab]?)(?:[ceghf])?', cleaned):
        pos = m.group(1)
        if not pos:
            continue
        # Strip trailing [ab] if it's not a known valid Sprinzl suffix (17a, 20a, 20b)
        if pos[-1] in "ab" and pos not in VALID_AB_POS:
            pos = pos[:-1]
        positions.add(pos)

    return positions


def parse_known_IEs(ie_csv: str) -> dict[tuple[str, str], set[str]]:
    """
    Returns {(isotype, domain): set_of_sprinzl_labels}.
    domain is 'Bacteria' or 'Archaea'.
    """
    result: dict[tuple[str, str], set[str]] = {}
    current_iso = ""

    with open(ie_csv, encoding="utf-8") as fh:
        lines = fh.readlines()

    for line in lines[2:]:  # skip 2 header rows
        parts = [p.strip().strip('"') for p in line.rstrip("\n").split(",")]
        # Pad to at least 5 columns
        while len(parts) < 5:
            parts.append("")

        iso_raw, dom_raw = parts[0], parts[1]
        cells = parts[2:]  # acceptor branch, core, anticodon branch

        if iso_raw:
            current_iso = _normalize_iso(iso_raw)

        domain = _DOMAIN_MAP.get(dom_raw)
        if domain is None or not current_iso:
            continue

        positions: set[str] = set()
        for cell in cells:
            positions |= _extract_positions(cell)

        if args.excl_anticodon:
            positions -= AC_POSITIONS

        if positions:
            key = (current_iso, domain)
            result.setdefault(key, set())
            result[key] |= positions

    return result


# ── ISM ranking ───────────────────────────────────────────────────────────────
def load_ism_data(iso: str, domain: str, max_len: int) -> np.ndarray | None:
    path = os.path.join(args.emb_dir, f"gnn_ism_saliency_{iso}_{domain}.npy")
    if not os.path.exists(path):
        return None
    sal = np.load(path)[:max_len].astype(np.float32)
    return sal


def rank_positions(sal: np.ndarray, raw_to_sp: dict[int, str],
                   excl_ac: bool) -> list[tuple[str, float]]:
    """
    Return list of (sprinzl_label, score) sorted by score descending.
    score = max|saliency| across all 4 nucleotides.
    """
    scores: list[tuple[str, float]] = []
    for raw_idx, sp_label in raw_to_sp.items():
        if raw_idx >= len(sal):
            continue
        if excl_ac and sp_label in AC_POSITIONS:
            continue
        score = float(np.abs(sal[raw_idx]).max())
        scores.append((sp_label, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def recall_at_k(ranked: list[tuple[str, float]],
                known_IEs: set[str], k: int) -> float:
    """Fraction of testable known IEs recovered in top-k ranked positions."""
    sp_labels = {sp for sp, _ in ranked}
    testable = known_IEs & sp_labels
    if not testable:
        return float("nan")
    top_k_labels = {sp for sp, _ in ranked[:k]}
    return len(testable & top_k_labels) / len(testable)


def auroc(ranked: list[tuple[str, float]], known_IEs: set[str]) -> float:
    """AUROC treating known IEs as positives, ISM rank as score."""
    sp_labels = {sp for sp, _ in ranked}
    testable = known_IEs & sp_labels
    if not testable or len(testable) == len(sp_labels):
        return float("nan")
    labels = [1 if sp in testable else 0 for sp, _ in ranked]
    scores_arr = [s for _, s in ranked]
    # Simple trapezoidal AUROC
    from sklearn.metrics import roc_auc_score as _roc
    try:
        return float(_roc(labels, scores_arr))
    except Exception:
        return float("nan")


# ── Per-isotype bar plot ───────────────────────────────────────────────────────
_COLOR_IE      = "#e05c2a"   # orange-red for known IEs
_COLOR_HIGH    = "#3a7dbf"   # blue for high-scoring non-IEs
_COLOR_LOW     = "#aec6e0"   # light blue for low-scoring
_COLOR_AC      = "#888888"   # grey for anticodon

def plot_ism_vs_ie(iso: str, domain: str,
                  sal: np.ndarray, raw_to_sp: dict[int, str],
                  known_IEs: set[str],
                  ranked: list[tuple[str, float]],
                  ks: list[int]) -> None:

    sp_to_raw = {v: k for k, v in raw_to_sp.items()}
    all_labels = sorted(raw_to_sp.values(), key=_sp_sort_key)

    scores_by_sp: dict[str, float] = {}
    for raw_idx, sp_label in raw_to_sp.items():
        if raw_idx < len(sal):
            scores_by_sp[sp_label] = float(np.abs(sal[raw_idx]).max())

    # Filter to positions that pass min-sal OR are known IEs
    sp_labels = {sp for sp, _ in ranked}
    testable_IEs = known_IEs & sp_labels

    display_labels = [
        lbl for lbl in all_labels
        if lbl in scores_by_sp and (
            scores_by_sp[lbl] >= args.min_sal or lbl in testable_IEs
        )
    ]
    if not display_labels:
        return

    xs = np.arange(len(display_labels))
    ys = np.array([scores_by_sp.get(lbl, 0.0) for lbl in display_labels])

    # Compute rank for each displayed position (1-indexed)
    rank_map = {sp: i + 1 for i, (sp, _) in enumerate(ranked)}

    colors = []
    for lbl in display_labels:
        if lbl in AC_POSITIONS:
            colors.append(_COLOR_AC)
        elif lbl in testable_IEs:
            colors.append(_COLOR_IE)
        else:
            colors.append(_COLOR_HIGH if ys[list(display_labels).index(lbl)] >= args.min_sal else _COLOR_LOW)

    fig, ax = plt.subplots(figsize=(max(8, len(display_labels) * 0.28 + 2), 3.8))
    bars = ax.bar(xs, ys, color=colors, width=0.75, linewidth=0)

    # Rank labels above IE bars
    for i, lbl in enumerate(display_labels):
        if lbl in testable_IEs:
            r = rank_map.get(lbl, "?")
            ax.text(i, ys[i] + 0.003, f"#{r}", ha="center", va="bottom",
                    fontsize=6, color=_COLOR_IE, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(display_labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel("max|ISM Δlogit|", fontsize=9)

    # Recall stats in subtitle
    recall_strs = []
    for k in ks:
        r = recall_at_k(ranked, testable_IEs, k)
        if not np.isnan(r):
            recall_strs.append(f"R@{k}={r:.0%}")
    auc = auroc(ranked, testable_IEs)
    auc_str = f"  AUROC={auc:.2f}" if not np.isnan(auc) else ""

    n_ie_shown = sum(1 for l in display_labels if l in testable_IEs)
    ax.set_title(
        f"ISM saliency — tRNA-{iso} ({domain})\n"
        f"Known IEs: {n_ie_shown}/{len(testable_IEs)} shown  "
        f"({len(known_IEs)} total in literature)  |  "
        f"{',  '.join(recall_strs)}{auc_str}",
        fontsize=8.5,
    )

    # Anticodon region shading
    ac_xs = [i for i, lbl in enumerate(display_labels) if lbl in AC_POSITIONS]
    if ac_xs:
        ax.axvspan(min(ac_xs) - 0.4, max(ac_xs) + 0.4,
                   color="lightgrey", alpha=0.3, zorder=0)

    # Legend
    legend_patches = [
        mpatches.Patch(color=_COLOR_IE, label="Known IE (literature)"),
        mpatches.Patch(color=_COLOR_HIGH, label="ISM signal (non-IE)"),
        mpatches.Patch(color=_COLOR_AC, label="Anticodon loop"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right",
              framealpha=0.8, borderpad=0.5)

    ax.set_xlim(-0.6, len(display_labels) - 0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    fname = f"{iso}_{domain}_ism_vs_IE.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fname}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
npz_path = os.path.join(args.emb_dir, "perpos_embeddings.npz")
if not os.path.exists(npz_path):
    raise SystemExit(f"ERROR: {npz_path} not found")
data    = np.load(npz_path, allow_pickle=True)
max_len = int(data["max_len"])
classes = list(data["classes"].astype(str))
print(f"max_len={max_len}  classes={classes}", flush=True)

# Resolve isotype list from available .npy files
def find_iso_domains() -> list[tuple[str, str]]:
    pairs = []
    for fname in sorted(os.listdir(args.emb_dir)):
        m = re.match(r"gnn_ism_saliency_(.+)_(Bacteria|Archaea)\.npy$", fname)
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs

all_iso_domains = find_iso_domains()
if args.isotypes:
    all_iso_domains = [(i, d) for i, d in all_iso_domains if i in args.isotypes]
if args.domains:
    all_iso_domains = [(i, d) for i, d in all_iso_domains if d in args.domains]

# Load known IEs
ie_csv_path = args.ie_csv
if not os.path.isabs(ie_csv_path):
    ie_csv_path = os.path.join(DATA_ROOT, ie_csv_path)
known_IEs_db = parse_known_IEs(ie_csv_path)
print(f"Loaded known IEs for {len(known_IEs_db)} (isotype, domain) pairs", flush=True)

# ── Per-isotype loop ──────────────────────────────────────────────────────────
stats_rows: list[dict] = []

for iso, domain in all_iso_domains:
    sal = load_ism_data(iso, domain, max_len)
    if sal is None:
        continue

    raw_to_sp = get_sprinzl_map(iso, domain)
    if not raw_to_sp:
        print(f"[{iso}/{domain}] no Sprinzl map — skipping", flush=True)
        continue

    ranked = rank_positions(sal, raw_to_sp, args.excl_anticodon)
    if not ranked:
        continue

    known_IEs = known_IEs_db.get((iso, domain), set())
    if args.excl_anticodon:
        known_IEs = known_IEs - AC_POSITIONS

    sp_in_map = {sp for sp, _ in ranked}
    testable = known_IEs & sp_in_map
    not_in_map = known_IEs - sp_in_map

    row: dict = {"isotype": iso, "domain": domain,
                 "n_known_IEs": len(known_IEs),
                 "n_testable": len(testable),
                 "not_in_sprinzl_map": "|".join(sorted(not_in_map)),
                 "auroc": round(auroc(ranked, testable), 3)}

    for k in args.top_k:
        r = recall_at_k(ranked, testable, k)
        row[f"recall@{k}"] = round(r, 3) if not np.isnan(r) else float("nan")

    # Top-5 positions and whether they are known IEs
    top5 = ranked[:5]
    row["top5_positions"] = "|".join(f"{sp}({'IE' if sp in testable else '-'})"
                                      for sp, _ in top5)
    row["known_IEs"] = "|".join(sorted(known_IEs))
    row["testable_IEs"] = "|".join(sorted(testable))

    stats_rows.append(row)

    # Print summary
    if testable:
        recall_str = "  ".join(
            f"R@{k}={row.get(f'recall@{k}', float('nan')):.0%}"
            for k in args.top_k
        )
        print(f"[{iso:8s}/{domain:10s}]  IEs={len(known_IEs)} testable={len(testable)}  "
              f"{recall_str}  AUROC={row['auroc']:.2f}", flush=True)
    else:
        print(f"[{iso:8s}/{domain:10s}]  no testable IEs (known={len(known_IEs)})", flush=True)

    # Individual plot
    plot_ism_vs_ie(iso, domain, sal, raw_to_sp, known_IEs, ranked, args.top_k)

# ── Summary CSV ───────────────────────────────────────────────────────────────
if stats_rows:
    stats_df = pd.DataFrame(stats_rows)

    # Full detailed table (includes top5_positions, known_IEs, testable_IEs)
    detail_path = os.path.join(outdir, "rank_stats.csv")
    stats_df.to_csv(detail_path, index=False)
    print(f"\nDetailed stats → {detail_path}", flush=True)

    # Clean summary table: key metrics only
    summary_cols = (["isotype", "domain", "n_known_IEs", "n_testable", "auroc"]
                    + [f"recall@{k}" for k in args.top_k])
    summary_df = stats_df[summary_cols].copy()
    summary_path = os.path.join(outdir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary         → {summary_path}", flush=True)

    # ── Recall@k heatmap ─────────────────────────────────────────────────────
    recall_cols = [f"recall@{k}" for k in args.top_k]
    # Separate heatmaps per domain
    for dom in sorted(stats_df["domain"].unique()):
        sub = stats_df[stats_df["domain"] == dom].copy()
        sub = sub[sub["n_testable"] > 0].sort_values("isotype")
        if sub.empty:
            continue

        mat = sub[recall_cols].values.astype(float)
        isos = sub["isotype"].tolist()

        fig, ax = plt.subplots(figsize=(max(4, len(recall_cols) * 1.0 + 1.5),
                                        max(4, len(isos) * 0.45 + 1.5)))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_xticks(range(len(recall_cols)))
        ax.set_xticklabels([f"Top-{k}" for k in args.top_k], fontsize=10)
        ax.set_yticks(range(len(isos)))
        ax.set_yticklabels(isos, fontsize=9)
        ax.set_xlabel("Rank threshold k", fontsize=10)
        ax.set_title(f"Recall@k — {dom}\n(fraction of testable known IEs in top-k ISM positions)",
                     fontsize=10)

        # Annotate cells
        for i in range(len(isos)):
            for j in range(len(recall_cols)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=8, color="black" if v < 0.6 else "white")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Recall")
        fig.tight_layout()
        fname = f"recall_heatmap_{dom}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap → {fname}", flush=True)

    # ── Recall@k curves ──────────────────────────────────────────────────────
    # Rebuild ranked lists for curve generation
    print("\nGenerating recall@k curves ...", flush=True)

    COLORS_CURVE = plt.cm.tab20.colors  # type: ignore[attr-defined]

    domains_available = sorted(stats_df["domain"].unique())
    fig, axes = plt.subplots(1, len(domains_available),
                             figsize=(5.5 * len(domains_available), 4.5),
                             squeeze=False)

    for col_i, dom in enumerate(domains_available):
        ax = axes[0][col_i]
        sub = stats_df[(stats_df["domain"] == dom) & (stats_df["n_testable"] > 0)]

        for row_i, (_, row) in enumerate(sub.sort_values("isotype").iterrows()):
            iso = row["isotype"]
            sal = load_ism_data(iso, dom, max_len)
            if sal is None:
                continue
            raw_to_sp = get_sprinzl_map(iso, dom)
            if not raw_to_sp:
                continue
            ranked_local = rank_positions(sal, raw_to_sp, args.excl_anticodon)
            testable = known_IEs_db.get((iso, dom), set()) & {sp for sp, _ in ranked_local}
            if not testable:
                continue

            ks_curve = list(range(1, len(ranked_local) + 1))
            recalls = [recall_at_k(ranked_local, testable, k) for k in ks_curve]

            color = COLORS_CURVE[row_i % len(COLORS_CURVE)]
            ax.plot(ks_curve, recalls, color=color, linewidth=1.4,
                    label=iso, alpha=0.85)
            # Mark the required top-k thresholds with dots
            for k in args.top_k:
                r = recall_at_k(ranked_local, testable, k)
                if not np.isnan(r):
                    ax.plot(k, r, "o", color=color, markersize=4, zorder=5)

        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
        for k in args.top_k:
            ax.axvline(k, color="lightgrey", linestyle=":", linewidth=0.7)

        ax.set_xlim(0, max(args.top_k) + 2)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("Rank threshold k", fontsize=10)
        ax.set_ylabel("Recall (fraction of known IEs recovered)", fontsize=9)
        ax.set_title(dom, fontsize=11, fontweight="bold")
        ax.legend(fontsize=6.5, ncol=2, loc="lower right", framealpha=0.8)

    fig.suptitle("ISM Recall@k vs Known Identity Elements", fontsize=12, y=1.01)
    fig.tight_layout()
    fname = "recall_curves.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curves → {fname}", flush=True)

print(f"\nDone. Results in: {outdir}/")
