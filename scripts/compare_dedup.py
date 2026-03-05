"""
compare_dedup.py
----------------
Visualises the difference between the original and deduplicated trnascan CSVs.

Usage:
    python compare_dedup.py                          # defaults
    python compare_dedup.py original.csv dedup.csv   # custom paths

Figures saved to /data/roma/figures/dedup_comparison/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("original", nargs="?", default="/data/roma/trnascan_GTDB_r226.csv")
parser.add_argument("dedup",    nargs="?", default="/data/roma/trnascan_GTDB_r226_dedup.csv")
args = parser.parse_args()

OUT = Path("/data/roma/figures/dedup_comparison")
OUT.mkdir(parents=True, exist_ok=True)

ISOTYPE_ORDER = [
    "Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His",
    "Ile","Ile2","Leu","Lys","Met","Phe","Pro","SeC","Ser",
    "Thr","Trp","Tyr","Val","fMet",
]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {"Original": "#4C72B0", "Deduplicated": "#DD8452"}


# ── load ──────────────────────────────────────────────────────────────────────
def load(path, label):
    print(f"Loading {label} …  ({path})")
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    df = df[df["tRNAscanID"] != "tRNAscanID"].copy()
    for c in ["Score", "conf_score", "intron_begin", "intron_end"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["seq_len"]   = df["primary_sequence"].str.len()
    df["has_intron"] = df["intron_begin"] > 0
    df["is_pseudo"]  = df["Note"].fillna("").str.contains("pseudo")
    df["label"]      = label
    print(f"  {len(df):,} rows, {df['GenomeID'].nunique():,} genomes")
    return df

orig  = load(args.original, "Original")
dedup = load(args.dedup,    "Deduplicated")

removed = len(orig) - len(dedup)
pct_removed = removed / len(orig) * 100


# ── Fig 1: high-level overview bar chart ─────────────────────────────────────
def fig_overview():
    metrics = {
        "Total tRNAs":       (len(orig),                    len(dedup)),
        "Unique genomes":    (orig["GenomeID"].nunique(),   dedup["GenomeID"].nunique()),
        "Unique sequences":  (orig["primary_sequence"].nunique(), dedup["primary_sequence"].nunique()),
        "Unique contigs":    (orig["ContigID"].nunique(),   dedup["ContigID"].nunique()),
    }
    labels  = list(metrics.keys())
    orig_v  = [metrics[m][0] for m in labels]
    dedup_v = [metrics[m][1] for m in labels]

    x     = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, orig_v,  width, label="Original",     color=COLORS["Original"],     alpha=0.85)
    b2 = ax.bar(x + width/2, dedup_v, width, label="Deduplicated",  color=COLORS["Deduplicated"], alpha=0.85)

    for bar in list(b1) + list(b2):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.01,
                f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.1f}k",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))
    ax.set_ylabel("Count")
    ax.set_title(f"Overview: original vs deduplicated  "
                 f"({removed:,} rows removed, {pct_removed:.1f}%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "01_overview.png")
    plt.close()
    print("  saved 01_overview.png")


# ── Fig 2: isotype distribution – absolute and relative change ────────────────
def fig_isotype_counts():
    orig_iso  = orig["Anticodon_predicted_isotype"].value_counts().reindex(ISOTYPE_ORDER, fill_value=0)
    dedup_iso = dedup["Anticodon_predicted_isotype"].value_counts().reindex(ISOTYPE_ORDER, fill_value=0)
    pct_change = (dedup_iso - orig_iso) / orig_iso * 100   # negative = removed

    x     = np.arange(len(ISOTYPE_ORDER))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # top: absolute counts side-by-side
    ax = axes[0]
    ax.bar(x - width/2, orig_iso.values  / 1e3, width, label="Original",    color=COLORS["Original"],     alpha=0.85)
    ax.bar(x + width/2, dedup_iso.values / 1e3, width, label="Deduplicated", color=COLORS["Deduplicated"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ISOTYPE_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Count (thousands)")
    ax.set_title("Isotype counts: original vs deduplicated")
    ax.legend()

    # bottom: % reduction per isotype
    ax2 = axes[1]
    bar_colors = ["#c0392b" if v < 0 else "#27ae60" for v in pct_change.values]
    ax2.bar(x, pct_change.values, color=bar_colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(pct_change.mean(), color="grey", linewidth=1, linestyle="--",
                label=f"mean = {pct_change.mean():.1f}%")
    ax2.set_xticks(x); ax2.set_xticklabels(ISOTYPE_ORDER, rotation=45, ha="right")
    ax2.set_ylabel("% change after dedup")
    ax2.set_title("% change in isotype count after deduplication")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(OUT / "02_isotype_counts.png")
    plt.close()
    print("  saved 02_isotype_counts.png")


# ── Fig 3: sequence length distribution overlay ───────────────────────────────
def fig_seq_len():
    fig, ax = plt.subplots(figsize=(9, 5))
    for df, label in [(orig, "Original"), (dedup, "Deduplicated")]:
        data = df["seq_len"].dropna()
        ax.hist(data, bins=60, alpha=0.55, color=COLORS[label], label=label, density=True)
        ax.axvline(data.median(), color=COLORS[label], lw=1.5, ls="--",
                   label=f"{label} median={data.median():.0f} nt")
    ax.set_xlabel("Sequence length (nt)")
    ax.set_ylabel("Density")
    ax.set_title("Sequence length distribution: original vs deduplicated")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "03_seq_len.png")
    plt.close()
    print("  saved 03_seq_len.png")


# ── Fig 4: score distributions overlay ───────────────────────────────────────
def fig_scores():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(axes,
                               ["Score", "conf_score"],
                               ["tRNAscan Score", "Confidence score"]):
        for df, label in [(orig, "Original"), (dedup, "Deduplicated")]:
            data = df[col].dropna()
            ax.hist(data, bins=100, alpha=0.55, color=COLORS[label],
                    label=f"{label} (med={data.median():.1f})", density=True)
        ax.set_xlabel(col); ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "04_score_distributions.png")
    plt.close()
    print("  saved 04_score_distributions.png")


# ── Fig 5: tRNAs per genome – before vs after ─────────────────────────────────
def fig_trnas_per_genome():
    orig_pg  = orig["GenomeID"].value_counts()
    dedup_pg = dedup["GenomeID"].value_counts()

    # align on genomes present in both
    common = orig_pg.index.intersection(dedup_pg.index)
    orig_c  = orig_pg.reindex(common)
    dedup_c = dedup_pg.reindex(common)
    delta   = dedup_c - orig_c   # always ≤ 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # left: distributions overlay
    ax = axes[0]
    ax.hist(orig_pg.values,  bins=80, alpha=0.55, color=COLORS["Original"],
            label=f"Original (med={orig_pg.median():.0f})", density=True)
    ax.hist(dedup_pg.values, bins=80, alpha=0.55, color=COLORS["Deduplicated"],
            label=f"Deduplicated (med={dedup_pg.median():.0f})", density=True)
    ax.set_xlabel("tRNAs per genome"); ax.set_ylabel("Density")
    ax.set_title("tRNAs per genome")
    ax.legend(fontsize=8)

    # middle: scatter orig vs dedup per genome (sample 5k if large)
    ax2 = axes[1]
    if len(common) > 10_000:
        idx = np.random.default_rng(0).choice(len(common), 10_000, replace=False)
        ox, dx = orig_c.values[idx], dedup_c.values[idx]
    else:
        ox, dx = orig_c.values, dedup_c.values
    ax2.hexbin(ox, dx, gridsize=60, cmap="Blues", mincnt=1, bins="log")
    lim = max(ox.max(), dx.max())
    ax2.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax2.set_xlabel("Original tRNAs/genome")
    ax2.set_ylabel("Deduplicated tRNAs/genome")
    ax2.set_title("Per-genome tRNA count: original vs dedup\n(log-scale hex, sample)")
    ax2.legend(fontsize=8)

    # right: histogram of reduction per genome
    ax3 = axes[2]
    ax3.hist(delta.values, bins=60, color="#c0392b", alpha=0.8)
    ax3.axvline(delta.median(), color="black", lw=1.5, ls="--",
                label=f"median Δ = {delta.median():.0f}")
    ax3.set_xlabel("tRNAs removed per genome (dedup − original)")
    ax3.set_ylabel("Number of genomes")
    ax3.set_title("Reduction in tRNA count per genome")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "05_trnas_per_genome.png")
    plt.close()
    print("  saved 05_trnas_per_genome.png")


# ── Fig 6: codon usage – absolute and relative change ────────────────────────
def fig_codon_usage():
    top_codons = orig["Codon"].value_counts().head(25).index.tolist()
    orig_c  = orig["Codon"].value_counts().reindex(top_codons, fill_value=0)
    dedup_c = dedup["Codon"].value_counts().reindex(top_codons, fill_value=0)

    # normalise to fraction
    orig_f  = orig_c  / orig_c.sum()  * 100
    dedup_f = dedup_c / dedup_c.sum() * 100
    delta_f = dedup_f - orig_f

    x     = np.arange(len(top_codons))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    ax.bar(x - width/2, orig_f.values,  width, label="Original",    color=COLORS["Original"],     alpha=0.85)
    ax.bar(x + width/2, dedup_f.values, width, label="Deduplicated", color=COLORS["Deduplicated"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(top_codons, rotation=45, ha="right")
    ax.set_ylabel("% of all tRNAs")
    ax.set_title("Codon usage (% of total): original vs deduplicated  (top 25)")
    ax.legend()

    ax2 = axes[1]
    bar_colors = ["#c0392b" if v < 0 else "#27ae60" for v in delta_f.values]
    ax2.bar(x, delta_f.values, color=bar_colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(top_codons, rotation=45, ha="right")
    ax2.set_ylabel("Δ percentage points")
    ax2.set_title("Shift in codon frequency after deduplication")

    plt.tight_layout()
    fig.savefig(OUT / "06_codon_usage.png")
    plt.close()
    print("  saved 06_codon_usage.png")


# ── Fig 7: quality flags comparison ──────────────────────────────────────────
def fig_quality_flags():
    def flag_rates(df):
        return {
            "Pseudo":     df["is_pseudo"].mean() * 100,
            "Has intron": df["has_intron"].mean() * 100,
        }

    orig_f  = flag_rates(orig)
    dedup_f = flag_rates(dedup)
    flags   = list(orig_f.keys())

    x     = np.arange(len(flags))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - width/2, [orig_f[f]  for f in flags], width,
                label="Original",    color=COLORS["Original"],     alpha=0.85)
    b2 = ax.bar(x + width/2, [dedup_f[f] for f in flags], width,
                label="Deduplicated", color=COLORS["Deduplicated"], alpha=0.85)
    for bar in list(b1) + list(b2):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(flags)
    ax.set_ylabel("% of tRNAs")
    ax.set_title("Quality flag rates: original vs deduplicated")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "07_quality_flags.png")
    plt.close()
    print("  saved 07_quality_flags.png")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'─'*60}")
    print(f"  Original:     {len(orig):,} rows")
    print(f"  Deduplicated: {len(dedup):,} rows")
    print(f"  Removed:      {removed:,} ({pct_removed:.1f}%)")
    print(f"{'─'*60}\n")

    print("Generating figures …")
    fig_overview()
    fig_isotype_counts()
    fig_seq_len()
    fig_scores()
    fig_trnas_per_genome()
    fig_codon_usage()
    fig_quality_flags()
    print(f"\nAll figures saved to {OUT}/")
