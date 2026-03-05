"""
EDA visualizations for trnascan_GTDB_r226.csv
~6M tRNA entries across ~142,794 GTDB r226 genomes.

Key findings:
- 6,068,145 tRNAs (after removing duplicate headers)
- 142,794 unique genomes, 2,747,054 unique contigs
- Dominant isotypes: Leu, Arg, Ser
- ~40K pseudo tRNAs, ~14K with introns
- Mean ~42 tRNAs per genome
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="EDA for trnascan GTDB CSV")
parser.add_argument("csv", nargs="?", default="/data/roma/trnascan_GTDB_r226.csv",
                    help="Path to input CSV (default: trnascan_GTDB_r226.csv)")
args = parser.parse_args()

DATA = args.csv
OUT  = Path("/data/roma/figures") / Path(DATA).stem
OUT.mkdir(parents=True, exist_ok=True)
print(f"Input:  {DATA}")
print(f"Output: {OUT}/")

ISOTYPE_ORDER = [
    "Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His",
    "Ile","Ile2","Leu","Lys","Met","Phe","Pro","SeC","Ser",
    "Thr","Trp","Tyr","Val","fMet","Undet","Sup"
]
AA_SCORE_COLS = [
    "Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His",
    "Ile","Ile2","Leu","Lys","Met","Phe","Pro","SeC","Ser",
    "Thr","Trp","Tyr","Val","fMet"
]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── load ─────────────────────────────────────────────────────────────────────
def load():
    print("Loading …")
    df = pd.read_csv(DATA, low_memory=False, on_bad_lines="skip")
    df = df[df["tRNAscanID"] != "tRNAscanID"].copy()   # drop repeated headers
    num_cols = (
        ["Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His",
         "Ile","Ile2","Leu","Lys","Met","Phe","Pro","SeC","Ser",
         "Thr","Trp","Tyr","Val","fMet",
         "tRNA #","Begin","End","intron_begin","intron_end",
         "Score","conf_score"]
    )
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["seq_len"] = df["primary_sequence"].str.len()
    df["has_intron"] = df["intron_begin"] > 0
    df["is_pseudo"] = df["Note"].fillna("").str.contains("pseudo")
    df["is_trunc"]  = df["Note"].fillna("").str.contains("trunc")
    print(f"  {len(df):,} tRNAs across {df['GenomeID'].nunique():,} genomes")
    return df


# ── Fig 1: isotype counts ────────────────────────────────────────────────────
def fig_isotype_counts(df):
    counts = (df["Anticodon_predicted_isotype"]
              .value_counts()
              .reindex(ISOTYPE_ORDER, fill_value=0))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values / 1e6,
                  color=sns.color_palette("tab20", len(counts)))
    ax.set_xlabel("Isotype")
    ax.set_ylabel("Count (millions)")
    ax.set_title("tRNA isotype distribution  (GTDB r226, n=6.1M)")
    ax.tick_params(axis="x", rotation=45)
    for b, v in zip(bars, counts.values):
        if v > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{v/1e3:.0f}k", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "01_isotype_counts.png")
    plt.close()
    print("  saved 01_isotype_counts.png")


# ── Fig 2: predicted isotype vs CM classifier agreement ─────────────────────
def fig_isotype_vs_cm(df):
    sub = df[df["CM"].notna() & df["Anticodon_predicted_isotype"].notna()].copy()
    iso_order = [i for i in ISOTYPE_ORDER if i not in ("Undet", "Sup")]
    cm_order   = [i for i in iso_order if i in sub["CM"].unique()]
    iso_order2 = [i for i in iso_order if i in sub["Anticodon_predicted_isotype"].unique()]

    ct = pd.crosstab(sub["Anticodon_predicted_isotype"],
                     sub["CM"],
                     normalize="index") * 100
    ct = ct.reindex(index=iso_order2, columns=cm_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(ct, ax=ax, cmap="YlOrRd", linewidths=0.3,
                cbar_kws={"label": "% of row"}, fmt=".0f", annot=False)
    ax.set_xlabel("CM classifier isotype")
    ax.set_ylabel("Anticodon-predicted isotype")
    ax.set_title("Predicted isotype vs CM classifier  (row-normalised %)")
    plt.tight_layout()
    fig.savefig(OUT / "02_isotype_vs_CM_heatmap.png")
    plt.close()
    print("  saved 02_isotype_vs_CM_heatmap.png")


# ── Fig 3: score distributions ───────────────────────────────────────────────
def fig_scores(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, color, title in zip(
        axes,
        ["Score", "conf_score"],
        ["steelblue", "tomato"],
        ["tRNAscan Score distribution", "Confidence score distribution"],
    ):
        data = df[col].dropna()
        ax.hist(data, bins=120, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(data.median(), color="black", lw=1.5, ls="--",
                   label=f"median={data.median():.1f}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))

    plt.tight_layout()
    fig.savefig(OUT / "03_score_distributions.png")
    plt.close()
    print("  saved 03_score_distributions.png")


# ── Fig 4: score by isotype (violin) ─────────────────────────────────────────
def fig_score_by_isotype(df):
    keep = [i for i in ISOTYPE_ORDER if i not in ("Undet", "Sup")]
    sub  = df[df["Anticodon_predicted_isotype"].isin(keep)].copy()
    medians = sub.groupby("Anticodon_predicted_isotype")["Score"].median().reindex(keep)
    order   = medians.sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=sub, x="Anticodon_predicted_isotype", y="Score",
        order=order, palette="tab20", ax=ax, inner="quartile", cut=0,
    )
    ax.set_xlabel("Isotype")
    ax.set_ylabel("tRNAscan Score")
    ax.set_title("tRNAscan score distribution by isotype")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUT / "04_score_by_isotype.png")
    plt.close()
    print("  saved 04_score_by_isotype.png")


# ── Fig 5: tRNAs per genome distribution ─────────────────────────────────────
def fig_per_genome(df):
    per_genome = df.groupby("GenomeID").size()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(per_genome, bins=100, color="mediumseagreen", edgecolor="none", alpha=0.85)
    ax.axvline(per_genome.median(), color="black", lw=1.5, ls="--",
               label=f"median={per_genome.median():.0f}")
    ax.axvline(per_genome.mean(), color="red", lw=1.5, ls=":",
               label=f"mean={per_genome.mean():.0f}")
    ax.set_xlabel("tRNAs per genome")
    ax.set_ylabel("Number of genomes")
    ax.set_title(f"tRNA count per genome  (n={per_genome.shape[0]:,} genomes)")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1e3 else str(int(x))))
    plt.tight_layout()
    fig.savefig(OUT / "05_trnas_per_genome.png")
    plt.close()
    print("  saved 05_trnas_per_genome.png")


# ── Fig 6: sequence length distribution ──────────────────────────────────────
def fig_seq_len(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    data = df["seq_len"].dropna()
    ax.hist(data, bins=60, color="orchid", edgecolor="none", alpha=0.85)
    ax.axvline(data.median(), color="black", lw=1.5, ls="--",
               label=f"median={data.median():.0f} nt")
    ax.set_xlabel("Sequence length (nt)")
    ax.set_ylabel("Count")
    ax.set_title("tRNA sequence length distribution")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"))
    plt.tight_layout()
    fig.savefig(OUT / "06_sequence_length.png")
    plt.close()
    print("  saved 06_sequence_length.png")


# ── Fig 7: pseudo / truncated / intron fractions ──────────────────────────────
def fig_quality_flags(df):
    cats = {
        "Normal": (~df["is_pseudo"] & ~df["is_trunc"] & ~df["has_intron"]).sum(),
        "Pseudo": (df["is_pseudo"] & ~df["is_trunc"]).sum(),
        "Truncated": (df["is_trunc"] & ~df["is_pseudo"]).sum(),
        "Pseudo+Trunc": (df["is_pseudo"] & df["is_trunc"]).sum(),
        "Has intron": df["has_intron"].sum(),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # pie
    ax = axes[0]
    labels = list(cats.keys())
    vals   = list(cats.values())
    colors = ["#4CAF50","#F44336","#FF9800","#9C27B0","#2196F3"]
    ax.pie(vals, labels=labels, autopct="%1.1f%%", colors=colors,
           startangle=140, textprops={"fontsize": 9})
    ax.set_title("tRNA quality flags")

    # bar with counts
    ax2 = axes[1]
    ax2.barh(labels[::-1], [v/1e3 for v in vals[::-1]], color=colors[::-1])
    ax2.set_xlabel("Count (thousands)")
    ax2.set_title("tRNA quality flags – absolute counts")
    for i, v in enumerate(vals[::-1]):
        ax2.text(v/1e3 + 0.5, i, f"{v:,}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "07_quality_flags.png")
    plt.close()
    print("  saved 07_quality_flags.png")


# ── Fig 8: top-30 codon usage ────────────────────────────────────────────────
def fig_codon_usage(df):
    codon_counts = df["Codon"].value_counts().head(30)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=codon_counts.index, y=codon_counts.values / 1e3, ax=ax,
                palette="viridis")
    ax.set_xlabel("Anticodon (Codon column)")
    ax.set_ylabel("Count (thousands)")
    ax.set_title("Top-30 anticodon sequences across all tRNAs")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUT / "08_codon_usage.png")
    plt.close()
    print("  saved 08_codon_usage.png")


# ── Fig 9: isotype score heatmap (mean score of each AA col per predicted iso) ─
def fig_isotype_score_heatmap(df):
    keep = [i for i in ISOTYPE_ORDER if i not in ("Undet", "Sup")]
    sub  = df[df["Anticodon_predicted_isotype"].isin(keep)].copy()

    mat = sub.groupby("Anticodon_predicted_isotype")[AA_SCORE_COLS].mean()
    mat = mat.reindex(keep).dropna(how="all")
    # replace -999 sentinel with NaN before plotting
    mat = mat.replace(-999, np.nan)

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(mat, ax=ax, cmap="RdYlGn", center=70,
                cbar_kws={"label": "Mean isotype score"},
                linewidths=0.2, annot=True, fmt=".0f", annot_kws={"size": 7})
    ax.set_xlabel("Isotype score column")
    ax.set_ylabel("Predicted isotype")
    ax.set_title("Mean tRNAscan isotype scores  (rows = predicted class, cols = score model)")
    plt.tight_layout()
    fig.savefig(OUT / "09_isotype_score_heatmap.png")
    plt.close()
    print("  saved 09_isotype_score_heatmap.png")


# ── Fig 10: isotypes per genome (stacked histogram or mean composition) ───────
def fig_isotype_composition(df):
    keep = [i for i in ISOTYPE_ORDER if i not in ("Undet", "Sup")]
    sub  = df[df["Anticodon_predicted_isotype"].isin(keep)]
    comp = (sub.groupby(["GenomeID","Anticodon_predicted_isotype"])
               .size()
               .unstack(fill_value=0))
    # normalise to fractions
    comp_frac = comp.div(comp.sum(axis=1), axis=0)
    mean_frac = comp_frac.mean().reindex(keep)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("tab20", len(mean_frac))
    bars = ax.bar(mean_frac.index, mean_frac.values * 100, color=colors)
    ax.set_xlabel("Isotype")
    ax.set_ylabel("Mean % of genome tRNA repertoire")
    ax.set_title("Mean isotype composition per genome  (averaged over 142k genomes)")
    ax.tick_params(axis="x", rotation=45)
    for b, v in zip(bars, mean_frac.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                f"{v*100:.1f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT / "10_isotype_composition_per_genome.png")
    plt.close()
    print("  saved 10_isotype_composition_per_genome.png")


# ── Fig 11: Score vs conf_score scatter (hexbin) ─────────────────────────────
def fig_score_scatter(df):
    sub = df[["Score","conf_score"]].dropna().sample(200_000, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(sub["Score"], sub["conf_score"], gridsize=80,
                   cmap="plasma", mincnt=1, bins="log")
    plt.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_xlabel("tRNAscan Score")
    ax.set_ylabel("Confidence score")
    ax.set_title("Score vs Confidence score  (200k random sample)")
    plt.tight_layout()
    fig.savefig(OUT / "11_score_vs_confscore.png")
    plt.close()
    print("  saved 11_score_vs_confscore.png")


# ── Fig 12: intron length distribution ───────────────────────────────────────
def fig_introns(df):
    intr = df[df["has_intron"]].copy()
    intr["intron_len"] = (intr["intron_end"] - intr["intron_begin"]).abs()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(intr["intron_len"].clip(0, 500), bins=80,
            color="cornflowerblue", edgecolor="none")
    ax.set_xlabel("Intron length (nt, clipped at 500)")
    ax.set_ylabel("Count")
    ax.set_title(f"Intron length distribution  (n={len(intr):,} intron-containing tRNAs)")

    ax2 = axes[1]
    iso_intr = (intr["Anticodon_predicted_isotype"]
                .value_counts()
                .reindex(ISOTYPE_ORDER, fill_value=0))
    ax2.bar(iso_intr.index, iso_intr.values, color="cornflowerblue")
    ax2.set_xlabel("Isotype")
    ax2.set_ylabel("Count")
    ax2.set_title("Intron-containing tRNAs by isotype")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(OUT / "12_introns.png")
    plt.close()
    print("  saved 12_introns.png")


# ── write statistics text file ───────────────────────────────────────────────
def write_stats(df):
    out_path = OUT / "stats.txt"
    per_genome   = df.groupby("GenomeID").size()
    per_contig   = df.groupby("ContigID").size()
    isotype_cts  = df["Anticodon_predicted_isotype"].value_counts()
    codon_cts    = df["Codon"].value_counts()
    cm_cts       = df["CM"].value_counts()
    note_cts     = df["Note"].fillna("none").value_counts()

    lines = []
    def h(title):
        lines.append("\n" + "=" * 60)
        lines.append(title)
        lines.append("=" * 60)

    # ── overview ──────────────────────────────────────────────────
    h("OVERVIEW")
    lines.append(f"Total tRNA rows (after header dedup): {len(df):,}")
    lines.append(f"Unique genomes:                       {df['GenomeID'].nunique():,}")
    lines.append(f"Unique contigs:                       {df['ContigID'].nunique():,}")
    lines.append(f"Columns:                              {len(df.columns)}")
    lines.append(f"Duplicate header rows removed:        142,794")

    # ── missing values ─────────────────────────────────────────────
    h("MISSING VALUES")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for col, n in missing.items():
        lines.append(f"  {col:<30} {n:>10,}  ({n/len(df)*100:.2f}%)")

    # ── per-genome tRNA count ──────────────────────────────────────
    h("tRNAs PER GENOME")
    desc = per_genome.describe()
    for stat, val in desc.items():
        lines.append(f"  {stat:<10} {val:>10.2f}")
    lines.append(f"  {'max genome':<10} {per_genome.idxmax()}  ({per_genome.max()} tRNAs)")
    lines.append(f"  {'min genome':<10} {per_genome.idxmin()}  ({per_genome.min()} tRNAs)")

    # ── per-contig tRNA count ──────────────────────────────────────
    h("tRNAs PER CONTIG")
    desc2 = per_contig.describe()
    for stat, val in desc2.items():
        lines.append(f"  {stat:<10} {val:>10.2f}")

    # ── isotype distribution ───────────────────────────────────────
    h("ISOTYPE DISTRIBUTION  (Anticodon_predicted_isotype)")
    for iso, n in isotype_cts.items():
        lines.append(f"  {iso:<10} {n:>10,}  ({n/len(df)*100:.2f}%)")

    # ── CM classifier distribution ─────────────────────────────────
    h("CM CLASSIFIER DISTRIBUTION")
    valid_cm = cm_cts[cm_cts.index.isin(ISOTYPE_ORDER)]
    for iso, n in valid_cm.items():
        lines.append(f"  {iso:<10} {n:>10,}  ({n/valid_cm.sum()*100:.2f}%)")
    lines.append(f"  (noise/malformed rows with numeric CM values: {cm_cts[~cm_cts.index.isin(ISOTYPE_ORDER)].sum():,})")

    # ── top-30 codon distribution ──────────────────────────────────
    h("TOP-30 CODON (ANTICODON) DISTRIBUTION")
    for codon, n in codon_cts.head(30).items():
        lines.append(f"  {codon:<6} {n:>10,}  ({n/len(df)*100:.2f}%)")

    # ── score statistics ───────────────────────────────────────────
    h("SCORE STATISTICS")
    for col in ["Score", "conf_score"]:
        lines.append(f"\n  {col}:")
        for stat, val in df[col].describe().items():
            lines.append(f"    {stat:<10} {val:>10.3f}")

    # ── sequence length statistics ─────────────────────────────────
    h("SEQUENCE LENGTH (nt)")
    for stat, val in df["seq_len"].describe().items():
        lines.append(f"  {stat:<10} {val:>10.2f}")

    # ── quality flags ──────────────────────────────────────────────
    h("QUALITY FLAGS")
    pseudo   = df["is_pseudo"].sum()
    trunc    = df["is_trunc"].sum()
    intron   = df["has_intron"].sum()
    ps_tr    = (df["is_pseudo"] & df["is_trunc"]).sum()
    normal   = (~df["is_pseudo"] & ~df["is_trunc"] & ~df["has_intron"]).sum()
    lines.append(f"  {'Normal':<20} {normal:>10,}  ({normal/len(df)*100:.2f}%)")
    lines.append(f"  {'Pseudo':<20} {pseudo:>10,}  ({pseudo/len(df)*100:.2f}%)")
    lines.append(f"  {'Truncated':<20} {trunc:>10,}  ({trunc/len(df)*100:.2f}%)")
    lines.append(f"  {'Pseudo+Truncated':<20} {ps_tr:>10,}  ({ps_tr/len(df)*100:.2f}%)")
    lines.append(f"  {'Has intron':<20} {intron:>10,}  ({intron/len(df)*100:.2f}%)")

    # ── top-20 Note categories ─────────────────────────────────────
    h("TOP-20 NOTE CATEGORIES")
    for note, n in note_cts.head(20).items():
        lines.append(f"  {n:>10,}  {note}")

    # ── isotype score summary (mean of each AA model col) ──────────
    h("MEAN ISOTYPE MODEL SCORES  (per predicted isotype, top diagonal)")
    keep = [i for i in ISOTYPE_ORDER if i not in ("Undet","Sup")]
    sub  = df[df["Anticodon_predicted_isotype"].isin(keep)]
    mat  = sub.groupby("Anticodon_predicted_isotype")[AA_SCORE_COLS].mean()
    mat  = mat.replace(-999, float("nan"))
    lines.append(f"  {'Isotype':<10} {'Own score (mean)':>18}  {'Max-score col':>15}  {'Max score':>10}")
    for iso in keep:
        if iso not in mat.index or iso not in mat.columns:
            continue
        row      = mat.loc[iso]
        own      = row.get(iso, float("nan"))
        best_col = row.idxmax()
        best_val = row.max()
        lines.append(f"  {iso:<10} {own:>18.2f}  {best_col:>15}  {best_val:>10.2f}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load()

    print("\nWriting statistics …")
    write_stats(df)

    print("\nGenerating figures …")
    fig_isotype_counts(df)
    fig_isotype_vs_cm(df)
    fig_scores(df)
    fig_score_by_isotype(df)
    fig_per_genome(df)
    fig_seq_len(df)
    fig_quality_flags(df)
    fig_codon_usage(df)
    fig_isotype_score_heatmap(df)
    fig_isotype_composition(df)
    fig_score_scatter(df)
    fig_introns(df)

    print(f"\nAll figures saved to {OUT}/")
