"""
eda_taxonomy.py
---------------
EDA figures leveraging the per-rank taxonomy columns added by add_taxonomy.py.
Requires: trnascan_GTDB_r226_taxonomy.csv

Figures produced
────────────────
01  Genome count per domain (bar)
02  tRNAs per genome – violin by domain
03  Isotype composition heatmap – mean % per phylum (top 30 phyla by genome count)
04  SeC tRNA prevalence by phylum (% genomes that carry ≥1 SeC)
05  Ile2 tRNA prevalence by domain and top phyla
06  Intron frequency by domain (% tRNAs with intron)
07  Pseudo tRNA rate by domain and top 20 phyla
08  tRNAscan score distribution by domain (violin)
09  Top 10 phyla × isotype heatmap (mean count per genome)
10  Genus-level tRNA count richness (top 30 genera, box-style via violin)
11  fMet count per genome by domain
12  Codon usage divergence: Archaea vs Bacteria (top-20 codons, grouped bar)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Taxonomy EDA for trnascan GTDB CSV")
parser.add_argument("csv", nargs="?",
                    default="/data/roma/trnascan_GTDB_r226_taxonomy.csv",
                    help="Path to taxonomy-enriched CSV (default: trnascan_GTDB_r226_taxonomy.csv)")
args = parser.parse_args()

DATA = args.csv
OUT  = Path("/data/roma/figures") / Path(DATA).stem
OUT.mkdir(parents=True, exist_ok=True)
print(f"Input:  {DATA}")
print(f"Output: {OUT}/")

ISOTYPE_ORDER = [
    "Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His",
    "Ile","Ile2","Leu","Lys","Met","Phe","Pro","SeC","Ser",
    "Thr","Trp","Tyr","Val","fMet",
]
TAX_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]

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
    df = df[df["tRNAscanID"] != "tRNAscanID"].copy()

    num_cols = [
        "Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His",
        "Ile","Ile2","Leu","Lys","Met","Phe","Pro","SeC","Ser",
        "Thr","Trp","Tyr","Val","fMet",
        "tRNA #","Begin","End","intron_begin","intron_end",
        "Score","conf_score",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["has_intron"] = df["intron_begin"] > 0
    df["is_pseudo"]  = df["Note"].fillna("").str.contains("pseudo")
    df["is_trunc"]   = df["Note"].fillna("").str.contains("trunc")

    # Clean domain labels (strip trailing whitespace / NaN)
    df["domain"] = df["domain"].fillna("Unknown").str.strip()

    print(f"  {len(df):,} tRNAs, {df['GenomeID'].nunique():,} genomes")
    print(f"  Domains: {df['domain'].value_counts().to_dict()}")
    return df


# ── helpers ──────────────────────────────────────────────────────────────────
def top_phyla(df, n=30):
    return (df.groupby("phylum")["GenomeID"]
              .nunique()
              .nlargest(n)
              .index.tolist())

def top_genera(df, n=30):
    return (df.groupby("genus")["GenomeID"]
              .nunique()
              .nlargest(n)
              .index.tolist())

DOMAIN_PALETTE = {"Bacteria": "#4C72B0", "Archaea": "#DD8452", "Unknown": "#999999"}


# ── Fig 01: genome count per domain ──────────────────────────────────────────
def fig_genome_count_by_domain(df):
    counts = df.groupby("domain")["GenomeID"].nunique().sort_values(ascending=False)
    colors = [DOMAIN_PALETTE.get(d, "#aaaaaa") for d in counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=colors)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 300,
                f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of genomes")
    ax.set_title("Genome count by domain  (GTDB r226)")
    plt.tight_layout()
    fig.savefig(OUT / "tax_01_genomes_by_domain.png")
    plt.close()
    print("  saved tax_01_genomes_by_domain.png")


# ── Fig 02: tRNAs per genome – violin by domain ───────────────────────────────
def fig_trnas_per_genome_by_domain(df):
    per_genome = (df.groupby(["GenomeID","domain"])
                    .size()
                    .reset_index(name="n_tRNA"))
    domains = per_genome["domain"].value_counts().index.tolist()
    colors  = [DOMAIN_PALETTE.get(d, "#aaaaaa") for d in domains]

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(
        [per_genome.loc[per_genome["domain"]==d, "n_tRNA"].values for d in domains],
        positions=range(len(domains)), showmedians=True,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    for part in ("cmedians","cbars","cmins","cmaxes"):
        parts[part].set_color("black"); parts[part].set_linewidth(1.2)

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains)
    ax.set_ylabel("tRNAs per genome")
    ax.set_title("tRNA count per genome by domain")

    # annotate medians
    for i, d in enumerate(domains):
        med = per_genome.loc[per_genome["domain"]==d, "n_tRNA"].median()
        ax.text(i, med + 1, f"med={med:.0f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT / "tax_02_trnas_per_genome_by_domain.png")
    plt.close()
    print("  saved tax_02_trnas_per_genome_by_domain.png")


# ── Fig 03: isotype composition heatmap – top 30 phyla ───────────────────────
def fig_isotype_heatmap_by_phylum(df):
    keep_iso = [i for i in ISOTYPE_ORDER if i != "SeC"]   # SeC handled separately
    phyla = top_phyla(df, 30)
    sub   = df[df["phylum"].isin(phyla) & df["Anticodon_predicted_isotype"].isin(keep_iso)]

    # mean fraction of isotype per genome, averaged over genomes in each phylum
    comp = (sub.groupby(["GenomeID","phylum","Anticodon_predicted_isotype"])
               .size()
               .unstack(fill_value=0))
    comp = comp.div(comp.sum(axis=1), axis=0)  # fractions per genome
    comp = comp.reset_index()
    mean_by_phylum = comp.groupby("phylum")[keep_iso].mean() * 100  # percent

    # sort phyla by domain then genome count
    phylum_domain = df.groupby("phylum")["domain"].first()
    genome_ct     = df.groupby("phylum")["GenomeID"].nunique()
    order = (pd.DataFrame({"domain": phylum_domain, "n": genome_ct})
               .loc[phyla]
               .sort_values(["domain","n"], ascending=[True, False])
               .index.tolist())
    mean_by_phylum = mean_by_phylum.reindex(order)

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(mean_by_phylum, ax=ax, cmap="YlOrRd",
                cbar_kws={"label": "Mean % of genome tRNA repertoire"},
                linewidths=0.3, annot=True, fmt=".1f", annot_kws={"size": 7})
    ax.set_xlabel("Isotype")
    ax.set_ylabel("Phylum")
    ax.set_title("Mean isotype composition per genome  –  top 30 phyla (sorted by domain, genome count)")
    plt.tight_layout()
    fig.savefig(OUT / "tax_03_isotype_heatmap_by_phylum.png")
    plt.close()
    print("  saved tax_03_isotype_heatmap_by_phylum.png")


# ── Fig 04: SeC prevalence by phylum ─────────────────────────────────────────
def fig_sec_by_phylum(df):
    phyla  = top_phyla(df, 40)
    sub    = df[df["phylum"].isin(phyla)]

    # genomes that carry ≥1 SeC tRNA
    has_sec = (sub[sub["Anticodon_predicted_isotype"] == "SeC"]
               .groupby("phylum")["GenomeID"].nunique()
               .rename("has_sec"))
    total   = sub.groupby("phylum")["GenomeID"].nunique().rename("total")
    pct     = (has_sec / total * 100).fillna(0).reindex(phyla, fill_value=0)
    pct     = pct.sort_values(ascending=False)

    # colour by domain
    phylum_domain = df.groupby("phylum")["domain"].first()
    colors = [DOMAIN_PALETTE.get(phylum_domain.get(p, "Unknown"), "#aaaaaa")
              for p in pct.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(pct)), pct.values, color=colors)
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels(pct.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of genomes with ≥1 SeC tRNA")
    ax.set_title("Selenocysteine (SeC) tRNA prevalence by phylum  (top 40 phyla by genome count)")
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=d) for d, c in DOMAIN_PALETTE.items() if d != "Unknown"]
    ax.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    fig.savefig(OUT / "tax_04_SeC_prevalence_by_phylum.png")
    plt.close()
    print("  saved tax_04_SeC_prevalence_by_phylum.png")


# ── Fig 05: Ile2 prevalence by domain and top phyla ──────────────────────────
def fig_ile2_by_domain(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: % genomes with ≥1 Ile2 by domain
    ax = axes[0]
    domains = df["domain"].value_counts().index.tolist()
    for_dom = []
    for d in domains:
        sub = df[df["domain"] == d]
        total = sub["GenomeID"].nunique()
        has   = sub[sub["Anticodon_predicted_isotype"]=="Ile2"]["GenomeID"].nunique()
        for_dom.append(has / total * 100)
    colors = [DOMAIN_PALETTE.get(d, "#aaaaaa") for d in domains]
    ax.bar(domains, for_dom, color=colors)
    ax.set_ylabel("% genomes with ≥1 Ile2 tRNA")
    ax.set_title("Ile2 tRNA prevalence by domain")

    # right: top 30 phyla
    ax2 = axes[1]
    phyla = top_phyla(df, 30)
    sub   = df[df["phylum"].isin(phyla)]
    has_ile2  = (sub[sub["Anticodon_predicted_isotype"]=="Ile2"]
                 .groupby("phylum")["GenomeID"].nunique())
    total_p   = sub.groupby("phylum")["GenomeID"].nunique()
    pct       = (has_ile2 / total_p * 100).fillna(0).reindex(phyla, fill_value=0)
    pct       = pct.sort_values(ascending=False)
    phylum_domain = df.groupby("phylum")["domain"].first()
    colors2   = [DOMAIN_PALETTE.get(phylum_domain.get(p,"Unknown"),"#aaaaaa") for p in pct.index]
    ax2.bar(range(len(pct)), pct.values, color=colors2)
    ax2.set_xticks(range(len(pct)))
    ax2.set_xticklabels(pct.index, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("% genomes with ≥1 Ile2 tRNA")
    ax2.set_title("Ile2 tRNA prevalence by phylum  (top 30)")

    plt.tight_layout()
    fig.savefig(OUT / "tax_05_Ile2_prevalence.png")
    plt.close()
    print("  saved tax_05_Ile2_prevalence.png")


# ── Fig 06: intron frequency by domain ───────────────────────────────────────
def fig_introns_by_domain(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: % tRNAs with intron by domain
    ax = axes[0]
    domains = df["domain"].value_counts().index.tolist()
    pcts = [df[df["domain"]==d]["has_intron"].mean()*100 for d in domains]
    colors = [DOMAIN_PALETTE.get(d, "#aaaaaa") for d in domains]
    bars = ax.bar(domains, pcts, color=colors)
    for b, v in zip(bars, pcts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_ylabel("% tRNAs with intron")
    ax.set_title("Intron prevalence by domain")

    # right: top 30 phyla
    ax2 = axes[1]
    phyla = top_phyla(df, 30)
    sub   = df[df["phylum"].isin(phyla)]
    pct_p = sub.groupby("phylum")["has_intron"].mean() * 100
    pct_p = pct_p.reindex(phyla).sort_values(ascending=False)
    phylum_domain = df.groupby("phylum")["domain"].first()
    colors2 = [DOMAIN_PALETTE.get(phylum_domain.get(p,"Unknown"),"#aaaaaa") for p in pct_p.index]
    ax2.bar(range(len(pct_p)), pct_p.values, color=colors2)
    ax2.set_xticks(range(len(pct_p)))
    ax2.set_xticklabels(pct_p.index, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("% tRNAs with intron")
    ax2.set_title("Intron prevalence by phylum  (top 30)")

    plt.tight_layout()
    fig.savefig(OUT / "tax_06_intron_prevalence_by_domain.png")
    plt.close()
    print("  saved tax_06_intron_prevalence_by_domain.png")


# ── Fig 07: pseudo tRNA rate by phylum ───────────────────────────────────────
def fig_pseudo_by_phylum(df):
    phyla = top_phyla(df, 30)
    sub   = df[df["phylum"].isin(phyla)]
    pct_p = sub.groupby("phylum")["is_pseudo"].mean() * 100
    pct_p = pct_p.reindex(phyla).sort_values(ascending=False)
    phylum_domain = df.groupby("phylum")["domain"].first()
    colors = [DOMAIN_PALETTE.get(phylum_domain.get(p,"Unknown"),"#aaaaaa") for p in pct_p.index]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(pct_p)), pct_p.values, color=colors)
    ax.set_xticks(range(len(pct_p)))
    ax.set_xticklabels(pct_p.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% pseudo tRNAs")
    ax.set_title("Pseudo tRNA rate by phylum  (top 30 phyla by genome count)")
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=d) for d, c in DOMAIN_PALETTE.items() if d != "Unknown"]
    ax.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    fig.savefig(OUT / "tax_07_pseudo_rate_by_phylum.png")
    plt.close()
    print("  saved tax_07_pseudo_rate_by_phylum.png")


# ── Fig 08: tRNAscan score by domain ─────────────────────────────────────────
def fig_score_by_domain(df):
    domains = df["domain"].value_counts().index.tolist()
    colors  = [DOMAIN_PALETTE.get(d, "#aaaaaa") for d in domains]

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(
        [df.loc[df["domain"]==d, "Score"].dropna().values for d in domains],
        positions=range(len(domains)), showmedians=True,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    for part in ("cmedians","cbars","cmins","cmaxes"):
        parts[part].set_color("black"); parts[part].set_linewidth(1.2)

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains)
    ax.set_ylabel("tRNAscan Score")
    ax.set_title("tRNAscan score distribution by domain")
    plt.tight_layout()
    fig.savefig(OUT / "tax_08_score_by_domain.png")
    plt.close()
    print("  saved tax_08_score_by_domain.png")


# ── Fig 09: mean tRNA count per genome – top 10 phyla × isotype ──────────────
def fig_count_heatmap_top_phyla(df):
    phyla = top_phyla(df, 10)
    keep  = [i for i in ISOTYPE_ORDER]
    sub   = df[df["phylum"].isin(phyla) & df["Anticodon_predicted_isotype"].isin(keep)]

    mean_count = (sub.groupby(["GenomeID","phylum","Anticodon_predicted_isotype"])
                     .size()
                     .unstack(fill_value=0)
                     .reset_index()
                     .groupby("phylum")[keep]
                     .mean())
    mean_count = mean_count.reindex(phyla)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(mean_count, ax=ax, cmap="Blues",
                cbar_kws={"label": "Mean tRNA count per genome"},
                linewidths=0.3, annot=True, fmt=".1f", annot_kws={"size": 7})
    ax.set_title("Mean tRNA copy number per genome  –  top 10 phyla × isotype")
    ax.set_xlabel("Isotype"); ax.set_ylabel("Phylum")
    plt.tight_layout()
    fig.savefig(OUT / "tax_09_count_heatmap_top10_phyla.png")
    plt.close()
    print("  saved tax_09_count_heatmap_top10_phyla.png")


# ── Fig 10: tRNA count per genome – top 30 genera ────────────────────────────
def fig_trnas_per_genome_by_genus(df):
    genera = top_genera(df, 30)
    sub    = df[df["genus"].isin(genera)]
    per_gm = sub.groupby(["GenomeID","genus"]).size().reset_index(name="n_tRNA")
    medians = per_gm.groupby("genus")["n_tRNA"].median().reindex(genera).sort_values(ascending=False)
    order   = medians.index.tolist()
    genus_domain = df.groupby("genus")["domain"].first()
    palette  = {g: DOMAIN_PALETTE.get(genus_domain.get(g,"Unknown"),"#aaaaaa") for g in order}

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.violinplot(data=per_gm, x="genus", y="n_tRNA", order=order,
                   palette=palette, ax=ax, inner="quartile", cut=0, scale="width")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Genus"); ax.set_ylabel("tRNAs per genome")
    ax.set_title("tRNA count per genome  –  top 30 genera by genome count")
    plt.tight_layout()
    fig.savefig(OUT / "tax_10_trnas_per_genome_by_genus.png")
    plt.close()
    print("  saved tax_10_trnas_per_genome_by_genus.png")


# ── Fig 11: fMet copy number per genome by domain ────────────────────────────
def fig_fmet_by_domain(df):
    fmet = (df[df["Anticodon_predicted_isotype"] == "fMet"]
            .groupby(["GenomeID","domain"])
            .size()
            .reset_index(name="n_fMet"))
    domains = fmet["domain"].value_counts().index.tolist()
    colors  = [DOMAIN_PALETTE.get(d,"#aaaaaa") for d in domains]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: violin
    ax = axes[0]
    parts = ax.violinplot(
        [fmet.loc[fmet["domain"]==d,"n_fMet"].values for d in domains],
        positions=range(len(domains)), showmedians=True,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    for part in ("cmedians","cbars","cmins","cmaxes"):
        parts[part].set_color("black")
    ax.set_xticks(range(len(domains))); ax.set_xticklabels(domains)
    ax.set_ylabel("fMet tRNA copies per genome")
    ax.set_title("fMet tRNA copy number by domain")

    # right: histogram overlaid
    ax2 = axes[1]
    for d, color in zip(domains, colors):
        vals = fmet.loc[fmet["domain"]==d,"n_fMet"]
        ax2.hist(vals, bins=30, alpha=0.6, color=color, label=d, density=True)
    ax2.set_xlabel("fMet copies per genome")
    ax2.set_ylabel("Density")
    ax2.set_title("fMet copy number distribution by domain")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(OUT / "tax_11_fMet_by_domain.png")
    plt.close()
    print("  saved tax_11_fMet_by_domain.png")


# ── Fig 12: codon usage – Archaea vs Bacteria ────────────────────────────────
def fig_codon_usage_by_domain(df):
    top_codons = df["Codon"].value_counts().head(20).index.tolist()
    sub = df[df["Codon"].isin(top_codons) & df["domain"].isin(["Bacteria","Archaea"])]

    codon_frac = (sub.groupby(["domain","Codon"])
                     .size()
                     .unstack(fill_value=0))
    codon_frac = codon_frac.div(codon_frac.sum(axis=1), axis=0) * 100
    codon_frac = codon_frac.reindex(columns=top_codons)

    x     = np.arange(len(top_codons))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (domain, color) in enumerate(zip(["Bacteria","Archaea"],
                                             [DOMAIN_PALETTE["Bacteria"], DOMAIN_PALETTE["Archaea"]])):
        if domain in codon_frac.index:
            ax.bar(x + (i-0.5)*width, codon_frac.loc[domain, top_codons],
                   width, label=domain, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(top_codons, rotation=45, ha="right")
    ax.set_ylabel("% of domain's tRNAs")
    ax.set_title("Anticodon usage – Bacteria vs Archaea  (top 20 codons globally)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "tax_12_codon_usage_bacteria_vs_archaea.png")
    plt.close()
    print("  saved tax_12_codon_usage_bacteria_vs_archaea.png")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load()
    print("\nGenerating figures …")
    fig_genome_count_by_domain(df)
    fig_trnas_per_genome_by_domain(df)
    fig_isotype_heatmap_by_phylum(df)
    fig_sec_by_phylum(df)
    fig_ile2_by_domain(df)
    fig_introns_by_domain(df)
    fig_pseudo_by_phylum(df)
    fig_score_by_domain(df)
    fig_count_heatmap_top_phyla(df)
    fig_trnas_per_genome_by_genus(df)
    fig_fmet_by_domain(df)
    fig_codon_usage_by_domain(df)
    print(f"\nAll figures saved to {OUT}/")
