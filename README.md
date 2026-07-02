# DIGtRNA: Discriminative element Identification via GCN for tRNAs


DIGtRNA is a genome-scale deep learning pipeline for discovering tRNA identity elements: the sequence and structural features that govern aminoacyl-tRNA synthetase recognition. It trains a graph convolutional network (GCN) on frozen RiNALMo embeddings from 855,166 deduplicated prokaryotic tRNA sequences, then uses in silico mutagenesis (ISM) to generate signed sequence logos that identify which nucleotide positions drive isotype-specific classification. Logos are reported in the Sprinzl coordinate system, stratified by domain (Bacteria / Archaea) and optionally by taxonomic order.

## Overview

Each tRNA must be accurately charged by its cognate aminoacyl-tRNA synthetase. The sequence and structural features that determine this specificity are well-characterized in a handful of model organisms but poorly mapped across the full diversity of prokaryotic life. These features are called identity elements. DIGtRNA addresses this gap by training a discriminative classifier on millions of tRNA sequences from 142,794 genomes in GTDB r226 and applying ISM to extract the learned recognition logic as interpretable sequence logos for all 22 isotype classes.

The approach recovers established identity elements (e.g. the G3:U70 wobble pair for tRNA-Ala, A73 for tRNA-Val) while nominating candidate elements in lineages underrepresented in classical biochemical studies.

## Methods at a glance

```
GTDB r226 genomes (142,794)
    → tRNAscan-SE 2.0 annotation (6.07M raw tRNAs)
    → Sprinzl alignment, confidence filtering, deduplication
    → 855,166 curated tRNA sequences (22 isotypes, 5,861 families)
    → RiNALMo (650M-param RNA LM, frozen) per-position embeddings
    → Graph construction (backbone edges + self-loops)
    → 3-layer dense GCN classifier + phylogenetic one-hot features
    → In silico mutagenesis (ISM) at every Sprinzl position
    → Signed ISM logos per isotype × domain
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for the full pipeline walkthrough, script reference, and HPC job details.

## Key results

| Model | Bacteria accuracy | Archaea accuracy |
|---|---|---|
| GCN (baseline) | 95.4% | 97.4% |
| GCN + FiLM (order-conditioned) | 98.9% | 99.0% |

- **RiNALMo outperforms RNA-FM** at every taxonomic level when the anticodon is masked, with an 18% accuracy gap at the domain level and ~5% advantage at phylum–order resolution.
- **GCN architecture recovers known identity elements** better than transformer or MLP classifiers with comparable accuracy, demonstrating that inductive bias aligned with tRNA structure matters more than raw model capacity for mechanistic interpretability.
- **ISM logos for all 22 isotypes** are included in [`figures/ism_logos_film/`](figures/ism_logos_film/), with frequency logos for comparison in [`figures/freq_logos/`](figures/freq_logos/).

## Repository structure

```
dsthesis/
├── scripts/
│   ├── pipeline/       data preparation and CSV construction
│   ├── embeddings/     RiNALMo embedding extraction
│   ├── train/          GCN training and hyperparameter sweeps
│   ├── ism/            ISM computation and clade-level analysis
│   ├── figures/        figure generation
│   └── utils/          shared constants (isotype classes, merges)
├── figures/
│   ├── ism_logos_film/ ISM sequence logos — FiLM GCN (main results)
│   ├── freq_logos/     nucleotide frequency logos per isotype × domain
│   ├── gnn_eda_5iso/   architecture comparison (accuracy + IE score)
│   ├── classification/ per-taxon softmax accuracy
│   ├── dedup/          deduplication overview and accuracy-vs-level
│   ├── ie_comparison_final/  Pelagibacterales His comparison
│   └── misc/           UMAP embeddings
├── data/               input CSVs — not tracked; see DOCUMENTATION.md
├── setup_env.sh        conda environment bootstrap
├── run_film_pipeline.sh  end-to-end FiLM pipeline runner
└── DOCUMENTATION.md    full reproducibility guide
```

Large files (embeddings, model weights, saliency maps, data CSVs) are excluded from the repository via `.gitignore`. See [DOCUMENTATION.md](DOCUMENTATION.md) for paths and reconstruction instructions.

## Setup

```bash
bash setup_env.sh        # creates conda env 'myenv'
conda activate myenv
export TRNA_DATA_ROOT=/path/to/dsthesis   # default: /data/roma/dsthesis
```

**Key dependencies:** `torch` ≥ 2.6 (CUDA 12.4), `rinalmo` (git install), `flash-attn` ≥ 2.8, `pandas`, `numpy`, `scikit-learn`, `logomaker`, `matplotlib`, `wandb`

**External tools:** `mmseqs2` (sequence deduplication)
