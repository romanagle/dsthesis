# Roma — Reproducibility Document
**Last updated: 2026-05-07**

---

## Overview

**Roma** is a deep-learning pipeline for extracting tRNA identity elements at
genome-scale. It trains a graph neural network (GNN) on per-position RiNALMo
embeddings from ~850 K deduplicated bacterial and archaeal tRNA sequences, then
uses in-silico mutagenesis (ISM) to identify sequence positions that drive
isotype classification. Identity elements are visualised as signed sequence logos
in the Sprinzl coordinate system, stratified by domain and taxonomic clade.

The pipeline uses RiNALMo as the only language model. The active classifier is
the GNN (`scripts/train/train_gnn.py`). Softmax baselines exist in
`scripts/figures/plot_ablation_comparison.py` for comparison only.

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [Prerequisites](#2-prerequisites)
3. [Isotype Classes](#3-isotype-classes)
4. [Pipeline Workflow](#4-pipeline-workflow)
   - [Phase 1 — Data Preparation](#phase-1--data-preparation)
   - [Phase 2 — Embedding Extraction](#phase-2--embedding-extraction)
   - [Phase 3 — GNN Training and Global ISM](#phase-3--gnn-training-and-global-ism)
   - [Phase 4 — Clade-level ISM](#phase-4--clade-level-ism)
   - [Phase 5 — Figures and EDA](#phase-5--figures-and-eda)
5. [HPC / SLURM Jobs](#5-hpc--slurm-jobs)
6. [Results Layout](#6-results-layout)
7. [Script Reference](#7-script-reference)

---

## 1. Repository Layout

```
dsthesis/
├── data/                   # Input CSVs (large; not in git)
│   ├── trnascan_GTDB_r226_dedup_taxonomy.csv   # Primary training/analysis file (~1.5 GB)
│   ├── sprinzl_filtered.csv                    # Sprinzl position occupancy filter
│   ├── training_dataset.csv                    # Filtered GNN training set
│   └── known_IEs.csv                           # Reference identity elements from literature
│
├── embeddings/
│   └── gnn/                # Per-position RiNALMo embeddings (not in git)
│       ├── perpos_embeddings.npz   # sequences, labels, classes, phylo_ohe, max_len
│       ├── emb_mmap.npy            # (N, max_len, 1280) float16 memory-map
│       └── nuc_embeddings.npy      # (4, 1280) per-nucleotide mean embeddings
│
├── gnn_final/              # Baseline trained GNN + ISM outputs
│   ├── gnn_model.pt
│   ├── phylo_encoder.pkl
│   ├── gnn_results.json
│   ├── gnn_ism_saliency_{iso}_{domain}.npy
│   ├── order_ism/
│   └── order_ism_culturable/
│
├── gnn_film/               # FiLM-conditioned GNN + ISM outputs (after run_film_pipeline.sh)
│   ├── gnn_model.pt
│   ├── gnn_results.json
│   └── gnn_ism_saliency_{iso}_{domain}.npy
│
├── figures/                # Publication figures (PNG + SVG only — no PDFs)
│   ├── classification/     # Softmax ablation accuracy
│   ├── dedup/              # Deduplication overview and accuracy-vs-level
│   ├── film_comparison/    # Baseline vs FiLM GNN comparison
│   ├── freq_logos/         # Nucleotide frequency logos
│   ├── gnn_eda/            # GNN training EDA
│   ├── ie_comparison/      # ISM logos vs known identity elements
│   ├── ism_logos/          # Main ISM sequence logos (baseline GNN)
│   ├── ism_logos_film/     # ISM sequence logos (FiLM GNN)
│   ├── misc/               # Miscellaneous figures
│   └── taxonomy/           # Taxonomy EDA figures
│
├── scripts/
│   ├── pipeline/           # Data preparation and CSV construction
│   ├── embeddings/         # RiNALMo embedding extraction
│   ├── train/              # GNN classifier training and sweeps
│   ├── ism/                # ISM computation and clade analysis
│   ├── figures/            # All figure generation
│   ├── utils/              # Shared constants and helpers
│   │   ├── __init__.py
│   │   └── iso_classes.py
│   └── hpc/                # SLURM sbatch job files
│
├── logs/                   # Pipeline run logs
├── DOCUMENTATION.md
├── run_film_pipeline.sh    # End-to-end FiLM pipeline runner
└── setup_env.sh            # Conda environment bootstrap
```

---

## 2. Prerequisites

### 2.1 Data root

Every script derives its file paths from a single environment variable:

```bash
export TRNA_DATA_ROOT=/data/roma/dsthesis   # default if unset
```

### 2.2 Conda environment

```bash
bash setup_env.sh        # creates conda env 'myenv'
conda activate myenv
```

**Key dependencies:**

| Package | Purpose |
|---|---|
| `torch` ≥ 2.6, CUDA 12.4 | Deep learning |
| `rinalmo` (git install) | RiNALMo RNA foundation model |
| `flash-attn` ≥ 2.8 | Efficient attention (required by RiNALMo) |
| `pandas`, `numpy` | Data I/O |
| `scikit-learn` | Metrics and train/test splits |
| `logomaker` | Sequence logo rendering |
| `matplotlib`, `seaborn` | Plotting |
| `wandb` | Hyperparameter sweep logging |

**External tools** (must be on `PATH`):
- `mmseqs2` — sequence deduplication

---

## 3. Isotype Classes

All class definitions live in `scripts/utils/iso_classes.py`.

**21 active classes** (alphabetical):
```
Ala  Arg  Asn  Asp  Cys  Gln  Glu  Gly  His
Ile  iMet Leu  Lys  Met  Phe  Pro  Ser
Thr  Trp  Tyr  Val
```

**Merge rules applied at load time:**

| Raw tRNAscan label | Merged into |
|---|---|
| `Ile2` | `Ile` |
| `fMet` | `Met` |

**Dropped:** `SeC`, `Undet`, `Sup`

---

## 4. Pipeline Workflow

### Phase 1 — Data Preparation

**Goal:** Build the primary analysis CSV from raw tRNAscan-SE predictions.
Run once on the full GTDB r226 genome set; outputs already exist.

| Step | Script | Output |
|---|---|---|
| 1.1 | `pipeline/build_trnascan_csv.py` | Per-phylum CSV fragments |
| 1.2 | `pipeline/add_taxonomy.py` | + GTDB taxonomy columns |
| 1.3 | `pipeline/add_sprinzl.py` | + Sprinzl position columns |
| 1.4 | `pipeline/dedup_mmseqs2.sh` | `trnascan_GTDB_r226_dedup_taxonomy.csv` |
| 1.5 | `pipeline/filter_structural_variants.py` | `data/sprinzl_filtered.csv` |
| 1.6 | `pipeline/build_training_dataset.py` | `data/training_dataset.csv` |

```bash
# Rebuild training dataset from existing CSVs (steps 1.5–1.6 already done):
python scripts/pipeline/build_training_dataset.py \
    --csv     data/trnascan_GTDB_r226_dedup_taxonomy.csv \
    --sprinzl data/sprinzl_filtered.csv \
    --output  data/training_dataset.csv
```

---

### Phase 2 — Embedding Extraction

**Already complete.** Re-run only if sequences change.

```bash
python scripts/embeddings/extract_rinalmo_perpos.py
python scripts/embeddings/npz_to_mmap.py --npz embeddings/gnn/perpos_embeddings.npz
```

**Outputs:** `embeddings/gnn/perpos_embeddings.npz`, `emb_mmap.npy`, `nuc_embeddings.npy`

---

### Phase 3 — GNN Training and Global ISM

```bash
# Baseline GNN (already trained → gnn_final/)
python scripts/train/train_gnn.py \
    --emb-dir embeddings/gnn \
    --csv     data/trnascan_GTDB_r226_dedup_taxonomy.csv \
    --outdir  gnn_final

# FiLM-conditioned GNN (run_film_pipeline.sh handles this)
python scripts/train/train_gnn.py \
    --emb-dir       embeddings/gnn \
    --csv           data/trnascan_GTDB_r226_dedup_taxonomy.csv \
    --outdir        gnn_film \
    --film \
    --phylo-encoder gnn_final/phylo_encoder.pkl
```

**Architecture:** 3-layer dense GCN → mean pool → [optional FiLM bottleneck] →
concat phylo OHE → linear head over 21 classes.

**FiLM option:** `--film` enables a two-stage training: stage 1 trains the full
model with FiLM zero-initialised (no-op); stage 2 freezes the GNN backbone and
fine-tunes only the FiLM + head to learn order-specific residuals.

**Key outputs in `gnn_final/` or `gnn_film/`:**
- `gnn_model.pt` — model weights
- `phylo_encoder.pkl` — fitted OneHotEncoder for phylogenetic features
- `gnn_results.json` — test accuracy, IE scores, hyperparams
- `gnn_ism_saliency_{iso}_{domain}.npy` — ISM maps (shape: `max_len × 4`)

---

### Phase 4 — Clade-level ISM

```bash
# Distribute ISM across GPUs for all eligible orders
python scripts/ism/run_order_ism.py --n-gpus 2 --min-seqs-bac 200

# Preview commands without running
python scripts/ism/run_order_ism.py --dry-run

# Single clade manually
python scripts/ism/compute_ism_clade.py \
    --clades Pseudomonadales Pelagibacterales \
    --clade-col order \
    --outdir gnn_final/order_ism_culturable
```

**Output:** `gnn_final/order_ism_culturable/gnn_ism_saliency_{iso}_{order}.npy`

---

### Phase 5 — Figures and EDA

All figure scripts write **PNG + SVG** only. No PDFs. No titles or subtitles in figures.

```bash
# ISM logos (baseline)
python scripts/figures/plot_ism_logos.py \
    --ism-dir gnn_final \
    --outdir  figures/ism_logos

# ISM logos (FiLM)
python scripts/figures/plot_ism_logos.py \
    --ism-dir gnn_film \
    --outdir  figures/ism_logos_film

# Baseline vs FiLM comparison
python scripts/figures/compare_film_vs_base.py \
    --base-dir gnn_final \
    --film-dir gnn_film \
    --outdir   figures/film_comparison

# Order-level logos
python scripts/figures/plot_ism_logos.py \
    --ism-dir   gnn_final/order_ism_culturable \
    --clade-col order \
    --outdir    figures/ism_logos_orders
```

---

## 5. HPC / SLURM Jobs

```bash
sbatch scripts/hpc/sbatch/01_embed_full.sbatch   # RiNALMo full embeddings
sbatch scripts/hpc/sbatch/03_plot.sbatch          # dedup accuracy plots
```

---

## 6. Results Layout

```
gnn_final/                              # Baseline GNN
├── gnn_model.pt
├── phylo_encoder.pkl
├── gnn_results.json                    # accuracy: 0.9689
├── gnn_ism_saliency_{iso}_{domain}.npy # 45 files (21 iso × 2 domain + merges)
├── order_ism/
└── order_ism_culturable/
    ├── {order}.done
    └── gnn_ism_saliency_{iso}_{order}.npy

gnn_film/                               # FiLM GNN (after run_film_pipeline.sh)
├── gnn_model.pt
├── gnn_results.json
└── gnn_ism_saliency_{iso}_{domain}.npy

figures/
├── classification/     softmax ablation accuracy
├── dedup/              dedup overview, accuracy-vs-level plots
├── film_comparison/    accuracy_ie_comparison.png, saliency_heatmap_comparison.png
├── freq_logos/         nucleotide frequency logos per isotype × domain
├── gnn_eda/            GNN training EDA
├── ie_comparison/      ISM logos vs known IEs, recall heatmaps
├── ism_logos/          signed ISM logos (baseline GNN)
├── ism_logos_film/     signed ISM logos (FiLM GNN)
├── misc/               miscellaneous one-off figures
└── taxonomy/           genome counts, isotype distributions by taxonomy
```

---

## 7. Script Reference

### `scripts/pipeline/`

| Script | Input | Output |
|---|---|---|
| `build_trnascan_csv.py` | tRNAscan-SE per-genome files | Per-phylum CSV fragments |
| `add_taxonomy.py` | Base CSV + GTDB taxonomy TSV | + taxonomy columns |
| `add_sprinzl.py` | Taxonomy CSV | + Sprinzl position columns |
| `dedup_by_taxlevel.py` | Dedup+taxonomy CSV | `data/dedup_{level}.csv` × 7 |
| `filter_structural_variants.py` | Sprinzl CSV | `data/sprinzl_filtered.csv` |
| `build_training_dataset.py` | Taxonomy CSV + sprinzl_filtered.csv | `data/training_dataset.csv` |
| `build_order_dataset.py` | Taxonomy CSV | Per-order training CSVs |

### `scripts/embeddings/`

| Script | Input | Output |
|---|---|---|
| `extract_rinalmo_perpos.py` | Taxonomy CSV | `embeddings/gnn/perpos_embeddings.npz` |
| `npz_to_mmap.py` | NPZ | `emb_mmap.npy`, `nuc_embeddings.npy` |
| `extract_rinalmo_full.py` | CSV | Mean-pool embeddings (ablation) |

### `scripts/train/`

| Script | Input | Output |
|---|---|---|
| `train_gnn.py` | NPZ + mmap + CSV | `{outdir}/gnn_model.pt` + ISM saliency |
| `train_gnn_clade.py` | NPZ + mmap + CSV | Clade-specific GNN model |
| `gnn_wandb_sweep.py` | NPZ + mmap | WandB sweep logs |

### `scripts/ism/`

| Script | Input | Output |
|---|---|---|
| `run_order_ism.py` | Taxonomy CSV | Launches `compute_ism_clade.py` across GPUs |
| `compute_ism_all.py` | NPZ + mmap + model + CSV | `gnn_ism_saliency_*.npy` |
| `compute_ism_clade.py` | NPZ + mmap + model + CSV | Per-clade saliency `.npy` |
| `compute_pairwise_ism.py` | NPZ + mmap + model | Pairwise ISM distance matrix |
| `build_ism_embeddings.py` | Saliency files | ISM-derived embedding matrix |
| `aggregate_clade_ism.py` | Clade saliency files | Aggregated embeddings |
| `predict_isotype.py` | Model + sequences | Isotype predictions |
| `run_clade_analysis.py` | Saliency files + CSV | Clade ISM analysis |
| `run_family_ism.py` | NPZ + model + CSV | Family-level ISM |

### `scripts/figures/`

| Script | Output directory |
|---|---|
| `plot_ism_logos.py` | `figures/ism_logos/` or `figures/ism_logos_film/` |
| `plot_freq_logos.py` | `figures/freq_logos/` |
| `plot_ism_logos_highlight.py` | `figures/ie_comparison/` |
| `plot_ism_logos_known_IEs.py` | `figures/ie_comparison/` |
| `rank_ism_vs_known_IEs.py` | `figures/ie_comparison/` |
| `compare_film_vs_base.py` | `figures/film_comparison/` |
| `eda_trnascan.py` | `figures/misc/` |
| `eda_taxonomy.py` | `figures/taxonomy/` |
| `compare_dedup.py` | `figures/dedup/` |
| `plot_dedup_accuracy.py` | `figures/dedup/` |
| `plot_dedup_sprinzl_accuracy.py` | `figures/dedup/` |
| `plot_softmax_phylo_accuracy.py` | `figures/classification/` |
| `plot_linear_probe_accuracy.py` | `figures/classification/` |
| `plot_ablation_comparison.py` | `figures/classification/` |
| `plot_ablation_umap.py` | `figures/misc/` |
| `plot_umap_phylogeny.py` | `figures/misc/` |
| `plot_sequence_length_dist.py` | `figures/misc/` |
| `plot_centroid_heatmap.py` | `figures/misc/` |
| `plot_centroid_distances.py` | `figures/misc/` |
| `analyze_ism_order_embeddings.py` | `figures/ie_comparison/` |
| `identity_order_embeddings.py` | `figures/ie_comparison/` |
| `family_ism_analysis.py` | `figures/ie_comparison/` |
| `plot_ecoli_cross_domain.py` | `figures/misc/` |
| `plot_pairwise_ism_logos.py` | `figures/ie_comparison/` |
| `plot_pairwise_ism_comparison_logos.py` | `figures/ie_comparison/` |
| `plot_pelagib_his_comparison.py` | `figures/ie_comparison/` |
| `regen_overview.py` | `figures/dedup/` |
| `ablation_dataset_filter.py` | `figures/classification/` |

### `scripts/utils/`

| File | Contents |
|---|---|
| `iso_classes.py` | `ISOTYPE_ORDER`, `MERGE_ISOS`, `DROP_ISOS`, `normalize_isotypes()`, `remap_isotypes()` |
