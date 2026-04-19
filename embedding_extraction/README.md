# Embedding Extraction and Analysis

This directory contains the pipeline for extracting contextualised embeddings from MacBERTh for the six target commodity terms, clustering them to identify usage types, extracting representative snippets, and computing Jensen–Shannon Divergence (JSD) between consecutive decades.

The embedding extraction and clustering analysis code is adapted from the [LatinISE project](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change) by Barbara McGillivray and colleagues, which provides a pipeline for extracting and analysing contextualised embeddings from BERT models for historical semantic change research. The JSD computation follows [Kutuzov & Giulianelli (2020)](https://aclanthology.org/2020.semeval-1.14/) and is based on their [released code](https://github.com/akutuzov/semeval2020/blob/master/code/jsd.py).

This is the core analytical pipeline of the paper, producing the data behind Tables 3, 4, 5–10, and Figures 8 and 9.

## Directory Structure

```
embedding_extraction/
├── README.md
├── gen_macberth.py                         # MacBERTh wrapper for embedding extraction
├── embeddings_extraction.py                # Extract target-word embeddings per decade
└── analysis/
    ├── commodity_analysis_finetuned.py     # Clustering + visualisation (domain-adapted model)
    ├── commodity_analysis_pretrained.py    # Clustering + visualisation (pretrained model)
    ├── extract_contexts_visible_only.py    # Centroid-closest snippets with target-word filtering
    ├── jsd.py                              # JSD computation via Affinity Propagation
    └── output/
        ├── csv/
        │   ├── cluster_summary_detailed.csv        # Clustering statistics (Table 4)
        │   ├── jsd_consecutive_decades.csv         # JSD between consecutive decades (Table 3)
        │   └── jsd_vs_1910.csv                     # JSD of each decade vs 1910s reference
        ├── PCA/
        │   ├── pca_coffee_all_decades.png
        │   ├── pca_tea_all_decades.png
        │   ├── pca_sugar_all_decades.png           # Figure 9a in the paper
        │   ├── pca_opium_all_decades.png
        │   ├── pca_cocoa_all_decades.png           # Figure 9b in the paper
        │   └── pca_tobacco_all_decades.png
        ├── clusters/
        │   ├── coffee/
        │   │   ├── clusters_coffee_1840.png        # Figure 8a in the paper
        │   │   ├── clusters_coffee_1850.png
        │   │   ├── ...
        │   │   └── clusters_coffee_1910.png        # Figure 8c in the paper
        │   ├── tea/
        │   │   ├── clusters_tea_1840.png
        │   │   └── ...
        │   ├── sugar/
        │   ├── opium/
        │   ├── cocoa/
        │   └── tobacco/
        │       ├── clusters_tobacco_1840.png
        │       └── ...
        └── cluster_contexts/
            ├── coffee/
            │   ├── contexts_coffee_1840.csv
            │   ├── contexts_coffee_1850.csv
            │   ├── ...
            │   └── contexts_coffee_1910.csv
            ├── tea/
            ├── sugar/
            ├── opium/
            ├── cocoa/
            │   ├── contexts_cocoa_1840.csv
            │   └── ...
            └── tobacco/
```

## Files

### Embedding extraction

| File | Description |
|---|---|
| `gen_macberth.py` | MacBERTh wrapper class. Handles tokenisation, subword-to-word alignment via transform matrices, and batched embedding extraction from the last hidden layer. Adapted from the `gen_berts.py` LatinBERT wrapper in the [LatinISE project](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change). Supports both the pretrained and domain-adapted MacBERTh |
| `embeddings_extraction.py` | Main extraction script. For each decade, finds all sentences containing target words (*coffee*, *tea*, *sugar*, *opium*, *cocoa*, *tobacco*), extracts the last-hidden-layer embedding at the target word's position, and saves embeddings with snippets to HDF5 files. Supports checkpointing (skips words already extracted) |

### Analysis

| File | Description |
|---|---|
| `commodity_analysis_finetuned.py` | Full analysis pipeline for the **domain-adapted** model: K-means clustering with silhouette-selected k ∈ [2, 10], per-decade cluster visualisations (PCA-projected), cross-decade PCA plots, and example snippets per cluster. Produces Table 4 and Figures 8–9 |
| `commodity_analysis_pretrained.py` | Identical pipeline for the **pretrained** model. Used for the domain-adaptation evaluation (Appendix) |
| `extract_contexts_visible_only.py` | Extracts **all** cluster contexts ranked by distance to centroid, filtered to keep only snippets where the target word is visible within the display window. Re-centres snippets around the target word (±150 characters). Produces the data behind Tables 5–10 |
| `jsd.py` | Computes JSD between consecutive decades and JSD vs. 1910 reference using Affinity Propagation on the combined usage matrix. Based on [Kutuzov & Giulianelli (2020)](https://github.com/akutuzov/semeval2020/blob/master/code/jsd.py). Produces Table 3. Includes checkpointing so interrupted runs can be resumed |

## Pipeline

### Step 1: Extract embeddings

Requires the domain-adapted MacBERTh model (from `domain_adaptation/`) and the decade sub-corpora (from `data/en_decade_corpus/`).

```bash
# Extract embeddings using the domain-adapted model
python embeddings_extraction.py --model finetuned

# Extract embeddings using the pretrained model (for domain-adaptation evaluation)
python embeddings_extraction.py --model pretrained
```

**What it does:**
1. Loads MacBERTh (pretrained or domain-adapted) via the `gen_macberth.py` wrapper
2. For each decade file, scans all lines for sentences containing target words
3. Extracts the last-hidden-layer embedding (768-d) at the target word's token position, using a transform matrix to handle subword tokenisation
4. Saves to HDF5: one file per decade (`commodity_embeddings_{decade}.h5`), with embeddings, snippets, and positions organised by word

**Output format** (HDF5):
```
commodity_embeddings_1840.h5
├── coffee/
│   ├── usage_0/
│   │   └── embedding    # (768,) float32, gzip-compressed
│   │   └── attrs: snippet, position
│   ├── usage_1/
│   └── ...
├── tea/
└── ...
```

**Note:** Update `dir_in` at the top of the script to point to your local data directory. Extraction processes ~50 sentences per batch and clears GPU memory between batches to avoid OOM on large decades (e.g., tea in the 1890s has ~42,000 usages).

### Step 2: Clustering analysis

```bash
# Run analysis on domain-adapted embeddings
python analysis/commodity_analysis_finetuned.py

# Run analysis on pretrained embeddings (for comparison)
python analysis/commodity_analysis_pretrained.py
```

**What it does:**
1. Loads all decade HDF5 files
2. For each word–decade pair, finds optimal k via silhouette score over k ∈ [2, 10]
3. Runs K-means with the selected k (random_state=42, n_init=10)
4. Saves cluster labels, centroid vectors, and PCA-projected cluster plots
5. Prints example snippets per cluster

### Step 3: Extract centroid-closest snippets

```bash
python analysis/extract_contexts_visible_only.py
```

**What it does:**
1. For each word–decade pair, runs the same K-means clustering as Step 2
2. Ranks all snippets within each cluster by Euclidean distance to the cluster centroid
3. Re-centres each snippet around the target word (±150 characters) to ensure visibility
4. Filters out snippets where the target word falls outside the display window
5. Saves all contexts (ranked by centroid proximity) to CSV files

The **five closest visible snippets per cluster** from these CSVs are what appear in the paper's appendix tables (Tables 5–10). The thematic category labels in those tables are the authors' interpretive readings, not algorithmically derived.

### Step 4: Compute JSD

```bash
python analysis/jsd.py
```

**What it does:**
1. For each pair of consecutive decades, pools embeddings from both decades into a combined matrix [U^t1; U^t2]
2. Computes cosine similarity and clusters via Affinity Propagation (damping=0.9, precomputed affinity)
3. Derives per-decade probability distributions from normalised cluster counts
4. Computes Jensen–Shannon Divergence using `scipy.spatial.distance.jensenshannon`
5. Also computes JSD vs. the 1910s reference decade for all words

## Output

### `output/csv/`

| File | Description | Paper element |
|---|---|---|
| `cluster_summary_detailed.csv` | Per word-decade clustering statistics: n_clusters, silhouette score, total usages, cluster sizes, and number of visible snippets after filtering | Table 4 |
| `jsd_consecutive_decades.csv` | JSD between each pair of consecutive decades for all six words | Table 3 |
| `jsd_vs_1910.csv` | JSD of each decade against the 1910s reference for all six words | Not in current paper; available for alternative analysis |

**Sample from `cluster_summary_detailed.csv`:**

| word | decade | n_clusters | silhouette | total_usages | cluster_sizes | visible_snippets |
|---|---|---|---|---|---|---|
| coffee | 1840 | 3 | 0.1764 | 2,651 | 1391\|603\|657 | 779 |
| coffee | 1890 | 7 | 0.1517 | 7,727 | 306\|979\|2198\|1116\|980\|699\|1449 | 3,101 |
| tea | 1890 | 3 | 0.1660 | 42,616 | 17755\|13202\|11659 | 19,540 |
| opium | 1880 | 2 | 0.2536 | 613 | 173\|440 | 318 |

Note that `visible_snippets` is always less than `total_usages` because some snippets have the target word outside the ±150 character display window and are filtered out.

### `output/PCA/`

Six cross-decade PCA projection plots, one per commodity. Each plot projects all of a word's 768-dimensional contextualised embeddings across all eight decades into a shared 2D PCA space, with points coloured by decade. **Figure 9** in the paper uses `pca_sugar_all_decades.png` and `pca_cocoa_all_decades.png`.

### `output/clusters/`

Per-decade cluster visualisation plots organised by commodity (6 subdirectories × 8 decades = 48 plots). Each plot shows the PCA-projected embeddings for one word in one decade, coloured by K-means cluster assignment with centroids marked. **Figure 8** in the paper uses `clusters_coffee_1840.png`, `clusters_coffee_1890.png`, and `clusters_coffee_1910.png`.

### `output/cluster_contexts/`

CSV files with all cluster contexts ranked by distance to centroid, organised by commodity (6 subdirectories × 8 decades = 48 files). Each CSV contains:

| Column | Description |
|---|---|
| `word` | Target commodity term |
| `decade` | Decade |
| `cluster` | Cluster ID (0-indexed) |
| `cluster_size` | Total number of usages in this cluster |
| `rank_in_cluster` | Rank by distance to centroid (1 = closest) |
| `distance_to_centroid` | Euclidean distance to the cluster centroid |
| `target_word_visible` | Whether the target word appears in the display window (always `True` after filtering) |
| `snippet` | Text snippet re-centred around the target word (±150 characters) |

These files are the evidence base for **Tables 5–10** in the paper. The five snippets closest to each centroid (i.e., rows with `rank_in_cluster` ≤ 5) were used for qualitative cluster interpretation.

## Relationship to the paper

| Paper element | Produced by | Output file(s) |
|---|---|---|
| Table 3 (JSD values) | `jsd.py` | `output/csv/jsd_consecutive_decades.csv` |
| Table 4 (clustering statistics) | `extract_contexts_visible_only.py` | `output/csv/cluster_summary_detailed.csv` |
| Tables 5–10 (cluster contexts) | `extract_contexts_visible_only.py` | `output/cluster_contexts/{word}/` |
| Figure 8 (coffee clusters) | `commodity_analysis_finetuned.py` | `output/clusters/coffee/` |
| Figure 9 (sugar/cocoa PCA) | `commodity_analysis_finetuned.py` | `output/PCA/pca_sugar_all_decades.png`, `pca_cocoa_all_decades.png` |
| Appendix (domain-adaptation comparison) | `commodity_analysis_pretrained.py` + `commodity_analysis_finetuned.py` | Compared qualitatively in the paper |

## Dependencies

Key packages beyond the standard scientific Python stack:

- `transformers` (HuggingFace) — for loading MacBERTh
- `h5py` — for HDF5 embedding storage
- `scikit-learn` — for K-means, Affinity Propagation, PCA, silhouette score
- `scipy` — for `jensenshannon` distance
- `torch` — for GPU-accelerated model inference

See the root `requirements.txt` for version pinning.
