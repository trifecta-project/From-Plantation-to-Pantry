# Embedding Extraction and Analysis

This directory contains the pipeline that extracts a contextualised embedding for every occurrence of the target commodity words from MacBERTh, and the analysis that clusters these embeddings and measures change between periods.

The extraction code is adapted from the [LatinISE project](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change) by Barbara McGillivray and Valentina Lunardi.

## Directory Structure

```
embedding_extraction/
├── README.md
├── gen_macberth.py            # MacBERTh wrapper: word-level embeddings from chosen hidden layers
├── embeddings_extraction.py   # Extracts target-word embeddings per decade
└── analysis/                  # Clustering, change measures and their outputs (see "Analysis" below)
```

## Embedding Extraction

### Files

| File | Description |
|---|---|
| `gen_macberth.py` | Wrapper around a HuggingFace BERT model (pretrained MacBERTh or a domain-adapted checkpoint). Tokenises each input into subwords, maps subword vectors back to words by averaging, and returns one 768-dimensional vector per word. `layer_mode` selects the hidden layers used: `last` (last layer), `last4` (mean of the last four layers), `first4` (mean of the first four) or `all` (mean of all layers). Adapted from the `gen_berts.py` LatinBERT wrapper of the LatinISE project |
| `embeddings_extraction.py` | Finds every occurrence of the target words in each decade sub-corpus, embeds a window of words centred on it, and saves the target word's embedding with its context to one HDF5 file per decade |

### Requirements

- The decade sub-corpora `1800.txt` … `1910.txt` in `data/en_decade_corpus/` (see `corpus_preparation/`).
- The model: the domain-adapted checkpoint `output/scaling_experiment/config_A/epoch_3` (see `domain_adaptation/`) or the pretrained [`emanjavacas/MacBERTh`](https://huggingface.co/emanjavacas/MacBERTh).

Set `dir_in` at the top of `embeddings_extraction.py` to your local data directory; the corpus is read from `dir_in/en_decade_corpus/` and the output written to `dir_in/output/`.

### Running it

The settings used in the paper:

```bash
# Domain-adapted model (Config A, epoch 3)
python embeddings_extraction.py --model scaling_best --layer-mode last4 \
    --variants spelling --context-window 50

# Pretrained MacBERTh, for the comparison with the domain-adapted model
python embeddings_extraction.py --model pretrained --layer-mode last4 \
    --variants spelling --context-window 50
```

| Option | Values | Meaning |
|---|---|---|
| `--model` | `scaling_best` (default), `pretrained` | Domain-adapted checkpoint or pretrained MacBERTh |
| `--layer-mode` | `last` (default), `last4`, `first4`, `all` | Hidden layers averaged into each embedding. The paper uses `last4`; the default `last` is kept for compatibility with earlier runs, so pass `--layer-mode last4` explicitly |
| `--variants` | `none`, `spelling` (default) | `none` matches the singular only (*coffee*); `spelling` also matches the plural (*coffees*, *teas*, *sugars*) |
| `--context-window` | integer, default 50 | Words of context kept on each side of the target |

The output folder name records the settings, e.g. `output/embeddings_macberth_scaling_best_windowed_last4_spelling_w50/`. Words already present in a decade's HDF5 file are skipped, so an interrupted run can be resumed.

### What it does

1. **Finds occurrences.** Each line of a decade file (one article) is lower-cased and split on spaces; free-standing punctuation tokens are dropped. A token matches if it equals one of the target word's forms exactly. A token with punctuation attached, such as `coffee,` or `tea.`, therefore does not match, so occurrences directly followed or preceded by punctuation are not included.
2. **Cuts a context window.** For each occurrence, the script keeps up to 50 words on each side of the target, so every occurrence is embedded in its own local context and the window stays well within the model's 512-token limit. An article containing a word several times yields one window per occurrence.
3. **Embeds the window.** The window is passed through MacBERTh (via `gen_macberth.py`), and the vector at the target word's position is kept: the mean of the last four hidden layers (`last4`), averaged over the target word's subword tokens.
4. **Saves** the embedding, the window text and the target's position in the window, in batches of 50 occurrences.

The script extracts six commodity words: *coffee*, *tea*, *sugar*, *opium*, *cocoa* and *tobacco*. The paper analyses *coffee*, *tea* and *sugar*.

### Output

One HDF5 file per decade, `commodity_embeddings_{decade}.h5`:

```
commodity_embeddings_1840.h5          attrs: model, model_path, decade, variant_mode, context_window, layer_mode
├── coffee/                           attrs: count, variants, variant_mode, context_window, layer_mode
│   ├── usage_0/
│   │   └── embedding                 (768,) float32, gzip-compressed
│   │                                 attrs: snippet (the context window, lower-cased), position (target index in the window)
│   ├── usage_1/
│   └── ...
├── tea/
└── ...
```

The `usage_*` groups are numbered from 0 in the order the occurrences were found: article by article, and within an article form by form. HDF5 lists them in string order (`usage_10` before `usage_2`), so sort them numerically when reading embeddings and snippets together.

`frequency_summary.csv` in the same folder has one row per word and decade:

| Column | Description |
|---|---|
| `model` | `scaling_best` or `pretrained` |
| `variant_mode` | `none` or `spelling` |
| `context_window` | Words of context on each side |
| `word` | Target word |
| `decade` | Decade |
| `forms` | Surface forms searched, separated by `\|` |
| `count` | Occurrences found |

`count` is the number of occurrences found; the number of embeddings saved is in each word group's `count` attribute. The two can differ only for an occurrence whose window cannot be embedded, which the script skips.

## Analysis

### Files

| File | Description |
|---|---|
| `commodity_analysis_finetuned.py` | Full analysis pipeline for the **domain-adapted** model: K-means clustering with silhouette-selected k ∈ [2, 10], per-decade cluster visualisations (PCA-projected), cross-decade PCA plots, and example snippets per cluster. Produces Table 4 and Figures 8–9 |
| `commodity_analysis_pretrained.py` | Identical pipeline for the **pretrained** model. Used for the domain-adaptation evaluation (Appendix) |
| `extract_contexts_visible_only.py` | Extracts **all** cluster contexts ranked by distance to centroid, filtered to keep only snippets where the target word is visible within the display window. Re-centres snippets around the target word (±150 characters). Produces the data behind Tables 5–10 |
| `jsd.py` | Computes JSD between consecutive decades and JSD vs. 1910 reference using Affinity Propagation on the combined usage matrix. Based on [Kutuzov & Giulianelli (2020)](https://github.com/akutuzov/semeval2020/blob/master/code/jsd.py). Produces Table 3. Includes checkpointing so interrupted runs can be resumed |


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
