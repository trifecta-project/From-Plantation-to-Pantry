# Analysis: Sense Clustering, Change Measurement and Dependency Parsing

This directory contains the analysis of the occurrence embeddings extracted in `embedding_extraction/`:

1. **Sense clustering and change measurement** (`cluster_jsd_by_period.py`). All occurrence embeddings of a word are clustered once, across all periods, into a shared inventory of clusters. Each period's cluster distribution is then compared with the next period's by Jensen–Shannon divergence (JSD), tested against sampling noise by permutation.
2. **Plotting the change** (`plot_jsd.py`): the JSD series per word, with the permutation noise floor.
3. **Dependency parsing** (`dep_profile_across_periods.py`): the syntactic relations of each word per cluster or per period, as independent evidence for what the clusters and their shifts represent.

## Files

| File | Description |
|---|---|
| `cluster_jsd_by_period.py` | Shared k-means clustering of one word's embeddings over all periods; cluster distributions per period; JSD between consecutive periods with a permutation test; representative snippets per cluster; stacked composition plot |
| `plot_jsd.py` | Per-word and combined JSD plots from the CSVs of `cluster_jsd_by_period.py` |
| `dep_profile_across_periods.py` | spaCy dependency profiles of each word per cluster or per period, from the per-occurrence CSV of `cluster_jsd_by_period.py` |
| `results/` | The outputs reported in the paper: `clusters/<word>/` (Steps 1 and 2), `dependencies_by_period/` and `dependencies_by_cluster/` (Step 3) |

## Requirements

- The HDF5 embeddings from `embedding_extraction/embeddings_extraction.py`, one folder per model and setting (e.g. the yearly `last4` embeddings of the domain-adapted model).
- `numpy`, `scipy`, `scikit-learn`, `h5py`, `matplotlib`, `pandas`, and `spacy` with an English model (`python -m spacy download en_core_web_sm`).

## Step 1: Sense clustering and JSD

```bash
for w in coffee tea sugar; do
  OMP_NUM_THREADS=1 python cluster_jsd_by_period.py --word $w \
      --emb-dir /path/to/embeddings \
      --silhouette-sample 100000000 \
      --balance-clustering --permute 999 --all-snippets \
      --out-dir results/clusters/$w
done
```

`--silhouette-sample 100000000` exceeds every word's number of occurrences, so k is selected on all embeddings, as in the paper (see step 2 below). For a quick exploratory run, leave it at the default of 2,000.

`OMP_NUM_THREADS=1` makes the k-means cluster ids identical across reruns, which matters when clusters are labelled by id.

### What it does

1. **Loads** every period's embeddings of the word from `commodity_embeddings_<period>.h5`. Periods are found from the file names, so a folder of 119 yearly files and a folder of 12 decade files both work; `--slice-width` groups years into wider slices.
2. **Chooses k** once, on the pooled embeddings of all periods: k-means is fitted for each k in `--k-range` (default 2–10) and scored by its silhouette, and the k with the highest score is selected. `--k` fixes k instead, and its silhouette score is still computed and reported. k is never chosen per period, so that all periods are described over the same clusters.

   The candidate fits and silhouette scores use `--silhouette-sample` occurrences, drawn at random (default 2,000), or all of them when the value is at least the number of pooled embeddings. **The results in the paper and in this repository use all embeddings**, i.e. a `--silhouette-sample` larger than the pooled count. This is much slower: the silhouette score compares every occurrence with every other, so its cost grows with the square of the number of occurrences, which runs to over a million for *tea*. The silhouette score is computed without the `--balance-clustering` weights.
3. **Clusters** all pooled embeddings with k-means (`n_init=10`, seed 42). With `--balance-clustering`, each embedding is weighted by the inverse of its period's number of occurrences, so that every period contributes equally to where the centroids lie while all embeddings are used.
4. **Describes each period** by its cluster distribution: the share of its occurrences in each cluster.
5. **Measures change** between consecutive periods as the Jensen–Shannon divergence between their cluster distributions (base 2, so between 0 and 1). The CSV also gives the Jensen–Shannon distance, its square root, which is what `scipy.spatial.distance.jensenshannon` returns.
6. **Tests each JSD against sampling noise** (`--permute r`): the cluster labels of the two periods are pooled, shuffled and split again at the two original sizes, r times, giving the JSD expected from sampling noise alone. The p-value is the share of the r + 1 values (observed included) that reach the observed JSD, so it cannot fall below 1/(r + 1); p-values are corrected across a word's transitions with Benjamini–Hochberg (`--alpha`, default 0.05). Because p reaches its floor easily with large samples, the CSV also gives the excess over the null mean and the corresponding z-score.
7. **Lists representative snippets**: for each cluster, the occurrences closest to its centroid (by cosine), with near-duplicates (e.g. reprinted advertisements) collapsed for display only (`--near-dup-threshold`, default 0.75).

Other options: `--canonical-only` or `--exclude-forms` restrict the surface forms used; `--exclude-next-token` drops occurrences followed by given tokens (e.g. `nut` after *cocoa*); `--dedup-embeddings` removes byte-identical embeddings before clustering; `--max-per-period` caps the occurrences per period; `--minibatch` uses MiniBatchKMeans for very large pools.

### Sense labels

The clusters are interpreted by reading their representative snippets. The labels are entered in `SENSE_LABELS_BY_SHARE` at the top of the script, each with the cluster's share of all occurrences as printed in `cluster_snippets_<word>.txt`, and are matched to cluster ids by share for the legend of the stacked plot. Where two clusters have similar shares, `LABELS_BY_CLUSTER_ID` assigns labels to cluster ids directly.

### Output (per word)

| File | Contents |
|---|---|
| `counts_<word>.csv` | Occurrences per period after any filtering |
| `silhouette_<word>.csv` | Silhouette score per candidate k, and the k selected |
| `cluster_snippets_<word>.txt` | The occurrences nearest each centroid, with each cluster's size and share |
| `cluster_all_snippets_<word>.csv` | With `--all-snippets`: every occurrence with its cluster, period, cosine to the centroid and context window (input to the dependency parsing). Not included in `results/`: it has one row per occurrence (over a million for *tea*) and is regenerated by rerunning with `--all-snippets` |
| `props_<word>.csv` | Cluster distribution per period |
| `jsd_<word>.csv` | JSD between consecutive periods: `js_distance`, `js_divergence` and, with `--permute`, `null_mean_divergence`, `excess`, `z`, `p_value`, `p_value_bh`, `significant` |
| `stacked_<word>.png` | Cluster distribution over time, as a stacked area plot |
| `facets_<word>.png` | Per-period scatter of the clusters in a shared PCA projection; drawn only for at most `--max-facets` periods (default 16) |
| `dedup_report_<word>.csv` | With `--dedup-embeddings`: embeddings removed as exact duplicates |

The period columns are called `decade`, `decade_start` and `decade_end` whatever the granularity; at year level they hold years.

## Step 2: Plot the JSD series

```bash
python plot_jsd.py --folder results/clusters --words coffee tea sugar --combined --smooth 5
```

It finds `jsd_<word>.csv` in `results/clusters/<word>/` and writes `perword_jsd_<word>.png` for each word and, with `--combined`, `jsd_all_words_clusters.png` with all words overlaid, to `results/clusters/` (or `--out-dir`). When the CSVs come from a run with `--permute`, the permutation noise floor is drawn in grey and transitions significant after Benjamini–Hochberg are marked. `--smooth N` adds a centred rolling median over N periods; `--metric distance` plots the Jensen–Shannon distance instead of the divergence.

## Step 3: Dependency parsing

Requires `cluster_all_snippets_<word>.csv` from Step 1 (run with `--all-snippets`).

```bash
for w in coffee tea sugar; do
  # per decade (10-year bins of the yearly clusters)
  python dep_profile_across_periods.py --words $w \
      --csv-dir results/clusters/$w --group-by period --period-width 10 \
      --out-dir results/dependencies_by_period

  # per cluster
  python dep_profile_across_periods.py --words $w \
      --csv-dir results/clusters/$w --group-by cluster \
      --out-dir results/dependencies_by_cluster
done
```

`--csv-dir` must be the folder that holds `cluster_all_snippets_<word>.csv` itself; with one folder per word, as in `results/clusters/`, run the script once per word.

These are the two runs reported in the paper (with the default `--min-count 3`, `--relations amod compound compound-of verb` and `en_core_web_sm`).

### What it does

Each occurrence's context window, as written by Step 1 (±50 tokens around the target with its default `--window`), is parsed with spaCy (`--model`, default `en_core_web_sm`), the target token is located, and its relations are counted:

| Relation | Meaning | Example |
|---|---|---|
| `amod:<lemma>` | Adjective modifying the target | *fine* coffee |
| `compound:<lemma>` | Noun modifying the target | *breakfast* coffee |
| `compound-of:<lemma>` | Target modifying a noun | coffee *house* |
| `<dep>-of:<lemma>` | Target as an argument of a verb, e.g. its object | `dobj-of:roast` |
| `nummod`, `nmod-of` | Optional, with `--relations` | |

Lemmas must be alphabetic, at least two characters long and not the target itself; modifiers and verbs that are stop words are excluded too. Relations seen fewer than `--min-count` times in a group (default 3) are dropped, to filter out one-off OCR errors, before each relation's share of the group's relations is computed.

Groups (`--group-by`): `cluster`; `period` (bins of `--period-width` years, starting on a multiple of the width, e.g. 1800s, 1810s); or `cluster_period`. `--max-per-group` samples a fixed number of occurrences per group.

### Output (per word)

| File | Contents |
|---|---|
| `deps_by_<group>_<word>.csv` | Every relation per group, with count and share |
| `deps_table_by_<group>_<word>.txt` | The `--topn` most frequent relations per group (default 10) |
| `deps_table_by_<group>_<word>.tex` | The same as a LaTeX table; written one row per group when there are more groups than `--max-table-cols` (default 8) |

`<group>` is `cluster`, or `period` followed by the bin width (e.g. `period10`).
