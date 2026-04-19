import os
import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# Configuration
# =============================================================================

dir_in = "..." #change it to your path
dir_emb = os.path.join(dir_in, "output", "embeddings_macberth_finetuned")

# New output directory — only visible-target snippets
dir_out = os.path.join(dir_in, "output", "analysis")

os.makedirs(os.path.join(dir_out, "cluster_contexts"), exist_ok=True)
os.makedirs(os.path.join(dir_out, "csv"), exist_ok=True)

decades = [1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910]
TARGET_WORDS = ['coffee', 'tea', 'sugar', 'opium', 'cocoa', 'tobacco']

# Characters to show either side of the target word in the snippet
SNIPPET_RADIUS = 150


# =============================================================================
# Snippet fix: recentre around target word
# =============================================================================

def fix_snippet(snippet, target_word, radius=SNIPPET_RADIUS):
    """
    Recentre the snippet window around the target word so it is always visible.
    Returns (fixed_snippet, word_found).
    If the target word is not found (OCR corruption), returns original snippet
    unchanged and word_found=False.
    """
    lower = snippet.lower()
    pos = lower.find(target_word.lower())

    if pos == -1:
        return snippet, False

    start = max(0, pos - radius)
    end = min(len(snippet), pos + len(target_word) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(snippet) else ""

    return prefix + snippet[start:end] + suffix, True


# =============================================================================
# Data loading
# =============================================================================

def load_commodity_embeddings(decade, dir_emb):
    """Load commodity embeddings from HDF5 file."""
    file_path = os.path.join(dir_emb, f"commodity_embeddings_{decade}.h5")
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return None

    word_data = {}
    with h5py.File(file_path, 'r') as f:
        for word in f.keys():
            count = f[word].attrs['count']
            embeddings = []
            snippets = []

            for i in range(count):
                usage = f[word][f'usage_{i}']
                emb = usage['embedding'][:]
                snippet = usage.attrs.get('snippet', '')
                if isinstance(snippet, bytes):
                    snippet = snippet.decode('utf-8')
                embeddings.append(emb)
                snippets.append(snippet)

            word_data[word] = {
                'embeddings': embeddings,
                'snippets': snippets,
                'count': count
            }

    return word_data


# =============================================================================
# Clustering
# =============================================================================

def optimal_num_clusters(embeddings, min_k=2, max_k=10, random_state=42):
    """Find optimal number of clusters using silhouette score."""
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    if len(embeddings) < min_k:
        return 1, None

    best_k = min_k
    best_score = -1

    for k in range(min_k, min(max_k + 1, len(embeddings))):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        if len(set(labels)) > 1 and len(set(labels)) < len(embeddings):
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k

    return best_k, best_score


# =============================================================================
# Main extraction — ALL contexts, filtered to visible only
# =============================================================================

def extract_all_cluster_contexts(word, decade, dir_emb):
    """
    Extract ALL contexts for each cluster of a word in a given decade.
    Sorted by distance to centroid (closest = most representative first).
    Only keeps rows where the target word is visible in the snippet.
    """
    data = load_commodity_embeddings(decade, dir_emb)
    if data is None or word not in data:
        print(f"  No data for {word} in {decade}")
        return None

    embeddings = np.array(data[word]['embeddings'])
    snippets = data[word]['snippets']

    if len(embeddings) < 5:
        print(f"  Not enough data for clustering: {len(embeddings)} usages")
        return None

    # Find optimal number of clusters
    n_clusters, silhouette = optimal_num_clusters(embeddings)
    sil_str = f"{silhouette:.4f}" if silhouette is not None else "N/A"
    print(f"  {word} {decade}: {n_clusters} clusters (silhouette: {sil_str}), "
          f"{len(embeddings)} total usages")

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Distance to centroid
    distances = np.zeros(len(embeddings))
    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        centroid = kmeans.cluster_centers_[label]
        distances[i] = np.linalg.norm(emb - centroid)

    # Build results — ALL rows, no n_examples cap
    results = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_distances = distances[cluster_mask]

        # Sort by distance to centroid
        sorted_idx = np.argsort(cluster_distances)

        for rank, idx in enumerate(sorted_idx):
            original_idx = cluster_indices[idx]
            raw_snippet = snippets[original_idx]
            fixed_snippet, word_found = fix_snippet(raw_snippet, word)

            results.append({
                'word': word,
                'decade': decade,
                'cluster': cluster_id,
                'cluster_size': int(cluster_mask.sum()),
                'rank_in_cluster': rank + 1,
                'distance_to_centroid': round(float(cluster_distances[idx]), 4),
                'target_word_visible': word_found,
                'snippet': fixed_snippet,
            })

    df = pd.DataFrame(results)

    # Count before filtering
    total_rows = len(df)
    n_missing = (~df['target_word_visible']).sum()

    # Filter: keep only rows where target word is visible
    df = df[df['target_word_visible']].copy()

    n_kept = len(df)
    print(f"    Filtered: {n_kept}/{total_rows} snippets kept "
          f"({n_missing} removed — target word not visible)")

    return df, n_clusters, silhouette


# =============================================================================
# Run
# =============================================================================

def main():
    all_results = []
    summary_results = []

    for word in TARGET_WORDS:
        print(f"\n{'='*60}")
        print(f"Processing: {word}")
        print('='*60)

        for decade in decades:
            result = extract_all_cluster_contexts(word, decade, dir_emb)

            if result is not None:
                df, n_clusters, silhouette = result
                all_results.append(df)

                cluster_sizes = df.groupby('cluster')['cluster_size'].first().tolist()
                n_visible = len(df)

                summary_results.append({
                    'word': word,
                    'decade': decade,
                    'n_clusters': n_clusters,
                    'silhouette': round(silhouette, 4) if silhouette else None,
                    'total_usages': sum(cluster_sizes),
                    'cluster_sizes': '|'.join(map(str, cluster_sizes)),
                    'visible_snippets': n_visible,
                })

                # Save per word-decade file
                out_file = os.path.join(
                    dir_out, "cluster_contexts", f"contexts_{word}_{decade}.csv"
                )
                df.to_csv(out_file, index=False)
                print(f"    Saved {len(df)} rows -> {out_file}")

    # Save combined file
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = os.path.join(dir_out, "cluster_contexts", "all_cluster_contexts.csv")
        combined_df.to_csv(combined_file, index=False)
        total_rows = len(combined_df)
        print(f"\nSaved combined contexts ({total_rows:,} rows): {combined_file}")

    # Save summary
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_file = os.path.join(dir_out, "csv", "cluster_summary_detailed.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary: {summary_file}")

        print("\n" + "="*60)
        print("CLUSTERING SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
