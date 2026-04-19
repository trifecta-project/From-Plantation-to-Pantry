import os
import csv
import pickle
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# =============================================================================
# Configuration
# =============================================================================

# Define paths
dir_in = "..." #change it to your path
dir_emb = os.path.join(dir_in, "output", "embeddings_macberth_finetuned")
dir_out = os.path.join(dir_in, "output", "analysis_finetuned")

# Create output directories
os.makedirs(os.path.join(dir_out, "csv"), exist_ok=True)
os.makedirs(os.path.join(dir_out, "PCA"), exist_ok=True)
os.makedirs(os.path.join(dir_out, "TSNE"), exist_ok=True)
os.makedirs(os.path.join(dir_out, "clusters"), exist_ok=True)

# Define decades and target words
decades = [1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910]
TARGET_WORDS = ['coffee', 'tea', 'sugar', 'opium', 'cocoa', 'tobacco']

rebuild = True  # Set to False to load from cached files

# =============================================================================
# Data Loading Functions
# =============================================================================

def load_commodity_embeddings(decade, dir_emb):
    """
    Load commodity embeddings from HDF5 file for a specific decade.
    
    Returns:
        dict: {word: [(embedding, snippet, position), ...]}
    """
    file_path = os.path.join(dir_emb, f"commodity_embeddings_{decade}.h5")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    print(f"Loading embeddings for {decade}...")
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
            print(f"  {word}: {count} usages")
    
    return word_data


def load_all_decades(decades, dir_emb):
    """Load embeddings for all decades."""
    all_data = {}
    for decade in decades:
        data = load_commodity_embeddings(decade, dir_emb)
        if data:
            all_data[decade] = data
    return all_data


# =============================================================================
# Analysis Functions (from bert_latinise.ipynb)
# =============================================================================

def compute_average_embedding(embeddings):
    """Compute normalized average embedding."""
    if len(embeddings) == 0:
        return None
    embeddings_array = np.array(embeddings)
    avg = np.mean(embeddings_array, axis=0)
    return normalize(avg.reshape(1, -1))[0]

def optimal_num_clusters(embeddings, min_k=2, max_k=10, random_state=42):
    """
    Find optimal number of clusters using silhouette score.
    """
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


def cluster_word_embeddings(embeddings, num_clusters=None):
    """
    Cluster word embeddings and return cluster centroids and labels.
    If num_clusters is None, find optimal number.
    """
    embeddings_array = np.array(embeddings)
    
    if num_clusters is None:
        num_clusters, score = optimal_num_clusters(embeddings_array)
        print(f"  Optimal clusters: {num_clusters} (silhouette: {score:.4f})")
    
    if len(embeddings_array) < num_clusters:
        return embeddings_array, np.arange(len(embeddings_array)), num_clusters
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_array)
    centroids = kmeans.cluster_centers_
    
    return centroids, labels, num_clusters


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_embeddings_pca_tsne(embeddings, title="t-SNE Visualization", 
                                   pca_components=50, tsne_perplexity=30,
                                   save_path=None, word=None, decade=None):
    """
    Reduce dimensionality with PCA + t-SNE and visualize.
    """
    if len(embeddings) < pca_components:
        pca_components = len(embeddings) - 1
    
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(embeddings)
    
    perplexity = min(tsne_perplexity, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    
    if save_path and word and decade:
        filename = f"tsne_{word}_{decade}.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {full_path}")
    
    plt.close()
    return X_pca, X_tsne


def plot_embeddings_across_decades(word, all_data, save_path=None):
    """
    Plot embeddings for a word across all decades using shared PCA space.
    """
    all_embeddings = []
    decade_labels = []
    
    for decade in sorted(all_data.keys()):
        if word in all_data[decade]:
            embs = all_data[decade][word]['embeddings']
            all_embeddings.extend(embs)
            decade_labels.extend([decade] * len(embs))
    
    if not all_embeddings:
        print(f"No embeddings found for '{word}'")
        return
    
    all_embeddings = np.array(all_embeddings)
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeddings)
    
    plt.figure(figsize=(12, 8))
    unique_decades = sorted(set(decade_labels))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_decades)))
    
    for i, decade in enumerate(unique_decades):
        mask = [d == decade for d in decade_labels]
        plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                   c=[colors[i]], label=str(decade), alpha=0.6)
    
    plt.title(f"PCA of '{word}' embeddings across decades")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Decade", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        filename = f"pca_{word}_all_decades.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {full_path}")
    
    plt.close()


def plot_cluster_visualization(word, embeddings, labels, centroids, decade, save_path=None):
    """
    Visualize clusters for a word's embeddings.
    """
    embeddings_array = np.array(embeddings)
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings_array)
    centroids_reduced = pca.transform(centroids)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], 
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    
    plt.title(f"Clusters of '{word}' in {decade}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    
    if save_path:
        filename = f"clusters_{word}_{decade}.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {full_path}")
    
    plt.close()


# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    # Load all data
    print("Loading embeddings from all decades...")
    all_data = load_all_decades(decades, dir_emb)
    
    print(f"\nLoaded {len(all_data)} decades")
    
    # ==========================================================================
    # 1. Clustering analysis
    # ==========================================================================
    print("\n" + "="*60)
    print("5. CLUSTERING ANALYSIS")
    print("="*60)
    
    cluster_results = {}
    
    for word in TARGET_WORDS:
        print(f"\n{word}:")
        cluster_results[word] = {}
        
        for decade in sorted(all_data.keys()):
            if word in all_data[decade]:
                embeddings = all_data[decade][word]['embeddings']
                if len(embeddings) >= 5:  # Need enough data for clustering
                    print(f"  {decade} ({len(embeddings)} usages):")
                    centroids, labels, k = cluster_word_embeddings(embeddings)
                    cluster_results[word][decade] = {
                        'centroids': centroids,
                        'labels': labels,
                        'n_clusters': k
                    }
                    
                    # Plot clusters
                    plot_cluster_visualization(
                        word, embeddings, labels, centroids, decade,
                        save_path=os.path.join(dir_out, "clusters")
                    )
    
    # Save clustering summary to CSV
    cluster_csv = os.path.join(dir_out, "csv", "cluster_summary.csv")
    with open(cluster_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Word", "Decade", "N_Usages", "N_Clusters", "Cluster_Sizes"])
        for word, decades_data in cluster_results.items():
            for decade, data in sorted(decades_data.items()):
                labels = data['labels']
                n_clusters = data['n_clusters']
                n_usages = len(labels)
                cluster_sizes = "|".join([str(np.sum(labels == i)) for i in range(n_clusters)])
                writer.writerow([word, decade, n_usages, n_clusters, cluster_sizes])
    print(f"\nSaved clustering summary to {cluster_csv}")
    
    # ==========================================================================
    # 2. Visualize embeddings across all decades
    # ==========================================================================
    print("\n" + "="*60)
    print("6. CROSS-DECADE VISUALIZATION")
    print("="*60)
    
    for word in TARGET_WORDS:
        print(f"\nPlotting {word} across decades...")
        plot_embeddings_across_decades(word, all_data, save_path=os.path.join(dir_out, "PCA"))
    
    # ==========================================================================
    # 3. Get example snippets for each cluster
    # ==========================================================================
    print("\n" + "="*60)
    print("7. EXAMPLE SNIPPETS PER CLUSTER")
    print("="*60)
    
    for word in TARGET_WORDS:
        print(f"\n{'='*40}")
        print(f"{word.upper()}")
        print(f"{'='*40}")
        
        for decade in sorted(all_data.keys()):
            if word in all_data[decade] and decade in cluster_results.get(word, {}):
                labels = cluster_results[word][decade]['labels']
                snippets = all_data[decade][word]['snippets']
                n_clusters = cluster_results[word][decade]['n_clusters']
                
                print(f"\n{decade} ({n_clusters} clusters):")
                for cluster_id in range(n_clusters):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    print(f"  Cluster {cluster_id+1} ({len(cluster_indices)} usages):")
                    # Show first 2 examples
                    for idx in cluster_indices[:2]:
                        snippet = snippets[idx][:100] + "..." if len(snippets[idx]) > 100 else snippets[idx]
                        print(f"    - {snippet}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {dir_out}")
    print("="*60)
