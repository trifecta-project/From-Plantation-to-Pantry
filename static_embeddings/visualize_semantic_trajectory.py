"""
Visualize Semantic Change Trajectory with Diachronic Word Embeddings
=====================================================================

This script visualizes how word meanings shift over time using Word2Vec 
embeddings trained on historical newspaper corpora. It extracts nearest 
neighbors across decades, applies t-SNE dimensionality reduction, and 
plots the semantic trajectory showing movement through embedding space.

Adapted from the Living with Machines project pipeline.

Requirements:
    pip install numpy pandas matplotlib scikit-learn gensim tqdm adjustText pyspellchecker

Usage:
    python visualize_semantic_trajectory.py --word coffee --vectors_dir ./vectors --output_dir ./output
    python visualize_semantic_trajectory.py --word tea --decades 1840s 1860s 1880s 1900s
    python visualize_semantic_trajectory.py --word sugar --topn 30 

"""

import os
import argparse
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings('ignore')

# Optional dependencies
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("Note: adjustText not installed. Text labels may overlap.")
    print("      Install with: pip install adjustText")

try:
    from spellchecker import SpellChecker
    HAS_SPELLCHECKER = True
except ImportError:
    HAS_SPELLCHECKER = False
    print("Note: pyspellchecker not installed. OCR error filtering disabled.")
    print("      Install with: pip install pyspellchecker")


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Example target words (colonial commodities from 19th-century British newspapers)
EXAMPLE_WORDS = ['coffee', 'tea', 'sugar', 'opium', 'cocoa', 'tobacco']

# Default time slices (decades)
DEFAULT_DECADES = ['1840s', '1850s', '1860s', '1870s', '1880s', '1890s', '1900s', '1910s']

# Color palette for decades (colorblind-friendly)
DECADE_COLORS = {
    '1840s': '#1b9e77',  # teal
    '1850s': '#d95f02',  # orange
    '1860s': '#7570b3',  # purple
    '1870s': '#e7298a',  # pink
    '1880s': '#66a61e',  # green
    '1890s': '#e6ab02',  # gold
    '1900s': '#a6761d',  # brown
    '1910s': '#666666',  # gray
}

# t-SNE parameters
TSNE_RANDOM_STATE = 42  # For reproducibility


# =============================================================================
# LOAD VECTORS
# =============================================================================

def load_all_models(vectors_dir, file_pattern=None):
    """
    Load all Word2Vec models from a directory.
    
    Args:
        vectors_dir: Path to directory containing vector files
        file_pattern: Optional glob pattern (default: tries common patterns)
    
    Returns:
        models: Dict mapping decade name to KeyedVectors model
        slices_names: List of decade names in sorted order
    """
    print("=" * 70)
    print("LOADING VECTORS")
    print("=" * 70)
    
    # Try common file patterns
    patterns = [
        f'{vectors_dir}/*-vectors.txt',
        f'{vectors_dir}/*.txt',
        f'{vectors_dir}/*.vec',
        f'{vectors_dir}/*.bin',
    ]
    
    if file_pattern:
        patterns = [file_pattern]
    
    allmodels_paths = []
    for pattern in patterns:
        allmodels_paths = sorted(glob(pattern))
        if allmodels_paths:
            break
    
    if not allmodels_paths:
        print(f"ERROR: No vector files found in {vectors_dir}")
        print("Expected formats: *-vectors.txt, *.txt, *.vec, *.bin")
        return None, None
    
    # Extract decade names from filenames
    slices_names = []
    for path in allmodels_paths:
        filename = os.path.basename(path)
        # Remove common suffixes
        slice_name = filename
        for suffix in ['-vectors.txt', '-vectors', '.txt', '.vec', '.bin']:
            slice_name = slice_name.replace(suffix, '')
        slices_names.append(slice_name)
    
    print(f"Found {len(allmodels_paths)} vector files:")
    for name in slices_names:
        print(f"  - {name}")
    
    # Load models
    models = {}
    for path, name in tqdm(zip(allmodels_paths, slices_names), 
                           total=len(allmodels_paths), desc="Loading"):
        try:
            # Try text format first, then binary
            if path.endswith('.bin'):
                model = KeyedVectors.load_word2vec_format(path, binary=True)
            else:
                model = KeyedVectors.load_word2vec_format(path, binary=False)
            models[name] = model
            print(f"  {name}: {len(model):,} words")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    
    return models, slices_names


# =============================================================================
# EXTRACT NEAREST NEIGHBORS
# =============================================================================

def get_nearest_neighbors(models, keyword, timeslices, topn=20):
    """
    Extract nearest neighbors for a keyword across selected time slices.
    
    Args:
        models: Dict of KeyedVectors models
        keyword: Target word
        timeslices: List of decade names to analyze
        topn: Number of neighbors per decade
    
    Returns:
        vocab: Union of all neighbor words
        neighbors_by_slice: Dict mapping decade to list of (word, score) tuples
    """
    print(f"\nExtracting neighbors for '{keyword}'...")
    
    vocab = []
    neighbors_by_slice = {}
    
    for slice_name in timeslices:
        if slice_name not in models:
            print(f"  WARNING: {slice_name} not found in models")
            continue
        
        model = models[slice_name]
        
        if keyword not in model:
            print(f"  WARNING: '{keyword}' not in {slice_name} vocabulary")
            continue
        
        try:
            similar = model.most_similar(positive=keyword, topn=topn)
            neighbors = [word for word, score in similar]
            neighbors_by_slice[slice_name] = similar
            vocab.extend(neighbors)
            print(f"  {slice_name}: {len(neighbors)} neighbors")
        except KeyError:
            print(f"  {slice_name}: keyword not found")
    
    # Remove duplicates while preserving order
    vocab = list(dict.fromkeys(vocab))
    
    return vocab, neighbors_by_slice


# =============================================================================
# FILTER OCR ERRORS
# =============================================================================

def filter_ocr_errors(vocab, enabled=True):
    """
    Filter out likely OCR errors using spellchecker.
    
    Args:
        vocab: List of words to filter
        enabled: Whether to apply filtering
    
    Returns:
        Filtered vocabulary list
    """
    print(f"\nFiltering OCR errors from {len(vocab)} words...")
    
    if not enabled:
        print("  OCR filtering disabled")
        return vocab.copy()
    
    if not HAS_SPELLCHECKER:
        print("  Spellchecker not available, skipping filter")
        return vocab.copy()
    
    spell = SpellChecker()
    newvocab = []
    known = set(spell.known(vocab))
    
    for word in vocab:
        if word in known:
            newvocab.append(word)
        else:
            # Try to correct
            correction = spell.correction(word)
            if correction and correction != word:
                newvocab.append(correction)
            # If no correction, skip (likely OCR error)
    
    # Remove duplicates
    newvocab = list(dict.fromkeys(newvocab))
    print(f"  Kept {len(newvocab)} words after filtering")
    
    return newvocab


# =============================================================================
# COLLECT VECTORS FOR t-SNE
# =============================================================================

def collect_vectors_for_tsne(models, keyword, neighbors_by_slice, timeslices):
    """
    Collect vectors for t-SNE visualization.
    
    Args:
        models: Dict of KeyedVectors models
        keyword: Target word
        neighbors_by_slice: Dict from get_nearest_neighbors
        timeslices: List of decades to include
    
    Returns:
        X: numpy array of vectors
        full_vocab: list of word labels
        n_keyword: number of keyword vectors (for trajectory)
        word_decade_map: dict mapping words to source decades
    """
    print(f"\nCollecting vectors for t-SNE...")
    
    # Use last time slice as reference for neighbor positions
    reference_slice = timeslices[-1]
    
    if reference_slice not in models:
        print(f"  ERROR: Reference slice {reference_slice} not found")
        return None, None, None, None
    
    ref_model = models[reference_slice]
    
    # Build map of which decades each neighbor appears in
    word_decade_map = {}
    all_neighbors = set()
    
    for slice_name, neighbors in neighbors_by_slice.items():
        for word, score in neighbors:
            all_neighbors.add(word)
            if word not in word_decade_map:
                word_decade_map[word] = []
            word_decade_map[word].append(slice_name)
    
    # Collect neighbor vectors (only those in reference model)
    fortsne = []
    final_vocab = []
    notpresent = []
    
    for word in all_neighbors:
        if word != keyword:
            if word in ref_model:
                fortsne.append(ref_model[word])
                final_vocab.append(word)
            else:
                notpresent.append(word)
    
    if notpresent:
        print(f"  Words not in {reference_slice}: {len(notpresent)}")
    
    # Collect keyword vectors for each time slice (for trajectory)
    keyword_labels = []
    for slice_name in timeslices:
        if slice_name in models and keyword in models[slice_name]:
            fortsne.append(models[slice_name][keyword])
            keyword_labels.append(f"{keyword}_{slice_name}")
    
    full_vocab = final_vocab + keyword_labels
    word_decade_map = {w: d for w, d in word_decade_map.items() if w in final_vocab}
    
    print(f"  Neighbor vectors: {len(final_vocab)}")
    print(f"  Keyword vectors: {len(keyword_labels)}")
    print(f"  Total vectors: {len(fortsne)}")
    
    return np.array(fortsne), full_vocab, len(keyword_labels), word_decade_map


# =============================================================================
# VISUALIZATION
# =============================================================================

def get_decade_color(decades_list, timeslices):
    """Assign color based on which decade(s) a word appears as neighbor."""
    if len(decades_list) == 1:
        return DECADE_COLORS.get(decades_list[0], 'gray')
    elif len(decades_list) == len(timeslices):
        return '#2c2c2c'  # Dark gray for stable words
    else:
        # Use color of earliest decade
        for ts in timeslices:
            if ts in decades_list:
                return DECADE_COLORS.get(ts, 'gray')
        return 'gray'


def visualize_trajectory(X, vocab, n_keyword_vectors, keyword, timeslices, 
                        output_dir, word_decade_map=None):
    """
    Apply t-SNE and create trajectory visualization.
    
    Args:
        X: Vector array from collect_vectors_for_tsne
        vocab: Word labels
        n_keyword_vectors: Number of trajectory points
        keyword: Target word
        timeslices: Decades analyzed
        output_dir: Where to save output
        word_decade_map: Decade mapping for coloring
    
    Returns:
        X_embedded: 2D t-SNE coordinates
    """
    print(f"\nApplying t-SNE...")
    
    X_embedded = TSNE(
        n_components=2,
        random_state=TSNE_RANDOM_STATE,
        metric='euclidean',
        learning_rate='auto',
        init='pca'
    ).fit_transform(X)
    
    # Split into neighbors and trajectory
    n_neighbors = len(X) - n_keyword_vectors
    
    neighbor_coords = X_embedded[:n_neighbors]
    keyword_coords = X_embedded[n_neighbors:]
    
    neighbor_labels = vocab[:n_neighbors]
    keyword_labels = vocab[n_neighbors:]
    
    # Assign colors to neighbors
    neighbor_colors = []
    if word_decade_map:
        for word in neighbor_labels:
            decades = word_decade_map.get(word, [])
            neighbor_colors.append(get_decade_color(decades, timeslices))
    else:
        neighbor_colors = ['lightgray'] * len(neighbor_labels)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12), dpi=100, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot neighbors
    for x, y, color in zip(neighbor_coords[:, 0], neighbor_coords[:, 1], neighbor_colors):
        ax.scatter(x, y, c=color, s=80, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # Add neighbor labels
    texts_neighbors = []
    for i, (label, color) in enumerate(zip(neighbor_labels, neighbor_colors)):
        txt = ax.text(neighbor_coords[i, 0], neighbor_coords[i, 1], label,
                     color=color, fontsize=9, ha='center', va='center', fontweight='bold')
        texts_neighbors.append(txt)
    
    # Plot trajectory points
    keyword_colors = [DECADE_COLORS.get(ts, 'red') for ts in timeslices if f"{keyword}_{ts}" in keyword_labels]
    
    for i, (x, y) in enumerate(keyword_coords):
        color = keyword_colors[i] if i < len(keyword_colors) else 'red'
        ax.scatter(x, y, c=color, s=300, zorder=5, edgecolors='black', linewidths=2)
    
    # Add trajectory labels
    texts_keyword = []
    for i, label in enumerate(keyword_labels):
        txt = ax.text(keyword_coords[i, 0], keyword_coords[i, 1], label,
                     color='black', fontsize=11, fontweight='bold', ha='center', va='bottom')
        texts_keyword.append(txt)
    
    # Draw arrows between trajectory points
    for i in range(len(keyword_coords) - 1):
        ax.annotate('',
            xy=(keyword_coords[i+1, 0], keyword_coords[i+1, 1]),
            xytext=(keyword_coords[i, 0], keyword_coords[i, 1]),
            arrowprops=dict(arrowstyle='->', color='black', lw=2)
        )
    
    # Adjust overlapping text
    if HAS_ADJUSTTEXT:
        adjust_text(texts_neighbors, arrowprops=dict(arrowstyle='-', color='gray', lw=0.3))
        adjust_text(texts_keyword)
    
    # Create legend
    legend_elements = []
    for ts in timeslices:
        if ts in DECADE_COLORS:
            legend_elements.append(
                plt.scatter([], [], c=DECADE_COLORS[ts], s=100, label=ts, edgecolors='white')
            )
    
    if word_decade_map:
        stable_count = sum(1 for d in word_decade_map.values() if len(d) == len(timeslices))
        if stable_count > 0:
            legend_elements.append(
                plt.scatter([], [], c='#2c2c2c', s=100, 
                           label=f'Stable ({stable_count} words)', edgecolors='white')
            )
    
    ax.legend(handles=legend_elements, loc='upper left', 
              title='Neighbor appears in:', fontsize=10, title_fontsize=11)
    
    # Labels and title
    ax.set_xlabel('t-SNE dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE dimension 2', fontsize=12)
    ax.set_title(f'Semantic Trajectory: {keyword.upper()}\n({" → ".join(timeslices)})', fontsize=14)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'trajectory_{keyword}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return X_embedded


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_neighbors_table(neighbors_by_slice, keyword, output_dir):
    """Save nearest neighbors for each decade as CSV."""
    rows = []
    for slice_name, neighbors in neighbors_by_slice.items():
        for rank, (word, score) in enumerate(neighbors, 1):
            rows.append({
                'word': keyword,
                'decade': slice_name,
                'rank': rank,
                'neighbor': word,
                'similarity': round(score, 4)
            })
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, f'neighbors_{keyword}.csv')
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    return df


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_word(models, keyword, timeslices, output_dir, topn=20, filter_ocr=True):
    """
    Process a single word: extract neighbors, apply t-SNE, visualize.
    
    Args:
        models: Dict of loaded models
        keyword: Target word
        timeslices: Decades to analyze
        output_dir: Output directory
        topn: Number of neighbors per decade
        filter_ocr: Whether to filter OCR errors
    """
    print("\n" + "=" * 70)
    print(f"PROCESSING: {keyword.upper()}")
    print("=" * 70)
    
    # Extract neighbors
    vocab, neighbors_by_slice = get_nearest_neighbors(models, keyword, timeslices, topn=topn)
    
    if not vocab:
        print(f"  ERROR: No neighbors found for '{keyword}'")
        return False
    
    # Save neighbors
    save_neighbors_table(neighbors_by_slice, keyword, output_dir)
    
    # Filter OCR errors
    filtered_vocab = filter_ocr_errors(vocab, enabled=filter_ocr)
    
    if not filtered_vocab:
        print(f"  ERROR: No words left after filtering")
        return False
    
    # Collect vectors
    X, full_vocab, n_keyword, word_decade_map = collect_vectors_for_tsne(
        models, keyword, neighbors_by_slice, timeslices
    )
    
    if X is None or len(X) < 5:
        print(f"  ERROR: Not enough vectors for t-SNE (need at least 5)")
        return False
    
    # Visualize
    visualize_trajectory(X, full_vocab, n_keyword, keyword, timeslices, 
                        output_dir, word_decade_map=word_decade_map)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Visualize semantic change trajectories using diachronic word embeddings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --word coffee --vectors_dir ./vectors --output_dir ./output
  %(prog)s --word tea --decades 1840s 1860s 1880s 1900s
  %(prog)s --word sugar --topn 30 --no-filter-ocr
  %(prog)s --all-commodities --vectors_dir ./lwm_vectors

Available example words: coffee, tea, sugar, opium, cocoa, tobacco
        """
    )
    
    # Input/output
    parser.add_argument('--vectors_dir', type=str, default='./vectors',
                        help='Directory containing Word2Vec vector files (default: ./vectors)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for results (default: ./output)')
    
    # Word selection
    parser.add_argument('--word', type=str, 
                        help='Target word to analyze')
    parser.add_argument('--words', type=str, nargs='+',
                        help='Multiple target words to analyze')
    parser.add_argument('--all-commodities', action='store_true',
                        help='Analyze all example commodity words')
    
    # Analysis parameters
    parser.add_argument('--decades', type=str, nargs='+', default=DEFAULT_DECADES,
                        help=f'Decades to compare (default: {" ".join(DEFAULT_DECADES)})')
    parser.add_argument('--topn', type=int, default=20,
                        help='Number of nearest neighbors per decade (default: 20)')
    parser.add_argument('--no-filter-ocr', action='store_true',
                        help='Disable OCR error filtering')
    
    args = parser.parse_args()
    
    # Determine target words
    if args.all_commodities:
        target_words = EXAMPLE_WORDS
    elif args.words:
        target_words = args.words
    elif args.word:
        target_words = [args.word]
    else:
        parser.error("Please specify --word, --words, or --all-commodities")
    
    print("=" * 70)
    print("SEMANTIC TRAJECTORY VISUALIZATION")
    print("=" * 70)
    print(f"\nVectors directory: {args.vectors_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Target words:      {', '.join(target_words)}")
    print(f"Decades:           {' → '.join(args.decades)}")
    print(f"Neighbors per decade: {args.topn}")
    print(f"OCR filtering:     {'disabled' if args.no_filter_ocr else 'enabled'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    models, slices_names = load_all_models(args.vectors_dir)
    
    if not models:
        print("\nERROR: No models loaded!")
        return 1
    
    # Process each word
    successful = []
    failed = []
    
    for keyword in target_words:
        success = process_word(
            models, keyword, args.decades, args.output_dir,
            topn=args.topn, filter_ocr=not args.no_filter_ocr
        )
        if success:
            successful.append(keyword)
        else:
            failed.append(keyword)
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    
    if successful:
        print(f"\nSuccessfully processed ({len(successful)}):")
        for word in successful:
            print(f"  - trajectory_{word}.png")
            print(f"  - neighbors_{word}.csv")
    
    if failed:
        print(f"\nFailed ({len(failed)}): {', '.join(failed)}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    exit(main())
