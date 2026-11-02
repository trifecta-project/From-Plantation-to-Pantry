"""
Extract Nearest Neighbors from Word2Vec Vectors
================================================
For each target word and decade, list its TOPN nearest neighbours by cosine
similarity in that decade's Word2Vec space (the pretrained, aligned vectors of
Pedrazzini & McGillivray 2022). The target word's own forms (singular and
plural) are removed from its neighbours. Also records each target word's
vocabulary rank per decade, as a proxy for frequency, and compares the
neighbour sets of the first and last decade.

Set VECTORS_DIR and OUTPUT_DIR below, then run:
    python extract_neighbors.py

Author: Jiaqi (TRIFECTA project)
"""

import os
from glob import glob
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to folder containing decade vector files
VECTORS_DIR = "..."  # change it to your path

# Output directory
OUTPUT_DIR = "results"

# Target words (canonical forms)
TARGET_WORDS = ['coffee', 'tea', 'sugar']

# Number of nearest neighbors to extract
TOPN = 1000

# All decades
DECADES = ['1800s', '1810s', '1820s', '1830s', '1840s', '1850s', 
           '1860s', '1870s', '1880s', '1890s', '1900s', '1910s']

# =============================================================================
# SURFACE FORMS OF THE TARGET WORDS
# =============================================================================
# Maps canonical form -> its forms (singular and plural). The first form found
# in a decade's vocabulary is used as the query word, i.e. the singular whenever
# it is present; all forms are removed from the neighbour lists.

TARGET_VARIANTS = {
    'coffee': [
        'coffee', 'coffees'
    ],
    'tea': [
        'tea', 'teas'
    ],
    'sugar': [
        'sugar', 'sugars'
    ],
    'opium': [
        'opium'       # Standard spelling
    ],
    'cocoa': [
        'cocoa'       # Standard spelling
    ],
    'tobacco': [
        'tobacco'       # Standard spelling
    ]
}


def find_word_in_vocab(model, canonical_word):
    """
    Find the best matching variant of a word in the model vocabulary.
    
    Args:
        model: KeyedVectors model
        canonical_word: The canonical form of the word (e.g., 'coffee')
    
    Returns:
        tuple: (found_variant, canonical_word) or (None, canonical_word) if not found
    """
    variants = TARGET_VARIANTS.get(canonical_word, [canonical_word])
    
    for variant in variants:
        if variant in model:
            return variant, canonical_word
    
    return None, canonical_word


# =============================================================================
# FREQUENCY EXTRACTION
# =============================================================================

def extract_frequencies_from_raw(vectors_dir, target_words):
    """
    Extract word frequencies by reading raw vector files.
    
    In Word2Vec text format, words are typically ordered by frequency
    (most frequent first). The rank can serve as a proxy for frequency.
    
    Updated to search for spelling variants.
    """
    print("\n" + "="*70)
    print("EXTRACTING VOCABULARY RANKS (proxy for frequency)")
    print("="*70)
    print("Note: In Word2Vec, words are typically ordered by frequency.")
    print("      Lower rank = more frequent word.\n")
    
    freq_data = []
    
    for decade in DECADES:
        filepath = os.path.join(vectors_dir, f"{decade}-vectors.txt")
        if not os.path.exists(filepath):
            continue
        
        # Read file to get word order (= frequency rank)
        word_ranks = {}
        total_words = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip().split()
            total_words = int(first_line[0])
            
            for rank, line in enumerate(f, 1):
                word = line.split()[0]
                word_ranks[word] = rank
                
                # Only read enough to find our target words + some buffer
                if rank > 100000:  # Safety limit
                    break
        
        for canonical_word in target_words:
            # Search for variants
            variants = TARGET_VARIANTS.get(canonical_word, [canonical_word])
            found_variant = None
            found_rank = None
            
            for variant in variants:
                if variant in word_ranks:
                    found_variant = variant
                    found_rank = word_ranks[variant]
                    break
            
            if found_variant:
                freq_data.append({
                    'word': canonical_word,
                    'variant_used': found_variant,
                    'decade': decade,
                    'vocab_rank': found_rank,
                    'total_vocab': total_words,
                    'rank_percentile': round(100 * found_rank / total_words, 2)
                })
                variant_info = f" (as '{found_variant}')" if found_variant != canonical_word else ""
                print(f"  {decade} - {canonical_word}{variant_info}: rank {found_rank:,} / {total_words:,} ({100*found_rank/total_words:.2f}%)")
            else:
                freq_data.append({
                    'word': canonical_word,
                    'variant_used': None,
                    'decade': decade,
                    'vocab_rank': None,
                    'total_vocab': total_words,
                    'rank_percentile': None
                })
                print(f"  {decade} - {canonical_word}: NOT IN VOCABULARY (tried: {variants})")
    
    return pd.DataFrame(freq_data)


def display_frequency_table(freq_df, target_words):
    """Display frequency ranks as a formatted table."""
    
    print("\n" + "="*70)
    print("VOCABULARY RANK TABLE (lower = more frequent)")
    print("="*70)
    
    # Pivot to show decades as columns
    pivot = freq_df.pivot(index='word', columns='decade', values='vocab_rank')
    
    # Reorder columns by decade
    ordered_cols = [d for d in DECADES if d in pivot.columns]
    pivot = pivot[ordered_cols]
    
    print(pivot.to_string())
    
    # Also show which variants were used
    print("\n" + "-"*70)
    print("VARIANTS USED:")
    print("-"*70)
    variants_pivot = freq_df.pivot(index='word', columns='decade', values='variant_used')
    variants_pivot = variants_pivot[ordered_cols]
    print(variants_pivot.to_string())
    
    return pivot


def save_frequency_data(freq_df, output_dir):
    """Save frequency data to files."""
    
    # Save full data
    freq_path = os.path.join(output_dir, "word_frequencies.csv")
    freq_df.to_csv(freq_path, index=False)
    print(f"\nSaved: {freq_path}")
    
    # Save wide format
    pivot = freq_df.pivot(index='word', columns='decade', values='vocab_rank')
    ordered_cols = [d for d in DECADES if d in pivot.columns]
    pivot = pivot[ordered_cols]
    wide_path = os.path.join(output_dir, "word_frequencies_wide.csv")
    pivot.to_csv(wide_path)
    print(f"Saved: {wide_path}")

# =============================================================================
# LOAD VECTORS
# =============================================================================

def load_all_models(vectors_dir):
    """Load all Word2Vec models from text files."""
    print("="*70)
    print("LOADING VECTORS")
    print("="*70)
    
    models = {}
    
    for decade in DECADES:
        # Try different file naming conventions
        possible_names = [
            f"{decade}-vectors.txt",
            f"{decade}.txt",
        ]
        
        filepath = None
        for name in possible_names:
            candidate = os.path.join(vectors_dir, name)
            if os.path.exists(candidate):
                filepath = candidate
                break
        
        if filepath is None:
            print(f"  {decade}: NOT FOUND")
            continue
        
        try:
            model = KeyedVectors.load_word2vec_format(filepath, binary=False)
            models[decade] = model
            print(f"  {decade}: {len(model):,} words, {model.vector_size} dimensions")
        except Exception as e:
            print(f"  {decade}: ERROR - {e}")
    
    return models

# =============================================================================
# EXTRACT NEIGHBORS
# =============================================================================

def is_variant_of_word(neighbor, canonical_word):
    """
    Check if a neighbor is a spelling variant of the target word.
    
    Args:
        neighbor: The neighbor word to check
        canonical_word: The canonical target word (e.g., 'coffee')
    
    Returns:
        True if neighbor is a variant of the target word
    """
    variants = TARGET_VARIANTS.get(canonical_word, [canonical_word])
    return neighbor.lower() in [v.lower() for v in variants]


def extract_neighbors(models, target_words, topn=1000):
    """
    Extract nearest neighbors for each word in each decade.
    
    Updated to:
    - Search for spelling variants of target words
    - EXCLUDE variants of the target word from neighbors
    
    Returns:
        DataFrame with columns: word, variant_used, decade, rank, neighbor, similarity
    """
    print("\n" + "="*70)
    print("EXTRACTING NEAREST NEIGHBORS")
    print("(Excluding spelling variants of target word from results)")
    print("="*70)
    
    all_rows = []
    
    for canonical_word in target_words:
        print(f"\n{canonical_word.upper()}:")
        variants_to_exclude = [v.lower() for v in TARGET_VARIANTS.get(canonical_word, [canonical_word])]
        
        for decade in DECADES:
            if decade not in models:
                print(f"  {decade}: model not loaded")
                continue
            
            model = models[decade]
            
            # Find the word variant in vocabulary
            found_variant, _ = find_word_in_vocab(model, canonical_word)
            
            if found_variant is None:
                variants = TARGET_VARIANTS.get(canonical_word, [canonical_word])
                print(f"  {decade}: '{canonical_word}' not in vocabulary (tried: {variants})")
                continue
            
            try:
                # Request more neighbors than needed to account for filtered variants
                neighbors_raw = model.most_similar(positive=found_variant, topn=topn)
                
                # Filter out variants of the target word
                neighbors_filtered = [
                    (neighbor, similarity) 
                    for neighbor, similarity in neighbors_raw 
                    if neighbor.lower() not in variants_to_exclude
                ]
                
                # Take only topn after filtering
                neighbors = neighbors_filtered[:topn]
                
                variant_info = f" (as '{found_variant}')" if found_variant != canonical_word else ""
                
                for rank, (neighbor, similarity) in enumerate(neighbors, 1):
                    all_rows.append({
                        'word': canonical_word,  # Always use canonical form
                        'variant_used': found_variant,
                        'decade': decade,
                        'rank': rank,
                        'neighbor': neighbor,
                        'similarity': round(similarity, 4)
                    })
                
                # Print top 5 for quick preview
                top5 = [n for n, s in neighbors[:5]]
                print(f"  {decade}{variant_info}: {', '.join(top5)}")
                
            except Exception as e:
                print(f"  {decade}: ERROR - {e}")
    
    return pd.DataFrame(all_rows)

# =============================================================================
# DISPLAY AND SAVE
# =============================================================================

def display_neighbors_table(df, word):
    """Display neighbors for a word as a formatted table."""
    
    word_df = df[df['word'] == word]
    
    if word_df.empty:
        print(f"No data for '{word}'")
        return
    
    # Pivot to show decades as columns
    pivot = word_df.pivot(index='rank', columns='decade', values='neighbor')
    
    # Reorder columns by decade
    ordered_cols = [d for d in DECADES if d in pivot.columns]
    pivot = pivot[ordered_cols]
    
    print(f"\n{'='*70}")
    print(f"NEAREST NEIGHBORS: {word.upper()}")
    print("="*70)
    
    # Show which variants were used
    variants_used = word_df.groupby('decade')['variant_used'].first().to_dict()
    variant_str = ", ".join([f"{d}: {v}" for d, v in sorted(variants_used.items()) if v != word])
    if variant_str:
        print(f"(Variants used: {variant_str})")
    
    print(pivot.to_string())


def save_results(df, output_dir):
    """Save results to CSV files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete results
    full_path = os.path.join(output_dir, "all_neighbors.csv")
    df.to_csv(full_path, index=False)
    print(f"\nSaved: {full_path}")
    
    # Save per-word files
    for word in df['word'].unique():
        word_df = df[df['word'] == word]
        word_path = os.path.join(output_dir, f"neighbors_{word}.csv")
        word_df.to_csv(word_path, index=False)
        print(f"Saved: {word_path}")
    
    # Save wide format (decades as columns)
    for word in df['word'].unique():
        word_df = df[df['word'] == word]
        pivot = word_df.pivot(index='rank', columns='decade', values='neighbor')
        ordered_cols = [d for d in DECADES if d in pivot.columns]
        pivot = pivot[ordered_cols]
        wide_path = os.path.join(output_dir, f"neighbors_{word}_wide.csv")
        pivot.to_csv(wide_path)
        print(f"Saved: {wide_path}")


def save_neighbor_changes(df, output_dir):
    """
    Save stable/new/lost neighbors analysis to files.
    Compares first and last decade for each word.
    """
    
    print("\n" + "="*70)
    print("SAVING NEIGHBOR CHANGES ANALYSIS")
    print("="*70)
    
    all_changes = []
    
    for word in df['word'].unique():
        word_df = df[df['word'] == word]
        n_decades = word_df['decade'].nunique()
        
        if n_decades < 2:
            continue
        
        # Get first and last decade data
        first_decade_name = word_df['decade'].min()
        last_decade_name = word_df['decade'].max()
        
        first_decade = word_df[word_df['decade'] == first_decade_name]
        last_decade = word_df[word_df['decade'] == last_decade_name]
        
        first_neighbors = set(first_decade['neighbor'].tolist())
        last_neighbors = set(last_decade['neighbor'].tolist())
        
        # Calculate sets
        stable = first_neighbors & last_neighbors
        new = last_neighbors - first_neighbors
        lost = first_neighbors - last_neighbors
        
        # Create DataFrame for this word
        changes_data = {
            'word': word,
            'first_decade': first_decade_name,
            'last_decade': last_decade_name,
            'stable_neighbors': ', '.join(sorted(stable)),
            'new_neighbors': ', '.join(sorted(new)),
            'lost_neighbors': ', '.join(sorted(lost)),
            'n_stable': len(stable),
            'n_new': len(new),
            'n_lost': len(lost)
        }
        all_changes.append(changes_data)
        
        # Save individual word file with detailed breakdown
        word_changes = []
        for neighbor in sorted(stable):
            word_changes.append({'neighbor': neighbor, 'status': 'stable', 
                                'in_first': True, 'in_last': True})
        for neighbor in sorted(new):
            word_changes.append({'neighbor': neighbor, 'status': 'new', 
                                'in_first': False, 'in_last': True})
        for neighbor in sorted(lost):
            word_changes.append({'neighbor': neighbor, 'status': 'lost', 
                                'in_first': True, 'in_last': False})
        
        word_changes_df = pd.DataFrame(word_changes)
        word_path = os.path.join(output_dir, f"neighbor_changes_{word}.csv")
        word_changes_df.to_csv(word_path, index=False)
        print(f"Saved: {word_path}")
    
    # Save summary of all words
    if all_changes:
        summary_df = pd.DataFrame(all_changes)
        summary_path = os.path.join(output_dir, "neighbor_changes_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
    
    return all_changes

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("EXTRACT NEAREST NEIGHBORS FROM WORD2VEC")
    print("With spelling variation handling")
    print("="*70)
    print(f"\nVectors directory: {VECTORS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target words: {TARGET_WORDS}")
    print(f"Top N neighbors: {TOPN}")
    
    # Print spelling variants
    print("\nSpelling variants to search:")
    for word, variants in TARGET_VARIANTS.items():
        print(f"  {word}: {variants}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extract word frequencies first
    freq_df = extract_frequencies_from_raw(VECTORS_DIR, TARGET_WORDS)
    
    if not freq_df.empty:
        display_frequency_table(freq_df, TARGET_WORDS)
        save_frequency_data(freq_df, OUTPUT_DIR)
    
    # Load models
    models = load_all_models(VECTORS_DIR)
    
    if not models:
        print("\nERROR: No models loaded!")
        return
    
    # Extract neighbors
    df = extract_neighbors(models, TARGET_WORDS, topn=TOPN)
    
    if df.empty:
        print("\nERROR: No neighbors extracted!")
        return
    
    # Display tables
    for word in TARGET_WORDS:
        display_neighbors_table(df, word)
    
    # Save results
    save_results(df, OUTPUT_DIR)
    
    # Save neighbor changes analysis
    all_changes = save_neighbor_changes(df, OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for word in TARGET_WORDS:
        word_df = df[df['word'] == word]
        n_decades = word_df['decade'].nunique()
        print(f"\n{word.upper()}:")
        print(f"  Decades with data: {n_decades}")
        
        # Show which variants were used
        if not word_df.empty:
            variants_used = word_df.groupby('decade')['variant_used'].first().to_dict()
            unique_variants = set(variants_used.values())
            if len(unique_variants) > 1 or (len(unique_variants) == 1 and list(unique_variants)[0] != word):
                print(f"  Variants used: {variants_used}")
        
        # Show frequency info
        if not freq_df.empty:
            word_freq = freq_df[freq_df['word'] == word]
            if not word_freq.empty:
                first_rank = word_freq[word_freq['decade'] == word_freq['decade'].min()]['vocab_rank'].values
                last_rank = word_freq[word_freq['decade'] == word_freq['decade'].max()]['vocab_rank'].values
                if len(first_rank) > 0 and len(last_rank) > 0 and first_rank[0] and last_rank[0]:
                    print(f"  Vocab rank: {int(first_rank[0]):,} ({word_freq['decade'].min()}) → {int(last_rank[0]):,} ({word_freq['decade'].max()})")
        
        # Show how neighbors change
        if n_decades >= 2:
            first_decade = word_df[word_df['decade'] == word_df['decade'].min()]
            last_decade = word_df[word_df['decade'] == word_df['decade'].max()]
            
            first_neighbors = set(first_decade['neighbor'].tolist())
            last_neighbors = set(last_decade['neighbor'].tolist())
            
            stable = first_neighbors & last_neighbors
            new = last_neighbors - first_neighbors
            lost = first_neighbors - last_neighbors
            
            print(f"  Stable neighbors (in both {word_df['decade'].min()} and {word_df['decade'].max()}): {len(stable)}")
            print(f"    {list(stable)[:5]}{'...' if len(stable) > 5 else ''}")
            print(f"  New neighbors (only in {word_df['decade'].max()}): {len(new)}")
            print(f"    {list(new)[:5]}{'...' if len(new) > 5 else ''}")
            print(f"  Lost neighbors (only in {word_df['decade'].min()}): {len(lost)}")
            print(f"    {list(lost)[:5]}{'...' if len(lost) > 5 else ''}")
    
    print(f"\n\nAll results saved to: {OUTPUT_DIR}")
    print("\nOutput files:")
    print("  - word_frequencies.csv (vocabulary ranks, includes variant_used)")
    print("  - word_frequencies_wide.csv (ranks by decade)")
    print("  - all_neighbors.csv (all neighbors, includes variant_used)")
    print("  - neighbor_changes_summary.csv (all words summary)")
    for word in TARGET_WORDS:
        print(f"  - neighbors_{word}.csv, neighbors_{word}_wide.csv")
        print(f"  - neighbor_changes_{word}.csv")


if __name__ == "__main__":
    main()
