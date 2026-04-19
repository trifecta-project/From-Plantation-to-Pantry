# Packages - Extract embeddings ONLY for target commodity words
# Usage:
#   python embeddings_extraction.py --model finetuned
#   python embeddings_extraction.py --model pretrained
import os 
import argparse
import h5py
import torch
from gen_macberth import MacBERTh
from tqdm import tqdm
import time
import gc

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract commodity word embeddings using MacBERTh')
parser.add_argument('--model', type=str, choices=['pretrained', 'finetuned'], default='finetuned',
                    help='Which model to use: pretrained (original MacBERTh) or finetuned (on HMD corpus)')
args = parser.parse_args()

# Define paths
dir_in = "..." #change it to your path
dir_corpus = os.path.join(dir_in, "en_decade_corpus")

# Model selection based on argument
if args.model == 'finetuned':
    macberth_model = os.path.join(dir_in, "output", "fine_tuned_macberth")
    dir_out = os.path.join(dir_in, "output", "embeddings_macberth_finetuned")
    print(f"Using FINE-TUNED MacBERTh model")
else:
    macberth_model = "emanjavacas/MacBERTh"
    dir_out = os.path.join(dir_in, "output", "embeddings_macberth_pretrained")
    print(f"Using PRE-TRAINED MacBERTh model")

# Target words for commodity analysis
TARGET_WORDS = {'coffee', 'tea', 'sugar', 'opium', 'cocoa', 'tobacco'}

# Ensure output directory exists
os.makedirs(dir_out, exist_ok=True)

# Define decades
decades = [1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910]

# Punctuation to filter
punctuation = {'.', ',', '...', ';', ':', '?', '(', ')', '-', '!', '[', ']', '"', "'", '""', '\n', ''}


def find_target_sentences(file_path, target_words, max_sentences=None):
    """
    Find sentences containing target words.
    
    Args:
        file_path: Path to corpus file
        target_words: Set of target words to search for
        max_sentences: Optional limit on sentences per word
        
    Returns:
        Dict mapping target word to list of (sentence_tokens, word_position) tuples
    """
    results = {word: [] for word in target_words}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            tokens = [token.lower() for token in line.split(" ") if token not in punctuation]
            
            # Check for each target word
            for word in target_words:
                if word in tokens:
                    # Find position(s) of word in sentence
                    positions = [i for i, t in enumerate(tokens) if t == word]
                    for pos in positions:
                        if max_sentences is None or len(results[word]) < max_sentences:
                            results[word].append((tokens, pos))
    
    return results


def extract_target_embeddings(macberth, sentences_with_positions):
    """
    Extract embeddings for target words from their sentences.
    
    Args:
        macberth: MacBERTh model instance
        sentences_with_positions: List of (tokens, position) tuples
        
    Returns:
        List of dicts with 'embedding', 'snippet', 'position'
    """
    if not sentences_with_positions:
        return []
    
    # Flatten sentences to strings (truncation handled in gen_macberth.convert_to_toks)
    sentences = [' '.join(tokens) for tokens, pos in sentences_with_positions]
    positions = [pos for tokens, pos in sentences_with_positions]
    
    # Get all embeddings
    bert_sents = macberth.get_berts(sentences)
    
    # Extract target word embeddings
    results = []
    for idx, (emb_list, pos) in enumerate(zip(bert_sents, positions)):
        # emb_list contains (token, embedding) tuples including [CLS] and [SEP]
        # Position in emb_list is pos + 1 because of [CLS] token at index 0
        target_pos = pos + 1
        
        # Check bounds (sentence may have been truncated)
        if target_pos < len(emb_list) - 1:  # -1 to exclude [SEP]
            token, embedding = emb_list[target_pos]
            results.append({
                'embedding': embedding,
                'snippet': sentences[idx][:200],  # Truncate snippet for storage
                'position': pos
            })
        # If target word was truncated, skip this occurrence
    
    return results


# Initialize MacBERTh once
print(f"\nLoading MacBERTh model from: {macberth_model}")
macberth = MacBERTh(model_name=macberth_model)
print("Model loaded.\n")

# Process each decade
for decade in decades:
    file_path = os.path.join(dir_corpus, f"{decade}.txt")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping...")
        continue
    
    output_path = os.path.join(dir_out, f"commodity_embeddings_{decade}.h5")
    
    # Check which words need to be processed
    if os.path.exists(output_path):
        with h5py.File(output_path, 'r') as f:
            existing_words = set(f.keys())
        words_to_process = TARGET_WORDS - existing_words
        if not words_to_process:
            print(f"Skipping {decade} - all target words already exist")
            continue
        print(f"\n{'='*60}")
        print(f"Processing decade {decade}...")
        print(f"Existing words: {existing_words}")
        print(f"New words to process: {words_to_process}")
        print(f"{'='*60}")
        file_mode = 'a'  # Append mode
    else:
        words_to_process = TARGET_WORDS
        print(f"\n{'='*60}")
        print(f"Processing decade {decade} (new file)...")
        print(f"Words to process: {words_to_process}")
        print(f"{'='*60}")
        file_mode = 'w'  # Write mode
    
    start_time = time.time()
    
    # Step 1: Find all sentences with target words (only for words we need)
    print("Finding sentences with target words...")
    target_sentences = find_target_sentences(file_path, words_to_process)
    
    for word in words_to_process:
        print(f"  {word}: {len(target_sentences[word])} occurrences")
    
    # Step 2: Extract embeddings for each target word
    print("\nExtracting embeddings...")
    
    with h5py.File(output_path, file_mode) as f:
        # Store/update model info as metadata
        f.attrs['model'] = args.model
        f.attrs['model_path'] = macberth_model
        f.attrs['decade'] = decade
        
        for word in words_to_process:
            sentences_with_pos = target_sentences[word]
            
            if not sentences_with_pos:
                print(f"  {word}: no occurrences, skipping")
                continue
            
            print(f"  Processing {word} ({len(sentences_with_pos)} sentences)...")
            
            # Process in small batches to avoid GPU OOM
            batch_size = 50
            all_results = []
            
            for i in tqdm(range(0, len(sentences_with_pos), batch_size), desc=f"  {word}"):
                batch = sentences_with_pos[i:i+batch_size]
                batch_results = extract_target_embeddings(macberth, batch)
                all_results.extend(batch_results)
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Save to HDF5
            word_grp = f.create_group(word)
            word_grp.attrs['count'] = len(all_results)
            
            for idx, item in enumerate(all_results):
                usage_grp = word_grp.create_group(f"usage_{idx}")
                usage_grp.create_dataset("embedding", data=item['embedding'], compression="gzip")
                usage_grp.attrs['snippet'] = item['snippet']
                usage_grp.attrs['position'] = item['position']
            
            print(f"  {word}: saved {len(all_results)} embeddings")
    
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\nDecade {decade} completed in {int(minutes)}m {seconds:.2f}s")
    
    # Cleanup
    del target_sentences
    gc.collect()

print("\n" + "="*60)
print("All decades processed!")
print(f"Output saved to: {dir_out}")
print("="*60)
