"""
embeddings_extraction.py - extract contextualised embeddings of the target
commodity words from the decade sub-corpora.

For every occurrence of a target word, the script embeds a window of
+/-CONTEXT_WINDOW words centred on it with MacBERTh (via gen_macberth.py) and
stores the target word's embedding, the window text and the target's position
in one HDF5 file per decade. It also appends the number of occurrences found
per word and decade to frequency_summary.csv.

Usage (the settings used in the paper):
    python embeddings_extraction.py --model scaling_best --layer-mode last4 \
        --variants spelling --context-window 50
    python embeddings_extraction.py --model pretrained --layer-mode last4 \
        --variants spelling --context-window 50

Words are matched as space-delimited tokens after lower-casing; a token with
punctuation attached (e.g. "coffee,") does not match.
"""

import os
import argparse
import csv
import h5py
import torch
from gen_macberth import MacBERTh   # generic BERT embedding extractor (loads any HF BERT by name)
from tqdm import tqdm
import time
import gc

dir_in = "..."  # change it to your path
dir_corpus = os.path.join(dir_in, "en_decade_corpus")

# ---------------------------------------------------------------------------
# Models available for extraction. 'path' is passed to the MacBERTh wrapper
# (a local dir or a HuggingFace id); 'dir' is the output folder name.
# ---------------------------------------------------------------------------
MODELS = {
    'pretrained':     {'path': "emanjavacas/MacBERTh",
                       'dir':  "embeddings_macberth_pretrained"},
    'scaling_best':   {'path': os.path.join(dir_in, "output", "scaling_experiment", "config_A", "epoch_3"),
                       'dir':  "embeddings_macberth_scaling_best_windowed"},
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract commodity word embeddings from a historical BERT')
parser.add_argument('--model', type=str,
                    choices=list(MODELS.keys()),
                    default='scaling_best',
                    help='Which model to use')
parser.add_argument('--variants', type=str,
                    choices=['none', 'spelling'],
                    default='spelling',
                    help="Which surface forms to match. "
                         "'none' = canonical singular only (baseline); "
                         "'spelling' = singular + plural forms (see TARGET_FORMS).")
parser.add_argument('--context-window', type=int, default=50,
                    help="Words of context kept on EACH side of the target word. "
                         "Each occurrence is embedded from this centred window "
                         "rather than the whole document, so the target is always "
                         "within the model's length budget and is never dropped. "
                         "50 keeps every window well under the model's 510-subword limit.")
parser.add_argument(
    '--layer-mode',
    type=str,
    choices=['last','first4','last4','all'],
    default='last',
    help="Which hidden layers to average (see gen_macberth.py). The paper uses last4."
)
args = parser.parse_args()

model_path = MODELS[args.model]['path']
model_tag = args.model
CONTEXT_WINDOW = args.context_window
LAYER_MODE = args.layer_mode
# Output dir encodes model, variant mode AND window radius so each condition is
# isolated and the (windowed) embeddings never mix with older whole-document runs.
dir_out = os.path.join(
    dir_in,
    "output",
    f"{MODELS[args.model]['dir']}_{LAYER_MODE}_{args.variants}_w{CONTEXT_WINDOW}"
)

print(f"Model:          {model_tag}  ({model_path})")
print(f"Variant mode:   {args.variants}")
print(f"Layer mode:     {LAYER_MODE}")
print(f"Context window: +/-{CONTEXT_WINDOW} words around the target")
print(f"Output dir:     {dir_out}")

# ---------------------------------------------------------------------------
# Target forms, split so the frequency contribution of the additional forms can
# be attributed against the canonical-only baseline:
#   canonical : the standard modern singular (always searched)
#   spelling  : additional surface forms, added with --variants spelling;
#               for these words, the plural forms
# ---------------------------------------------------------------------------
TARGET_FORMS = {
    'coffee':  {'canonical': 'coffee',  'spelling': ['coffees']},
    'tea':     {'canonical': 'tea',     'spelling': ['teas']},
    'sugar':   {'canonical': 'sugar',   'spelling': ['sugars']},
}


def build_variants(mode):
    """
    Build the {canonical -> [surface forms]} map for a given variant mode.
    The canonical form is always included; the additional forms are added for
    mode='spelling'. Order is preserved and duplicates removed.
    """
    result = {}
    for word, forms in TARGET_FORMS.items():
        surface = [forms['canonical']]
        if mode == 'spelling':
            surface += forms['spelling']
        # Deduplicate while preserving order
        seen = set()
        result[word] = [s for s in surface if not (s in seen or seen.add(s))]
    return result


# Resolve the surface forms for this run
TARGET_VARIANTS = build_variants(args.variants)
TARGET_WORDS = set(TARGET_VARIANTS.keys())  # canonical words (HDF5 keys, resume logic)

print("Surface forms searched this run:")
for word in TARGET_WORDS:
    print(f"  {word}: {TARGET_VARIANTS[word]}")

# Ensure output directory exists
os.makedirs(dir_out, exist_ok=True)

# Per-run frequency log for cross-mode / cross-model comparison
summary_path = os.path.join(dir_out, "frequency_summary.csv")
summary_needs_header = not os.path.exists(summary_path)

# Define decades
decades = [1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910]

# Punctuation to filter
punctuation = {'.', ',', '...', ';', ':', '?', '(', ')', '-', '!', '[', ']', '"', "'", '""', '\n', ''}


def find_target_sentences(file_path, variants_map, context_window, max_sentences=None):
    """
    Find occurrences of target words (or any surface form) and return a CENTRED
    context window around each one.

    Args:
        file_path: Path to corpus file
        variants_map: Dict mapping canonical word -> list of surface forms
        context_window: words of context kept on each side of the target
        max_sentences: Optional limit on occurrences per canonical word

    Returns:
        Dict mapping canonical word to list of (window_tokens, position_in_window)
        tuples. `window_tokens` is the +/-context_window slice of the document
        centred on the occurrence; `position_in_window` is the target's index
        within that slice. Each occurrence gets its own window, so multiple
        occurrences in one document are embedded independently in their own
        local contexts.
    """
    results = {word: [] for word in variants_map}

    # Build reverse lookup: surface form (lowercased) -> canonical word
    variant_to_canonical = {}
    for canonical, variants in variants_map.items():
        for v in variants:
            variant_to_canonical[v.lower()] = canonical

    all_variants = set(variant_to_canonical.keys())

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            tokens = [token.lower() for token in line.split(" ") if token not in punctuation]

            # Which surface forms appear in this document line?
            present = all_variants.intersection(tokens)
            for variant in present:
                canonical = variant_to_canonical[variant]
                # Find position(s) of this form in the document
                positions = [i for i, t in enumerate(tokens) if t == variant]
                for pos in positions:
                    if max_sentences is not None and len(results[canonical]) >= max_sentences:
                        continue
                    # Centre a +/-context_window slice on this occurrence so the
                    # target is always inside the model's length budget.
                    lo = max(0, pos - context_window)
                    hi = min(len(tokens), pos + context_window + 1)
                    window = tokens[lo:hi]
                    results[canonical].append((window, pos - lo))  # local target index

    return results


def extract_target_embeddings(embedder, sentences_with_positions):
    """
    Extract embeddings for target words from their (windowed) contexts.

    Args:
        embedder: BERT extractor instance (MacBERTh wrapper, any HF BERT)
        sentences_with_positions: List of (window_tokens, position_in_window) tuples

    Returns:
        List of dicts with 'embedding', 'snippet', 'position'. `snippet` is the
        full windowed context (space-joined, lowercased); `position` is the
        target's index within that window.
    """
    if not sentences_with_positions:
        return []

    # Each input is already a short centred window, so no document-length
    # truncation occurs downstream (convert_to_toks / get_batches caps do not bind).
    sentences = [' '.join(tokens) for tokens, pos in sentences_with_positions]
    positions = [pos for tokens, pos in sentences_with_positions]

    # Get all embeddings
    bert_sents = embedder.get_berts(sentences)

    # Extract target word embeddings
    results = []
    for idx, (emb_list, pos) in enumerate(zip(bert_sents, positions)):
        # emb_list contains (token, embedding) tuples including [CLS] and [SEP]
        # Position in emb_list is pos + 1 because of [CLS] token at index 0
        target_pos = pos + 1

        # Bounds check (should essentially always pass for windowed inputs;
        # only fails for a degenerate window, which we then skip).
        if target_pos < len(emb_list) - 1:  # -1 to exclude [SEP]
            token, embedding = emb_list[target_pos]
            results.append({
                'embedding': embedding,
                'snippet': sentences[idx],   # full centred window (target always visible)
                'position': pos              # index of the target WITHIN the window
            })

    return results


# Initialize the model once (same wrapper for every model -> identical extraction)
print(f"\nLoading model from: {model_path}")
embedder = MacBERTh(
    model_name=model_path,
    layer_mode=args.layer_mode
)
print("Model loaded.")
# Report tokenizer casing if the wrapper exposes it (the pipeline lowercases
# input, so a CASED model would see lowercased text).
try:
    tk = getattr(embedder, 'tokenizer', None)
    if tk is not None:
        print(f"Tokenizer do_lower_case: {getattr(tk, 'do_lower_case', 'n/a')}")
except Exception:
    pass
print()

# Track totals across decades (for words processed this run)
run_totals = {word: 0 for word in TARGET_WORDS}

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

    # Step 1: Find all occurrences with target forms (only for words we need),
    # each returned as a centred +/-CONTEXT_WINDOW context window.
    print(f"Finding occurrences with target forms (mode={args.variants}, "
          f"window=+/-{CONTEXT_WINDOW})...")
    variants_to_process = {word: TARGET_VARIANTS[word] for word in words_to_process}
    target_sentences = find_target_sentences(file_path, variants_to_process,
                                             context_window=CONTEXT_WINDOW)

    for word in words_to_process:
        print(f"  {word} {TARGET_VARIANTS[word]}: {len(target_sentences[word])} occurrences")

    # Step 2: Extract embeddings for each target word
    print("\nExtracting embeddings...")

    with h5py.File(output_path, file_mode) as f:
        # Store/update model info as metadata
        f.attrs['model'] = model_tag
        f.attrs['model_path'] = model_path
        f.attrs['decade'] = decade
        f.attrs['variant_mode'] = args.variants
        f.attrs['context_window'] = CONTEXT_WINDOW
        f.attrs['layer_mode'] = LAYER_MODE

        for word in words_to_process:
            sentences_with_pos = target_sentences[word]

            if not sentences_with_pos:
                print(f"  {word}: no occurrences, skipping")
                continue

            print(f"  Processing {word} ({len(sentences_with_pos)} occurrences)...")

            # Process in small batches to avoid GPU OOM
            batch_size = 50
            all_results = []

            for i in tqdm(range(0, len(sentences_with_pos), batch_size), desc=f"  {word}"):
                batch = sentences_with_pos[i:i+batch_size]
                batch_results = extract_target_embeddings(embedder, batch)
                all_results.extend(batch_results)

                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Save to HDF5 (keyed by canonical word)
            word_grp = f.create_group(word)
            word_grp.attrs['count'] = len(all_results)
            word_grp.attrs['variants'] = ','.join(TARGET_VARIANTS[word])
            word_grp.attrs['variant_mode'] = args.variants
            word_grp.attrs['context_window'] = CONTEXT_WINDOW
            word_grp.attrs['layer_mode'] = LAYER_MODE

            for idx, item in enumerate(all_results):
                usage_grp = word_grp.create_group(f"usage_{idx}")
                usage_grp.create_dataset("embedding", data=item['embedding'], compression="gzip")
                usage_grp.attrs['snippet'] = item['snippet']
                usage_grp.attrs['position'] = item['position']

            run_totals[word] += len(all_results)
            print(f"  {word}: saved {len(all_results)} embeddings")

    # Log per-decade frequency for cross-mode / cross-model comparison
    with open(summary_path, 'a', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        if summary_needs_header:
            writer.writerow(['model', 'variant_mode', 'context_window',
                             'word', 'decade', 'forms', 'count'])
            summary_needs_header = False
        for word in words_to_process:
            writer.writerow([
                model_tag, args.variants, CONTEXT_WINDOW, word, decade,
                '|'.join(variants_to_process[word]),
                len(target_sentences[word]),
            ])

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\nDecade {decade} completed in {int(minutes)}m {seconds:.2f}s")

    # Cleanup
    del target_sentences
    gc.collect()

print("\n" + "="*60)
print("All decades processed!")
print(f"Model:          {model_tag}")
print(f"Variant mode:   {args.variants}")
print(f"Context window: +/-{CONTEXT_WINDOW} words")
print(f"Output saved to: {dir_out}")
print(f"Frequency log:  {summary_path}")
print("-"*60)
print("Run totals (occurrences saved this run, words processed this run):")
for word in sorted(TARGET_WORDS):
    print(f"  {word:<8} {run_totals[word]}")
print("="*60)
