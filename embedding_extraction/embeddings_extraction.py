"""
embeddings_extraction.py - extract contextualised embeddings of target words
from a historical BERT, one centred context window per occurrence.

For every occurrence of a target word in each period sub-corpus (one file per
year or per decade), the script embeds a window of +/-context_window tokens
centred on it with MacBERTh (via gen_macberth.py), checks that the vector taken
is the target word's, and stores the vector, the window and the target's
position in one HDF5 file per period. It also writes frequency_summary.csv
with the occurrences found and embeddings saved per word and period.

Tokenisation: each line is lower-cased and split into runs of letters/digits,
optionally joined by internal hyphens or apostrophes. Punctuation attached to a
word is therefore removed ("tea," -> tea), while hyphenated compounds stay
whole and so do not match ("tea-table", "cocoa-nut").

Target verification: the vector taken at the target's expected index is
checked against the expected surface form; if it does not match, the nearest
matching token in the window is used ('realigned'); if none matches, the
occurrence is skipped ('missing'). Before extracting, an alignment probe on a
sample of occurrences stops the run if fewer than --min-align-rate verify.

Usage (yearly embeddings with the domain-adapted model):
    python embeddings_extraction.py --corpus-dir ../data/en_year_corpus \
        --model scaling_best --layer-mode last4 --forms plural \
        --words coffee tea sugar

    # control words, through the same script and settings
    python embeddings_extraction.py --corpus-dir ../data/en_year_corpus \
        --model scaling_best --layer-mode last4 --forms canonical \
        --words presence child difference

    # check alignment on one dense year first, writing nothing
    python embeddings_extraction.py --corpus-dir ../data/en_year_corpus \
        --model scaling_best --layer-mode last4 --probe-only --periods 1880
"""

import os
import re
import sys
import argparse
import csv
import random
import h5py
import torch
from gen_macberth import MacBERTh   # generic BERT embedding extractor (loads any HF BERT by name)
from tqdm import tqdm
import time
import gc

dir_in = "..."  # change it to your path
DEFAULT_CORPUS_SUBDIR = "en_decade_corpus"

# Bump this whenever the tokenisation rule changes, so corrected runs land in
# their own directory instead of being skipped by the resume logic.
TOKENISATION_TAG = "striptok"

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
parser.add_argument('--forms', type=str,
                    choices=['canonical', 'plural'],
                    default='plural',
                    help="Which surface forms to match. "
                         "'canonical' = the singular only (e.g. tea); "
                         "'plural' = singular + plural (e.g. tea, teas). "
                         "No historical or OCR spelling variants are searched "
                         "in either mode.")
parser.add_argument('--variants', type=str,
                    choices=['none', 'spelling'],
                    default=None,
                    help=argparse.SUPPRESS)   # deprecated alias for --forms
parser.add_argument('--context-window', type=int, default=50,
                    help="Words of context kept on EACH side of the target word. "
                         "Each occurrence is embedded from this centred window "
                         "rather than the whole document, so the target is always "
                         "within the model's length budget and is never dropped. "
                         "50 keeps every window well under the model's 510-subword limit.")
parser.add_argument('--layer-mode', type=str,
                    choices=['last', 'first4', 'last4', 'all'],
                    default='last',
                    help="Which hidden layers to average (see gen_macberth.py). "
                         "The paper uses last4.")
parser.add_argument('--batch-size', type=int, default=50,
                    help='Occurrences per forward pass (default 50).')
parser.add_argument('--decades', type=int, nargs='+', default=None,
                    help='Decades to process (default: all). Useful for a short '
                         'validation run before committing to the full corpus.')
parser.add_argument('--probe-only', action='store_true',
                    help='Tokenise and extract a small sample, report the alignment '
                         'counters, write nothing, and exit. Run this first.')
parser.add_argument('--probe-size', type=int, default=200,
                    help='Occurrences per word used by --probe-only and by the '
                         'early-abort check (default 200).')
parser.add_argument('--min-align-rate', type=float, default=0.90,
                    help='Abort if fewer than this fraction of the first '
                         '--probe-size occurrences verify against the target '
                         '(default 0.90). Set to 0 to disable.')
parser.add_argument('--corpus-dir', type=str, default=None,
                    help='Directory of period sub-corpora. Default: '
                         'en_decade_corpus under the project root. Files may be '
                         'named <period>.txt or en_<period>.txt, so both '
                         '"1880.txt" and "en_1880.txt" are found.')
parser.add_argument('--periods', type=int, nargs='+', default=None,
                    help='Periods to process. Default: every period found in '
                         '--corpus-dir. Works for years as well as decades; '
                         '--decades is kept as an alias.')
parser.add_argument('--granularity', type=str, choices=['decade', 'year', 'auto'],
                    default='auto',
                    help='Recorded in the output as metadata. "auto" calls it a '
                         'year when more than 20 periods are present.')
parser.add_argument('--words', type=str, nargs='+', default=None,
                    help='Which terms to extract, from TARGET_FORMS. Default: '
                         'the six commodity terms. Control terms go through this '
                         'same script, and therefore the same model, window, '
                         'tokenisation and verification, so that any difference '
                         'in the measures cannot come from the pipeline.')
parser.add_argument('--out-dir', type=str, default=None,
                    help='Output directory. Default: output/<model>_<layer>_'
                         '<forms>_w<window>_<tokenisation>. Give an explicit path '
                         'to keep a set of terms separate, e.g. output/pepper.')
parser.add_argument('--max-per-word-decade', type=int, default=None,
                    help='Cap the occurrences extracted per word per decade. The '
                         'cap is applied by random sampling AFTER all occurrences '
                         'have been located, not by taking the first N, so the '
                         'sample is unbiased with respect to position in the '
                         'corpus file. Useful for high-frequency control terms.')
parser.add_argument('--sample-seed', type=int, default=20261006,
                    help='Seed for --max-per-word-decade sampling.')
args = parser.parse_args()

# Deprecated --variants -> --forms
if args.variants is not None:
    legacy = {'none': 'canonical', 'spelling': 'plural'}
    args.forms = legacy[args.variants]
    print(f"NOTE: --variants {args.variants} is deprecated; using --forms {args.forms}.\n"
          f"      ('spelling' never searched spelling variants, only plurals.)")

model_path = MODELS[args.model]['path']
model_tag = args.model
CONTEXT_WINDOW = args.context_window
LAYER_MODE = args.layer_mode
FORMS_MODE = args.forms
BATCH_SIZE = args.batch_size

# Output dir encodes model, forms mode, window radius AND tokenisation, so each
# condition is isolated and a corrected run never collides with an older one.
if args.out_dir:
    dir_out = os.path.expanduser(args.out_dir)
    if not os.path.isabs(dir_out):
        dir_out = os.path.join(dir_in, "output", dir_out)
else:
    dir_out = os.path.join(
        dir_in,
        "output",
        f"{MODELS[args.model]['dir']}_{LAYER_MODE}_{FORMS_MODE}_w{CONTEXT_WINDOW}_{TOKENISATION_TAG}"
    )

print(f"Model:          {model_tag}  ({model_path})")
print(f"Forms mode:     {FORMS_MODE}")
print(f"Layer mode:     {LAYER_MODE}")
print(f"Context window: +/-{CONTEXT_WINDOW} words around the target")
print(f"Tokenisation:   {TOKENISATION_TAG}")
if args.max_per_word_decade:
    print(f"Cap:            {args.max_per_word_decade:,} occurrences per word-decade "
          f"(random sample, seed {args.sample_seed})")
print(f"Output dir:     {dir_out}")

# ---------------------------------------------------------------------------
# Target forms.
#   canonical : the standard modern singular (always searched)
#   plural    : the plural form, added for --forms plural
# These are NOT spelling variants. No historical or OCR forms are searched.
# ---------------------------------------------------------------------------
TARGET_FORMS = {
    # --- commodity terms under study ---------------------------------------
    'coffee':  {'canonical': 'coffee',  'plural': ['coffees']},
    'tea':     {'canonical': 'tea',     'plural': ['teas']},
    'sugar':   {'canonical': 'sugar',   'plural': ['sugars']},
    'opium':   {'canonical': 'opium',   'plural': []},
    'cocoa':   {'canonical': 'cocoa',   'plural': ['cocoas']},
    'tobacco': {'canonical': 'tobacco', 'plural': ['tobaccos']},

    # --- control terms ------------------------------------------------------
    # Frequency-matched controls for the commodity terms, matched on the
    # singular only.
    'presence':   {'canonical': 'presence',   'plural': []},
    'difference': {'canonical': 'difference', 'plural': []},
    'marble':     {'canonical': 'marble',     'plural': []},
    'child':      {'canonical': 'child',      'plural': []},

    # Several controls are matched on the SINGULAR ONLY, because their plural
    # carries a different sense and importing a second sense is precisely what
    # a stability control must avoid: "peppers" are capsicums, "salts" are
    # smelling or Epsom salts. For the commodity terms above the plural is the
    # same sense ("teas" = kinds of tea), so it is included there.
    'pepper':  {'canonical': 'pepper',  'plural': []},
    'cheese':  {'canonical': 'cheese',  'plural': ['cheeses']},
    'salt':    {'canonical': 'salt',    'plural': []},
    'flour':   {'canonical': 'flour',   'plural': []},
    'rice':    {'canonical': 'rice',    'plural': []},
    'butter':  {'canonical': 'butter',  'plural': []},
    'bread':   {'canonical': 'bread',   'plural': []},
    'machine': {'canonical': 'machine', 'plural': ['machines']},
    'engine':  {'canonical': 'engine',  'plural': ['engines']},
    'water':   {'canonical': 'water',   'plural': []},
    'stone':   {'canonical': 'stone',   'plural': []},
}

# Extracted when --words is not given: the commodity terms under study.
DEFAULT_WORDS = ['coffee', 'tea', 'sugar', 'opium', 'cocoa', 'tobacco']


def build_forms(mode, words):
    """
    Build the {canonical -> [surface forms]} map for a given forms mode.
    The canonical form is always included; the plural is added for
    mode='plural'. Order is preserved and duplicates removed.
    """
    result = {}
    for word in words:
        forms = TARGET_FORMS[word]
        surface = [forms['canonical']]
        if mode == 'plural':
            surface += forms['plural']
        seen = set()
        result[word] = [s for s in surface if not (s in seen or seen.add(s))]
    return result


if args.words is None:
    args.words = list(DEFAULT_WORDS)

unknown = [w for w in args.words if w not in TARGET_FORMS]
if unknown:
    sys.exit(f"Unknown target word(s): {unknown}\n"
             f"Add them to TARGET_FORMS with their surface forms first. "
             f"Known: {sorted(TARGET_FORMS)}")

TARGET_VARIANTS = build_forms(FORMS_MODE, args.words)
TARGET_WORDS = set(TARGET_VARIANTS.keys())   # canonical words (HDF5 keys, resume logic)

print("Surface forms searched this run:")
for word in sorted(TARGET_WORDS):
    print(f"  {word}: {TARGET_VARIANTS[word]}")

os.makedirs(dir_out, exist_ok=True)

summary_path = os.path.join(dir_out, "frequency_summary.csv")
summary_needs_header = not os.path.exists(summary_path)

dir_corpus = (os.path.expanduser(args.corpus_dir) if args.corpus_dir
              else os.path.join(dir_in, DEFAULT_CORPUS_SUBDIR))
if not os.path.isdir(dir_corpus):
    sys.exit(f"Corpus directory not found: {dir_corpus}")

# Accept both naming conventions: 1880.txt (decade files) and en_1880.txt (year files)
PERIOD_FILE_RE = re.compile(r"^(?:en[_-])?(\d{4})\.txt$", re.IGNORECASE)
period_files = {}
for name in sorted(os.listdir(dir_corpus)):
    m = PERIOD_FILE_RE.match(name)
    if m:
        period_files[int(m.group(1))] = os.path.join(dir_corpus, name)

if not period_files:
    sys.exit(f"No <period>.txt or en_<period>.txt files in {dir_corpus}")

requested = args.periods or args.decades
if requested:
    missing = [p for p in requested if p not in period_files]
    if missing:
        print(f"WARNING: no corpus file for {missing}; those periods are skipped.")
    decades = [p for p in requested if p in period_files]
else:
    decades = sorted(period_files)

GRANULARITY = (args.granularity if args.granularity != 'auto'
               else ('year' if len(period_files) > 20 else 'decade'))

print(f"Corpus dir:     {dir_corpus}")
print(f"Granularity:    {GRANULARITY}  ({len(decades)} periods to process, "
      f"{len(period_files)} found)")

# A gap in a yearly run is easy to miss among 119 files, so name it.
span = range(min(period_files), max(period_files) + 1)
gaps = [p for p in span if p not in period_files]
if GRANULARITY == 'year' and gaps:
    print(f"NOTE: no corpus file for {len(gaps)} year(s) in "
          f"{min(period_files)}-{max(period_files)}: {gaps}")

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
# A token is a run of letters/digits, optionally joined by INTERNAL hyphens or
# apostrophes. Attached punctuation is therefore removed, while hyphenated
# compounds stay whole (and so stay excluded from matching, as intended).
#   "tea,"          -> ['tea']
#   "coffee.—this"  -> ['coffee', 'this']
#   "coffee-"       -> ['coffee']
#   "tea-table"     -> ['tea-table']
#   "don't"         -> ["don't"]
# [^\W_] is "word character except underscore", i.e. unicode letters and digits.
TOKEN_RE = re.compile(r"[^\W_]+(?:[-'’ʼ][^\W_]+)*", re.UNICODE)


def tokenise(line):
    """Lowercase and tokenise one corpus line."""
    return TOKEN_RE.findall(line.lower())


def normalise_model_token(tok):
    """
    Normalise a token as returned by the embedding wrapper so it can be
    compared with our surface forms. Handles a WordPiece continuation prefix
    if the wrapper happens to return one.
    """
    if tok is None:
        return ''
    tok = str(tok)
    if tok.startswith('##'):
        tok = tok[2:]
    return tok.lower().strip("▁ ")


def find_target_sentences(file_path, variants_map, context_window, max_per_word=None):
    """
    Find occurrences of target words (or any surface form) and return a CENTRED
    context window around each one.

    Args:
        file_path: Path to corpus file
        variants_map: Dict mapping canonical word -> list of surface forms
        context_window: words of context kept on each side of the target
        max_per_word: Optional cap on occurrences per canonical word

    Returns:
        Dict mapping canonical word to a list of (window_text, position_in_window,
        surface_form) tuples. `window_text` is the +/-context_window slice of the
        document centred on the occurrence, space-joined; `position_in_window` is
        the target's index within that slice; `surface_form` is the form that
        matched, carried through so the extraction step can verify it.

        Occurrences are returned in document order.
    """
    results = {word: [] for word in variants_map}

    # Build reverse lookup: surface form (lowercased) -> canonical word
    variant_to_canonical = {}
    for canonical, variants in variants_map.items():
        for v in variants:
            variant_to_canonical[v.lower()] = canonical

    if not variant_to_canonical:
        return results

    full = {w for w in results}          # words that have hit max_per_word

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            if not line.strip():
                continue

            tokens = tokenise(line)
            if not tokens:
                continue

            # One pass over the line, in document order.
            for pos, tok in enumerate(tokens):
                canonical = variant_to_canonical.get(tok)
                if canonical is None:
                    continue
                if max_per_word is not None and len(results[canonical]) >= max_per_word:
                    continue
                lo = max(0, pos - context_window)
                hi = min(len(tokens), pos + context_window + 1)
                results[canonical].append((' '.join(tokens[lo:hi]), pos - lo, tok))

            if max_per_word is not None:
                full = {w for w in results if len(results[w]) >= max_per_word}
                if len(full) == len(results):
                    break

    return results


def locate_target(emb_list, expected_index, surface_form):
    """
    Find the entry of emb_list that holds the target word.

    emb_list is a list of (token, embedding) pairs including [CLS] and [SEP],
    so the target should sit at expected_index = position_in_window + 1.
    That assumption holds only if the wrapper returns exactly one entry per
    input word; it is verified rather than trusted.

    Returns (index, status) where status is 'exact', 'realigned' or 'missing'.
    """
    last = len(emb_list) - 1          # index of [SEP]

    if 0 < expected_index < last:
        if normalise_model_token(emb_list[expected_index][0]) == surface_form:
            return expected_index, 'exact'

    # Fall back to the nearest entry that does hold the target. Picking the
    # nearest matters when the form occurs more than once in the window.
    best = None
    for i in range(1, last):
        if normalise_model_token(emb_list[i][0]) == surface_form:
            if best is None or abs(i - expected_index) < abs(best - expected_index):
                best = i
    if best is not None:
        return best, 'realigned'

    return None, 'missing'


def extract_target_embeddings(embedder, batch, counters=None, examples=None):
    """
    Extract embeddings for target words from their (windowed) contexts.

    Args:
        embedder: BERT extractor instance (MacBERTh wrapper, any HF BERT)
        batch: list of (window_text, position_in_window, surface_form) tuples
        counters: optional dict with keys 'exact', 'realigned', 'missing'
        examples: optional list collecting a few mismatch examples for reporting

    Returns:
        List of dicts with 'embedding', 'snippet', 'position'. `snippet` is the
        full windowed context; `position` is the target's index within it.
        Occurrences whose target cannot be located are skipped.
    """
    if not batch:
        return []

    sentences = [b[0] for b in batch]
    positions = [b[1] for b in batch]
    forms = [b[2] for b in batch]

    bert_sents = embedder.get_berts(sentences)

    results = []
    for idx, (emb_list, pos, form) in enumerate(zip(bert_sents, positions, forms)):
        target_index, status = locate_target(emb_list, pos + 1, form)

        if counters is not None:
            counters[status] = counters.get(status, 0) + 1

        if status != 'exact' and examples is not None and len(examples) < 5:
            got = (normalise_model_token(emb_list[pos + 1][0])
                   if 0 < pos + 1 < len(emb_list) - 1 else '<out of range>')
            examples.append(f"expected '{form}' at index {pos + 1}, found '{got}' "
                            f"({status}) in: {sentences[idx][:90]}...")

        if target_index is None:
            continue

        _token, embedding = emb_list[target_index]
        results.append({
            'embedding': embedding,
            'snippet': sentences[idx],   # full centred window (target always visible)
            'position': pos              # index of the target WITHIN the window
        })

    return results


def run_batches(embedder, occurrences, desc, counters, examples):
    """Extract a list of occurrences in batches, clearing GPU memory as it goes."""
    out = []
    for i in tqdm(range(0, len(occurrences), BATCH_SIZE), desc=desc):
        out.extend(extract_target_embeddings(
            embedder, occurrences[i:i + BATCH_SIZE], counters, examples))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return out


def word_is_complete(h5file, word):
    """
    True only if the word group exists AND was fully written. An interrupted
    run can leave a group with some of its usages, which must be redone.
    """
    if word not in h5file:
        return False
    return bool(h5file[word].attrs.get('complete', False))


# ---------------------------------------------------------------------------
# Load the model
# ---------------------------------------------------------------------------
print(f"\nLoading model from: {model_path}")
embedder = MacBERTh(
    model_name=model_path,
    layer_mode=LAYER_MODE
)
print("Model loaded.")
try:
    tk = getattr(embedder, 'tokenizer', None)
    if tk is not None:
        print(f"Tokenizer do_lower_case: {getattr(tk, 'do_lower_case', 'n/a')}")
except Exception:
    pass
print()

# ---------------------------------------------------------------------------
# Alignment probe. Verifies that the wrapper returns one entry per input word
# before any real work is done. Catches a silent off-by-one across the whole
# run in about a minute rather than after several GPU hours.
# ---------------------------------------------------------------------------
probe_decade = next((d for d in decades if d in period_files), None)

if probe_decade is None:
    sys.exit("No corpus file found for any requested decade.")

print("=" * 60)
print(f"ALIGNMENT PROBE on {probe_decade} "
      f"({args.probe_size} occurrences per word)")
print("=" * 60)

probe_occurrences = find_target_sentences(
    period_files[probe_decade],
    TARGET_VARIANTS, CONTEXT_WINDOW, max_per_word=args.probe_size)

probe_counters = {'exact': 0, 'realigned': 0, 'missing': 0}
probe_examples = []

for word in sorted(TARGET_WORDS):
    occ = probe_occurrences.get(word, [])
    if not occ:
        print(f"  {word}: no occurrences in the probe sample")
        continue
    per_word = {'exact': 0, 'realigned': 0, 'missing': 0}
    run_batches(embedder, occ, f"  probe {word}", per_word, probe_examples)
    for k in probe_counters:
        probe_counters[k] += per_word[k]
    total = sum(per_word.values())
    print(f"  {word:<8} {total:>5} probed | exact {per_word['exact']:>5} "
          f"realigned {per_word['realigned']:>4} missing {per_word['missing']:>4}")

probe_total = sum(probe_counters.values())
probe_ok = (probe_counters['exact'] + probe_counters['realigned'])
probe_rate = probe_ok / probe_total if probe_total else 0.0

print("-" * 60)
print(f"Probe: {probe_total} occurrences, "
      f"{probe_counters['exact']} exact, "
      f"{probe_counters['realigned']} realigned, "
      f"{probe_counters['missing']} missing "
      f"-> verified rate {probe_rate:.1%}")

if probe_examples:
    print("\nFirst mismatches:")
    for line in probe_examples:
        print(f"  {line}")

if probe_total == 0:
    # Possible on a single-year file for a rare word; it is not an alignment
    # failure, so it must not trigger the abort below.
    print("\nWARNING: the probe found no occurrences at all in "
          f"{probe_decade}. Alignment is therefore UNVERIFIED. Probe a denser "
          f"period with --periods before trusting the run.")
elif args.min_align_rate > 0 and probe_rate < args.min_align_rate:
    print("\n" + "!" * 60)
    print(f"ABORTING: verified rate {probe_rate:.1%} is below "
          f"--min-align-rate {args.min_align_rate:.0%}.")
    print("The wrapper is not returning one entry per input word, so the")
    print("position arithmetic is wrong. Fix that before extracting; do not")
    print("lower the threshold to get past this.")
    print("!" * 60)
    sys.exit(1)

if probe_counters['realigned']:
    print("\nNOTE: some occurrences needed realignment. They were recovered "
          "correctly, but the off-by-one is worth understanding.")

del probe_occurrences
gc.collect()

if args.probe_only:
    print("\n--probe-only set: nothing written. Exiting.")
    sys.exit(0)

print("\nProbe passed. Starting extraction.\n")

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
run_totals = {word: 0 for word in TARGET_WORDS}
run_counters = {word: {'exact': 0, 'realigned': 0, 'missing': 0} for word in TARGET_WORDS}

for decade in decades:
    file_path = period_files[decade]

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping...")
        continue

    output_path = os.path.join(dir_out, f"commodity_embeddings_{decade}.h5")

    # Which words still need doing? A group that exists but was never marked
    # complete is treated as unfinished and redone.
    if os.path.exists(output_path):
        with h5py.File(output_path, 'r') as f:
            complete_words = {w for w in f.keys() if word_is_complete(f, w)}
            partial_words = set(f.keys()) - complete_words
        words_to_process = TARGET_WORDS - complete_words
        if not words_to_process:
            print(f"Skipping {decade} - all target words already complete")
            continue
        print(f"\n{'='*60}")
        print(f"Processing {GRANULARITY} {decade}...")
        print(f"Complete words: {sorted(complete_words) if complete_words else 'none'}")
        if partial_words:
            print(f"Incomplete from an earlier run (will be redone): {sorted(partial_words)}")
        print(f"To process: {sorted(words_to_process)}")
        print(f"{'='*60}")
        file_mode = 'a'
    else:
        words_to_process = set(TARGET_WORDS)
        print(f"\n{'='*60}")
        print(f"Processing {GRANULARITY} {decade} (new file)...")
        print(f"To process: {sorted(words_to_process)}")
        print(f"{'='*60}")
        file_mode = 'w'

    start_time = time.time()

    print(f"Finding occurrences (forms={FORMS_MODE}, window=+/-{CONTEXT_WINDOW})...")
    variants_to_process = {word: TARGET_VARIANTS[word] for word in words_to_process}
    target_sentences = find_target_sentences(file_path, variants_to_process,
                                             context_window=CONTEXT_WINDOW)

    found_before_cap = {w: len(target_sentences[w]) for w in words_to_process}

    # Apply the cap by RANDOM SAMPLE of everything found, not by stopping the
    # scan early. Taking the first N would sample whichever newspapers happen to
    # sit at the top of the decade file.
    if args.max_per_word_decade:
        rng = random.Random(f"{args.sample_seed}-{decade}")
        for word in words_to_process:
            occ = target_sentences[word]
            if len(occ) > args.max_per_word_decade:
                target_sentences[word] = rng.sample(occ, args.max_per_word_decade)

    for word in sorted(words_to_process):
        n = len(target_sentences[word])
        note = (f" (sampled from {found_before_cap[word]:,})"
                if n < found_before_cap[word] else "")
        print(f"  {word} {TARGET_VARIANTS[word]}: {n:,} occurrences{note}")

    print("\nExtracting embeddings...")

    decade_counters = {}
    saved_counts = {}

    with h5py.File(output_path, file_mode) as f:
        f.attrs['model'] = model_tag
        f.attrs['model_path'] = model_path
        f.attrs['decade'] = decade          # kept for existing readers
        f.attrs['period'] = decade
        f.attrs['granularity'] = GRANULARITY
        f.attrs['forms_mode'] = FORMS_MODE
        f.attrs['variant_mode'] = FORMS_MODE      # kept for older readers
        f.attrs['context_window'] = CONTEXT_WINDOW
        f.attrs['layer_mode'] = LAYER_MODE
        f.attrs['tokenisation'] = TOKENISATION_TAG

        for word in sorted(words_to_process):
            occurrences = target_sentences[word]
            counters = {'exact': 0, 'realigned': 0, 'missing': 0}
            decade_counters[word] = counters

            if not occurrences:
                print(f"  {word}: no occurrences, skipping")
                saved_counts[word] = 0
                continue

            print(f"  Processing {word} ({len(occurrences)} occurrences)...")
            examples = []
            all_results = run_batches(embedder, occurrences, f"  {word}", counters, examples)

            # Remove any incomplete group left by an interrupted run
            if word in f:
                del f[word]

            word_grp = f.create_group(word)
            word_grp.attrs['count'] = len(all_results)
            word_grp.attrs['occurrences_found'] = len(occurrences)
            word_grp.attrs['variants'] = ','.join(TARGET_VARIANTS[word])
            word_grp.attrs['forms_mode'] = FORMS_MODE
            word_grp.attrs['variant_mode'] = FORMS_MODE
            word_grp.attrs['context_window'] = CONTEXT_WINDOW
            word_grp.attrs['layer_mode'] = LAYER_MODE
            word_grp.attrs['tokenisation'] = TOKENISATION_TAG
            word_grp.attrs['align_exact'] = counters['exact']
            word_grp.attrs['align_realigned'] = counters['realigned']
            word_grp.attrs['align_missing'] = counters['missing']

            for idx, item in enumerate(all_results):
                usage_grp = word_grp.create_group(f"usage_{idx}")
                usage_grp.create_dataset("embedding", data=item['embedding'],
                                         compression="gzip")
                usage_grp.attrs['snippet'] = item['snippet']
                usage_grp.attrs['position'] = item['position']

            # Mark complete only once every usage is on disk
            word_grp.attrs['complete'] = True
            f.flush()

            saved_counts[word] = len(all_results)
            run_totals[word] += len(all_results)
            for k in counters:
                run_counters[word][k] += counters[k]

            note = ""
            if counters['missing']:
                note = f"  ({counters['missing']} could not be located and were skipped)"
            print(f"  {word}: saved {len(all_results)} embeddings{note}")
            if examples:
                print(f"    first mismatch: {examples[0]}")

    # Per-decade log. Records occurrences found AND embeddings saved, which are
    # not the same number, plus the alignment counters.
    with open(summary_path, 'a', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        if summary_needs_header:
            writer.writerow(['model', 'forms_mode', 'tokenisation', 'context_window',
                             'word', 'decade', 'forms',
                             'occurrences_found', 'embeddings_saved',
                             'align_exact', 'align_realigned', 'align_missing'])
            summary_needs_header = False
        for word in sorted(words_to_process):
            c = decade_counters.get(word, {'exact': 0, 'realigned': 0, 'missing': 0})
            writer.writerow([
                model_tag, FORMS_MODE, TOKENISATION_TAG, CONTEXT_WINDOW, word, decade,
                '|'.join(variants_to_process[word]),
                found_before_cap.get(word, len(target_sentences[word])),
                saved_counts.get(word, 0),
                c['exact'], c['realigned'], c['missing'],
            ])

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\n{GRANULARITY.capitalize()} {decade} completed in "
          f"{int(minutes)}m {seconds:.2f}s")

    del target_sentences
    gc.collect()

print("\n" + "=" * 60)
print(f"All {len(decades)} {GRANULARITY}s processed!")
print(f"Model:          {model_tag}")
print(f"Forms mode:     {FORMS_MODE}")
print(f"Tokenisation:   {TOKENISATION_TAG}")
print(f"Context window: +/-{CONTEXT_WINDOW} words")
print(f"Output saved to: {dir_out}")
print(f"Frequency log:  {summary_path}")
print("-" * 60)
print("Run totals (embeddings saved this run, words processed this run):")
for word in sorted(TARGET_WORDS):
    c = run_counters[word]
    print(f"  {word:<8} {run_totals[word]:>8}   "
          f"exact {c['exact']:>8}  realigned {c['realigned']:>6}  missing {c['missing']:>6}")
print("=" * 60)
