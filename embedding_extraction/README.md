# Embedding Extraction

This directory contains the pipeline that extracts a contextualised embedding for every occurrence of the target words from MacBERTh. The analysis of these embeddings (clustering, change measurement and dependency parsing) is in [`analysis/`](analysis/), with its own README.

The extraction code is adapted from the [LatinISE project](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change) by Barbara McGillivray and Valentina Lunardi.

## Directory Structure

```
embedding_extraction/
├── README.md
├── gen_macberth.py            # MacBERTh wrapper: word-level embeddings from chosen hidden layers
├── embeddings_extraction.py   # Extracts target-word embeddings per period (year or decade)
└── analysis/                  # Clustering, JSD and dependency parsing (see analysis/README.md)
```

## Files

| File | Description |
|---|---|
| `gen_macberth.py` | Wrapper around a HuggingFace BERT model (pretrained MacBERTh or a domain-adapted checkpoint). Tokenises each input into subwords, maps subword vectors back to words by averaging, and returns one 768-dimensional vector per word. `layer_mode` selects the hidden layers used: `last` (last layer), `last4` (mean of the last four layers), `first4` (mean of the first four) or `all` (mean of all layers). Adapted from the `gen_berts.py` LatinBERT wrapper of the LatinISE project |
| `embeddings_extraction.py` | Finds every occurrence of the target words in each period sub-corpus, embeds a window of tokens centred on it, verifies that the vector taken belongs to the target word, and saves it with its context to one HDF5 file per period |

## Requirements

- The period sub-corpora (see `corpus_preparation/`). `--corpus-dir` must contain the period files themselves, named `<period>.txt` or `en_<period>.txt` (e.g. `1880.txt`, `en_1880.txt`); files in sub-folders are not found.
- The model: the domain-adapted checkpoint `output/scaling_experiment/config_A/epoch_3` (see `domain_adaptation/`) or the pretrained [`emanjavacas/MacBERTh`](https://huggingface.co/emanjavacas/MacBERTh).

Set `dir_in` at the top of `embeddings_extraction.py` to your local data directory. The model checkpoint and the output folder are found under `dir_in/output/`; without `--corpus-dir` the corpus is read from `dir_in/en_decade_corpus`.

## Running it

Year-level embeddings of the three commodity terms with the domain-adapted model:

```bash
python embeddings_extraction.py --corpus-dir /path/to/en_year_corpus \
    --model scaling_best --layer-mode last4 --forms plural \
    --words coffee tea sugar
```

The control words go through the same script with the same model, window, tokenisation and verification, matched on the singular only:

```bash
python embeddings_extraction.py --corpus-dir /path/to/en_year_corpus \
    --model scaling_best --layer-mode last4 --forms canonical \
    --words presence child difference
```

Before a long run, check the alignment on one dense year without writing anything:

```bash
python embeddings_extraction.py --corpus-dir /path/to/en_year_corpus \
    --model scaling_best --layer-mode last4 --probe-only --periods 1880
```

| Option | Values | Meaning |
|---|---|---|
| `--corpus-dir` | directory | Folder holding the period files |
| `--model` | `scaling_best` (default), `pretrained` | Domain-adapted checkpoint (Config A, epoch 3) or pretrained MacBERTh |
| `--layer-mode` | `last` (default), `last4`, `first4`, `all` | Hidden layers averaged into each embedding. The paper uses `last4`; pass it explicitly |
| `--forms` | `plural` (default), `canonical` | `canonical` matches the singular only (*coffee*); `plural` also matches the plural (*coffees*, *teas*, *sugars*). No spelling or OCR variants are searched |
| `--words` | target words | Default: *coffee*, *tea*, *sugar*, *opium*, *cocoa*, *tobacco*. Every word must be listed in `TARGET_FORMS` |
| `--context-window` | integer, default 50 | Tokens of context kept on each side of the target |
| `--periods` | years or decades | Subset of periods; default all files in `--corpus-dir` |
| `--max-per-word-decade` | integer | Cap per word and period, applied by random sampling after all occurrences are found (seed `--sample-seed`) |
| `--probe-only` | flag | Run the alignment probe only and write nothing |
| `--min-align-rate` | fraction, default 0.90 | The run stops if fewer probe occurrences than this verify |
| `--out-dir` | directory | Output folder; default `output/<model>_<layer>_<forms>_w<window>_striptok` |

A word already marked complete in a period's HDF5 file is skipped, so an interrupted run can be resumed; a word left incomplete by an interruption is redone.

## What it does

1. **Tokenises.** Each line of a period file (one article) is lower-cased and split into runs of letters and digits, optionally joined by internal hyphens or apostrophes. Punctuation attached to a word is removed (`tea,` → `tea`; `coffee.—this` → `coffee`, `this`), while hyphenated compounds stay whole and therefore do not match (`tea-table`, `cocoa-nut`).
2. **Finds occurrences** of the requested surface forms, in document order.
3. **Cuts a context window** of up to 50 tokens on each side of each occurrence, so every occurrence is embedded in its own local context, well within the model's 512-token limit.
4. **Embeds the window** with MacBERTh and takes the vector at the target's position: the mean of the last four hidden layers (`last4`), averaged over the target's subword tokens.
5. **Verifies the target.** The token at the expected position must be the expected surface form. If it is not, the nearest matching token in the window is used (*realigned*); if there is none, the occurrence is skipped (*missing*). The counts of each outcome are recorded. Before extraction, a probe of 200 occurrences per word stops the run if fewer than 90% verify.
6. **Saves** the embedding, the window text and the target's position.

## Output

One HDF5 file per period, `commodity_embeddings_<period>.h5`:

```
commodity_embeddings_1880.h5    attrs: model, model_path, period, granularity, forms_mode,
│                                      context_window, layer_mode, tokenisation
├── coffee/                     attrs: count, occurrences_found, variants, forms_mode,
│   │                                  align_exact, align_realigned, align_missing, complete, ...
│   ├── usage_0/
│   │   └── embedding           (768,) float32, gzip-compressed
│   │                           attrs: snippet (the context window), position (target index in it)
│   ├── usage_1/
│   └── ...
├── tea/
└── ...
```

The `usage_*` groups are numbered from 0 in document order. HDF5 lists them in string order (`usage_10` before `usage_2`), so sort them numerically when pairing embeddings with snippets. The target's surface form is `snippet.split(" ")[position]`.

`frequency_summary.csv` in the same folder has one row per word and period:

| Column | Description |
|---|---|
| `model`, `forms_mode`, `tokenisation`, `context_window` | Run settings |
| `word` | Target word |
| `decade` | Period (a year for yearly runs; the column name is kept for compatibility) |
| `forms` | Surface forms searched, separated by `\|` |
| `occurrences_found` | Occurrences found, before any `--max-per-word-decade` cap |
| `embeddings_saved` | Embeddings written to the HDF5 file |
| `align_exact`, `align_realigned`, `align_missing` | Outcome of the target verification |

`corpus_preparation/plot_counts_by_year.py` plots `occurrences_found` (or `embeddings_saved`) per year from this file.
