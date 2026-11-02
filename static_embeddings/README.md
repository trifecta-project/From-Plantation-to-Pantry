# Static Embeddings: Nearest Neighbours

This directory contains the validation of the sense trajectories with static word embeddings. The main analysis (`embedding_extraction/`) rests on contextualised occurrence embeddings from a domain-adapted MacBERTh. To test whether the shifts it finds are visible outside that family of representations, we look at the same target words in static Word2Vec embeddings, which differ from the occurrence embeddings in every respect that matters here: they assign one vector to each word type rather than one per occurrence, they come from a shallow predictive model rather than a transformer, and they were trained independently of this study.

For each target word and decade, we list its ten nearest neighbours by cosine similarity. Neighbours characterise a word through the company it keeps in the vector space, so a change in a word's neighbours over time indicates a change in the contexts in which it was used. The question put to them is whether the neighbours of *coffee*, *tea* and *sugar* shift from vocabulary of cultivation, shipping, duties and wholesale trade towards vocabulary of preparation, serving and domestic consumption.

A static model has one vector per word and decade, so it cannot separate a word's senses and cannot reproduce the sense-level trajectories of the main analysis. What it can show is whether the same direction of change is visible in a representation that is independent of our model, our domain adaptation and our clustering.

## Directory Structure

```
static_embeddings/
├── README.md
├── extract_neighbors.py          # Nearest neighbours of each target word per decade
└── results/                      # Output of extract_neighbors.py
    ├── all_neighbors.csv
    ├── neighbors_<word>.csv
    ├── neighbors_<word>_wide.csv
    ├── neighbor_changes_<word>.csv
    ├── neighbor_changes_summary.csv
    ├── word_frequencies.csv
    └── word_frequencies_wide.csv
```

## The Word2Vec vectors

We use the pretrained, aligned decade-specific Word2Vec vectors released by:

> Nilo Pedrazzini and Barbara McGillivray (2022). *Diachronic word embeddings from 19th-century British newspapers* [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.7181682](https://doi.org/10.5281/zenodo.7181682)

They were trained on the same Heritage Made Digital and Living with Machines newspaper data that underlies our analysis (see `data/`), one model per decade from the 1800s to the 1910s, with the Gensim implementation of Word2Vec: skip-gram, 5 epochs, 200 dimensions, a context window of 3 and a minimum word count of 1, with hyperparameters selected by grid search. The decade-specific spaces were aligned by Orthogonal Procrustes (Schönemann, 1966), so that vectors from different decades are directly comparable. We did not train or modify these vectors.

To download them:

```bash
wget https://zenodo.org/records/7181682/files/lwm_vectors.zip
unzip lwm_vectors.zip -d lwm_vectors/
```

The vectors are not included in this repository.

## Running it

Set `VECTORS_DIR` at the top of `extract_neighbors.py` to the folder holding the decade files (`1800s-vectors.txt` … `1910s-vectors.txt`), then run:

```bash
python extract_neighbors.py
```

The target words (`TARGET_WORDS`: *coffee*, *tea*, *sugar*), the number of neighbours (`TOPN`, 1,000) and the output folder (`OUTPUT_DIR`, `results/`) are also set at the top of the script. Requires `gensim` and `pandas`.

## What it does

1. **Records each target word's vocabulary rank per decade**: its position in the decade's vector file, as a proxy for frequency, on the assumption that the file lists words from most to least frequent, as Word2Vec usually writes them.
2. **Loads each decade's vectors** with Gensim.
3. **Finds the query word.** Each target word has its singular and plural forms (*coffee*, *coffees*); the first one present in a decade's vocabulary is used as the query word, which is the singular whenever it is present.
4. **Lists the nearest neighbours**: the 1,000 words with the highest cosine similarity to the query word in that decade's space. Forms of the target word itself (e.g. *coffees* for *coffee*) are removed and the remaining neighbours ranked from 1. Each decade's neighbours are computed among that decade's vocabulary.
5. **Compares the first and last decade**: which neighbours are in both the 1800s and the 1910s lists (stable), only in the 1910s list (new), or only in the 1800s list (lost).

**The paper reports the ten nearest neighbours per decade**, i.e. the rows with `rank` ≤ 10. The stable, new and lost neighbours in the `neighbor_changes` files are computed over all 1,000 neighbours.

## Output

| File | Contents |
|---|---|
| `all_neighbors.csv` | All neighbours of all words: `word`, `variant_used` (the query word), `decade`, `rank`, `neighbor`, `similarity` (cosine) |
| `neighbors_<word>.csv` | The same, for one word |
| `neighbors_<word>_wide.csv` | One word's neighbours with one column per decade and one row per rank |
| `neighbor_changes_<word>.csv` | Each neighbour of the 1800s or 1910s list, marked `stable`, `new` or `lost` |
| `neighbor_changes_summary.csv` | Per word: the stable, new and lost neighbours and their numbers |
| `word_frequencies.csv` | Per word and decade: `vocab_rank`, the size of the vocabulary and the rank as a percentile |
| `word_frequencies_wide.csv` | Vocabulary ranks with one column per decade |

## Acknowledgements

The vectors and the alignment are by Pedrazzini and McGillivray (2022); see also the [Living with Machines DiachronicEmb-BigHistData repository](https://github.com/Living-with-machines/DiachronicEmb-BigHistData).
