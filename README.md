# From Plantation to Pantry

Code and data for the paper *From Plantation to Pantry*, submitted to *Digital Scholarship in the Humanities*.

## Overview

The paper traces lexical semantic change in three commodity terms, *coffee*, *tea* and *sugar*, in British newspapers from 1801 to 1919. The corpus combines the British Library's [Heritage Made Digital](https://bl.iro.bl.uk/concern/datasets/2800eb7d-8b49-4398-a6e9-c2c5692a1304) (HMD) and [Living with Machines](https://bl.iro.bl.uk/concern/datasets/99dc570a-9460-48ac-baed-9d2b8c4c13c0) (LwM) collections: 7,927,079 articles and about 5.6 billion tokens, divided into one sub-corpus per year.

The pipeline:

1. **Domain adaptation.** [MacBERTh](https://huggingface.co/emanjavacas/MacBERTh), a BERT model pretrained on historical English, is adapted to the newspaper corpus by continued masked-language-model training. A grid search on a 1% sample and a scaling experiment on the full corpus select the final model.
2. **Occurrence embeddings.** Every occurrence of a target word is embedded in a context window of ±50 tokens, using the mean of the model's last four hidden layers.
3. **Senses and change.** For each word, the occurrence embeddings of all 119 years are clustered together with k-means into one shared set of clusters, which are interpreted as senses. Each year's cluster distribution is compared with the next year's by Jensen–Shannon divergence (JSD), tested against sampling noise by permutation.
4. **Validation.** Each target is compared with a control word of similar yearly frequency (*presence*, *child*, *difference*); the clusters and their shifts are checked against dependency relations (spaCy) and against nearest neighbours in independently trained static Word2Vec embeddings; and the analysis is repeated with the pretrained MacBERTh to compare it with the domain-adapted model.

## Repository Structure

```
From-Plantation-to-Pantry/
├── README.md
├── requirements.txt
├── data/                        # Corpus documentation, statistics and metadata
│   ├── en_year_corpus/          #   year sub-corpora (generated locally) and their statistics
│   └── en_decade_corpus/        #   decade sub-corpora (generated locally), corpus metadata, newspaper inventory
├── corpus_preparation/          # Build the year and decade sub-corpora; extract metadata; plot term counts
├── domain_adaptation/           # Grid search and scaling experiment for domain-adapting MacBERTh
├── embedding_extraction/        # Extract occurrence embeddings of the target and control words
│   └── analysis/                # Clustering, JSD, dependency parsing and control-word comparison
│       ├── results/             #   results with the domain-adapted model
│       ├── results_pretrained/  #   results with the pretrained model
│       └── controls/            #   targets compared with their control words
└── static_embeddings/           # Nearest neighbours in pretrained Word2Vec embeddings
```

Each folder has its own README describing its files, how to run them and what they produce.

## Pipeline

| Step | Folder | What it does |
|---|---|---|
| 1 | `corpus_preparation/` | Aggregates the HMD and LwM plaintext files into one sub-corpus per year (and per decade), and extracts corpus metadata |
| 2 | `domain_adaptation/` | Domain-adapts MacBERTh: grid search on a 1% subcorpus, then a scaling experiment on the full corpus; selects Config A, epoch 3 |
| 3 | `embedding_extraction/` | Extracts an embedding for every occurrence of the target and control words, per year |
| 4 | `embedding_extraction/analysis/` | Shared k-means clustering per word, cluster distributions per year, JSD between consecutive years with a permutation test, dependency profiles per cluster and per decade |
| 5 | `embedding_extraction/analysis/controls/` | Compares the JSD of each target with that of its control word, against their frequencies |
| 6 | `static_embeddings/` | Lists the nearest neighbours of each target per decade in the pretrained Word2Vec embeddings of Pedrazzini and McGillivray (2022) |

## Data

The HMD and LwM collections are released by the British Library and are too large to include here; the sub-corpora are generated locally from them (see `data/README.md`). The repository includes the corpus statistics, the newspaper inventory and the results reported in the paper. The domain-adapted model and the occurrence embeddings are not included because of their size; they are regenerated with the scripts in `domain_adaptation/` and `embedding_extraction/`. The Word2Vec vectors are available from [Zenodo](https://doi.org/10.5281/zenodo.7181682) (see `static_embeddings/README.md`).

## Requirements

Python 3 with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

A GPU is needed for domain adaptation and embedding extraction. Corpus preparation, the analysis and the static embeddings run on CPU, although clustering with the silhouette score on all embeddings takes a long time for *tea* (see `embedding_extraction/analysis/README.md`).

## Acknowledgements

This code builds on and adapts work from:

- **[LatinISE](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change)** by Barbara McGillivray and Valentina Lunardi: the embedding extraction wrapper and the domain-adaptation scripts.
- **[Pedrazzini and McGillivray (2022)](https://doi.org/10.5281/zenodo.7181682)**: the pretrained and aligned Word2Vec vectors trained on HMD and LwM ([DiachronicEmb-BigHistData](https://github.com/Living-with-machines/DiachronicEmb-BigHistData)).

## Funding

This research was supported by the European Union through the European Research Council under grant agreement 101088548 ([TRIFECTA](https://trifecta.dhlab.nl/)).
