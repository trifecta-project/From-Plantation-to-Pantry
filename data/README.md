# Data

This directory documents the newspaper corpus used in the paper: where the source data comes from, how to obtain it, and how it is divided into sub-corpora. The corpus text files are too large to host on GitHub and must be generated locally with the scripts in `corpus_preparation/`. The statistics and metadata describing the corpus are included.

## Directory Structure

```
data/
├── README.md
├── en_year_corpus/      # Year-level sub-corpora (1801–1919) and aggregation statistics
└── en_decade_corpus/    # Decade-level sub-corpora, corpus metadata and newspaper inventory
```

| Folder | Contents | Used for |
|---|---|---|
| [`en_year_corpus/`](en_year_corpus/) | One text file per year; articles, tokens and HMD/LwM split per year | Semantic change analysis |
| [`en_decade_corpus/`](en_decade_corpus/) | One text file per decade; corpus composition from the XML metadata, newspaper inventory, articles per newspaper | Domain adaptation of MacBERTh; description of the corpus |

Each folder has its own README describing its files and how to regenerate them.

## Source Corpora

The corpus combines two British Library collections:

| Collection | Full Name | Download |
|---|---|---|
| **HMD** | Heritage Made Digital | [https://bl.iro.bl.uk/concern/datasets/2800eb7d-8b49-4398-a6e9-c2c5692a1304](https://bl.iro.bl.uk/concern/datasets/2800eb7d-8b49-4398-a6e9-c2c5692a1304) |
| **LwM** | Living with Machines | [https://bl.iro.bl.uk/concern/datasets/99dc570a-9460-48ac-baed-9d2b8c4c13c0](https://bl.iro.bl.uk/concern/datasets/99dc570a-9460-48ac-baed-9d2b8c4c13c0) |

The HMD collection focused on digitising London metropolitan newspapers in poor physical condition with wider circulation ([Beelen and van Strien, 2022](https://arxiv.org/abs/2211.10086)). The LwM collection digitised predominantly provincial titles across England, Wales and Scotland, with 44% of articles from northern England, 41% from the south, 6% from Wales and 3% from Scotland ([Ridge and Pedrazzini, 2024](https://livingwithmachines.ac.uk/public-domain-newspaper-titles-in-living-with-machines/)).

Together the two collections contain about 9.3 million articles from 120 newspaper titles. After aggregation, the sub-corpora hold 7,927,079 articles and about 5.6 billion tokens for 1801–1919; see `en_decade_corpus/README.md` for how the counts at each stage relate.

## Downloading the Source Data

Download both collections from the links above and extract them. This gives two directories:

```
hmd-alto2txt/
├── plaintext/        ← used to build the sub-corpora
└── metadata/         ← used for corpus metadata extraction

lwm-alto2txt/
├── plaintext/
└── metadata/
```

The `plaintext/` directories contain one `.txt` file per article, organised by newspaper ID, year and date (e.g. `0003038/1899/0929/0003038_18990929_art0087.txt`). The `metadata/` directories contain matching XML files with publication metadata, OCR quality scores and article types.

To build the sub-corpora:

```bash
# Year sub-corpora
python corpus_preparation/aggregate_by_year.py \
    --lwm /path/to/lwm-alto2txt/plaintext \
    --hmd /path/to/hmd-alto2txt/plaintext \
    --output ./data/en_year_corpus \
    --workers 16

# Decade sub-corpora, built from the year sub-corpora
python corpus_preparation/build_decade_corpus.py \
    --year-dir ./data/en_year_corpus \
    --output ./data/en_decade_corpus
```

Building the year sub-corpora processes about 9.3 million files and takes several hours depending on disk speed. `corpus_preparation/README.md` describes all steps, including metadata extraction.

## Pretrained Word2Vec Vectors

The static embedding analysis uses pretrained and aligned Word2Vec vectors from:

> Nilo Pedrazzini and Barbara McGillivray (2022). *Diachronic word embeddings from 19th-century British newspapers* [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.7181682](https://doi.org/10.5281/zenodo.7181682)

These vectors were trained on the same HMD and LwM data (skip-gram, 5 epochs, 200 dimensions, context window 3, minimum count 1), with decade-specific vector spaces aligned using Orthogonal Procrustes (Schönemann, 1966).

```bash
wget https://zenodo.org/records/7181682/files/lwm_vectors.zip
unzip lwm_vectors.zip -d vectors/
```

## Licensing

The HMD and LwM datasets are released by the British Library. Please consult the dataset pages linked above for licensing terms and conditions of use.

