# Data

This directory contains the decade-level sub-corpora used in the paper and corpus metadata files. The corpus text files are too large to host on GitHub and must be generated locally following the instructions below. The metadata CSVs documenting the corpus composition are included.

## Directory Structure

```
data/
├── README.md
├── newspaper_inventory.csv                 # All 120 newspaper titles in the corpus
├── corpus_overview.csv                     # Per-decade summary statistics
├── top30_newspapers_by_article_count.csv   # Top 30 newspapers by volume
└── en_decade_corpus/                       # Generated — see instructions below
    ├── en_1800s.txt
    ├── en_1810s.txt
    ├── ...
    ├── en_1910s.txt
    ├── aggregation_report.txt
    └── aggregation_stats.json
```

## Corpus Metadata

### `newspaper_inventory.csv`

Complete inventory of all 120 newspaper titles in the corpus: BL identifier, title, location, source collection (HMD or LwM), and total article count. The corpus draws on 17 London-based titles (2,530,729 articles) and 103 provincial titles (6,795,441 articles). For the full documentation of these titles including normalised names, year spans, and word counts, see [Ridge and Pedrazzini (2024)](https://livingwithmachines.ac.uk/public-domain-newspaper-titles-in-living-with-machines/).

### `corpus_overview.csv`

Per-decade summary statistics extracted from the HMD and LwM XML metadata files:

| Column | Description |
|---|---|
| `Decade` | Decade label |
| `Articles` | Total number of articles |
| `Words` | Total word count (from metadata, not retokenised) |
| `HMD` | Articles from Heritage Made Digital |
| `LwM` | Articles from Living with Machines |
| `Papers` | Number of distinct newspaper titles |
| `London_percent` | Percentage of articles from London-based newspapers |
| `OCR_mean` | Mean OCR confidence score across all articles |
| `OCR_median` | Median OCR confidence score |

Key observations from this data (discussed in the paper's Data Bias limitation):

- **London dominance in early decades**: HMD titles (all London-based) contribute 100% of articles in the 1810s–1820s, declining to 56% by the 1840s. From the 1880s onward, the corpus is entirely LwM (0% London).
- **OCR quality improves over time**: mean OCR rises from 0.69 (1800s) to 0.86 (1870s+), supporting the paper's decision to restrict analysis to the 1840s–1910s period.
- **The Sun dominates**: a single London newspaper (BL ID 0002194) accounts for 1,625,117 articles — approximately 17% of the entire corpus.

### `top30_newspapers_by_article_count.csv`

The 30 largest newspapers by article count, with BL ID, collection, location, and title.

## Source Corpora

The underlying data comes from two British Library collections:

| Collection | Full Name | Download |
|---|---|---|
| **HMD** | Heritage Made Digital | [https://bl.iro.bl.uk/concern/datasets/2800eb7d-8b49-4398-a6e9-c2c5692a1304](https://bl.iro.bl.uk/concern/datasets/2800eb7d-8b49-4398-a6e9-c2c5692a1304) |
| **LwM** | Living with Machines | [https://bl.iro.bl.uk/concern/datasets/99dc570a-9460-48ac-baed-9d2b8c4c13c0](https://bl.iro.bl.uk/concern/datasets/99dc570a-9460-48ac-baed-9d2b8c4c13c0) |

The HMD collection focused on digitising London metropolitan newspapers in poor physical condition with wider circulation ([Beelen and van Strien, 2022](https://arxiv.org/abs/2211.10086)). The LwM collection digitised predominantly provincial titles across England, Wales, and Scotland, with 44% of articles from northern England, 41% from the south, 6% from Wales, and 3% from Scotland ([Ridge and Pedrazzini, 2024](https://livingwithmachines.ac.uk/public-domain-newspaper-titles-in-living-with-machines/)).

The combined corpus spans 1800–1919 and comprises approximately 5.87 billion words across 9.3 million articles from 120 newspaper titles. Our semantic change analysis uses the period 1840–1919; the earlier decades (1800s–1830s) are retained for domain-adaptation pretraining but excluded from the analysis due to sparser coverage.

## Generating the Decade Sub-Corpora

### Step 1: Download and extract the source data

Download both collections from the links above and extract them. After extraction you will have two directories:

```
hmd-alto2txt/
├── plaintext/        ← used for text aggregation
└── metadata/         ← used for corpus metadata extraction

lwm-alto2txt/
├── plaintext/        ← used for text aggregation
└── metadata/         ← used for corpus metadata extraction
```

The `plaintext/` subdirectories contain newspaper articles as individual `.txt` files organised by newspaper ID, year, and date (e.g., `0003038/1899/0929/0003038_18990929_art0087.txt`). The `metadata/` subdirectories contain corresponding XML files with publication metadata, OCR quality scores, and article type information.

### Step 2: Aggregate and partition by decade

Run the aggregation script from the `corpus_preparation/` directory:

```bash
python corpus_preparation/aggregate_and_partition.py \
    --lwm /path/to/lwm-alto2txt/plaintext \
    --hmd /path/to/hmd-alto2txt/plaintext \
    --output ./data/en_decade_corpus \
    --workers 4
```

This script:
1. Scans all `.txt` files in both plaintext directories
2. Extracts the publication year from each file's path (pattern: `{newspaper_id}_{YYYYMMDD}_art{number}.txt`)
3. Aggregates all text into decade-level files (`en_1800s.txt`, `en_1810s.txt`, ..., `en_1910s.txt`)
4. Generates an aggregation report (`aggregation_report.txt`) and statistics (`aggregation_stats.json`)

Processing ~9 million files takes several hours depending on hardware. The `--workers` flag controls the number of parallel processes.

### Step 3 (optional): Extract corpus metadata

To regenerate the metadata CSVs from the XML files:

```bash
python corpus_preparation/extract_corpus_metadata.py \
    --hmd /path/to/hmd-alto2txt/metadata \
    --lwm /path/to/lwm-alto2txt/metadata \
    --output ./data \
    --workers 8
```

This parses all `*_metadata.xml` files and produces the corpus overview and newspaper inventory CSVs. Processing ~9 million XML files takes 30–90 minutes depending on disk speed.

### Expected output

After running the aggregation script, `data/en_decade_corpus/` should contain one `.txt` file per decade with approximate token counts as follows:

| Decade | Approximate tokens |
|---|---|
| 1800s | 142,200,426 |
| 1810s | 191,189,208 |
| 1820s | 239,720,024 |
| 1830s | 183,986,090 |
| 1840s | 601,310,976 |
| 1850s | 667,080,887 |
| 1860s | 651,028,965 |
| 1870s | 539,171,959 |
| 1880s | 868,343,506 |
| 1890s | 747,699,750 |
| 1900s | 580,114,837 |
| 1910s | 318,302,058 |

Note: minor discrepancies between these token counts (from `count_tokens_and_terms.py`) and the word counts in `corpus_overview.csv` (from XML metadata) arise from different tokenisation methods.

## Pretrained Word2Vec Vectors

The static embedding analysis uses pretrained and aligned Word2Vec vectors from:

> Nilo Pedrazzini and Barbara McGillivray (2022). *Diachronic word embeddings from 19th-century British newspapers* [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.7181682](https://doi.org/10.5281/zenodo.7181682)

These vectors were trained on the same HMD and LwM data with the following hyperparameters: skip-gram architecture, 5 epochs, 200 dimensions, context window of 3, minimum word count of 1. Decade-specific vector spaces were aligned using Orthogonal Procrustes (Schönemann, 1966).

Download:
```bash
wget https://zenodo.org/records/7181682/files/lwm_vectors.zip
unzip lwm_vectors.zip -d vectors/
```

## Licensing

The HMD and LwM datasets are released by the British Library. Please consult the respective dataset pages linked above for licensing terms and conditions of use.
