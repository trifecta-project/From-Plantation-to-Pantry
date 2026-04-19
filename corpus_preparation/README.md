# Corpus Preparation

This directory contains scripts for aggregating the Heritage Made Digital (HMD) and Living with Machines (LwM) newspaper collections into decade-level sub-corpora, computing token and target-term frequencies, and extracting corpus metadata from the XML files.

## Files

| File | Description |
|---|---|
| `aggregate_and_partition.py` | Main aggregation script: scans both HMD and LwM plaintext directories, extracts publication years from file paths, and writes one text file per decade to `data/en_decade_corpus/` |
| `count_tokens_and_terms.py` | Counts total tokens and frequencies of the six target commodity terms (*coffee*, *tea*, *sugar*, *opium*, *cocoa*, *tobacco*) per decade, outputs raw and normalised (per-million) frequencies |
| `extract_corpus_metadata.py` | Parses all `*_metadata.xml` files from both HMD and LwM metadata directories to extract newspaper titles, locations, dates, OCR quality scores, and article types. Produces corpus overview statistics, a complete newspaper inventory, and London vs provincial breakdown |
| `decade_term_frequencies.csv` | Output of `count_tokens_and_terms.py`: token counts and term frequencies per decade (both raw and per-million), used to produce Table 1 in the paper |
| `aggregation_report.txt` | Output of `aggregate_and_partition.py`: summary statistics of the aggregation run, including article counts, token counts, and byte sizes per decade, broken down by source (HMD vs LwM) |

## Pipeline

### Step 1: Aggregate and partition

Requires the raw HMD and LwM plaintext directories (see `data/README.md` for download instructions).

```bash
python aggregate_and_partition.py \
    --lwm /path/to/lwm-alto2txt/plaintext \
    --hmd /path/to/hmd-alto2txt/plaintext \
    --output ../data/en_decade_corpus \
    --workers 4
```

**What it does:**
- Scans all `.txt` files under both plaintext directories (~9 million files total)
- Extracts the publication year from each file path using the British Library alto2txt naming convention (e.g., `0003038_18990929_art0087.txt` → year 1899)
- Writes all text from each decade into a single file: `en_1800s.txt`, `en_1810s.txt`, ..., `en_1910s.txt`
- Generates `aggregation_report.txt` (human-readable summary) and `aggregation_stats.json` (machine-readable statistics)

**Key statistics from our run** (from `aggregation_report.txt`):
- Files processed: 7,982,524
- Files skipped: 1,399,130 (empty, too short, or year not extractable)
- Total articles aggregated: 7,927,079
- Total tokens: ~5.62 billion
- Period covered: 1801–1919

### Step 2: Count tokens and term frequencies

Requires the decade sub-corpora generated in Step 1.

```bash
python count_tokens_and_terms.py
```

**Note:** Update the `corpus_dir` variable at the top of the script to point to your local `en_decade_corpus/` directory before running.

**What it does:**
- Reads each decade file and tokenises using whitespace and basic punctuation boundaries
- Counts total tokens per decade
- Counts occurrences of the six target terms: *coffee*, *tea*, *sugar*, *opium*, *cocoa*, *tobacco*
- Computes normalised frequencies (per million tokens)
- Saves all results to `decade_term_frequencies.csv`

**Output columns in `decade_term_frequencies.csv`:**

| Column | Description |
|---|---|
| `decade` | Decade label (e.g., `1800`, `1810`, ..., `1910`) |
| `total_tokens` | Total token count for that decade |
| `coffee`, `tea`, ..., `tobacco` | Raw frequency of each target term |
| `coffee_per_million`, ..., `tobacco_per_million` | Frequency normalised per million tokens |

### Step 3: Extract corpus metadata

Requires the HMD and LwM metadata directories (the `metadata/` subdirectories alongside the `plaintext/` directories).

```bash
python extract_corpus_metadata.py \
    --hmd /path/to/hmd-alto2txt/metadata \
    --lwm /path/to/lwm-alto2txt/metadata \
    --output ../data \
    --workers 8
```

**What it does:**
- Parses all `*_metadata.xml` files (~9 million files) from both collections
- Extracts per-article: newspaper title, BL identifier, location, publication date, article type, word count, and OCR confidence score
- Classifies each article as London or provincial based on location
- Aggregates by decade and by newspaper title

**Runtime:** 30–90 minutes depending on disk speed. The bottleneck is disk I/O (reading millions of small XML files); the parsing and aggregation are fast.

**Outputs** (saved to `data/`):

| File | Description |
|---|---|
| `corpus_overview.csv` | Per-decade summary: articles, words, HMD/LwM split, number of titles, London percentage, OCR quality (mean and median) |
| `newspaper_inventory.csv` | Complete inventory of all 120 newspaper titles with BL ID, location, collection, and article count |
| `top30_newspapers_by_article_count.csv` | The 30 largest newspapers by volume |

**Key findings from the metadata** (discussed in the paper's Data Bias limitation):
- The corpus comprises 120 newspaper titles: 17 London-based (HMD) and 103 provincial (LwM)
- London articles account for 100% of the 1810s–1820s but 0% from the 1880s onward
- Mean OCR quality rises from 0.69 (1800s) to 0.86 (1870s+)
- *The Sun* alone contributes 1.6 million articles (~17% of the corpus)

## Relationship to the Paper

- **Table 1** (Number of tokens per decade) is derived from the `total_tokens` column in `decade_term_frequencies.csv`. Note that minor differences between the token counts here and in Table 1 may arise from different tokenisation approaches (this script uses a simple regex tokeniser; the Word2Vec training used Gensim's tokenisation).
- **Section 3** (Data and Models) references the corpus composition statistics from `corpus_overview.csv` and cites the newspaper inventory documented in [Ridge and Pedrazzini (2024)](https://livingwithmachines.ac.uk/public-domain-newspaper-titles-in-living-with-machines/).
- **Section 7** (Data Bias limitation) draws on the London percentage, OCR quality trends, and the dominance of *The Sun* to characterise the corpus's asymmetric coverage.
- **`aggregation_report.txt`** documents the provenance of the corpus: how many files were processed, how many were skipped, and the contribution of each source (HMD vs LwM) per decade.
- The decade sub-corpora produced by Step 1 are the input to both the **static embedding analysis** (via the pretrained Word2Vec vectors from Pedrazzini & McGillivray 2022, which were trained on the same data) and the **domain adaptation** of MacBERTh (see `domain_adaptation/`).
