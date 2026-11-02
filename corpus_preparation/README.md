# Corpus Preparation

This directory contains the scripts that turn the British Library's Heritage Made Digital (HMD) and Living with Machines (LwM) newspaper collections into the sub-corpora used in the paper, and that describe the resulting corpus.

The corpus is built at year level: one sub-corpus per year, 1801–1919. The decade sub-corpora used for domain adaptation are built from the year sub-corpora, so both levels contain exactly the same articles.

## Files

| File | Description |
|---|---|
| `aggregate_by_year.py` | Builds the year sub-corpora from the HMD and LwM plaintext files and writes the aggregation report and statistics |
| `build_decade_corpus.py` | Concatenates the year sub-corpora into decade sub-corpora for domain adaptation |
| `extract_corpus_metadata.py` | Extracts newspaper titles, locations, dates, OCR quality and article types from the XML metadata files |
| `plot_counts_by_year.py` | Plots the number of occurrences of each target term per year |
| `aggregation_report.txt` | Human-readable summary of the aggregation run: articles, tokens and HMD/LwM split per year and per decade |

## Pipeline

```
HMD + LwM plaintext ──> aggregate_by_year.py ──> data/en_year_corpus/{year}/en_{year}.txt
                                                        │
                                                        └──> build_decade_corpus.py ──> data/en_decade_corpus/{decade}.txt

HMD + LwM metadata ───> extract_corpus_metadata.py ──> corpus composition CSVs

embedding extraction ─> frequency_summary.csv ──> plot_counts_by_year.py ──> occurrences per year figure
```

### Step 1: Build the year sub-corpora

Requires the HMD and LwM plaintext directories (see `data/README.md` for download instructions).

```bash
python aggregate_by_year.py \
    --lwm /path/to/lwm-alto2txt/plaintext \
    --hmd /path/to/hmd-alto2txt/plaintext \
    --output ../data/en_year_corpus \
    --workers 16
```

The script:

- scans all plaintext files in both collections (~9.3 million) in a single directory traversal;
- extracts each article's publication year from its file name, following the alto2txt naming convention (`0003038_18990929_art0087.txt` → 1899), with the `{newspaper_id}/{YYYY}/{MMDD}/` directory structure as a fallback;
- discards files that are empty, shorter than 50 characters, or whose year cannot be determined or falls outside 1800–1919;
- writes each remaining article as one line, with internal whitespace collapsed, to `{year}/en_{year}.txt`;
- writes `aggregation_report.txt` and `aggregation_stats.json` to the output directory.

`--start-year` and `--end-year` change the year range; `--keep-empty-years` keeps folders for years with no articles.

In our run, 7,927,079 articles (about 5.62 billion whitespace-delimited tokens) were written to 119 year sub-corpora, 1801–1919, and 1,399,131 files were skipped. The per-year counts are in `data/en_year_corpus/aggregation_stats.json`.

Processing takes several hours, depending mostly on disk speed.

### Step 2: Build the decade sub-corpora

Requires the year sub-corpora from Step 1.

```bash
python build_decade_corpus.py \
    --year-dir ../data/en_year_corpus \
    --output ../data/en_decade_corpus
```

This concatenates the year files of each decade, in year order, into `1800.txt`, `1810.txt`, …, `1910.txt` — the file names that `domain_adaptation/` and `embedding_extraction/` read. Each decade file contains exactly the articles of its years. Line order follows the years, so it can differ from that of decade files built directly from the source collections; random samples drawn from the files with a fixed seed can therefore differ too.

### Step 3: Extract corpus metadata

Requires the HMD and LwM metadata directories (the `metadata/` directories next to `plaintext/`).

```bash
python extract_corpus_metadata.py \
    --hmd /path/to/hmd-alto2txt/metadata \
    --lwm /path/to/lwm-alto2txt/metadata \
    --output ../data/en_decade_corpus \
    --workers 8
```

The script parses every `*_metadata.xml` file and extracts, per article, the newspaper title, BL identifier, place of publication, date, article type, word count and OCR confidence score. It classifies each article as London or provincial by place of publication and aggregates by decade and by title. It writes:

| File | Contents |
|---|---|
| `corpus_overview_by_decade.csv` | Articles, words, HMD/LwM split, number of titles, London/provincial split and OCR quality per decade |
| `newspaper_inventory.csv` | Every title with BL ID, location, collection, total articles, first and last decade, and articles per decade |
| `newspaper_summary_for_paper.csv` | Every title with BL ID, location, collection, total articles and decades covered |
| `ocr_quality_by_decade.csv` | OCR confidence statistics per decade |
| `item_types_by_decade.csv` | Article types per decade |

`--sample N` processes a random sample of N files, for testing. Processing the full collections takes 30–90 minutes; reading millions of small XML files is the bottleneck.

### Step 4: Plot occurrences per year

Requires the `frequency_summary.csv` files written by the embedding extraction (see `embedding_extraction/`), which record how many occurrences of each target word were found and embedded per year.

```bash
python plot_counts_by_year.py \
    --csv /path/to/commodity_extraction/frequency_summary.csv \
          /path/to/control_word_extraction/frequency_summary.csv \
    --words coffee tea sugar water stone \
    --out embeddings_over_time_yearly.png
```

The figure shows one line per term on a log scale; the control words (*water*, *stone*) are drawn dashed. `--column embeddings_saved` plots the number of embeddings written rather than the number of occurrences found, `--linear` uses a linear scale, and `--plain` draws the control words like the commodity terms. The script also prints, per word, the number of years, the minimum, maximum and total count, and the peak year.

## Outputs and Where They Are

| Output | Location |
|---|---|
| Year sub-corpora (`en_YYYY.txt`) | `data/en_year_corpus/` — generated locally, not in the repository |
| `aggregation_stats.json` | `data/en_year_corpus/` |
| `aggregation_report.txt` | this directory |
| Decade sub-corpora (`1800.txt` …) | `data/en_decade_corpus/` — generated locally, not in the repository |
| Corpus composition CSVs and newspaper figure | `data/en_decade_corpus/` |
