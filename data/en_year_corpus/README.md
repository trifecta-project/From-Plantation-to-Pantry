# Year Sub-Corpora

This directory documents the year-level sub-corpora used for the semantic change analysis in the paper. Each year from 1801 to 1919 is a separate sub-corpus, built from the British Library's Heritage Made Digital (HMD) and Living with Machines (LwM) newspaper collections (see `data/README.md` for download links).

The text files are too large to host on GitHub and must be generated locally; the statistics of our run are included.

## Directory Structure

```
en_year_corpus/
├── README.md
├── aggregation_stats.json      # Articles, tokens and HMD/LwM split per year
├── 1801/
│   └── en_1801.txt             # Not in repo — generated locally
├── 1802/
│   └── en_1802.txt
├── ...
└── 1919/
    └── en_1919.txt
```

Each `en_YYYY.txt` contains every article published in that year, one article per line, with internal whitespace collapsed.

## Generating the Year Sub-Corpora

```bash
python corpus_preparation/aggregate_by_year.py \
    --lwm /path/to/lwm-alto2txt/plaintext \
    --hmd /path/to/hmd-alto2txt/plaintext \
    --output ./data/en_year_corpus \
    --workers 16
```

See `corpus_preparation/README.md` for what the script does. Besides the year folders it writes `aggregation_stats.json` (kept here) and `aggregation_report.txt` (kept in `corpus_preparation/`).

The decade sub-corpora in `data/en_decade_corpus/` are built from these year files with `corpus_preparation/build_decade_corpus.py`.

## `aggregation_stats.json`

| Key | Description |
|---|---|
| `articles_by_year` | Number of articles per year |
| `tokens_by_year` | Whitespace-delimited tokens per year |
| `bytes_by_year` | Size of the source plaintext files per year, in bytes |
| `by_source` | Articles per year from each collection, under `HMD` and `LWM` |
| `files_processed` | Files written to the sub-corpora |
| `files_skipped` | Files discarded as empty, shorter than 50 characters, or undatable |
| `generated` | Timestamp of the run |

Our run wrote 7,927,079 articles (5,624,093,243 tokens) to 119 year sub-corpora and skipped 1,399,131 files. Every year from 1801 to 1919 has articles.

## Coverage Across Years

Year-level slicing exposes variation in coverage that decade aggregation smooths over. The early years are much thinner than the later ones: 10,000–26,000 articles per year from 1801 to 1839 (median about 17,000), against 69,000–159,000 per year from 1880 to 1909 (median about 122,000), with numbers falling again in the 1910s (42,000–109,000). The source balance shifts over the period: HMD titles, mostly London-based, supply 99% of articles from 1801 to 1839; LwM titles outnumber HMD titles from 1870, and from 1872 onward the corpus is entirely LwM.
