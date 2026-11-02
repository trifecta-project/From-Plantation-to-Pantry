# Decade Sub-Corpora and Corpus Metadata

This directory documents the composition of the newspaper corpus at decade level: corpus metadata extracted from the HMD and LwM XML metadata files, a newspaper inventory, and a figure showing how the largest titles contribute over time. It also describes the decade-level text files, which are built from the year sub-corpora and are the input to the domain adaptation of MacBERTh (see `domain_adaptation/`).

The year-level sub-corpora used for the semantic change analysis are documented separately in `data/en_year_corpus/`.

The decade text files are too large to host on GitHub and must be generated locally. The metadata CSVs, the figure and the script that draws it are included.

## Directory Structure

```
en_decade_corpus/
├── README.md
├── corpus_overview.csv                     # Per-decade corpus composition (from XML metadata)
├── newspaper_inventory.csv                 # All 120 newspaper titles in the corpus
├── top30_newspapers_by_article_count.csv   # The 30 largest titles by article count
├── articles_by_newspaper_decade.csv        # Articles per title per decade (input to the figure)
├── plot_articles_by_newspaper.py           # Draws articles_by_newspaper.png
├── articles_by_newspaper.png               # Articles per decade for the 10 largest titles
├── 1800.txt                                # Not in repo — generated locally
├── 1810.txt
├── ...
└── 1910.txt
```

## Corpus Metadata Files

### `corpus_overview.csv`

Per-decade summary of the corpus composition, derived from the output of `corpus_preparation/extract_corpus_metadata.py`, which reads the `*_metadata.xml` files of both collections.

| Column | Description |
|---|---|
| `Decade` | Decade label (e.g. `1800s`) |
| `Articles` | Number of articles |
| `Words` | Word count as recorded in the XML metadata (not re-tokenised) |
| `HMD` | Articles from Heritage Made Digital |
| `LwM` | Articles from Living with Machines |
| `Papers` | Number of distinct newspaper titles active in the decade |
| `London_percent` | Percentage of articles from London-based titles |
| `OCR_mean` | Mean OCR confidence score (0–1) |
| `OCR_median` | Median OCR confidence score |

The file contains a `1920s` row (64,669 articles). These articles are present in the source collections but fall outside the 1800–1919 period and are excluded from all sub-corpora.

### `newspaper_inventory.csv`

Inventory of all 120 newspaper titles in the source collections.

| Column | Description |
|---|---|
| `BL_ID` | British Library identifier |
| `Newspaper_Title` | Title as it appears in the metadata |
| `Location` | Place of publication |
| `Collection` | Source collection: `HMD` or `LwM` |
| `Articles` | Total number of articles from this title |

The inventory contains 21 HMD and 99 LwM titles. By place of publication, 17 titles (2,530,729 articles) are London-based and 103 are provincial; the London titles all come from HMD, while HMD also includes some provincial titles, such as the Liverpool papers. For normalised title names and full documentation of the LwM titles, see [Ridge and Pedrazzini (2024)](https://livingwithmachines.ac.uk/public-domain-newspaper-titles-in-living-with-machines/).

### `top30_newspapers_by_article_count.csv`

The 30 largest titles by article count, with BL ID, collection, location and title. The largest single contributor is *The Sun* (BL ID 0002194, London, HMD) with 1,625,117 articles, about 17% of all articles in the source collections.

## Newspaper Counts

### `articles_by_newspaper_decade.csv`

Number of articles per newspaper title per decade, counted from the plaintext file paths of both collections (`{newspaper_id}/{YYYY}/{MMDD}/...`) before any filtering. It is the input to the figure below.

| Column | Description |
|---|---|
| `newspaper_id` | British Library identifier |
| `source` | Collection directory the title was found in (`HMD` or `LWM`) |
| `1800s` … `1910s` | Articles in each decade |
| `total` | All articles of the title, including the 1920s (which have no column of their own) |

Because it counts files before filtering, its totals (9,326,201 articles, of which 64,669 from the 1920s) are close to the XML-metadata figures in `newspaper_inventory.csv` rather than to the aggregated corpus. It has 121 identifiers against the inventory's 120 titles: `0002090` (9,997 articles, 1850s) is counted separately here, whereas the metadata includes it under *The Liverpool Standard and General Commercial Advertiser* (`0002088`). A title found under both collection directories is assigned a single `source`, so for the per-collection split use `data/en_year_corpus/aggregation_stats.json` instead.

## Articles per Newspaper

`articles_by_newspaper.png` shows the number of articles per decade for the ten largest titles, one panel per title, with every other title drawn behind it in grey on a shared vertical scale. The dashed line marks the decade boundary (1860s/1870s) at which the dominant source collection switches from HMD to LwM, computed from the per-year source counts in `data/en_year_corpus/aggregation_stats.json`. The figure makes visible how strongly the early decades depend on a few London titles, *The Sun* above all, and how the corpus becomes a broad spread of provincial titles from the 1870s onward.

To redraw it:

```bash
python plot_articles_by_newspaper.py \
    --csv articles_by_newspaper_decade.csv \
    --top-n 10 \
    --name-map top30_newspapers_by_article_count.csv \
    --mark-source-transition \
    --aggregation-stats ../en_year_corpus/aggregation_stats.json \
    --out articles_by_newspaper.png
```

`--name-map` labels the panels with newspaper titles instead of BL identifiers, taken from `top30_newspapers_by_article_count.csv`. `--free-y` gives each panel its own vertical scale, and `--no-context` drops the grey background lines. Run `python plot_articles_by_newspaper.py --help` for all options.

## Article Counts at Different Stages

The corpus statistics in this repository come from two sources and differ for that reason:

| Source | Articles | Covers |
|---|---|---|
| XML metadata (`newspaper_inventory.csv`, `corpus_overview.csv`) | 9,326,170 | Every article in both collections, including the 1920s |
| XML metadata, 1800s–1910s only | 9,261,501 | As above, without the 1920s |
| File paths (`articles_by_newspaper_decade.csv`) | 9,326,201 | Every plaintext file, before filtering |
| Aggregation (`data/en_year_corpus/aggregation_stats.json`) | 7,927,079 | Articles written to the sub-corpora |

The aggregation step drops files that are empty, shorter than 50 characters, or whose publication year cannot be determined or falls outside 1800–1919 (1,399,131 files skipped in our run). Token counts likewise differ between the XML metadata (`Words`) and the whitespace-delimited counts of the aggregation.

## Decade Text Files

One plain-text file per decade, named by the decade's first year, containing all articles from that decade with one article per line. They are built by concatenating the year sub-corpora, so they contain exactly the same articles:

```bash
python corpus_preparation/build_decade_corpus.py \
    --year-dir ./data/en_year_corpus \
    --output ./data/en_decade_corpus
```

Expected contents (decade totals of `data/en_year_corpus/aggregation_stats.json`; tokens are whitespace-delimited):

| File | Years | Articles | Tokens |
|---|---|---|---|
| `1800.txt` | 1801–1809 | 129,303 | 137,760,980 |
| `1810.txt` | 1810–1819 | 163,987 | 186,315,494 |
| `1820.txt` | 1820–1829 | 209,329 | 235,859,159 |
| `1830.txt` | 1830–1839 | 169,439 | 181,018,792 |
| `1840.txt` | 1840–1849 | 498,147 | 593,285,339 |
| `1850.txt` | 1850–1859 | 763,035 | 657,024,751 |
| `1860.txt` | 1860–1869 | 831,504 | 639,663,893 |
| `1870.txt` | 1870–1879 | 708,228 | 529,158,049 |
| `1880.txt` | 1880–1889 | 1,175,959 | 850,837,590 |
| `1890.txt` | 1890–1899 | 1,326,660 | 732,560,675 |
| `1900.txt` | 1900–1909 | 1,191,786 | 568,077,287 |
| `1910.txt` | 1910–1919 | 759,702 | 312,531,234 |
| **Total** | | **7,927,079** | **5,624,093,243** |
