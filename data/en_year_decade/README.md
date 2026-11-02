# Year Sub-Corpora

This directory documents the year-level sub-corpora used for the analyses in the paper. Each year from 1800 to 1919 is a separate sub-corpus, built from the British Library's Heritage Made Digital (HMD) and Living with Machines (LwM) newspaper collections (see `data/README.md` for download links).

The text files (`en_YYYY.txt`) are too large to host on GitHub and must be generated locally — see the instructions below. The aggregation report and the corpus statistics CSVs are included in the repository.

## Directory Structure

```
en_year_corpus/
├── README.md
├── aggregation_report.txt          # Output of aggregate_by_year.py (human-readable)
├── aggregation_stats.json          # Output of aggregate_by_year.py (machine-readable)
├── year_corpus_statistics.csv      # Articles, HMD/LwM split and tokens per year
├── year_term_frequencies.csv       # Raw and per-million frequencies of the target terms per year
├── 1801/
│   └── en_1801.txt                 # Not in repo — generated locally
├── 1802/
│   └── en_1802.txt
├── ...
└── 1919/
    └── en_1919.txt
```

Each `en_YYYY.txt` contains every article published in that year, one article per line, with internal whitespace collapsed. Years with no surviving articles are not written.

## Generating the Year Sub-Corpora

### Step 1: Aggregate by year

```bash
python corpus_preparation/aggregate_by_year.py \
    --lwm /path/to/lwm-alto2txt/plaintext \
    --hmd /path/to/hmd-alto2txt/plaintext \
    --output ./data/en_year_corpus \
    --workers 16
```

The script:

1. Scans all plaintext files in both collections in a single directory traversal.
2. Extracts each article's publication year, primarily from the alto2txt filename pattern `{newspaper_id}_{YYYYMMDD}_art{number}.txt` (e.g. `0003038_18990929_art0087.txt` → 1899), falling back to the `{newspaper_id}/{YYYY}/{MMDD}/` directory structure.
3. Discards articles shorter than 50 characters or whose year cannot be determined or falls outside 1800–1919.
4. Writes each remaining article as one line to `{year}/en_{year}.txt`.
5. Writes `aggregation_report.txt` (decade roll-up, per-year table with HMD/LwM split, sparsity check) and `aggregation_stats.json`.

`--start-year` and `--end-year` change the year range; `--keep-empty-years` keeps folders for years with no articles.

### Step 2: Compute corpus statistics

```bash
python corpus_preparation/year_corpus_statistics.py \
    --corpus-dir ./data/en_year_corpus \
    --workers 8
```

This re-tokenises every year file (lower-cased, regex `\b\w+\b`, identical to `count_tokens_and_terms.py`, so year totals sum to the decade totals in `corpus_preparation/decade_term_frequencies.csv`) and writes the two CSVs below. Use `--term CANONICAL:form,form` (repeatable) to count other words.

## Statistics Files

### `year_corpus_statistics.csv`

| Column | Description |
|---|---|
| `year` | Publication year |
| `articles` | Articles written to that year's file (from `aggregation_stats.json`) |
| `articles_hmd` | Of which from Heritage Made Digital |
| `articles_lwm` | Of which from Living with Machines |
| `tokens` | Token count of `en_YYYY.txt` |

### `year_term_frequencies.csv`

| Column | Description |
|---|---|
| `year` | Publication year |
| `tokens` | Token count of `en_YYYY.txt` |
| `coffee`, `tea`, `sugar` | Raw count of the singular form |
| `coffee_all`, `tea_all`, `sugar_all` | Raw count of singular and plural forms combined (*coffee/coffees*, *tea/teas*, *sugar/sugars*) |
| `coffee_pm`, … | Singular count per million tokens |
| `coffee_all_pm`, … | Singular + plural count per million tokens |

The per-million columns are the normalised frequencies against which the year-level JSD series is correlated in the paper.

## Notes on Year-Level Coverage

Year-level slicing exposes variation in coverage that decade aggregation smooths over. The early years are much thinner than the later ones, and the HMD/LwM balance shifts over the period (HMD titles, all London-based, dominate the early decades; LwM provincial titles dominate from the 1880s). See the sparsity check in `aggregation_report.txt` and the per-year source split in `year_corpus_statistics.csv`.
