#!/usr/bin/env python3
"""
Build the decade sub-corpora from the year sub-corpora.

Concatenates the year files written by aggregate_by_year.py into one file per
decade, in year order:

    en_year_corpus/1801/en_1801.txt  ┐
    ...                              ├──>  en_decade_corpus/1800.txt
    en_year_corpus/1809/en_1809.txt  ┘

The decade files are named by the decade's first year (1800.txt, 1810.txt, ...,
1910.txt), which is what domain_adaptation/ and embedding_extraction/ read.

Every decade file holds exactly the articles of its years, one per line, with the
same filtering as aggregate_by_year.py. Lines are ordered by year, so the line
order is not identical to that of the original decade files; content is.

Usage:
    python build_decade_corpus.py \
        --year-dir ../data/en_year_corpus \
        --output ../data/en_decade_corpus
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path


def find_year_files(year_dir: Path):
    """Return {year: path} for every en_YYYY.txt under year_dir."""
    files = {}
    for p in year_dir.glob("**/en_*.txt"):
        m = re.fullmatch(r"en_(\d{4})\.txt", p.name)
        if m:
            files[int(m.group(1))] = p
    return files


def main():
    ap = argparse.ArgumentParser(description="Concatenate year sub-corpora into decade sub-corpora")
    ap.add_argument("--year-dir", type=Path, required=True,
                    help="en_year_corpus directory written by aggregate_by_year.py")
    ap.add_argument("--output", "-o", type=Path, required=True,
                    help="directory for the decade files")
    ap.add_argument("--start-year", type=int, default=1800)
    ap.add_argument("--end-year", type=int, default=1919)
    args = ap.parse_args()

    year_files = find_year_files(args.year_dir)
    if not year_files:
        raise SystemExit(f"No en_YYYY.txt files found under {args.year_dir}")

    by_decade = defaultdict(list)
    for year in sorted(year_files):
        if args.start_year <= year <= args.end_year:
            by_decade[year // 10 * 10].append(year)

    args.output.mkdir(parents=True, exist_ok=True)
    for decade, years in sorted(by_decade.items()):
        out_path = args.output / f"{decade}.txt"
        lines = 0
        with open(out_path, "wb") as out:
            for year in years:
                with open(year_files[year], "rb") as src:
                    while chunk := src.read(16 * 1024 * 1024):
                        out.write(chunk)
                        lines += chunk.count(b"\n")
        print(f"{out_path.name}: {len(years)} years ({years[0]}-{years[-1]}), {lines:,} articles")


if __name__ == "__main__":
    main()
