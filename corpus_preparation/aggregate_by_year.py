"""
Aggregate alto2txt plaintext output by YEAR.
 
alto2txt outputs plaintext files organized by newspaper title and date.
This script:
1. Scans LWM and HMD plaintext directories
2. Extracts publication year from file paths/names
3. Aggregates all text by year (1800, 1801, ..., 1919)
4. Writes one folder per year, each containing that year's plaintext
 
Usage:
    python aggregate_by_year.py \
        --lwm /path/to/lwm-alto2txt/plaintext \
        --hmd /path/to/hmd-alto2txt/plaintext \
        --output ../data/en_year_corpus \
        --workers 16
 
Output structure:
    en_year_corpus/
    ├── 1800/
    │   └── en_1800.txt
    ├── 1801/
    │   └── en_1801.txt
    ├── ...
    ├── 1919/
    │   └── en_1919.txt
    ├── aggregation_report.txt
    └── aggregation_stats.json
 
Each year file contains one article per line (internal whitespace collapsed).
Years with no surviving articles are removed at the end unless
--keep-empty-years is passed.
"""
 
import re
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
 
# Default corpus bounds (inclusive). Overridable on the command line.
DEFAULT_START_YEAR = 1800
DEFAULT_END_YEAR = 1919
 
TEXT_SUFFIXES = {'.txt', '.text', ''}
 
 
def extract_year_from_path(filepath: Path, start_year: int, end_year: int) -> int:
    """
    Extract publication year from file path.
 
    Optimized for British Library alto2txt output format:
    - Pattern: {newspaper_id}_{YYYYMMDD}_art{number}.txt
    - Example: 0003038_18990929_art0087.txt
    - Directory structure: {newspaper_id}/{YYYY}/{MMDD}/
    """
    filename = filepath.name
    path_str = str(filepath)
 
    # Priority 1: YYYYMMDD in filename (most reliable for this dataset)
    match = re.search(r'_(\d{4})(\d{2})(\d{2})_', filename)
    if match:
        year = int(match.group(1))
        if start_year <= year <= end_year:
            return year
 
    # Priority 2: Year as directory in path /1899/
    match = re.search(r'/(\d{4})/', path_str)
    if match:
        year = int(match.group(1))
        if start_year <= year <= end_year:
            return year
 
    # Priority 3: YYYYMMDD anywhere in filename
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        if start_year <= year <= end_year and 1 <= month <= 12 and 1 <= day <= 31:
            return year
 
    # Priority 4: Any 4-digit year in the path
    for match in re.finditer(r'(\d{4})', path_str):
        year = int(match.group(1))
        if start_year <= year <= end_year:
            return year
 
    return None
 
 
def extract_newspaper_title(filepath: Path, base_path: Path) -> str:
    """
    Extract newspaper ID from path.
 
    British Library structure: {newspaper_id}/{YYYY}/{MMDD}/filename.txt
    Example: 0003038/1899/0929/0003038_18990929_art0087.txt
    """
    try:
        rel_path = filepath.relative_to(base_path)
        if rel_path.parts:
            return rel_path.parts[0]
    except Exception:
        pass
 
    match = re.match(r'^(\d+)_', filepath.name)
    if match:
        return match.group(1)
 
    return "unknown"
 
 
def process_file(filepath: Path, base_path: Path, start_year: int, end_year: int) -> dict:
    """
    Process a single plaintext file.
    Returns dict with year, text, metadata or None if invalid.
    """
    try:
        year = extract_year_from_path(filepath, start_year, end_year)
        if not year:
            return None
 
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception:
            return None
 
        if not text or len(text.strip()) < 50:
            return None
 
        # Normalize whitespace so each article occupies exactly one line
        text = ' '.join(text.split())
 
        file_size = filepath.stat().st_size
 
        return {
            'year': year,
            'text': text,
            'tokens': len(text.split()),
            'file_size': file_size,
            'newspaper': extract_newspaper_title(filepath, base_path),
            'filepath': str(filepath.name)
        }
 
    except Exception:
        return None
 
 
def process_file_wrapper(args):
    """Wrapper for multiprocessing."""
    filepath, base_path, start_year, end_year = args
    return process_file(filepath, base_path, start_year, end_year)
 
 
def find_text_files(base_path: Path) -> list:
    """Find all text files under base_path in a single directory traversal."""
    return [
        f for f in base_path.rglob("*")
        if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES
    ]
 
 
def aggregate_by_year(lwm_path: Path,
                      hmd_path: Path,
                      output_path: Path,
                      workers: int = 4,
                      start_year: int = DEFAULT_START_YEAR,
                      end_year: int = DEFAULT_END_YEAR,
                      keep_empty_years: bool = False):
    """
    Main aggregation function.
    Optimized for British Library alto2txt output (~9 million files).
    """
    output_path.mkdir(parents=True, exist_ok=True)
 
    years = list(range(start_year, end_year + 1))
 
    # Collect all files from both sources
    all_files = []
    sources = {}
 
    for path, name in [(lwm_path, 'LWM'), (hmd_path, 'HMD')]:
        if path and path.exists():
            logger.info(f"Scanning {name} at {path}...")
            logger.info("  (This may take a few minutes for large directories)")
            files = find_text_files(path)
            logger.info(f"  Found {len(files):,} text files")
            for f in files:
                all_files.append((f, path, start_year, end_year))
                sources[str(f)] = name
        elif path:
            logger.warning(f"Path does not exist: {path}")
 
    if not all_files:
        logger.error("No files found!")
        return
 
    total_files = len(all_files)
    logger.info(f"\nTotal files to process: {total_files:,}")
    logger.info(f"Year range: {start_year}-{end_year} ({len(years)} years)")
 
    # Initialize statistics
    stats = {
        'articles_by_year': defaultdict(int),
        'tokens_by_year': defaultdict(int),
        'bytes_by_year': defaultdict(int),
        'by_source': {'LWM': defaultdict(int), 'HMD': defaultdict(int)},
        'by_newspaper': defaultdict(lambda: defaultdict(int)),
        'files_processed': 0,
        'files_skipped': 0,
    }
 
    # Create one folder per year and open its output file.
    # Buffer is kept modest because ~120 files are open simultaneously.
    year_files = {}
    for year in years:
        year_dir = output_path / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        year_files[year] = open(
            year_dir / f"en_{year}.txt",
            'w', encoding='utf-8', buffering=256 * 1024
        )
 
    logger.info("\nProcessing files...")
    batch_size = 50000
    total_batches = (total_files + batch_size - 1) // batch_size
    start_time = datetime.now()
 
    # The process pool is created once rather than per batch.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch_num, batch_start in enumerate(range(0, total_files, batch_size), 1):
            batch = all_files[batch_start:batch_start + batch_size]
            batch_processed = 0
            batch_skipped = 0
 
            results = executor.map(process_file_wrapper, batch, chunksize=200)
 
            pbar = tqdm(
                results,
                total=len(batch),
                desc=f"Batch {batch_num}/{total_batches}",
                unit="files"
            )
 
            for i, result in enumerate(pbar):
                if result is None:
                    batch_skipped += 1
                    stats['files_skipped'] += 1
                    continue
 
                batch_processed += 1
                stats['files_processed'] += 1
                year = result['year']
 
                # Write one article per line into that year's file
                year_files[year].write(result['text'] + '\n')
 
                stats['articles_by_year'][year] += 1
                stats['tokens_by_year'][year] += result['tokens']
                stats['bytes_by_year'][year] += result['file_size']
 
                filepath = batch[i][0]
                source = sources.get(str(filepath), 'unknown')
                if source in stats['by_source']:
                    stats['by_source'][source][year] += 1
 
                # Newspaper tracking (sampled to save memory)
                if stats['files_processed'] % 100 == 0:
                    stats['by_newspaper'][result['newspaper']][year] += 100
 
            elapsed = (datetime.now() - start_time).total_seconds()
            files_done = min(batch_start + len(batch), total_files)
            rate = files_done / elapsed if elapsed > 0 else 0
            eta_seconds = (total_files - files_done) / rate if rate > 0 else 0
 
            logger.info(f"  Batch {batch_num} complete: {batch_processed:,} processed, {batch_skipped:,} skipped")
            logger.info(f"  Overall progress: {files_done:,}/{total_files:,} ({100*files_done/total_files:.1f}%)")
            logger.info(f"  Rate: {rate:.0f} files/sec, ETA: {eta_seconds/60:.1f} minutes")
 
    for f in year_files.values():
        f.close()
 
    # Remove year folders that ended up with no articles
    if not keep_empty_years:
        removed = []
        for year in years:
            if stats['articles_by_year'].get(year, 0) == 0:
                shutil.rmtree(output_path / str(year), ignore_errors=True)
                removed.append(year)
        if removed:
            logger.info(f"\nRemoved {len(removed)} empty year folders: {removed}")
 
    generate_report(stats, output_path, years)
 
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nDone! Total time: {total_time/60:.1f} minutes")
    logger.info(f"Output written to {output_path}")
 
    return stats
 
 
def generate_report(stats: dict, output_path: Path, years: list):
    """Generate comprehensive statistics report."""
 
    lines = []
    lines.append("=" * 80)
    lines.append("BRITISH LIBRARY CORPUS AGGREGATION REPORT (BY YEAR)")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"Files processed: {stats['files_processed']:,}")
    lines.append(f"Files skipped: {stats['files_skipped']:,}")
 
    # Decade roll-up first, for readability
    lines.append("\n" + "=" * 80)
    lines.append("ROLL-UP BY DECADE")
    lines.append("=" * 80)
    lines.append(f"{'Decade':<10} {'Articles':>12} {'Tokens':>15} {'Size (MB)':>12}")
    lines.append("-" * 80)
 
    decade_articles = defaultdict(int)
    decade_tokens = defaultdict(int)
    decade_bytes = defaultdict(int)
    for year in years:
        d = (year // 10) * 10
        decade_articles[d] += stats['articles_by_year'].get(year, 0)
        decade_tokens[d] += stats['tokens_by_year'].get(year, 0)
        decade_bytes[d] += stats['bytes_by_year'].get(year, 0)
 
    for d in sorted(decade_articles):
        lines.append(
            f"{str(d) + 's':<10} {decade_articles[d]:>12,} "
            f"{decade_tokens[d]:>15,} {decade_bytes[d]/1024/1024:>12.1f}"
        )
 
    # Full year-level table
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY BY YEAR")
    lines.append("=" * 80)
    lines.append(f"{'Year':<8} {'Articles':>12} {'Tokens':>15} {'Size (MB)':>12} {'LWM':>10} {'HMD':>10}")
    lines.append("-" * 80)
 
    total_articles = 0
    total_tokens = 0
    total_bytes = 0
 
    for year in years:
        articles = stats['articles_by_year'].get(year, 0)
        tokens = stats['tokens_by_year'].get(year, 0)
        bytes_size = stats['bytes_by_year'].get(year, 0)
        lwm = stats['by_source']['LWM'].get(year, 0)
        hmd = stats['by_source']['HMD'].get(year, 0)
 
        total_articles += articles
        total_tokens += tokens
        total_bytes += bytes_size
 
        lines.append(
            f"{year:<8} {articles:>12,} {tokens:>15,} "
            f"{bytes_size/1024/1024:>12.1f} {lwm:>10,} {hmd:>10,}"
        )
 
    lines.append("-" * 80)
    lines.append(f"{'TOTAL':<8} {total_articles:>12,} {total_tokens:>15,} {total_bytes/1024/1024:>12.1f}")
 
    # Sparsity check: year-level slicing exposes thin years that
    # decade-level aggregation hides.
    empty_years = [y for y in years if stats['articles_by_year'].get(y, 0) == 0]
    thin_years = [y for y in years if 0 < stats['articles_by_year'].get(y, 0) < 100]
 
    lines.append("\n" + "=" * 80)
    lines.append("SPARSITY CHECK")
    lines.append("=" * 80)
    lines.append(f"Years with 0 articles ({len(empty_years)}): {empty_years}")
    lines.append(f"Years with <100 articles ({len(thin_years)}): {thin_years}")
 
    # Top newspapers
    lines.append("\n" + "=" * 80)
    lines.append("TOP 20 NEWSPAPERS BY ARTICLE COUNT (sampled)")
    lines.append("=" * 80)
 
    newspaper_totals = {np: sum(yrs.values()) for np, yrs in stats['by_newspaper'].items()}
    for newspaper, count in sorted(newspaper_totals.items(), key=lambda x: -x[1])[:20]:
        lines.append(f"  {newspaper}: {count:,}")
 
    report = '\n'.join(lines)
    print(report)
 
    with open(output_path / "aggregation_report.txt", 'w') as f:
        f.write(report)
 
    json_stats = {
        'articles_by_year': {str(k): v for k, v in sorted(stats['articles_by_year'].items())},
        'tokens_by_year': {str(k): v for k, v in sorted(stats['tokens_by_year'].items())},
        'bytes_by_year': {str(k): v for k, v in sorted(stats['bytes_by_year'].items())},
        'by_source': {
            src: {str(k): v for k, v in sorted(yrs.items())}
            for src, yrs in stats['by_source'].items()
        },
        'files_processed': stats['files_processed'],
        'files_skipped': stats['files_skipped'],
        'generated': datetime.now().isoformat()
    }
 
    with open(output_path / "aggregation_stats.json", 'w') as f:
        json.dump(json_stats, f, indent=2)
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate alto2txt plaintext by year, one folder per year"
    )
    parser.add_argument('--lwm', type=Path, default=None,
                        help="Path to LWM plaintext directory")
    parser.add_argument('--hmd', type=Path, default=None,
                        help="Path to HMD plaintext directory")
    parser.add_argument('--output', '-o', type=Path, default=Path('./en_year_corpus'),
                        help="Output directory")
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument('--start-year', type=int, default=DEFAULT_START_YEAR,
                        help=f"First year to include (default {DEFAULT_START_YEAR})")
    parser.add_argument('--end-year', type=int, default=DEFAULT_END_YEAR,
                        help=f"Last year to include, inclusive (default {DEFAULT_END_YEAR})")
    parser.add_argument('--keep-empty-years', action='store_true',
                        help="Keep folders for years with no articles")
 
    args = parser.parse_args()
 
    if not args.lwm and not args.hmd:
        parser.error("At least one of --lwm or --hmd must be provided")
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
 
    aggregate_by_year(
        args.lwm, args.hmd, args.output, args.workers,
        args.start_year, args.end_year, args.keep_empty_years
    )
 
 
if __name__ == "__main__":
    main()
