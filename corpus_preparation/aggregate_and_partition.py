#!/usr/bin/env python3
"""
Aggregate alto2txt plaintext output by decade for Word2Vec training.

alto2txt outputs plaintext files organized by newspaper title and date.
This script:
1. Scans LWM and HMD plaintext directories
2. Extracts publication year from file paths/names
3. Aggregates all text by decade (1800s, 1810s, ..., 1910s)
4. Outputs one large text file per decade for further steps

Usage:
    python aggregate_and_partition.py \
        --lwm ".../Downloads/lwm-alto2txt/plaintext" \ #change it to your path
        --hmd ".../Downloads/Daniel - hmd-alto2txt/plaintext" \ #change it to your path
        --output ./en_decade_corpus

Output structure:
    en_decade_corpus/
    ├── en_1800s.txt
    ├── en_1810s.txt
    ├── ...
    ├── en_1910s.txt
    ├── aggregation_report.txt
    └── aggregation_stats.json

"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Decades to process
DECADES = [(y, y+9) for y in range(1800, 1920, 10)]


def get_decade(year: int) -> str:
    """Convert year to decade string."""
    if year < 1800 or year > 1920:
        return None
    decade_start = (year // 10) * 10
    return f"{decade_start}s"


def extract_year_from_path(filepath: Path) -> int:
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
    # Pattern: _18990929_ (underscore + 8 digits + underscore)
    match = re.search(r'_(\d{4})(\d{2})(\d{2})_', filename)
    if match:
        year = int(match.group(1))
        if 1800 <= year <= 1920:
            return year
    
    # Priority 2: Year as directory in path /1899/
    match = re.search(r'/(\d{4})/', path_str)
    if match:
        year = int(match.group(1))
        if 1800 <= year <= 1920:
            return year
    
    # Priority 3: YYYYMMDD anywhere in filename (without underscore requirement)
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        # Validate it looks like a real date
        if 1800 <= year <= 1920 and 1 <= month <= 12 and 1 <= day <= 31:
            return year
    
    # Priority 4: Any 4-digit year in the path
    for match in re.finditer(r'(\d{4})', path_str):
        year = int(match.group(1))
        if 1800 <= year <= 1920:
            return year
    
    return None


def extract_newspaper_title(filepath: Path, base_path: Path) -> str:
    """
    Extract newspaper ID from path.
    
    British Library structure: {newspaper_id}/{YYYY}/{MMDD}/filename.txt
    Example: 0003038/1899/0929/0003038_18990929_art0087.txt
    
    The newspaper_id (e.g., 0003038) is a BL identifier.
    """
    try:
        rel_path = filepath.relative_to(base_path)
        # First component is newspaper ID
        if rel_path.parts:
            return rel_path.parts[0]
    except:
        pass
    
    # Fallback: extract from filename
    # Pattern: {newspaper_id}_{YYYYMMDD}_art{number}.txt
    match = re.match(r'^(\d+)_', filepath.name)
    if match:
        return match.group(1)
    
    return "unknown"


def process_file(filepath: Path, base_path: Path) -> dict:
    """
    Process a single plaintext file.
    Returns dict with year, text, metadata or None if invalid.
    """
    try:
        # Extract year
        year = extract_year_from_path(filepath)
        if not year:
            return None
        
        decade = get_decade(year)
        if not decade:
            return None
        
        # Read text
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            return None
        
        # Skip empty or very short files
        if not text or len(text.strip()) < 50:
            return None
        
        # Clean text: normalize whitespace, remove excessive newlines
        text = ' '.join(text.split())
        
        # Get file size for stats
        file_size = filepath.stat().st_size
        
        return {
            'decade': decade,
            'year': year,
            'text': text,
            'tokens': len(text.split()),
            'file_size': file_size,
            'newspaper': extract_newspaper_title(filepath, base_path),
            'filepath': str(filepath.name)
        }
    
    except Exception as e:
        return None


def process_file_wrapper(args):
    """Wrapper for multiprocessing."""
    filepath, base_path = args
    return process_file(filepath, base_path)


def find_text_files(base_path: Path) -> list:
    """Find all text files in directory."""
    text_files = []
    
    # Common extensions for alto2txt output
    extensions = ['.txt', '.text', '']
    
    for ext in extensions:
        if ext:
            text_files.extend(base_path.rglob(f"*{ext}"))
        else:
            # Files without extension - check if they're text
            for f in base_path.rglob("*"):
                if f.is_file() and not f.suffix:
                    text_files.append(f)
    
    # Filter to only files (not directories)
    text_files = [f for f in text_files if f.is_file()]
    
    return text_files


def aggregate_by_decade(lwm_path: Path, hmd_path: Path, output_path: Path, workers: int = 4):
    """
    Main aggregation function.
    Optimized for British Library alto2txt output (~9 million files).
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all files from both sources
    all_files = []
    sources = {}
    
    for path, name in [(lwm_path, 'LWM'), (hmd_path, 'HMD')]:
        if path and path.exists():
            logger.info(f"Scanning {name} at {path}...")
            logger.info(f"  (This may take a few minutes for large directories)")
            files = find_text_files(path)
            logger.info(f"  Found {len(files):,} text files")
            for f in files:
                all_files.append((f, path))
                sources[str(f)] = name
        elif path:
            logger.warning(f"Path does not exist: {path}")
    
    if not all_files:
        logger.error("No files found!")
        return
    
    total_files = len(all_files)
    logger.info(f"\nTotal files to process: {total_files:,}")
    logger.info(f"Estimated processing time: {total_files // 10000} - {total_files // 5000} minutes")
    
    # Initialize statistics
    stats = {
        'articles_by_decade': defaultdict(int),
        'tokens_by_decade': defaultdict(int),
        'bytes_by_decade': defaultdict(int),
        'by_source': {'LWM': defaultdict(int), 'HMD': defaultdict(int)},
        'by_newspaper': defaultdict(lambda: defaultdict(int)),
        'years': defaultdict(int),
        'files_processed': 0,
        'files_skipped': 0,
    }
    
    # Open output files for each decade
    decade_files = {}
    for start, end in DECADES:
        decade = f"{start}s"
        decade_files[decade] = open(output_path / f"en_{decade}.txt", 'w', encoding='utf-8', buffering=1024*1024)  # 1MB buffer
    
    # Process files in batches
    logger.info("\nProcessing files...")
    batch_size = 50000  # Larger batches for efficiency
    total_batches = (total_files + batch_size - 1) // batch_size
    
    start_time = datetime.now()
    
    for batch_num, batch_start in enumerate(range(0, total_files, batch_size), 1):
        batch = all_files[batch_start:batch_start + batch_size]
        batch_processed = 0
        batch_skipped = 0
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_file_wrapper, args): args[0] for args in batch}
            
            pbar = tqdm(
                as_completed(futures), 
                total=len(batch), 
                desc=f"Batch {batch_num}/{total_batches}",
                unit="files"
            )
            
            for future in pbar:
                filepath = futures[future]
                
                try:
                    result = future.result()
                    
                    if result is None:
                        batch_skipped += 1
                        stats['files_skipped'] += 1
                        continue
                    
                    batch_processed += 1
                    stats['files_processed'] += 1
                    decade = result['decade']
                    
                    # Write text to decade file
                    decade_files[decade].write(result['text'] + '\n')
                    
                    # Update statistics
                    stats['articles_by_decade'][decade] += 1
                    stats['tokens_by_decade'][decade] += result['tokens']
                    stats['bytes_by_decade'][decade] += result['file_size']
                    stats['years'][result['year']] += 1
                    
                    # Source tracking
                    source = sources.get(str(filepath), 'unknown')
                    stats['by_source'][source][decade] += 1
                    
                    # Newspaper tracking (sample to save memory)
                    if stats['files_processed'] % 100 == 0:
                        stats['by_newspaper'][result['newspaper']][decade] += 100
                    
                except Exception as e:
                    batch_skipped += 1
                    stats['files_skipped'] += 1
        
        # Progress update after each batch
        elapsed = (datetime.now() - start_time).total_seconds()
        files_done = batch_start + len(batch)
        rate = files_done / elapsed if elapsed > 0 else 0
        eta_seconds = (total_files - files_done) / rate if rate > 0 else 0
        
        logger.info(f"  Batch {batch_num} complete: {batch_processed:,} processed, {batch_skipped:,} skipped")
        logger.info(f"  Overall progress: {files_done:,}/{total_files:,} ({100*files_done/total_files:.1f}%)")
        logger.info(f"  Rate: {rate:.0f} files/sec, ETA: {eta_seconds/60:.1f} minutes")
    
    # Close all output files
    for f in decade_files.values():
        f.close()
    
    # Generate report
    generate_report(stats, output_path)
    
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nDone! Total time: {total_time/60:.1f} minutes")
    logger.info(f"Output written to {output_path}")
    
    return stats


def generate_report(stats: dict, output_path: Path):
    """Generate comprehensive statistics report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("BRITISH LIBRARY CORPUS AGGREGATION REPORT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"Files processed: {stats['files_processed']:,}")
    lines.append(f"Files skipped: {stats['files_skipped']:,}")
    
    # By decade summary
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY BY DECADE")
    lines.append("=" * 80)
    lines.append(f"{'Decade':<10} {'Articles':>12} {'Tokens':>15} {'Size (MB)':>12} {'LWM':>10} {'HMD':>10}")
    lines.append("-" * 80)
    
    total_articles = 0
    total_tokens = 0
    total_bytes = 0
    
    for start, _ in DECADES:
        decade = f"{start}s"
        articles = stats['articles_by_decade'].get(decade, 0)
        tokens = stats['tokens_by_decade'].get(decade, 0)
        bytes_size = stats['bytes_by_decade'].get(decade, 0)
        lwm = stats['by_source']['LWM'].get(decade, 0)
        hmd = stats['by_source']['HMD'].get(decade, 0)
        
        total_articles += articles
        total_tokens += tokens
        total_bytes += bytes_size
        
        lines.append(f"{decade:<10} {articles:>12,} {tokens:>15,} {bytes_size/1024/1024:>12.1f} {lwm:>10,} {hmd:>10,}")
    
    lines.append("-" * 80)
    lines.append(f"{'TOTAL':<10} {total_articles:>12,} {total_tokens:>15,} {total_bytes/1024/1024:>12.1f}")

    # Top newspapers
    lines.append("\n" + "=" * 80)
    lines.append("TOP 20 NEWSPAPERS BY ARTICLE COUNT")
    lines.append("=" * 80)
    
    newspaper_totals = {np: sum(decades.values()) for np, decades in stats['by_newspaper'].items()}
    top_newspapers = sorted(newspaper_totals.items(), key=lambda x: -x[1])[:20]
    
    for newspaper, count in top_newspapers:
        lines.append(f"  {newspaper}: {count:,}")
    
    # Year distribution
    lines.append("\n" + "=" * 80)
    lines.append("COVERAGE BY YEAR (sample)")
    lines.append("=" * 80)
    
    years = stats['years']
    if years:
        min_year = min(years.keys())
        max_year = max(years.keys())
        lines.append(f"Year range: {min_year} - {max_year}")
        
        # Show sparse years
        sparse_years = [y for y in range(1800, 1921) if years.get(y, 0) < 100]
        if sparse_years:
            lines.append(f"\nYears with <100 articles: {sparse_years[:20]}...")
    
    # Print and save report
    report = '\n'.join(lines)
    print(report)
    
    with open(output_path / "aggregation_report.txt", 'w') as f:
        f.write(report)
    
    # Save JSON stats (convert defaultdicts)
    def convert_defaultdict(d):
        if isinstance(d, defaultdict):
            return {k: convert_defaultdict(v) for k, v in d.items()}
        return d
    
    json_stats = {
        'articles_by_decade': dict(stats['articles_by_decade']),
        'tokens_by_decade': dict(stats['tokens_by_decade']),
        'bytes_by_decade': dict(stats['bytes_by_decade']),
        'by_source': {k: dict(v) for k, v in stats['by_source'].items()},
        'files_processed': stats['files_processed'],
        'files_skipped': stats['files_skipped'],
        'generated': datetime.now().isoformat()
    }
    
    with open(output_path / "aggregation_stats.json", 'w') as f:
        json.dump(json_stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate alto2txt plaintext by decade for Word2Vec training"
    )
    parser.add_argument(
        '--lwm',
        type=Path,
        default=None,
        help="Path to LWM plaintext directory"
    )
    parser.add_argument(
        '--hmd',
        type=Path,
        default=None,
        help="Path to HMD plaintext directory"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./en_decade_corpus'),
        help="Output directory"
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    
    args = parser.parse_args()
    
    if not args.lwm and not args.hmd:
        parser.error("At least one of --lwm or --hmd must be provided")
    
    aggregate_by_decade(args.lwm, args.hmd, args.output, args.workers)


if __name__ == "__main__":
    main()
