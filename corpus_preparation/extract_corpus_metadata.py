"""
extract_corpus_metadata.py

Extract metadata from HMD and LwM alto2txt XML metadata files to produce
a systematic corpus overview for the paper.

Usage:
    python extract_corpus_metadata.py \
        --hmd /path/to/hmd-alto2txt/metadata \
        --lwm /path/to/lwm-alto2txt/metadata \
        --output ./corpus_metadata

    # Quick test with 1000 files:
    python extract_corpus_metadata.py --hmd ... --lwm ... --sample 1000

Performance note:
    The bottleneck is disk I/O — reading millions of small XML files.
    The actual parsing and aggregation are fast. With --workers 8 on
    a machine with SSD, expect ~30-90 minutes for the full corpus
    (~9 million files). The --sample flag is useful for testing.
"""

import os
import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import statistics


def parse_metadata_xml(xml_path):
    """
    Parse a single alto2txt metadata XML file.
    Returns dict with extracted fields, or None if parsing fails.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        pub = root.find('.//publication')
        if pub is None:
            return None
        
        pub_id = pub.get('id', '')
        
        source_el = pub.find('source')
        source = source_el.text if source_el is not None and source_el.text else ''
        
        title_el = pub.find('title')
        title = title_el.text if title_el is not None and title_el.text else ''
        title = title.strip().rstrip(':').strip()
        
        location_el = pub.find('location')
        location = location_el.text if location_el is not None and location_el.text else ''
        
        issue = pub.find('.//issue')
        if issue is None:
            return None
        
        date_el = issue.find('date')
        date_str = date_el.text if date_el is not None and date_el.text else ''
        
        year = None
        if date_str and len(date_str) >= 4:
            try:
                year = int(date_str[:4])
            except ValueError:
                return None
        
        if year is None or year < 1800 or year > 1920:
            return None
        
        decade = (year // 10) * 10
        
        item = issue.find('.//item')
        if item is None:
            return None
        
        item_type_el = item.find('item_type')
        item_type = item_type_el.text if item_type_el is not None and item_type_el.text else ''
        
        word_count_el = item.find('word_count')
        word_count = 0
        if word_count_el is not None and word_count_el.text:
            try:
                word_count = int(word_count_el.text)
            except ValueError:
                word_count = 0
        
        ocr_mean_el = item.find('ocr_quality_mean')
        ocr_mean = None
        if ocr_mean_el is not None and ocr_mean_el.text:
            try:
                ocr_mean = float(ocr_mean_el.text)
            except ValueError:
                ocr_mean = None
        
        is_hmd = 'Heritage Made Digital' in source or 'HMD' in source
        collection = 'HMD' if is_hmd else 'LwM'
        
        location_lower = location.lower()
        is_london = 'london' in location_lower
        
        return {
            'pub_id': pub_id,
            'title': title,
            'location': location,
            'is_london': is_london,
            'date': date_str,
            'year': year,
            'decade': decade,
            'item_type': item_type,
            'word_count': word_count,
            'ocr_mean': ocr_mean,
            'collection': collection,
        }
    
    except (ET.ParseError, Exception):
        return None


def parse_wrapper(xml_path):
    return parse_metadata_xml(xml_path)


def find_xml_files(base_path):
    xml_files = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith('_metadata.xml'):
                xml_files.append(os.path.join(root, f))
    return xml_files


def main():
    parser = argparse.ArgumentParser(
        description='Extract corpus metadata from HMD and LwM XML files'
    )
    parser.add_argument('--hmd', type=str, default=None,
                        help='Path to HMD metadata directory')
    parser.add_argument('--lwm', type=str, default=None,
                        help='Path to LwM metadata directory')
    parser.add_argument('--output', type=str, default='./corpus_metadata',
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--sample', type=int, default=0,
                        help='Process only N files (0 = all, for testing)')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    # Collect all XML files
    all_xml = []
    for path, name in [(args.hmd, 'HMD'), (args.lwm, 'LwM')]:
        if path and os.path.exists(path):
            print(f"Scanning {name} metadata at {path}...")
            files = find_xml_files(path)
            print(f"  Found {len(files):,} metadata XML files")
            all_xml.extend(files)
        elif path:
            print(f"Warning: path not found: {path}")
    
    if not all_xml:
        print("No XML files found!")
        return
    
    if args.sample > 0:
        import random
        random.seed(42)
        all_xml = random.sample(all_xml, min(args.sample, len(all_xml)))
        print(f"Sampling {len(all_xml)} files for testing")
    
    print(f"\nTotal files to process: {len(all_xml):,}")
    
    # Parse all XML files in parallel
    results = []
    failed = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(parse_wrapper, f): f for f in all_xml}
        
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Parsing XML", unit="files"):
            result = future.result()
            if result is not None:
                results.append(result)
            else:
                failed += 1
    
    print(f"\nParsed: {len(results):,} articles")
    print(f"Failed/skipped: {failed:,}")
    
    # =========================================================================
    # Aggregate by decade
    # =========================================================================
    
    decade_stats = defaultdict(lambda: {
        'articles': 0,
        'total_words': 0,
        'hmd_articles': 0,
        'lwm_articles': 0,
        'newspapers': Counter(),  # title -> article count
        'newspaper_ids': {},      # title -> pub_id
        'newspaper_locations': {},  # title -> location
        'newspaper_collections': {},  # title -> collection
        'london_articles': 0,
        'provincial_articles': 0,
        'item_types': Counter(),
        'ocr_values': [],
    })
    
    for r in results:
        d = decade_stats[r['decade']]
        d['articles'] += 1
        d['total_words'] += r['word_count']
        d['newspapers'][r['title']] += 1
        d['newspaper_ids'][r['title']] = r['pub_id']
        d['newspaper_locations'][r['title']] = r['location']
        d['newspaper_collections'][r['title']] = r['collection']
        d['item_types'][r['item_type']] += 1
        
        if r['collection'] == 'HMD':
            d['hmd_articles'] += 1
        else:
            d['lwm_articles'] += 1
        
        if r['is_london']:
            d['london_articles'] += 1
        else:
            d['provincial_articles'] += 1
        
        if r['ocr_mean'] is not None:
            d['ocr_values'].append(r['ocr_mean'])
    
    # =========================================================================
    # Output 1: Main corpus overview table
    # =========================================================================
    
    overview_path = os.path.join(args.output, 'corpus_overview_by_decade.csv')
    with open(overview_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Decade', 'Articles', 'Total_Words',
            'HMD_Articles', 'LwM_Articles',
            'Newspaper_Titles',
            'London_Articles', 'Provincial_Articles', 'London_Pct',
            'OCR_Mean', 'OCR_Median', 'OCR_SD', 'OCR_Min',
        ])
        
        for decade in sorted(decade_stats.keys()):
            d = decade_stats[decade]
            ocr = d['ocr_values']
            
            ocr_mean = statistics.mean(ocr) if ocr else None
            ocr_median = statistics.median(ocr) if ocr else None
            ocr_sd = statistics.stdev(ocr) if len(ocr) > 1 else None
            ocr_min = min(ocr) if ocr else None
            
            total = d['articles']
            london_pct = (d['london_articles'] / total * 100) if total > 0 else 0
            
            writer.writerow([
                f"{decade}s",
                d['articles'],
                d['total_words'],
                d['hmd_articles'],
                d['lwm_articles'],
                len(d['newspapers']),
                d['london_articles'],
                d['provincial_articles'],
                f"{london_pct:.1f}",
                f"{ocr_mean:.4f}" if ocr_mean else '',
                f"{ocr_median:.4f}" if ocr_median else '',
                f"{ocr_sd:.4f}" if ocr_sd else '',
                f"{ocr_min:.4f}" if ocr_min else '',
            ])
    
    print(f"\nSaved: {overview_path}")
    
    # =========================================================================
    # Output 2: Complete newspaper inventory
    # =========================================================================
    
    # Build a global newspaper inventory across all decades
    all_newspapers = {}  # title -> {pub_id, location, collection, decade_counts}
    
    for decade in sorted(decade_stats.keys()):
        d = decade_stats[decade]
        for title, count in d['newspapers'].items():
            if title not in all_newspapers:
                all_newspapers[title] = {
                    'pub_id': d['newspaper_ids'].get(title, ''),
                    'location': d['newspaper_locations'].get(title, ''),
                    'collection': d['newspaper_collections'].get(title, ''),
                    'total_articles': 0,
                    'decade_counts': {},
                    'first_decade': decade,
                    'last_decade': decade,
                }
            all_newspapers[title]['total_articles'] += count
            all_newspapers[title]['decade_counts'][decade] = count
            all_newspapers[title]['last_decade'] = decade
    
    # Sort by total article count (descending)
    sorted_newspapers = sorted(all_newspapers.items(),
                               key=lambda x: -x[1]['total_articles'])
    
    all_decades = sorted(decade_stats.keys())
    
    inventory_path = os.path.join(args.output, 'newspaper_inventory.csv')
    with open(inventory_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'BL_ID', 'Newspaper_Title', 'Location', 'Collection',
            'Total_Articles', 'First_Decade', 'Last_Decade',
        ] + [f"{d}s" for d in all_decades]
        writer.writerow(header)
        
        for title, info in sorted_newspapers:
            row = [
                info['pub_id'],
                title,
                info['location'],
                info['collection'],
                info['total_articles'],
                f"{info['first_decade']}s",
                f"{info['last_decade']}s",
            ]
            for d in all_decades:
                row.append(info['decade_counts'].get(d, 0))
            writer.writerow(row)
    
    print(f"Saved: {inventory_path}")
    
    # =========================================================================
    # Output 3: Newspaper summary for the paper (compact)
    # =========================================================================
    
    paper_table_path = os.path.join(args.output, 'newspaper_summary_for_paper.csv')
    with open(paper_table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'BL_ID', 'Newspaper_Title', 'Location', 'Collection',
            'Total_Articles', 'Coverage',
        ])
        
        for title, info in sorted_newspapers:
            first = info['first_decade']
            last = info['last_decade']
            coverage = f"{first}s–{last}s" if first != last else f"{first}s"
            
            writer.writerow([
                info['pub_id'],
                title,
                info['location'],
                info['collection'],
                info['total_articles'],
                coverage,
            ])
    
    print(f"Saved: {paper_table_path}")
    
    # =========================================================================
    # Output 4: OCR quality by decade
    # =========================================================================
    
    ocr_path = os.path.join(args.output, 'ocr_quality_by_decade.csv')
    with open(ocr_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Decade', 'N_Articles_With_OCR', 'OCR_Mean', 'OCR_Median',
            'OCR_SD', 'OCR_Q25', 'OCR_Q75', 'OCR_Min', 'OCR_Max',
            'Pct_Below_0.5', 'Pct_Below_0.7', 'Pct_Above_0.9'
        ])
        
        for decade in sorted(decade_stats.keys()):
            ocr = sorted(decade_stats[decade]['ocr_values'])
            if not ocr:
                continue
            
            n = len(ocr)
            q25 = ocr[int(n * 0.25)]
            q75 = ocr[int(n * 0.75)]
            below_50 = sum(1 for v in ocr if v < 0.5) / n * 100
            below_70 = sum(1 for v in ocr if v < 0.7) / n * 100
            above_90 = sum(1 for v in ocr if v > 0.9) / n * 100
            
            writer.writerow([
                f"{decade}s", n,
                f"{statistics.mean(ocr):.4f}",
                f"{statistics.median(ocr):.4f}",
                f"{statistics.stdev(ocr):.4f}" if n > 1 else '',
                f"{q25:.4f}", f"{q75:.4f}",
                f"{min(ocr):.4f}", f"{max(ocr):.4f}",
                f"{below_50:.1f}", f"{below_70:.1f}", f"{above_90:.1f}",
            ])
    
    print(f"Saved: {ocr_path}")
    
    # =========================================================================
    # Output 5: Item types by decade
    # =========================================================================
    
    types_path = os.path.join(args.output, 'item_types_by_decade.csv')
    with open(types_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        all_types = set()
        for d in decade_stats.values():
            all_types.update(d['item_types'].keys())
        all_types = sorted(all_types)
        
        writer.writerow(['Decade'] + all_types)
        
        for decade in sorted(decade_stats.keys()):
            row = [f"{decade}s"]
            for t in all_types:
                row.append(decade_stats[decade]['item_types'].get(t, 0))
            writer.writerow(row)
    
    print(f"Saved: {types_path}")
    
    # =========================================================================
    # Print summary
    # =========================================================================
    
    print("\n" + "=" * 110)
    print("CORPUS OVERVIEW BY DECADE")
    print("=" * 110)
    print(f"{'Decade':<8} {'Articles':>10} {'Words':>14} {'HMD':>8} {'LwM':>8} "
          f"{'Papers':>7} {'London%':>8} {'OCR mean':>9} {'OCR med':>8}")
    print("-" * 110)
    
    total_articles = 0
    total_words = 0
    all_ocr = []
    
    for decade in sorted(decade_stats.keys()):
        d = decade_stats[decade]
        ocr = d['ocr_values']
        total = d['articles']
        total_articles += total
        total_words += d['total_words']
        all_ocr.extend(ocr)
        
        london_pct = (d['london_articles'] / total * 100) if total > 0 else 0
        ocr_mean = statistics.mean(ocr) if ocr else 0
        ocr_med = statistics.median(ocr) if ocr else 0
        
        print(f"{decade}s  {total:>10,} {d['total_words']:>14,} "
              f"{d['hmd_articles']:>8,} {d['lwm_articles']:>8,} "
              f"{len(d['newspapers']):>7} {london_pct:>7.1f}% "
              f"{ocr_mean:>9.4f} {ocr_med:>8.4f}")
    
    print("-" * 110)
    print(f"{'TOTAL':<8} {total_articles:>10,} {total_words:>14,}")
    
    if all_ocr:
        print(f"\nOverall OCR: mean={statistics.mean(all_ocr):.4f}, "
              f"median={statistics.median(all_ocr):.4f}, "
              f"sd={statistics.stdev(all_ocr):.4f}")
    
    all_np = set()
    for d in decade_stats.values():
        all_np.update(d['newspapers'].keys())
    print(f"Total unique newspaper titles: {len(all_np)}")
    
    # Print top newspapers
    print("\n" + "=" * 110)
    print("TOP 30 NEWSPAPERS BY ARTICLE COUNT")
    print("=" * 110)
    print(f"{'BL ID':<12} {'Collection':<6} {'Articles':>10} {'Location':<35} {'Title'}")
    print("-" * 110)
    
    for title, info in sorted_newspapers[:30]:
        loc = info['location'][:33] if len(info['location']) > 33 else info['location']
        print(f"{info['pub_id']:<12} {info['collection']:<6} "
              f"{info['total_articles']:>10,} {loc:<35} {title}")
    
    # Print London vs provincial breakdown
    print("\n" + "=" * 110)
    print("LONDON vs PROVINCIAL NEWSPAPERS")
    print("=" * 110)
    
    london_titles = set()
    provincial_titles = set()
    for title, info in all_newspapers.items():
        if 'london' in info['location'].lower():
            london_titles.add(title)
        else:
            provincial_titles.add(title)
    
    london_articles = sum(info['total_articles'] for t, info in all_newspapers.items()
                          if 'london' in info['location'].lower())
    provincial_articles = sum(info['total_articles'] for t, info in all_newspapers.items()
                              if 'london' not in info['location'].lower())
    
    print(f"London newspapers: {len(london_titles)} titles, {london_articles:,} articles")
    print(f"Provincial newspapers: {len(provincial_titles)} titles, {provincial_articles:,} articles")
    
    if london_titles:
        print(f"\nLondon titles:")
        for title in sorted(london_titles):
            info = all_newspapers[title]
            print(f"  {info['pub_id']:<12} {info['total_articles']:>8,}  {title}")
    
    if provincial_titles:
        print(f"\nProvincial titles:")
        for title in sorted(provincial_titles):
            info = all_newspapers[title]
            print(f"  {info['pub_id']:<12} {info['total_articles']:>8,}  {title} ({info['location']})")


if __name__ == '__main__':
    main()
