# create_subcorpus.py
# Randomly samples a small fraction of lines from each decade file
# and saves them as a single subcorpus file for grid search.
#
# Usage: python create_subcorpus.py --fraction 0.01
# Output: en_decade_corpus/subcorpus.txt

import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fraction", type=float, default=0.01,
                    help="Fraction of lines to sample per file (default: 0.01 = 1%%)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

dir_in     = "..." # change it to your path
dir_corpus = os.path.join(dir_in, "en_decade_corpus")
out_path   = os.path.join(dir_corpus, "subcorpus.txt")

random.seed(args.seed)

files = sorted([f for f in os.listdir(dir_corpus) if f.endswith('.txt')])
print(f"Found {len(files)} decade files")

total_seen = 0
total_kept = 0
all_lines  = []

for file_name in files:
    file_path = os.path.join(dir_corpus, file_name)
    kept = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_seen += 1
            if random.random() < args.fraction:
                all_lines.append(line)
                kept += 1
    print(f"  {file_name}: kept {kept:,} lines")
    total_kept += kept

# Shuffle so decade files are mixed (better for train/val split)
random.shuffle(all_lines)

with open(out_path, 'w', encoding='utf-8') as f:
    for line in all_lines:
        f.write(line + '\n')

print(f"\nDone. Sampled {total_kept:,} / {total_seen:,} lines ({args.fraction*100:.1f}%)")
print(f"Saved to: {out_path}")
