import os
from collections import Counter
import re

# Configuration
corpus_dir = ".../en_decade_corpus" #change it to your path
target_terms = ["coffee", "tea", "sugar", "opium", "cocoa", "tobacco"] #change it to your words

# Results storage
results = {}

# Process each decade file
for filename in sorted(os.listdir(corpus_dir)):
    if filename.endswith('.txt'):
        decade = filename.replace('.txt', '')
        filepath = os.path.join(corpus_dir, filename)
        
        print(f"Processing {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize (simple whitespace + basic punctuation handling)
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Count total tokens
        total_tokens = len(tokens)
        
        # Count target terms
        token_counts = Counter(tokens)
        term_frequencies = {term: token_counts[term] for term in target_terms}
        
        # Store results
        results[decade] = {
            'total_tokens': total_tokens,
            'term_frequencies': term_frequencies
        }
        
        print(f"  Total tokens: {total_tokens:,}")
        for term, freq in term_frequencies.items():
            print(f"  '{term}': {freq:,}")
        print()

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Decade':<10} {'Total Tokens':<15} {'coffee':<10} {'tea':<10} {'sugar':<10} {'opium':<10}")
print("-"*80)

for decade in sorted(results.keys()):
    data = results[decade]
    print(f"{decade:<10} {data['total_tokens']:<15,} ", end="")
    for term in target_terms:
        print(f"{data['term_frequencies'][term]:<10,} ", end="")
    print()

# Calculate normalized frequencies (per million tokens)
print("\n" + "="*80)
print("NORMALIZED FREQUENCIES (per million tokens)")
print("="*80)
print(f"{'Decade':<10} {'coffee':<15} {'tea':<15} {'sugar':<15} {'opium':<15}")
print("-"*80)

for decade in sorted(results.keys()):
    data = results[decade]
    total = data['total_tokens']
    print(f"{decade:<10} ", end="")
    for term in target_terms:
        normalized = (data['term_frequencies'][term] / total) * 1_000_000 if total > 0 else 0
        print(f"{normalized:<15.2f} ", end="")
    print()

# Save results to CSV
import csv

output_file = "decade_term_frequencies.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    header = ['decade', 'total_tokens'] + target_terms + [f'{term}_per_million' for term in target_terms]
    writer.writerow(header)
    
    # Write data
    for decade in sorted(results.keys()):
        data = results[decade]
        total = data['total_tokens']
        row = [decade, total]
        row.extend([data['term_frequencies'][term] for term in target_terms])
        row.extend([(data['term_frequencies'][term] / total) * 1_000_000 if total > 0 else 0 
                    for term in target_terms])
        writer.writerow(row)

print(f"\nResults saved to {output_file}")
