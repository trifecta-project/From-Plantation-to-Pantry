#!/usr/bin/env python3
"""
evaluate_validation.py - MLM loss and perplexity of several models on the
validation articles of the domain adaptation, under identical conditions; and,
optionally, the list of those validation articles.

The validation losses in candidate_models.csv were measured during training,
each with its own random masking. To compare the original MacBERTh with the
domain-adapted model (or several checkpoints), this script evaluates every
model on the same validation articles with the same masked positions:

  * the validation split is rebuilt with the training code itself
    (finetune_macberth_scaling.make_train_val_split, seed 42, 10% per decade
    file) and checked against split_info.json, written during training;
  * sequences are built and masked exactly as in training (first 256 subword
    tokens of each article, masking probability 0.15), and the masking is
    seeded, so every model sees the same masks;
  * the loss is averaged over batches of 32.

With --write-split, it also writes the validation articles, so that material
for evaluating the model can be drawn from articles it has not seen. With
--year-dir and --line-index (from corpus_preparation/build_line_index.py),
each validation article gets its article ID, newspaper ID and date.

Usage
-----
python evaluate_validation.py --data-dir /path/to/data \\
    --models MacBERTh=emanjavacas/MacBERTh \\
             A5=/path/to/data/output/scaling_experiment/config_A/epoch_5 \\
    --out-dir results/validation_eval

# also the list of validation articles, with article IDs
python evaluate_validation.py --data-dir /path/to/data --write-split \\
    --year-dir /path/to/en_year_corpus --line-index /path/to/line_index \\
    --out-dir results/validation_eval
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_macberth_scaling import (IndexedDataset, MAX_LENGTH,  # noqa: E402
                                       RANDOM_SEED, get_corpus_files,
                                       make_train_val_split)


def check_split(split_info_path, split):
    with open(split_info_path) as fh:
        saved = json.load(fh)["per_decade"]
    bad = {d: (saved.get(d), split.get(d)) for d in set(saved) | set(split)
           if saved.get(d) != split.get(d)}
    if bad:
        for d, (s, n) in sorted(bad.items()):
            print(f"  {d}: training {s}, now {n}")
        raise SystemExit("The rebuilt split differs from the one used in training: "
                         "the decade files have changed.")
    print("Split identical to the one used in training (split_info.json).")


def evaluate(model, dataset, tokenizer, device, seed, batch_size):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                               mlm_probability=0.15)
    def collate(batch):
        # masking drawn from a generator seeded per batch, so that every model
        # gets exactly the same masked positions
        torch.manual_seed(seed + collate.n)
        collate.n += 1
        return collator(batch)
    collate.n = 0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    model.eval()
    total, n = 0.0, 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(enabled=device == "cuda"):
                loss = model(**batch).loss
            total += loss.item()
            n += 1
    loss = total / max(n, 1)
    return loss, float(np.exp(loss)), n


def write_split(val_index, out_path, year_dir=None, line_index=None):
    """Validation articles: decade file, line number, byte offset and, with the
    year files and line index, article ID, newspaper ID and date."""
    by_file = {}
    for path, off in val_index:
        by_file.setdefault(path, set()).add(off)
    rows = []
    for path, offs in by_file.items():
        decade = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as fh:
            off = 0
            for ln, line in enumerate(fh):
                if off in offs:
                    rows.append([decade, ln, off])
                off += len(line)

    have_ids = bool(year_dir and line_index)
    if have_ids:
        ids = {}
        for decade in sorted({r[0] for r in rows}):
            d0 = int(decade)
            starts, total = [], 0
            for y in range(d0, d0 + 10):
                yf = None
                for cand in (os.path.join(year_dir, str(y), f"en_{y}.txt"),
                             os.path.join(year_dir, f"en_{y}.txt")):
                    if os.path.exists(cand):
                        yf = cand
                        break
                if yf is None:
                    continue
                with open(yf, "rb") as fh:
                    n = sum(1 for _ in fh)
                starts.append((total, y, n))
                total += n
            ids[decade] = starts
        lookup = {}
        for decade, starts in ids.items():
            for start, y, n in starts:
                p = os.path.join(line_index, f"line_index_{y}.tsv")
                if not os.path.exists(p):
                    continue
                with open(p, encoding="utf-8") as fh:
                    rd = csv.reader(fh, delimiter="\t")
                    next(rd)
                    for r in rd:
                        lookup[(decade, start + int(r[0]))] = (y, r[0], r[1], r[2], r[3])
        n_hit = 0
        for r in rows:
            hit = lookup.get((r[0], r[1]))
            if hit:
                n_hit += 1
            r.extend(hit if hit else ("", "", "", "", ""))
        print(f"Article IDs found for {n_hit:,} of {len(rows):,} validation articles")
        if n_hit < len(rows):
            print("  (a decade file whose line count differs from its year files "
                  "cannot be mapped; those rows have no article ID)")

    rows.sort(key=lambda r: (r[0], r[1]))
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["decade", "line", "byte_offset"] +
                   (["year", "year_line", "article_id", "newspaper_id", "date"]
                    if have_ids else []))
        w.writerows(rows)
    print(f"Wrote {len(rows):,} validation articles to {out_path}")


def evaluate_models(specs, index, label, tokenizer, device, args):
    dataset = IndexedDataset(index, tokenizer, MAX_LENGTH)
    print(f"\nEvaluating on {len(index):,} articles ({label}), batch size "
          f"{args.batch_size}, device {device}")
    rows = []
    for spec in specs:
        name, path = spec.split("=", 1)
        t1 = time.time()
        model = BertForMaskedLM.from_pretrained(path).to(device)
        loss, ppl, n = evaluate(model, dataset, tokenizer, device, args.seed, args.batch_size)
        rows.append(dict(model=name, articles=label, n_articles=len(index), batches=n,
                         val_loss=round(loss, 4), perplexity=round(ppl, 2),
                         minutes=round((time.time() - t1) / 60, 1), path=path))
        print(f"  {name}: loss {loss:.4f}, perplexity {ppl:.2f} "
              f"({(time.time() - t1) / 60:.0f} min)", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True,
                    help="folder holding en_decade_corpus/, as for training")
    ap.add_argument("--split-info", default=None,
                    help="split_info.json written during training (default "
                         "<data-dir>/output/scaling_experiment/split_info.json)")
    ap.add_argument("--models", nargs="+", default=[], metavar="NAME=PATH",
                    help="models to evaluate (a local checkpoint or a Hugging Face name)")
    ap.add_argument("--tokenizer", default="emanjavacas/MacBERTh",
                    help="tokeniser used for all models, as in training")
    ap.add_argument("--max-articles", type=int, default=0,
                    help="evaluate on at most this many articles, a fixed random "
                         "sample (default 0: all)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--write-split", action="store_true",
                    help="also write the list of validation articles")
    ap.add_argument("--year-dir", help="year files, to add article IDs to the list")
    ap.add_argument("--line-index", help="line index folder, to add article IDs")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    split_info = args.split_info or os.path.join(args.data_dir, "output",
                                                 "scaling_experiment", "split_info.json")

    files = get_corpus_files(os.path.join(args.data_dir, "en_decade_corpus"))
    _, val_index, split = make_train_val_split(files, val_fraction=0.1, seed=RANDOM_SEED)
    check_split(split_info, split)

    if args.write_split:
        write_split(val_index, os.path.join(args.out_dir, "validation_articles.tsv"),
                    args.year_dir, args.line_index)

    if not args.models:
        print(f"\nDone in {(time.time() - t0) / 60:.0f} min")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.max_articles and args.max_articles < len(val_index):
        val_index = random.Random(args.seed).sample(val_index, args.max_articles)
    rows = evaluate_models(args.models, val_index, "validation", tokenizer, device, args)

    out = os.path.join(args.out_dir, "validation_perplexity.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
