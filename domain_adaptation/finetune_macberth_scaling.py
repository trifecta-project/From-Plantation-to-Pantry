#!/usr/bin/env python3
"""
finetune_macberth_scaling.py - domain adaptation of MacBERTh on the full
newspaper corpus (continued masked-language-model training), for three
configurations along the learning-rate/epoch trade-off.

The 1% grid search (grid_search.py) found lr=5e-5, batch size 16 and 4 epochs
best on about 79,000 articles. The full corpus has about 7.1 million training
articles, so each epoch performs about 90 times as many parameter updates, and
the subcorpus optimum need not carry over. Three configurations with about the
same product of learning rate and number of epochs are therefore trained:

  Config A: lr=2e-5, 5 epochs 
  Config B: lr=3e-5, 3 epochs 
  Config C: lr=5e-5, 2 epochs 

All use batch size 16, start from the original
MacBERTh, and share one train/validation split: 10% of the articles of each
decade file, drawn at random (seed 42), are held out for validation; every
other article is a training sequence, read once per epoch. Each run's linear
learning-rate schedule (10% warmup) is planned for its own number of epochs,
and a checkpoint is saved after every epoch (5 + 3 + 2 = 10 checkpoints).

Each article (one line of a decade file) is tokenised with MacBERTh's
tokeniser and truncated to its first 256 subword tokens.

Usage
-----
python finetune_macberth_scaling.py --config A --data-dir /path/to/data
python finetune_macberth_scaling.py --config B --data-dir /path/to/data
python finetune_macberth_scaling.py --config C --data-dir /path/to/data
python finetune_macberth_scaling.py --config all --data-dir /path/to/data   # A, B, C in sequence
python finetune_macberth_scaling.py --config compare --data-dir /path/to/data

--data-dir holds en_decade_corpus/ (1800.txt ... 1910.txt); checkpoints go to
<data-dir>/output/scaling_experiment/config_<X>/epoch_<n>/ unless --out-dir is
given. --config compare ranks all checkpoints by validation loss and writes
candidate_models.csv. Runtime per epoch on one NVIDIA A40 (fp16): about 11 hours.
"""

import argparse
import csv
import gc
import json
import os
import random
import re
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, BertForMaskedLM,
                          DataCollatorForLanguageModeling, get_scheduler)

BERT_PATH = "emanjavacas/MacBERTh"

CONFIGS = {
    "A": {"lr": 2e-5, "epochs": 5, "bs": 16, "desc": "conservative (lr=2e-5, 5 epochs)"},
    "B": {"lr": 3e-5, "epochs": 3, "bs": 16, "desc": "moderate (lr=3e-5, 3 epochs)"},
    "C": {"lr": 5e-5, "epochs": 2, "bs": 16, "desc": "aggressive-brief (lr=5e-5, 2 epochs)"},
}

WARMUP_FRACTION = 0.10
MAX_LENGTH = 256
RANDOM_SEED = 42


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_corpus_files(corpus_dir):
    """Sorted decade files (1800.txt ... 1910.txt). Only files named by a
    four-digit decade are read, so other text files in the folder (such as
    the grid-search subcorpus) never enter the training data."""
    files = sorted(os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir)
                   if re.fullmatch(r"\d{4}\.txt", f))
    if not files:
        raise SystemExit(f"No decade files (e.g. 1850.txt) in {corpus_dir}")
    print(f"Found {len(files)} decade files: "
          f"{', '.join(os.path.basename(f) for f in files)}")
    return files


def make_train_val_split(files, val_fraction=0.1, seed=RANDOM_SEED):
    """Stratified split: within each decade file, every non-empty line is held
    out for validation with probability val_fraction. Returns the training and
    validation index as lists of (file path, byte offset), and the counts per
    decade."""
    rng = random.Random(seed)
    train_index, val_index, split_info = [], [], {}
    for file_path in files:
        decade = os.path.basename(file_path).replace(".txt", "")
        n_train = n_val = 0
        with open(file_path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    if rng.random() < val_fraction:
                        val_index.append((file_path, offset))
                        n_val += 1
                    else:
                        train_index.append((file_path, offset))
                        n_train += 1
                offset += len(line)
        split_info[decade] = {"train": n_train, "val": n_val}
        print(f"  {decade}: {n_train:,} train / {n_val:,} val")
    random.Random(seed + 1).shuffle(train_index)
    print(f"\nTotal: {len(train_index):,} train / {len(val_index):,} val")
    return train_index, val_index, split_info


class IndexedDataset(torch.utils.data.Dataset):
    """Reads each article from its (file path, byte offset) and tokenises it:
    the first MAX_LENGTH subword tokens, padded."""

    def __init__(self, index, tokenizer, max_length=MAX_LENGTH):
        self.index = index
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_path, offset = self.index[idx]
        with open(file_path, "rb") as f:
            f.seek(offset)
            line = f.readline().decode("utf-8", errors="replace").strip()
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
        token_ids = token_ids[:self.max_length]
        padding = self.max_length - len(token_ids)
        return {
            "input_ids": torch.tensor(token_ids + [0] * padding, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(token_ids) + [0] * padding,
                                           dtype=torch.long),
        }


# ─── Training ────────────────────────────────────────────────────────────────

def evaluate(model, val_loader, device):
    """Mean validation loss over batches, and perplexity."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                loss = model(**batch).loss
            total += loss.mean().item()
            n += 1
    loss = total / max(n, 1)
    return loss, float(np.exp(loss))


def train_config(name, config, train_index, val_index, tokenizer, out_dir):
    lr, epochs, bs = config["lr"], config["epochs"], config["bs"]
    config_dir = os.path.join(out_dir, f"config_{name}")
    os.makedirs(config_dir, exist_ok=True)
    print(f"\n{'=' * 70}\nCONFIG {name}: {config['desc']}\n"
          f"  lr={lr}, bs={bs}, epochs={epochs}, warmup={WARMUP_FRACTION:.0%}\n"
          f"  Output: {config_dir}\n{'=' * 70}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading MacBERTh from {BERT_PATH} ...")
    model = BertForMaskedLM.from_pretrained(BERT_PATH)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)
    model.to(device)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                               mlm_probability=0.15)
    train_loader = DataLoader(IndexedDataset(train_index, tokenizer), batch_size=bs,
                              shuffle=True, collate_fn=collator, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(IndexedDataset(val_index, tokenizer), batch_size=bs * 2,
                            collate_fn=collator, num_workers=4, pin_memory=True)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_FRACTION)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=warmup_steps,
                              num_training_steps=total_steps)
    scaler = GradScaler()
    print(f"Training plan: {len(train_loader):,} batches per epoch, {total_steps:,} "
          f"steps, {warmup_steps:,} warmup steps, {len(val_loader):,} validation batches")

    epoch_logs, best_loss, best_epoch = [], float("inf"), -1
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        running, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                loss = model(**batch).loss
                if n_gpus > 1:
                    loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            n_batches += 1
            if step % 500 == 0:
                pbar.set_postfix(loss=f"{running / n_batches:.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")
        train_loss = running / max(n_batches, 1)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        hours = (time.time() - t_epoch) / 3600
        print(f"\n  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_ppl={val_ppl:.2f}, time={hours:.1f}h")

        ckpt_dir = os.path.join(config_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        (model.module if hasattr(model, "module") else model).save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        meta = {"config_name": name, "config": config, "epoch": epoch,
                "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
                "val_perplexity": round(val_ppl, 4), "epoch_hours": round(hours, 2),
                "lr_at_end": scheduler.get_last_lr()[0]}
        with open(os.path.join(ckpt_dir, "checkpoint_info.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Checkpoint saved: {ckpt_dir}")
        epoch_logs.append(meta)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch

    summary = {"config_name": name, "config": config, "best_epoch": best_epoch,
               "best_val_loss": round(best_loss, 6),
               "best_val_perplexity": round(float(np.exp(best_loss)), 4),
               "total_hours": round((time.time() - t_start) / 3600, 2),
               "epoch_logs": epoch_logs}
    with open(os.path.join(config_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCONFIG {name} COMPLETE: best epoch {best_epoch}, val_loss {best_loss:.4f}, "
          f"perplexity {np.exp(best_loss):.2f}, {summary['total_hours']:.1f} h")

    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def compare_all_checkpoints(out_dir):
    """Rank all checkpoints of all configurations by validation loss and write
    candidate_models.csv."""
    rows = []
    for name in CONFIGS:
        path = os.path.join(out_dir, f"config_{name}", "summary.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            summary = json.load(f)
        for log in summary["epoch_logs"]:
            rows.append({"config": name, "learning_rate": summary["config"]["lr"],
                         "epoch": log["epoch"], "of_epochs": summary["config"]["epochs"],
                         "train_loss": log["train_loss"], "val_loss": log["val_loss"],
                         "val_ppl": log["val_perplexity"], "hours": log["epoch_hours"],
                         "path": os.path.join(out_dir, f"config_{name}",
                                              f"epoch_{log['epoch']}")})
    if not rows:
        raise SystemExit(f"No finished configuration in {out_dir}")
    rows.sort(key=lambda r: r["val_loss"])
    print(f"\n{'Checkpoint':<24} {'LR':<8} {'Train loss':<11} {'Val loss':<10} {'PPL':<8}")
    print("-" * 63)
    for r in rows:
        print(f"{'Config ' + r['config'] + ', epoch ' + str(r['epoch']):<24} "
              f"{r['learning_rate']:<8.0e} {r['train_loss']:<11.4f} "
              f"{r['val_loss']:<10.4f} {r['val_ppl']:<8.2f}")
    path = os.path.join(out_dir, "candidate_models.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nLowest validation loss: Config {rows[0]['config']}, epoch {rows[0]['epoch']}")
    print(f"Wrote {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, choices=["A", "B", "C", "all", "compare"],
                    help="configuration to train, all three in sequence, or compare "
                         "(rank all checkpoints)")
    ap.add_argument("--data-dir", required=True,
                    help="folder holding en_decade_corpus/")
    ap.add_argument("--out-dir", default=None,
                    help="output folder (default <data-dir>/output/scaling_experiment)")
    args = ap.parse_args()
    corpus_dir = os.path.join(args.data_dir, "en_decade_corpus")
    out_dir = args.out_dir or os.path.join(args.data_dir, "output", "scaling_experiment")

    if args.config == "compare":
        compare_all_checkpoints(out_dir)
        return

    to_run = list(CONFIGS) if args.config == "all" else [args.config]
    for name in to_run:
        done = os.path.join(out_dir, f"config_{name}", "summary.json")
        if os.path.exists(done):
            raise SystemExit(f"Config {name} is already finished ({done}); "
                             f"move that folder away to train it again.")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    print(f"Loading tokenizer from {BERT_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    print("Building the corpus index ...")
    train_index, val_index, split_info = make_train_val_split(get_corpus_files(corpus_dir))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "split_info.json"), "w") as f:
        json.dump({"total_train": len(train_index), "total_val": len(val_index),
                   "per_decade": split_info, "val_fraction": 0.1, "seed": RANDOM_SEED},
                  f, indent=2)

    for name in to_run:
        train_config(name, CONFIGS[name], train_index, val_index, tokenizer, out_dir)
    compare_all_checkpoints(out_dir)


if __name__ == "__main__":
    main()
