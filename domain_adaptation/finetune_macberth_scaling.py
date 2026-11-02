# finetune_macberth_scaling.py
#
# Trains MacBERTh on the FULL corpus with configurations informed by the
# parameter sensitivity of the 1% grid search.
#
# Rationale: the grid search on 1% of the data (~79K sentences) found lr=5e-5,
# bs=16, ep=4 as optimal. But the full corpus has ~7.1M training sentences
# (~90x more), so each epoch produces ~90x more gradient updates. To avoid
# overfitting, we test 3 configurations along the lr/epoch trade-off:
#
#   Config A: lr=2e-5, 3 epochs  (conservative)
#   Config B: lr=3e-5, 2 epochs  (moderate)
#   Config C: lr=5e-5, 1 epoch   (the grid-search learning rate, single pass)
#
# All use bs=16 (the grid-search winner) and 10% warmup, start from the original
# pretrained MacBERTh, and share one stratified 90/10 train/val split.
# Each saves a checkpoint after every epoch: 3+2+1 = 6 checkpoints.
#
#   Config A_continue: loads the Config A epoch-3 checkpoint and trains epochs
#   4-5 with a new optimizer and a new linear schedule planned for 5 epochs
#   (warmup back to lr=2e-5, then linear decay). 2 further checkpoints.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 python finetune_macberth_scaling.py --config A
#   CUDA_VISIBLE_DEVICES=1 python finetune_macberth_scaling.py --config B
#   CUDA_VISIBLE_DEVICES=1 python finetune_macberth_scaling.py --config C
#   CUDA_VISIBLE_DEVICES=0 python finetune_macberth_scaling.py --config A_continue   # after A
#
#   # Or run A, B and C sequentially:
#   CUDA_VISIBLE_DEVICES=0 python finetune_macberth_scaling.py --config all
#
#   # Rank all checkpoints and write candidate_models.csv:
#   python finetune_macberth_scaling.py --config compare
#
# Runtime per epoch (A40, bs=16, fp16): ~23-29 hours

import os
import gc
import json
import time
import argparse

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import (
    BertForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from torch.optim import AdamW
from tqdm import tqdm

from newspaper_dataset import NewspaperDataset

# ─── Paths ────────────────────────────────────────────────────────────────────
DIR_IN      = "..."  # change it to your path
DIR_CORPUS  = os.path.join(DIR_IN, "en_decade_corpus")
DIR_OUT     = os.path.join(DIR_IN, "output", "scaling_experiment")
BERT_PATH   = "emanjavacas/MacBERTh"

# ─── Configurations ──────────────────────────────────────────────────────────
CONFIGS = {
    "A": {
        "lr": 2e-5,
        "epochs": 3,
        "bs": 16,
        "desc": "conservative (lr=2e-5, 3ep)"
    },

    "A_continue": {
        "lr": 2e-5,
        "epochs": 5,
        "bs": 16,
        "desc": "continue A from epoch 3 to epoch 5"
    },

    "B": {
        "lr": 3e-5,
        "epochs": 2,
        "bs": 16,
        "desc": "moderate (lr=3e-5, 2ep)"
    },

    "C": {
        "lr": 5e-5,
        "epochs": 1,
        "bs": 16,
        "desc": "aggressive-brief (lr=5e-5, 1ep)"
    },
}

WARMUP_FRACTION = 0.10
MAX_LENGTH      = 256
RANDOM_SEED     = 42

# ─── Shared setup ─────────────────────────────────────────────────────────────

def get_corpus_files():
    """Get sorted list of decade corpus files (excluding subcorpus)."""
    files = sorted([
        os.path.join(DIR_CORPUS, f)
        for f in os.listdir(DIR_CORPUS)
        if f.endswith('.txt') and 'subcorpus' not in f
    ])
    print(f"Found {len(files)} decade files:")
    for f in files:
        print(f"  {os.path.basename(f)}")
    return files


def make_train_val_split(files, val_fraction=0.1, seed=42):
    """
    Stratified train/val split: for each decade file, hold out
    val_fraction of lines for validation. Returns two lists of
    (file_path, byte_offset) tuples.
    """
    import random
    rng = random.Random(seed)

    train_index = []
    val_index   = []
    split_info  = {}

    for file_path in files:
        decade_name = os.path.basename(file_path).replace('.txt', '')
        file_train  = []
        file_val    = []

        with open(file_path, 'rb') as f:
            offset = 0
            for line in f:
                stripped = line.strip()
                if stripped:
                    if rng.random() < val_fraction:
                        file_val.append((file_path, offset))
                    else:
                        file_train.append((file_path, offset))
                offset += len(line)

        train_index.extend(file_train)
        val_index.extend(file_val)
        split_info[decade_name] = {
            "train": len(file_train),
            "val": len(file_val)
        }
        print(f"  {decade_name}: {len(file_train):,} train / {len(file_val):,} val")

    # Shuffle train index
    rng2 = random.Random(seed + 1)
    rng2.shuffle(train_index)

    print(f"\nTotal: {len(train_index):,} train / {len(val_index):,} val")
    return train_index, val_index, split_info


class IndexedDataset(torch.utils.data.Dataset):
    """Dataset that reads from pre-built (file_path, byte_offset) index."""

    def __init__(self, index, tokenizer, max_length=256):
        self.index      = index
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_path, offset = self.index[idx]
        with open(file_path, 'rb') as f:
            f.seek(offset)
            line = f.readline().decode('utf-8', errors='replace').strip()

        tokens    = self.tokenizer.tokenize(line)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        padding_length = self.max_length - len(token_ids)
        input_ids      = token_ids + [0] * padding_length
        attention_mask  = [1] * len(token_ids) + [0] * padding_length

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask,  dtype=torch.long),
        }


def evaluate(model, val_loader, device):
    """Compute validation loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def train_config(config_name, config, train_index, val_index, tokenizer):
    """Train one configuration with per-epoch checkpoints."""

    lr     = config["lr"]
    epochs = config["epochs"]
    bs     = config["bs"]
    desc   = config["desc"]

    config_dir = os.path.join(DIR_OUT, f"config_{config_name}")
    os.makedirs(config_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CONFIG {config_name}: {desc}")
    print(f"  lr={lr}, bs={bs}, epochs={epochs}, warmup={WARMUP_FRACTION*100:.0f}%")
    print(f"  Output: {config_dir}")
    print(f"{'='*70}\n")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Load fresh model for each config ──────────────────────────────────
    if config_name == "A_continue":

        resume_path = os.path.join(
            DIR_OUT,
            "config_A",
            "epoch_3"
        )

        print(f"Resuming from checkpoint:")
        print(resume_path)

        model = BertForMaskedLM.from_pretrained(resume_path)

    else:

        print(f"Loading MacBERTh from {BERT_PATH}...")
        model = BertForMaskedLM.from_pretrained(BERT_PATH)

    # Multi-GPU if available
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # ── Datasets ──────────────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    train_dataset = IndexedDataset(train_index, tokenizer, MAX_LENGTH)
    val_dataset   = IndexedDataset(val_index,   tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        collate_fn=data_collator, num_workers=4, pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs * 2,
        collate_fn=data_collator, num_workers=4, pin_memory=True
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_FRACTION)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = GradScaler()

    print(f"\nTraining plan:")
    print(f"  Train batches/epoch: {len(train_loader):,}")
    print(f"  Total steps:         {total_steps:,}")
    print(f"  Warmup steps:        {warmup_steps:,}")
    print(f"  Val batches:         {len(val_loader):,}")

    # ── Training loop ─────────────────────────────────────────────────────
    epoch_logs    = []
    best_val_loss = float("inf")
    best_epoch    = -1
    training_start = time.time()

    if config_name == "A_continue":
        start_epoch = 4
    else:
        start_epoch = 1


    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches    = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
                if n_gpus > 1:
                    loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            n_batches += 1

            if step % 500 == 0:
                pbar.set_postfix({
                    "loss": f"{running_loss / max(n_batches, 1):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

        train_loss = running_loss / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────────────────
        val_loss, val_ppl = evaluate(model, val_loader, device)

        epoch_time = (time.time() - epoch_start) / 3600
        print(f"\n  Epoch {epoch}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}, "
              f"time={epoch_time:.1f}h")

        # ── Save checkpoint ───────────────────────────────────────────────
        ckpt_dir = os.path.join(config_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        # Save checkpoint metadata
        ckpt_meta = {
            "config_name":  config_name,
            "config":       config,
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_loss, 6),
            "val_perplexity": round(val_ppl, 4),
            "epoch_hours":  round(epoch_time, 2),
            "lr_at_end":    scheduler.get_last_lr()[0],
        }
        with open(os.path.join(ckpt_dir, "checkpoint_info.json"), "w") as f:
            json.dump(ckpt_meta, f, indent=2)

        print(f"  Checkpoint saved: {ckpt_dir}")

        epoch_logs.append(ckpt_meta)

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch

    # ── Config summary ────────────────────────────────────────────────────
    total_hours = (time.time() - training_start) / 3600

    summary = {
        "config_name":       config_name,
        "config":            config,
        "best_epoch":        best_epoch,
        "best_val_loss":     round(best_val_loss, 6),
        "best_val_perplexity": round(np.exp(best_val_loss), 4),
        "total_hours":       round(total_hours, 2),
        "epoch_logs":        epoch_logs,
    }

    with open(os.path.join(config_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"CONFIG {config_name} COMPLETE")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best val_loss:  {best_val_loss:.4f}")
    print(f"  Best perplexity: {np.exp(best_val_loss):.2f}")
    print(f"  Total time:     {total_hours:.1f} hours")
    print(f"  Best model:     {config_dir}/epoch_{best_epoch}/")
    print(f"{'='*70}\n")

    # Cleanup GPU memory before next config
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def compare_all_checkpoints():
    """
    After all configs are trained, print a comparison table of all
    checkpoints + pretrained MacBERTh for easy model selection.
    """
    print(f"\n{'='*70}")
    print("ALL CANDIDATE MODELS")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'Val Loss':<12} {'PPL':<10} {'Hours':<8}")
    print("-" * 65)

    # Pretrained baseline
    print(f"{'MacBERTh (pretrained)':<35} {'—':<12} {'—':<10} {'0':<8}")

    # Collect all checkpoints
    candidates = []
    for config_name in ["A", "A_continue", "B", "C"]:
        config_dir = os.path.join(DIR_OUT, f"config_{config_name}")
        summary_path = os.path.join(config_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            for log in summary["epoch_logs"]:
                label = f"Config {config_name} epoch {log['epoch']}"
                candidates.append({
                    "label":    label,
                    "val_loss": log["val_loss"],
                    "val_ppl":  log["val_perplexity"],
                    "hours":    log["epoch_hours"],
                    "path":     os.path.join(config_dir, f"epoch_{log['epoch']}"),
                })

    # Sort by val_loss
    candidates.sort(key=lambda x: x["val_loss"])
    for c in candidates:
        print(f"{c['label']:<35} {c['val_loss']:<12.4f} {c['val_ppl']:<10.2f} {c['hours']:<8.1f}")

    print(f"\n{'='*70}")
    print("The checkpoint with the lowest validation loss is the selected model.")
    print(f"{'='*70}")

    # Save comparison CSV
    import csv
    csv_path = os.path.join(DIR_OUT, "candidate_models.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "val_loss", "val_ppl", "hours", "path"])
        writer.writeheader()
        writer.writerows(candidates)
    print(f"\nComparison saved to: {csv_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MacBERTh on full corpus with scaling-informed configs"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        choices=["A", "A_continue", "B", "C", "all", "compare"],
        help="Which config to run: A, A_continue (after A), B, C, all (A, B and C in sequence), or compare (rank all checkpoints)"
    )
    args = parser.parse_args()

    # Compare mode: just print results from previous runs
    if args.config == "compare":
        compare_all_checkpoints()
        return

    # Set seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    # Load tokenizer (shared across all configs)
    print(f"Loading tokenizer from {BERT_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

    # Build corpus index (done once, reused across configs)
    print(f"\nBuilding corpus index...")
    corpus_files = get_corpus_files()
    train_index, val_index, split_info = make_train_val_split(
        corpus_files, val_fraction=0.1, seed=RANDOM_SEED
    )

    # Save split info
    os.makedirs(DIR_OUT, exist_ok=True)
    with open(os.path.join(DIR_OUT, "split_info.json"), "w") as f:
        json.dump({
            "total_train": len(train_index),
            "total_val":   len(val_index),
            "per_decade":  split_info,
            "val_fraction": 0.1,
            "seed":        RANDOM_SEED,
        }, f, indent=2)

    # Run config(s)
    if args.config == "all":
        configs_to_run = ["A", "B", "C"]
    else:
        configs_to_run = [args.config]

    summaries = {}
    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        summary = train_config(
            config_name, config, train_index, val_index, tokenizer
        )
        summaries[config_name] = summary

    # Print comparison
    compare_all_checkpoints()


if __name__ == "__main__":
    main()

