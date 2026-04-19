# Fine-tune MacBERTh on FULL CORPUS with best configuration from grid search
# Best config: lr=5e-5, batch_size=16, epochs=4
# Optimized with: Multi-GPU (DataParallel) + fp16 mixed precision
#
# Usage:
#   Single GPU:  CUDA_VISIBLE_DEVICES=0 python finetune_macberth_full_corpus.py
#   Multi-GPU:   CUDA_VISIBLE_DEVICES=0,1 python finetune_macberth_full_corpus.py

import os
import json
import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm

# ─── Configuration (winning hyperparameters from grid search) ─────────────────
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
EPOCHS = 4
WARMUP_FRACTION = 0.1  # Standard 10% warmup

# Gradient accumulation - increase if OOM (effective batch = BATCH_SIZE * GRAD_ACCUM)
GRADIENT_ACCUMULATION_STEPS = 1

# fp16 mixed precision - 2-3x speedup on A40/A100 Tensor Cores
USE_FP16 = True

RANDOM_SEED = 42

print("=" * 70)
print("Fine-tuning MacBERTh on FULL CORPUS")
print("=" * 70)
print(f"Configuration (from grid search winner):")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Warmup: {WARMUP_FRACTION * 100:.0f}% of training steps")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  fp16 mixed precision: {USE_FP16}")
print("=" * 70)

# ─── GPU Setup ────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. This script requires GPU(s).")

# Get actual number of visible GPUs (respects CUDA_VISIBLE_DEVICES)
n_gpus = torch.cuda.device_count()
print(f"\nDetected {n_gpus} GPU(s):")
for i in range(n_gpus):
    try:
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  [{i}] {name} - {mem:.1f} GB")
    except Exception as e:
        print(f"  [{i}] Could not get properties: {e}")

device = torch.device("cuda")

torch.set_num_threads(8)
torch.backends.cudnn.benchmark = True

# ─── Paths ────────────────────────────────────────────────────────────────────
dir_in = "..." #change it to your path
dir_corpus = os.path.join(dir_in, "en_decade_corpus")
dir_out = os.path.join(dir_in, "output")

os.makedirs(dir_out, exist_ok=True)

# Find corpus files (exclude subcorpus.txt)
files = os.listdir(dir_corpus)
files = [f for f in files if f.endswith('.txt') and f != 'subcorpus.txt']
files = sorted(files)

print(f"\nFound {len(files)} decade files: {files}")

# ─── Setup Tokenizer First (needed for streaming dataset) ────────────────────
bert_path = "emanjavacas/MacBERTh"

print("\nLoading MacBERTh tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(bert_path)

# ─── Create Train/Val Split Files (Stratified by Decade) ─────────────────────
# Instead of loading all 7.9M sentences into RAM, we:
# 1. Process each decade file separately
# 2. Split 90/10 within each decade (stratified)
# 3. Write to temporary train.txt and val.txt files
# 4. Use streaming dataset that reads line-by-line during training

print("\n" + "=" * 70)
print("Creating stratified train/val split (memory-safe)...")
print("=" * 70)

np.random.seed(RANDOM_SEED)

train_file_path = os.path.join(dir_corpus, "_train_split.txt")
val_file_path = os.path.join(dir_corpus, "_val_split.txt")

decade_counts = {}
total_train = 0
total_val = 0

# Open output files for writing
with open(train_file_path, 'w', encoding='utf-8') as train_f, \
     open(val_file_path, 'w', encoding='utf-8') as val_f:
    
    for file_name in files:
        decade = file_name.replace('.txt', '')
        file_path = os.path.join(dir_corpus, file_name)
        
        # Read one decade at a time (manageable: ~500k-1M sentences per decade)
        print(f"Processing {file_name}...")
        decade_sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    decade_sentences.append(line)
        
        n_total = len(decade_sentences)
        
        # Stratified 90/10 split within this decade
        decade_train, decade_val = train_test_split(
            decade_sentences, test_size=0.1, random_state=RANDOM_SEED
        )
        
        # Write to split files
        for sent in decade_train:
            train_f.write(sent + '\n')
        for sent in decade_val:
            val_f.write(sent + '\n')
        
        decade_counts[decade] = {
            "total": n_total,
            "train": len(decade_train),
            "val": len(decade_val)
        }
        total_train += len(decade_train)
        total_val += len(decade_val)
        
        print(f"  {decade}: {len(decade_train):,} train / {len(decade_val):,} val (total: {n_total:,})")
        
        # Free memory after each decade
        del decade_sentences, decade_train, decade_val
        gc.collect()

print(f"\nTotal: {total_train:,} train / {total_val:,} val")
print(f"Split files saved to:")
print(f"  Train: {train_file_path}")
print(f"  Val:   {val_file_path}")

# Save corpus info
corpus_info = {
    "total_train": total_train,
    "total_val": total_val,
    "decade_counts": decade_counts,
    "stratified": True,
    "config": {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "warmup_fraction": WARMUP_FRACTION,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "use_fp16": USE_FP16,
        "n_gpus": n_gpus
    }
}
with open(os.path.join(dir_out, "full_corpus_training_info.json"), "w") as f:
    json.dump(corpus_info, f, indent=2)

# ─── Memory-Safe Streaming Dataset ────────────────────────────────────────────
# Custom dataset that reads lines from file on-demand instead of loading all into RAM

class StreamingTextDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset that reads text lines on-demand.
    Only stores byte offsets, not the actual text.
    """
    def __init__(self, file_path, tokenizer, max_length=256):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Build index of line offsets (only stores integers, not text)
        print(f"  Indexing {os.path.basename(file_path)}...")
        self.line_offsets = []
        with open(file_path, 'rb') as f:
            offset = 0
            for line in f:
                if line.strip():
                    self.line_offsets.append(offset)
                offset += len(line)
        print(f"    Indexed {len(self.line_offsets):,} lines")
    
    def __len__(self):
        return len(self.line_offsets)
    
    def __getitem__(self, idx):
        # Read single line from file
        with open(self.file_path, 'rb') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline().decode('utf-8').strip()
        
        # Tokenize on-the-fly
        encoded = self.tokenizer(
            line,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

print("\n" + "=" * 70)
print("Building streaming datasets (memory-safe)...")
print("=" * 70)

train_dataset = StreamingTextDataset(train_file_path, tokenizer, max_length=256)
val_dataset = StreamingTextDataset(val_file_path, tokenizer, max_length=256)

print(f"\nTrain dataset: {len(train_dataset):,} sentences")
print(f"Val dataset: {len(val_dataset):,} sentences")

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=4,
    pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=4,
    pin_memory=True
)

# Load model with dropout configuration
config = BertConfig.from_pretrained(bert_path)
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

print("\nLoading MacBERTh model...")
model = BertForMaskedLM.from_pretrained(bert_path, config=config)

# ─── Multi-GPU Setup (DataParallel) ───────────────────────────────────────────
if n_gpus > 1:
    print(f"\nUsing DataParallel across {n_gpus} GPUs")
    model = torch.nn.DataParallel(model)
model.to(device)

# ─── fp16 Mixed Precision Setup ───────────────────────────────────────────────
scaler = GradScaler(enabled=USE_FP16)
if USE_FP16:
    print("Using fp16 mixed precision training")

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
num_training_steps = len(train_dataloader) * EPOCHS
warmup_steps = int(WARMUP_FRACTION * num_training_steps)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

print(f"\nTraining configuration:")
print(f"  Total training steps: {num_training_steps:,}")
print(f"  Warmup steps: {warmup_steps:,} ({WARMUP_FRACTION * 100:.0f}%)")
print(f"  Steps per epoch: {len(train_dataloader):,}")

# ─── Training Loop ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Starting training...")
print("=" * 70)

best_val_loss = float('inf')
best_epoch = 1
training_history = []
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # Training
    model.train()
    loop = tqdm(train_dataloader, leave=True, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    total_loss = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(loop):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass with autocast for fp16
        with autocast(enabled=USE_FP16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            
            # Handle DataParallel output (loss may be tensor with multiple values)
            if loss.dim() > 0:
                loss = loss.mean()
            
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass with gradient scaling for fp16
        scaler.scale(loss).backward()
        
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        loop.set_postfix(loss=loss.item() * GRADIENT_ACCUMULATION_STEPS)
    
    avg_train_loss = total_loss / len(train_dataloader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with autocast(enabled=USE_FP16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
            
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    val_perplexity = np.exp(avg_val_loss)
    epoch_time = time.time() - epoch_start
    
    print(f"\nEpoch {epoch + 1}/{EPOCHS} completed in {epoch_time / 60:.1f} min")
    print(f"  Train loss: {avg_train_loss:.4f}")
    print(f"  Val loss: {avg_val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
    
    training_history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_perplexity": val_perplexity,
        "epoch_time_minutes": round(epoch_time / 60, 1)
    })
    
    # Save checkpoint if best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        
        # Save best model (unwrap DataParallel if needed)
        checkpoint_path = os.path.join(dir_out, "fine_tuned_macberth")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        save_model = model.module if hasattr(model, 'module') else model
        save_model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"  *** New best model saved (epoch {best_epoch}) ***")
    
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

total_time = time.time() - start_time

# ─── Save Final Results ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Training complete!")
print("=" * 70)
print(f"Total training time: {total_time / 3600:.2f} hours")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best validation perplexity: {np.exp(best_val_loss):.2f}")

# Save training history
final_results = {
    "config": {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "warmup_fraction": WARMUP_FRACTION,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "use_fp16": USE_FP16,
        "n_gpus": n_gpus
    },
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "best_val_perplexity": np.exp(best_val_loss),
    "total_training_time_hours": round(total_time / 3600, 2),
    "training_history": training_history
}

results_path = os.path.join(dir_out, "full_corpus_training_results.json")
with open(results_path, "w") as f:
    json.dump(final_results, f, indent=2)

print(f"\nModel saved to: {os.path.join(dir_out, 'fine_tuned_macberth')}")
print(f"Results saved to: {results_path}")

print("\n" + "=" * 70)
print("TRAINING HISTORY")
print("=" * 70)
for h in training_history:
    print(f"Epoch {h['epoch']}: train_loss={h['train_loss']:.4f}, "
          f"val_loss={h['val_loss']:.4f}, ppl={h['val_perplexity']:.2f}, "
          f"time={h['epoch_time_minutes']:.1f}min")
