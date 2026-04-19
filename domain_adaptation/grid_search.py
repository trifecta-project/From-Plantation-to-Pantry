import os
import json
import time
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from itertools import product
from newspaper_dataset import NewspaperDataset

# ---- Parse arguments --------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config_idx", type=int, required=True, help="Index into the hyperparameter grid (0-47)")
# subcorpus size is fixed by create_subcorpus.py — no subsample flag needed here
args = parser.parse_args()

# ---- Hyperparameter Grid (must match submit script and merge script) ---------
LEARNING_RATES    = [2e-5, 3e-5, 4e-5, 5e-5]
BATCH_SIZES       = [16, 32]
EPOCHS_LIST       = [2, 3, 4]

grid = list(product(LEARNING_RATES, BATCH_SIZES, EPOCHS_LIST))
assert args.config_idx < len(grid), f"config_idx {args.config_idx} out of range (grid size: {len(grid)})"

lr, batch_size, epochs = grid[args.config_idx]
run_name = f"lr{lr}_bs{batch_size}_ep{epochs}"

print(f"Config {args.config_idx}: {run_name}")

# ---- Paths ------------------------------------------------------------------
dir_in     = "..." #change it to your path
dir_corpus = os.path.join(dir_in, "en_decade_corpus")
dir_out    = os.path.join(dir_in, "output", "grid_search")
run_out    = os.path.join(dir_out, run_name)
os.makedirs(run_out, exist_ok=True)

# ---- Load subcorpus (small file created by create_subcorpus.py) -------------
subcorpus_path = os.path.join(dir_corpus, "subcorpus.txt")
assert os.path.exists(subcorpus_path), (
    f"Subcorpus not found at {subcorpus_path}. "
    f"Run: python create_subcorpus.py --fraction 0.01 first."
)

print(f"Loading subcorpus from {subcorpus_path}...")
with open(subcorpus_path, 'r', encoding='utf-8') as f:
    corpus = [line.strip() for line in f if line.strip()]
print(f"Subcorpus size: {len(corpus):,} sentences")

# Standard random 90/10 train/val split on the small subcorpus
train_texts, val_texts = train_test_split(corpus, test_size=0.1, random_state=42)
print(f"Train: {len(train_texts):,} | Val: {len(val_texts):,}")

# ---- Device -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ---- Tokenizer & Datasets ---------------------------------------------------
bert_path = "emanjavacas/MacBERTh"
print("Loading MacBERTh tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(bert_path)

# Small enough to load fully into RAM
train_dataset = NewspaperDataset(train_texts, tokenizer, max_length=256)
val_dataset   = NewspaperDataset(val_texts,   tokenizer, max_length=256)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# ---- DataLoaders ------------------------------------------------------------
# num_workers=4 per process — with 4 parallel processes = 16 cores total,
# leaving ~48 cores free for other users on the shared server.
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=data_collator, num_workers=4, pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    collate_fn=data_collator, num_workers=4, pin_memory=True
)

# ---- Model ------------------------------------------------------------------
config = BertConfig.from_pretrained(bert_path)
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2

model = BertForMaskedLM.from_pretrained(bert_path, config=config)
model.to(device)
model.train()

# ---- Optimizer & Scheduler --------------------------------------------------
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
num_training_steps = len(train_dataloader) * epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,num_warmup_steps=0,
    num_training_steps=num_training_steps
)

print(f"\nStarting fine-tuning: {run_name}")
print(f"lr={lr} | batch_size={batch_size} | epochs={epochs}")
print(f"Total training steps: {num_training_steps}")

# ---- Training Loop ----------------------------------------------------------
epoch_logs = []
best_val_loss = float('inf')
best_epoch = 1
start_time = time.time()

for epoch in range(epochs):
    # Train
    model.train()
    loop = tqdm(train_dataloader, leave=True)
    total_loss = 0

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch + 1}/{epochs}")
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    val_perplexity = np.exp(avg_val_loss)
    print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f} | Perplexity: {val_perplexity:.2f}")

    epoch_logs.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_perplexity": val_perplexity
    })

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1

elapsed = time.time() - start_time

# ---- Save Results for This Run ----------------------------------------------
pd.DataFrame(epoch_logs).to_csv(os.path.join(run_out, "epoch_log.csv"), index=False)

result = {
    "config_idx": args.config_idx,
    "run_name": run_name,
    "learning_rate": lr,
    "batch_size": batch_size,
    "epochs": epochs,
    "best_val_loss": best_val_loss,
    "best_epoch": best_epoch,
    "final_val_perplexity": epoch_logs[-1]["val_perplexity"],
    "elapsed_seconds": round(elapsed, 1)
}

with open(os.path.join(run_out, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"\nRun complete in {elapsed/60:.1f} min")
print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
print(f"Results saved to {run_out}/result.json")
