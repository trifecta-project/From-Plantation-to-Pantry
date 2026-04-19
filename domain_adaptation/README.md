# Domain Adaptation

This directory contains the pipeline for domain-adapting MacBERTh ([Manjavacas Arevalo & Fonteyn, 2021](https://aclanthology.org/2021.nlp4dh-1.4/)) to our nineteenth-century British newspaper corpus via continued pretraining with masked language modelling (MLM).

The code is adapted from the fine-tuning scripts in the [LatinISE project](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change) by Barbara McGillivray and colleagues. Their pipeline performs grid search and domain adaptation for the LatinBERT model on classical Latin corpora. Because their classical Latin dataset is substantially smaller than ours (~4.65 billion tokens), we adapted the approach by conducting the hyperparameter grid search on a 1% stratified subcorpus rather than on the full data.

## Files

| File | Description |
|---|---|
| `newspaper_dataset.py` | PyTorch `Dataset` class for newspaper text. Supports two modes: **in-memory** (for the small grid search subcorpus, ~79k sentences stored in RAM) and **file-based** (for the full corpus, reads lines on demand via byte-offset indexing to avoid loading ~7.9M sentences into memory) |
| `create_subcorpus.py` | Creates the 1% stratified subcorpus for hyperparameter grid search. Randomly samples a fraction of lines from each decade file and writes them to a single shuffled file |
| `grid_search.py` | Runs one hyperparameter configuration of the grid search. Trains MacBERTh on the subcorpus with a given learning rate, batch size, and epoch count, evaluates on a 90/10 validation split, and saves per-epoch loss and perplexity |
| `finetune_macberth.py` | Full-corpus training using the best configuration from the grid search. Performs a stratified 90/10 train/val split across all decades, trains with multi-GPU support and fp16 mixed precision, and saves the best checkpoint |
| `results.csv` | Grid search results: all 24 configurations ranked by validation loss. The winning configuration is lr=5e-5, batch_size=16, epochs=4 (validation loss: 2.810, perplexity: 16.62) |
| `full_corpus_training_info.json` | Training metadata for the full-corpus run: per-decade train/val split counts (total: 7,134,367 train / 792,712 val sentences) and the configuration used |

## Pipeline

### Step 1: Create the subcorpus for grid search

Requires the decade sub-corpora in `data/en_decade_corpus/` (see `corpus_preparation/`).

```bash
python create_subcorpus.py --fraction 0.01 --seed 42
```

This samples 1% of lines from each decade file, shuffles them, and writes the result to `data/en_decade_corpus/subcorpus.txt`. With our corpus this produces approximately 79,000 sentences.

**Note:** Update the `dir_in` variable at the top of the script to point to your local data directory.

### Step 2: Run the hyperparameter grid search

The grid search explores 24 configurations (4 learning rates × 2 batch sizes × 3 epoch counts):

| Parameter | Values |
|---|---|
| Learning rate | 2e-5, 3e-5, 4e-5, 5e-5 |
| Batch size | 16, 32 |
| Epochs | 2, 3, 4 |

Each configuration is run as a separate job, indexed by `--config_idx` (0–23):

```bash
python grid_search.py --config_idx 0
python grid_search.py --config_idx 1
# ... through 23
```

On a SLURM cluster, you can submit all 24 as a job array. Each job trains MacBERTh on the subcorpus with MLM, evaluates on a held-out 10% validation set, and saves the results to `output/grid_search/<run_name>/result.json`.

**Grid search results** (from `results.csv`, top 5):

| Rank | Config | LR | Batch size | Epochs | Val loss | Perplexity |
|---|---|---|---|---|---|---|
| 1 | lr5e-05_bs16_ep4 | 5e-5 | 16 | 4 | 2.810 | 16.62 |
| 2 | lr4e-05_bs16_ep4 | 4e-5 | 16 | 4 | 2.831 | 16.96 |
| 3 | lr5e-05_bs32_ep4 | 5e-5 | 32 | 4 | 2.847 | 17.24 |
| 4 | lr3e-05_bs16_ep4 | 3e-5 | 16 | 4 | 2.857 | 17.42 |
| 5 | lr4e-05_bs32_ep4 | 4e-5 | 32 | 4 | 2.862 | 17.49 |

### Step 3: Train on the full corpus

Using the winning configuration (lr=5e-5, batch_size=16, epochs=4):

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python finetune_macberth.py

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1 python finetune_macberth.py
```

**What it does:**

1. Creates a stratified 90/10 train/val split (within each decade separately, preserving temporal coverage)
2. Builds a memory-safe streaming dataset that reads lines via byte-offset indexing rather than loading all ~7.9M sentences into RAM
3. Trains MacBERTh with MLM (masking probability 0.15), linear warmup (10% of steps), gradient clipping (max norm 1.0), and fp16 mixed precision
4. Saves the best checkpoint (by validation loss) to `output/fine_tuned_macberth/`
5. Writes training metadata to `output/full_corpus_training_info.json`

**Training data** (from `full_corpus_training_info.json`):

| Decade | Total sentences | Train | Val |
|---|---|---|---|
| 1800 | 129,303 | 116,372 | 12,931 |
| 1810 | 163,987 | 147,588 | 16,399 |
| 1820 | 209,329 | 188,396 | 20,933 |
| 1830 | 169,439 | 152,495 | 16,944 |
| 1840 | 498,147 | 448,332 | 49,815 |
| 1850 | 763,035 | 686,731 | 76,304 |
| 1860 | 831,504 | 748,353 | 83,151 |
| 1870 | 708,228 | 637,405 | 70,823 |
| 1880 | 1,175,959 | 1,058,363 | 117,596 |
| 1890 | 1,326,660 | 1,193,994 | 132,666 |
| 1900 | 1,191,786 | 1,072,607 | 119,179 |
| 1910 | 759,702 | 683,731 | 75,971 |
| **Total** | **7,927,079** | **7,134,367** | **792,712** |

**Note:** The full-corpus training takes several hours to days depending on GPU hardware. The script supports multi-GPU training via `DataParallel` and fp16 mixed precision for faster training on A40/A100 GPUs.

## Model details

| Parameter | Value |
|---|---|
| Base model | `emanjavacas/MacBERTh` (HuggingFace) |
| Pretraining task | Masked Language Modelling (MLM) |
| MLM masking probability | 0.15 |
| Hidden dropout | 0.2 |
| Attention dropout | 0.2 |
| Optimiser | AdamW (weight decay 0.01) |
| LR scheduler | Linear with 10% warmup |
| Gradient clipping | Max norm 1.0 |
| Max sequence length | 256 tokens |

## Output

The domain-adapted model is saved to `output/fine_tuned_macberth/` and can be loaded with:

```python
from transformers import BertForMaskedLM, AutoTokenizer

model = BertForMaskedLM.from_pretrained("path/to/output/fine_tuned_macberth")
tokenizer = AutoTokenizer.from_pretrained("path/to/output/fine_tuned_macberth")
```

## Relationship to the paper

- **Section 4.2** (Model Fine-tuning for Domain Adaptation) describes the grid search procedure, the 1% subcorpus rationale, and the winning configuration
- **Section 7** (Limitations — Model Configuration) acknowledges that the 1% subcorpus grid search is a trade-off and that configurations optimal for a smaller sample may not generalise perfectly to the full data
- **Appendix** (Domain-Adaptation Evaluation) compares PCA projections and clustering outcomes between the pretrained and domain-adapted models (see `domain_adaptation_evaluation/`)
