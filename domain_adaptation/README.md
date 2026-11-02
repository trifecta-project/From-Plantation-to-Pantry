# Domain Adaptation

This directory contains the pipeline for domain-adapting MacBERTh ([Manjavacas Arevalo & Fonteyn, 2021](https://aclanthology.org/2021.nlp4dh-1.4/)) to our nineteenth-century British newspaper corpus through continued pretraining with masked language modelling (MLM).

The code is adapted from the fine-tuning scripts of the [LatinISE project](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change) by Barbara McGillivray and Valentina Lunardi, which perform a grid search and domain adaptation for LatinBERT.

Domain adaptation proceeds in two stages:

1. **Grid search on a 1% subcorpus.** Searching hyperparameters on the full corpus is too expensive, so a grid of 24 configurations is compared on a 1% sample.
2. **Scaling experiment on the full corpus.** The full corpus is about 90 times larger than the subcorpus, so each epoch performs about 90 times more parameter updates, and the best configuration on the subcorpus need not be the best on the full corpus. Three configurations along the learning-rate/epoch trade-off are therefore trained on the full corpus, plus a continuation of the most conservative one, and the checkpoint with the lowest validation loss is selected.

The selected model is **Config A, epoch 3** (learning rate 2e-5, batch size 16; validation loss 2.093, perplexity 8.11). It is the model used for all embedding extraction (see `embedding_extraction/`).

## Files

| File | Description |
|---|---|
| `newspaper_dataset.py` | PyTorch `Dataset` for newspaper text, used by the grid search |
| `create_subcorpus.py` | Samples 1% of the lines of each decade file into one shuffled subcorpus for the grid search |
| `grid_search.py` | Trains and evaluates one grid-search configuration on the subcorpus |
| `grid_search_results.csv` | The grid-search results, ranked by validation loss (22 of the 24 configurations) |
| `finetune_macberth_scaling.py` | Trains Configs A, B, C and A_continue on the full corpus, saving a checkpoint after every epoch, and ranks all checkpoints |
| `candidate_models.csv` | All 8 full-corpus checkpoints, ranked by validation loss |

## Requirements

The decade sub-corpora (`1800.txt` … `1910.txt`) in `data/en_decade_corpus/`, built with `corpus_preparation/build_decade_corpus.py`. Together they hold all 7,927,079 articles of 1801–1919, one per line; each line is one training sequence.

Set `dir_in` (or `DIR_IN`) at the top of each script to your local data directory.

## Stage 1: Grid search on a 1% subcorpus

### Create the subcorpus

```bash
python create_subcorpus.py --fraction 0.01 --seed 42
```

This samples 1% of the lines of each decade file, shuffles them, and writes `data/en_decade_corpus/subcorpus.txt` (about 79,000 lines).

### Run the grid search

The grid covers 4 learning rates × 2 batch sizes × 3 epoch counts:

| Parameter | Values |
|---|---|
| Learning rate | 2e-5, 3e-5, 4e-5, 5e-5 |
| Batch size | 16, 32 |
| Epochs | 2, 3, 4 |

Each configuration is one job, selected by `--config_idx` (0–23):

```bash
python grid_search.py --config_idx 0
# ... through 23
```

On a SLURM cluster the 24 jobs can be submitted as a job array. Each job trains MacBERTh on a random 90% of the subcorpus, evaluates on the remaining 10%, and writes its result to `output/grid_search/<run_name>/result.json`. Training uses MLM with masking probability 0.15, dropout 0.2 (hidden and attention), AdamW with weight decay 0.01, a linear learning-rate schedule without warmup, and a maximum sequence length of 256 tokens.

### Results

`grid_search_results.csv` holds results for 22 of the 24 configurations; configurations 0 (lr 2e-5, batch size 16, 2 epochs) and 7 (lr 3e-5, batch size 16, 3 epochs) have none. The top 5:

| Rank | Configuration | Learning rate | Batch size | Epochs | Val loss | Perplexity |
|---|---|---|---|---|---|---|
| 1 | lr5e-05_bs16_ep4 | 5e-5 | 16 | 4 | 2.810 | 16.62 |
| 2 | lr4e-05_bs16_ep4 | 4e-5 | 16 | 4 | 2.831 | 16.96 |
| 3 | lr5e-05_bs32_ep4 | 5e-5 | 32 | 4 | 2.847 | 17.24 |
| 4 | lr3e-05_bs16_ep4 | 3e-5 | 16 | 4 | 2.857 | 17.42 |
| 5 | lr4e-05_bs32_ep4 | 4e-5 | 32 | 4 | 2.862 | 17.49 |

Batch size 16 outperforms 32 in all 10 learning-rate/epoch combinations where both were run, so all full-corpus configurations use batch size 16.

## Stage 2: Scaling experiment on the full corpus

### Configurations

| Config | Learning rate | Epochs | Rationale |
|---|---|---|---|
| A | 2e-5 | 3 | Conservative: low learning rate, more passes |
| B | 3e-5 | 2 | Moderate |
| C | 5e-5 | 1 | The grid-search learning rate, limited to a single pass |
| A_continue | 2e-5 | 4–5 | Config A trained for two further epochs |

All configurations use batch size 16, MLM with masking probability 0.15, AdamW with weight decay 0.01, a linear learning-rate schedule with 10% warmup, gradient clipping at norm 1.0, a maximum sequence length of 256 tokens and fp16 mixed precision, with MacBERTh's default dropout. They share one train/validation split: a random 90/10 split within each decade (seed 42), so that every decade is represented in the validation set in proportion to its size.

Configs A, B and C each start from the original pretrained MacBERTh. Config A_continue starts from the Config A epoch-3 checkpoint, with a new optimizer and a new learning-rate schedule planned for five epochs: the learning rate warms up again to 2e-5 over the first half epoch and then decays linearly. Training stops after epoch 5, at about two-thirds of the peak learning rate, so the schedule does not run to zero as it did for Config A.

### Running it

```bash
# One configuration per GPU
CUDA_VISIBLE_DEVICES=0 python finetune_macberth_scaling.py --config A
CUDA_VISIBLE_DEVICES=1 python finetune_macberth_scaling.py --config B
CUDA_VISIBLE_DEVICES=1 python finetune_macberth_scaling.py --config C

# Or A, B and C in sequence
CUDA_VISIBLE_DEVICES=0 python finetune_macberth_scaling.py --config all

# Continue Config A (requires config_A/epoch_3)
CUDA_VISIBLE_DEVICES=0 python finetune_macberth_scaling.py --config A_continue

# Rank all checkpoints and write candidate_models.csv
python finetune_macberth_scaling.py --config compare
```

Each epoch takes about 23–29 hours on one NVIDIA A40. Checkpoints are written to `output/scaling_experiment/config_<X>/epoch_<n>/`, each with a `checkpoint_info.json` recording its training loss, validation loss, perplexity and learning rate at the end of the epoch. The train/validation split is saved to `output/scaling_experiment/split_info.json`.

### Results

All 8 checkpoints (`candidate_models.csv`):

| Checkpoint | Learning rate | Val loss | Perplexity | Hours (epoch) |
|---|---|---|---|---|
| **Config A, epoch 3** | 2e-5 | **2.093** | **8.11** | 23.2 |
| Config A_continue, epoch 5 | 2e-5 | 2.108 | 8.23 | 28.3 |
| Config B, epoch 2 | 3e-5 | 2.110 | 8.25 | 23.1 |
| Config A, epoch 2 | 2e-5 | 2.147 | 8.56 | 23.2 |
| Config A_continue, epoch 4 | 2e-5 | 2.165 | 8.72 | 28.9 |
| Config C, epoch 1 | 5e-5 | 2.172 | 8.78 | 23.1 |
| Config B, epoch 1 | 3e-5 | 2.235 | 9.34 | 23.1 |
| Config A, epoch 1 | 2e-5 | 2.255 | 9.54 | 23.2 |

These validation losses are not comparable with those of the grid search: they are measured on a different validation set, after training on about 90 times more data.

The results confirm that the subcorpus optimum does not carry over to the full corpus:

- After one epoch, the highest learning rate is ahead (C: 2.172, B: 2.235, A: 2.255), but the lower learning rates overtake it with more epochs.
- Config A improves with every epoch (2.255 → 2.147 → 2.093); the gain from epoch 2 to 3 is about half that from epoch 1 to 2.
- Config A_continue does not improve on epoch 3. Raising the learning rate back to its peak first pushes the validation loss up (epoch 4: 2.165), and it falls again only partly (epoch 5: 2.108). This shows that restarting training at the peak learning rate does not help; it does not show how Config A would have done with a single schedule planned for more epochs.

Config A, epoch 3 has the lowest validation loss of all checkpoints and is used for all further analysis.

## Using the model

The selected checkpoint is a standard HuggingFace model directory and loads with `from_pretrained`:

```python
from transformers import AutoModel, AutoTokenizer

path = "output/scaling_experiment/config_A/epoch_3"
model = AutoModel.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
```

For embedding extraction, pass the same path to the `MacBERTh` wrapper in `embedding_extraction/gen_macberth.py`.
