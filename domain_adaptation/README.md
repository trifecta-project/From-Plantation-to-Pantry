# Domain Adaptation

This directory contains the pipeline for domain-adapting MacBERTh ([Manjavacas Arevalo & Fonteyn, 2021](https://aclanthology.org/2021.nlp4dh-1.4/)) to our nineteenth-century British newspaper corpus through continued pretraining with masked language modelling (MLM).

The code is adapted from the fine-tuning scripts of the [LatinISE project](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change) by Barbara McGillivray and Valentina Lunardi, which perform a grid search and domain adaptation for LatinBERT.

Domain adaptation proceeds in two stages:

1. **Grid search on a 1% subcorpus.** Searching hyperparameters on the full corpus is too expensive, so a grid of 24 configurations is compared on a 1% sample.
2. **Scaling experiment on the full corpus.** The full corpus is about 90 times larger than the subcorpus, so each epoch performs about 90 times more parameter updates, and the best configuration on the subcorpus need not be the best on the full corpus. Three configurations along the learning-rate/epoch trade-off are therefore trained on the full corpus, and the checkpoint with the lowest validation loss is selected.

The selected model is **Config A after five epochs** (learning rate 2e-5, batch size 16). It is the model used for all embedding extraction (see `embedding_extraction/`).

## Files

| File | Description |
|---|---|
| `create_subcorpus.py` | Samples 1% of the articles of each decade file into one shuffled subcorpus for the grid search |
| `newspaper_dataset.py` | PyTorch `Dataset` for newspaper text, used by the grid search |
| `grid_search.py` | Trains and evaluates one grid-search configuration on the subcorpus |
| `finetune_macberth_scaling.py` | Trains Configs A, B and C on the full corpus, saving a checkpoint after every epoch, and ranks all checkpoints |
| `evaluate_validation.py` | Evaluates several models (e.g. the original MacBERTh and the selected checkpoint) on the same validation articles with the same masks, and lists the validation articles |

## Requirements

- The decade sub-corpora `en_decade_corpus/1800.txt` … `1910.txt`, built with `corpus_preparation/build_decade_corpus.py`. Together they hold all 7,927,079 articles of 1801–1919, one per line; each line is one training sequence. All scripts take the folder that contains `en_decade_corpus/` as `--data-dir`, and write their output to `<data-dir>/output/`.
- `torch`, `transformers`, `numpy`, `pandas`, `scikit-learn`, `tqdm`.
- A GPU. The timings below are for one NVIDIA A40 with mixed precision.

The model, [`emanjavacas/MacBERTh`](https://huggingface.co/emanjavacas/MacBERTh), is downloaded from the Hugging Face Hub on first use.

## Stage 1: Grid search on a 1% subcorpus

### Create the subcorpus

```bash
python create_subcorpus.py --data-dir /path/to/data --fraction 0.01 --seed 42
```

This samples 1% of the articles of each decade file, shuffles them, and writes `en_decade_corpus/subcorpus.txt` (about 79,000 articles).

### Run the grid search

The grid covers 4 learning rates × 2 batch sizes × 3 epoch counts:

| Parameter | Values |
|---|---|
| Learning rate | 2e-5, 3e-5, 4e-5, 5e-5 |
| Batch size | 16, 32 |
| Epochs | 2, 3, 4, 5|

Each configuration is one job, selected by `--config_idx` (0–23):

```bash
for i in $(seq 0 23); do
    python grid_search.py --config_idx $i --data-dir /path/to/data
done
```

On a SLURM cluster the 24 jobs can be submitted as a job array. Each job trains MacBERTh on a random 90% of the subcorpus, evaluates on the remaining 10%, and writes its result to `output/grid_search/<run_name>/result.json`. Training uses MLM with masking probability 0.15, dropout 0.2 (hidden and attention), AdamW with weight decay 0.01, a linear learning-rate schedule without warmup, and a maximum sequence length of 256 tokens.

To collect the results into one table ranked by validation loss:

```bash
python -c "
import glob, json, pandas as pd
d = pd.DataFrame(json.load(open(p)) for p in glob.glob('/path/to/data/output/grid_search/*/result.json'))
d = d.sort_values('best_val_loss').reset_index(drop=True)
d.insert(0, 'rank', d.index + 1)
d.to_csv('grid_search_results.csv', index=False)
print(d.head())"
```

### Results

`grid_search_results.csv` 

## Stage 2: Scaling experiment on the full corpus

### Configurations

| Config | Learning rate | Epochs | 
|---|---|---|---|
| A | 2e-5 | 5 |
| B | 3e-5 | 3 | 
| C | 5e-5 | 2 | 

The three configurations have about the same product of learning rate and number of epochs, so they trade the size of each update against the number of passes over the data. All use batch size 16, MLM with masking probability 0.15, AdamW with weight decay 0.01, a linear learning-rate schedule with 10% warmup planned for the configuration's own number of epochs, gradient clipping at norm 1.0, a maximum sequence length of 256 tokens and fp16 mixed precision, with MacBERTh's default dropout. Each starts from the original pretrained MacBERTh.

All three share one train/validation split: within each decade file, 10% of the articles are drawn at random (seed 42) for validation, so that every decade is represented in the validation set in proportion to its size. The remaining articles, about 7.1 million, are the training data; each is read once per epoch, as its first 256 subword tokens. No validation article is used in training.

### Running it

```bash
# One configuration at a time
python finetune_macberth_scaling.py --config A --data-dir /path/to/data
python finetune_macberth_scaling.py --config B --data-dir /path/to/data
python finetune_macberth_scaling.py --config C --data-dir /path/to/data

# Or A, B and C in sequence
python finetune_macberth_scaling.py --config all --data-dir /path/to/data

# Rank all checkpoints and write candidate_models.csv
python finetune_macberth_scaling.py --config compare --data-dir /path/to/data
```

Use one GPU per configuration (`CUDA_VISIBLE_DEVICES=0`, `1`, …); several configurations on the same GPU share it and finish no sooner. Each epoch takes about 11 hours on one A40, so A takes about 55 hours, B about 33 and C about 22.

Checkpoints are written to `output/scaling_experiment/config_<X>/epoch_<n>/` (`--out-dir` to change this), each with a `checkpoint_info.json` recording its training loss, validation loss, perplexity and learning rate at the end of the epoch; `config_<X>/summary.json` collects them. The split is saved to `output/scaling_experiment/split_info.json`. A configuration that has finished (one with a `summary.json`) is not trained again.

### Results

`--config compare` ranks all ten checkpoints (A after epochs 1–5, B after epochs 1–3, C after epochs 1–2) by validation loss and writes them, with their training loss, to `output/scaling_experiment/candidate_models.csv`. A checkpoint after epoch *n* of a longer run is not the same as a run planned for *n* epochs, since its learning rate has not yet decayed to zero; the comparison of interest is between the final checkpoints of the three configurations, and the intermediate checkpoints show how each converges. The validation losses are not comparable with those of the grid search: they are measured on a different validation set, after training on about 90 times more data.

## Comparing with MacBERTh

The validation losses in `candidate_models.csv` were measured during training, each with its own random masking. `evaluate_validation.py` evaluates several models on the same validation articles with the same masked positions, so that the original MacBERTh and the domain-adapted checkpoints can be compared directly:

```bash
python evaluate_validation.py --data-dir /path/to/data \
    --models MacBERTh=emanjavacas/MacBERTh \
             A5=/path/to/data/output/scaling_experiment/config_A/epoch_5 \
             B3=/path/to/data/output/scaling_experiment/config_B/epoch_3 \
             C2=/path/to/data/output/scaling_experiment/config_C/epoch_2 \
    --out-dir results/validation_eval
```

It rebuilds the split with the training code, stops if it differs from `split_info.json`, builds and masks the sequences as in training, and writes `validation_perplexity.csv`. `--max-articles N` evaluates on a fixed random sample of N articles instead of all 0.8 million validation articles. With `--write-split` it writes `validation_articles.tsv`, the articles no model has been trained on, from which evaluation material can be drawn; with `--year-dir` and `--line-index` (from `corpus_preparation/build_line_index.py`), each gets its article ID, newspaper ID and date.

## Using the model

The selected checkpoint is a standard Hugging Face model directory and loads with `from_pretrained`:

```python
from transformers import AutoModel, AutoTokenizer

path = "output/scaling_experiment/config_A/epoch_5"
model = AutoModel.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
```

For embedding extraction, pass the same path to the `MacBERTh` wrapper in `embedding_extraction/gen_macberth.py`.
