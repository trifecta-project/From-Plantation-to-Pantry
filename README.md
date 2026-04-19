# From Plantation to Pantry

Code and data for the paper:

> Jiaqi Zhu and Marieke van Erp. "From Plantation to Pantry: Detecting Semantic Change in Commodity Terms through Static and Contextualised Embeddings in Nineteenth-Century British Newspapers." *Digital Scholarship in the Humanities* (submitted).

## Overview

This repository contains the full pipeline for detecting semantic change in six commodity terms — *coffee*, *tea*, *sugar*, *opium*, *cocoa*, and *tobacco* — across eight decades of British newspaper discourse (1840–1919). The pipeline combines static Word2Vec embeddings with a domain-adapted [MacBERTh](https://huggingface.co/emanjavacas/MacBERTh) model to address two research questions: whether the dominant usage of each term shifted (RQ1, via cosine similarity), and how the relative prominence of distinct usage types reorganised over time (RQ2, via K-means clustering and Jensen–Shannon Divergence).

The underlying corpus is drawn from the British Library's [Heritage Made Digital](https://bl.iro.bl.uk/concern/datasets/2800eb7d-8b49-4398-a6e9-c2c5692a1304) (HMD) and [Living with Machines](https://bl.iro.bl.uk/concern/datasets/99dc570a-9460-48ac-baed-9d2b8c4c13c0) (LwM) collections, comprising approximately 4.65 billion tokens across the analysis period.

## Repository Structure

```
From-Plantation-to-Pantry/
├── README.md
├── requirements.txt
│
├── data/                              # Corpus data and access instructions
│   └── 
│   ├── pretrained_word2vec_vectors
        ├── README.md
        ├── shared_vocab.py
        └── shared_vocab_output.txt
│   ├── en_decade_corpus/           # Decade-level sub-corpora (generated)
        ├── README.md
        ├── corpus_overview.csv                     # Per-decade corpus composition statistics
        ├── newspaper_inventory.csv                 # All 120 newspaper titles in the corpus
        └── top30_newspapers_by_article_count.csv   # Top 30 newspapers by volume
│
├── corpus_preparation/                # Aggregate HMD + LwM into decade files
│   ├── aggregate_and_partition.py
│   ├── count_tokens_and_terms.py
│   ├── decade_term_frequencies.csv
│   ├── extract_corpus_metadata.py
│   └── aggregation_report.txt
│
├── static_embeddings/                 # Static Word2Vec analysis (RQ1)
│                                      # Cosine similarity, nearest neighbours, t-SNE
│
├── domain_adaptation/                 # Domain-adapt MacBERTh on newspaper corpus
│   ├── create_subcorpus.py            # 1% stratified subcorpus for grid search
│   ├── grid_search.py                 # 24-configuration hyperparameter search
│   ├── finetune_macberth.py           # Full-corpus training (best config)
│   ├── newspaper_dataset.py           # PyTorch Dataset (memory-safe)
│   ├── results.csv                    # Grid search results
│   └── full_corpus_training_info.json
│
├── embedding_extraction/              # Extract and analyse contextualised embeddings (RQ2)
│   ├── gen_macberth.py                # MacBERTh wrapper
│   ├── embeddings_extraction.py       # Extract target-word embeddings
│   └── analysis/
│       ├── commodity_analysis_finetuned.py   # K-means clustering + visualisation
│       ├── commodity_analysis_pretrained.py  # Same for pretrained (comparison)
│       ├── extract_contexts_visible_only.py  # Centroid-closest snippets
│       ├── jsd.py                            # JSD via Affinity Propagation
│       └── output/                           # CSVs, plots, cluster contexts
│
└── domain_adaptation_evaluation/      # PCA comparison: pretrained vs adapted
    ├── pca_pretrained_vs_adapted.py
    └── pca_plots/
```

Each directory has its own `README.md` with detailed documentation of the files, usage instructions, and mapping to the paper.

## Pipeline Overview

The pipeline proceeds in six steps, each corresponding to a directory:

| Step | Directory | Description | Paper section |
|---|---|---|---|
| 1 | `corpus_preparation/` | Aggregate HMD + LwM corpora, partition into decade sub-corpora | Section 3 |
| 2 | `static_embeddings/` | Cosine similarity vs 1910s, nearest neighbours, t-SNE visualisations | Section 4.1 |
| 3 | `domain_adaptation/` | Grid search on 1% subcorpus, train MacBERTh on full corpus with best config | Section 4.2 |
| 4 | `embedding_extraction/` | Extract last-hidden-layer embeddings for target words per decade | Section 4.2 |
| 5 | `embedding_extraction/analysis/` | K-means clustering, centroid-closest snippets, JSD computation | Sections 4.2, 5 |
| 6 | `domain_adaptation_evaluation/` | PCA comparison of pretrained vs adapted embeddings | Appendix |

## Key Results

The pipeline produces the following outputs mapped to the paper:

| Paper element | Output |
|---|---|
| Table 1 (token counts) | `corpus_preparation/decade_term_frequencies.csv` |
| Table 2 (nearest neighbours) | `static_embeddings/` |
| Table 3 (JSD values) | `embedding_extraction/analysis/output/csv/jsd_consecutive_decades.csv` |
| Table 4 (clustering statistics) | `embedding_extraction/analysis/output/csv/cluster_summary_detailed.csv` |
| Tables 5–10 (cluster contexts) | `embedding_extraction/analysis/output/cluster_contexts/` |
| Figure 4 (cosine similarity) | `static_embeddings/` |
| Figures 5–7 (t-SNE trajectories) | `static_embeddings/` |
| Figure 8 (coffee clusters) | `embedding_extraction/analysis/output/clusters/coffee/` |
| Figure 9 (sugar/cocoa PCA) | `embedding_extraction/analysis/output/PCA/` |
| Figures 10–11 (domain adaptation) | `domain_adaptation_evaluation/pca_plots/` |

## Acknowledgements

This code builds on and adapts work from:

- **[LatinISE](https://github.com/BarbaraMcG/latinise/tree/master/christianity_semantic_change)** (Barbara McGillivray and colleagues) — embedding extraction wrapper, clustering analysis pipeline, and domain-adaptation architecture
- **[UiO-UvA SemEval-2020](https://github.com/akutuzov/semeval2020)** (Andrey Kutuzov and Mario Giulianelli) — Affinity Propagation-based JSD computation
- **[diachronicBert](https://github.com/wendyqiu/diachronicBert)** (Wendy Qiu) — PCA visualisation for domain-adaptation evaluation
- **[Pedrazzini & McGillivray (2022)](https://aclanthology.org/2022.nlp4dh-1.11/)** — pretrained and aligned Word2Vec vectors on HMD + LwM

## Data Access

The HMD and LwM corpora are released by the British Library and are too large to include in this repository. See `data/README.md` for download links and instructions for generating the decade sub-corpora.

The pretrained Word2Vec vectors are from Pedrazzini & McGillivray (2022). See `static_embeddings/README.md` for access details.

## Requirements

Python 3.9+ with the following key packages:

```
torch
transformers
scikit-learn
scipy
gensim
h5py
numpy
pandas
matplotlib
tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

**Note:** GPU access is required for embedding extraction (`embedding_extraction/`) and domain adaptation (`domain_adaptation/`). The static embedding analysis and clustering can run on CPU.

## Citation

If you use this code or data, please cite:

```bibtex
@article{zhu2026plantation,
  title={From Plantation to Pantry: Detecting Semantic Change in Commodity Terms 
         through Static and Contextualised Embeddings in Nineteenth-Century 
         British Newspapers},
  author={Zhu, Jiaqi and van Erp, Marieke},
  journal={Digital Scholarship in the Humanities},
  year={2026},
  note={Submitted}
}
```

## Funding

This research was supported by the European Union through the European Research Council under grant agreement 101088548 — [TRIFECTA](https://trifecta.dhlab.nl/).

## License

[TBD — add license before publication]
