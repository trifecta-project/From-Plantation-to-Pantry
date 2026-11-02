# Control Words: JSD versus Frequency

A JSD value on its own does not show that a word's meaning changed: a year with few occurrences gives a noisier estimate of the cluster distribution, and shifts in corpus composition change the contexts a word appears in. Each target is therefore paired with a control word, a noun with a similar yearly frequency whose meaning is not expected to change, and both are processed with the same pipeline:

| Target | Control |
|---|---|
| *coffee* | *presence* |
| *tea* | *child* |
| *sugar* | *difference* |

This directory contains the figure that compares each target with its control, and the script that draws it.

## Files

| File | Description |
|---|---|
| `plot_jsd_vs_frequency.py` | Plots the yearly JSD of each target and its control above their yearly frequencies, and prints the correlations between JSD and frequency |
| `results/jsd_vs_frequency.png` | The figure, one column per target–control pair |

## Requirements

- **JSD per word**: `jsd_<word>.csv` for each target and control, written by `../cluster_jsd_by_period.py`. The controls' embeddings are extracted with `embedding_extraction/embeddings_extraction.py` (`--words presence child difference --forms canonical`) and clustered with `cluster_jsd_by_period.py` exactly as the targets are.
- **Yearly frequencies**: a CSV with the columns `word`, `year` and `count`, giving each word's number of occurrences per year.
- `numpy`, `pandas`, `matplotlib`, `scipy`.

## Running it

```bash
python plot_jsd_vs_frequency.py \
    --freq-csv /path/to/yearly_frequencies.csv \
    --jsd-csv ../results/clusters/coffee/jsd_coffee.csv \
              ../results/clusters/tea/jsd_tea.csv \
              ../results/clusters/sugar/jsd_sugar.csv \
              ../results/clusters/presence/jsd_presence.csv \
              ../results/clusters/child/jsd_child.csv \
              ../results/clusters/difference/jsd_difference.csv \
    --pair coffee:presence --pair tea:child --pair sugar:difference \
    --smooth 5 --out results/jsd_vs_frequency.png
```

| Option | Default | Meaning |
|---|---|---|
| `--freq-csv` | required | Yearly frequency table (`word`, `year`, `count`) |
| `--jsd-csv` | required | One or more JSD CSVs |
| `--pair` | required, repeatable | A target and its control, as `TARGET:CONTROL`; one figure column per pair |
| `--years` | `1801 1919` | Years shown |
| `--smooth` | `5` | Centred rolling mean over this many years, applied to the plotted lines only |
| `--linear-freq` | off | Linear instead of logarithmic frequency axis |
| `--out` | `jsd_vs_frequency.png` | Output file |
| `--dpi` | `200` | Resolution |

## The figure

Each column shows one pair: the target and its control on the same axes, the JSD between consecutive years in the top panel and the number of occurrences per year (logarithmic scale) in the bottom panel, on a shared year axis. JSD and frequency are drawn in separate panels rather than on two vertical scales of one panel, since a dual-scale chart suggests a correspondence that the choice of scales rather than the data creates. Both series are smoothed with a 5-year centred rolling mean for display.

Each JSD value is plotted at the first year of its transition: the JSD between 1850 and 1851 is drawn at 1850.

The script reads the JSD columns written by `cluster_jsd_by_period.py`: the year from `decade_start` and the value from `js_divergence` (base 2). If a file holds only `js_distance`, it is squared into a divergence, so that all words are compared on the same measure.

## Printed correlations

After drawing the figure, the script prints for each word, on the unsmoothed series:

| Column | Meaning |
|---|---|
| JSD vs frequency | Spearman correlation between a year's JSD and the word's number of occurrences in that year, with its p-value |
| JSD vs d(freq) | Spearman correlation between a year's JSD and the absolute change in the word's number of occurrences from the previous year |
| median JSD | Median JSD over all transitions |

The JSD of a year and the frequency are matched by year; the JSD from year *t* to *t*+1 is stored under year *t*. "JSD vs d(freq)" therefore pairs the JSD from *t* to *t*+1 with the change in frequency from *t*−1 to *t*.
