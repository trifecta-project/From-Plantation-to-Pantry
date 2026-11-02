#!/usr/bin/env python3
"""
Number of occurrences per YEAR for each commodity term.

The data comes from the extraction's frequency_summary.csv rather than being
pasted into the source, because 119 years x several words is hundreds of numbers
and hardcoding them means the figure silently goes stale the next time the
extraction is re-run.

Note the CSV's column is still called `decade` but holds a year, which is how the
yearly extraction writes it.

Three choices follow from having ten times as many points as a decade chart:

  * No markers. A marker every year hides the lines.
  * A numeric x-axis with ticks every 10 years, instead of one rotated tick per
    period.
  * Labels at the right-hand end of each line rather than a legend, so the eye
    does not have to travel to a key and back.

--csv takes more than one file, because the control words (water, stone) are
extracted into their own folder and so have their own frequency_summary.csv. The
files are concatenated; if a word appears in two of them the first wins.

Control words are drawn as a dashed grey line, so the figure says at a glance
which line is the baseline and which lines are the commodities. --plain turns
that off and draws them like any other word.

Usage
-----
python plot_counts_by_year.py \\
    --csv output/embeddings_macberth_scaling_best_windowed_last4_spelling_w50_yearly/frequency_summary.csv \\
          output/embeddings_control_words_yearly/frequency_summary.csv \\
    --words coffee tea sugar water stone

python plot_counts_by_year.py --csv <...> --column embeddings_saved
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical order on a light background: every adjacent pair is
# distinguishable with normal and with colour-deficient vision.
COLOURS = {
    "coffee":  "#2a78d6",
    "tea":     "#eb6834",
    "sugar":   "#1baf7a",
    "cocoa":   "#eda100",
    "opium":   "#e87ba4",
    "tobacco": "#008300",
    "water":   "#6b6f76",
    "stone":   "#9a9289",
}

# Drawn dashed, to separate the control baseline from the commodity terms.
CONTROLS = {"water", "stone"}

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, nargs="+",
                help="one or more frequency_summary.csv files; the control "
                     "words live in their own extraction folder")
ap.add_argument("--words", nargs="+",
                default=["coffee", "tea", "sugar"])
ap.add_argument("--plain", action="store_true",
                help="draw the control words as solid lines like the rest")
ap.add_argument("--column", default="occurrences_found",
                help="occurrences_found (every match) or embeddings_saved "
                     "(what was actually written). They differ slightly.")
ap.add_argument("--linear", action="store_true",
                help="linear y-axis; the default is log, since the counts span "
                     "three orders of magnitude")
ap.add_argument("--out", default="embeddings_over_time_yearly.png")
args = ap.parse_args()

frames = []
for path in args.csv:
    part = pd.read_csv(path)
    # the yearly extraction keeps the column name `decade`, holding a year
    if "period" in part.columns and "decade" not in part.columns:
        part = part.rename(columns={"period": "decade"})
    frames.append(part)
    print(f"{path}: {len(part):,} rows, "
          f"{len(set(part['word']))} words "
          f"({', '.join(sorted(set(part['word'])))})")
df = pd.concat(frames, ignore_index=True)
df = df.drop_duplicates(subset=["word", "decade"], keep="first")

year_col = "decade"
col = args.column if args.column in df.columns else "occurrences_found"
if col not in df.columns:
    raise SystemExit(f"No column {args.column}. Found: {sorted(df.columns)}")

df[year_col] = df[year_col].astype(int)
have = set(df["word"])
words = [w for w in args.words if w in have]
missing = [w for w in args.words if w not in have]
if missing:
    print(f"\nnot in any of the files, skipped: {', '.join(missing)}")
if not words:
    raise SystemExit(f"None of {args.words} are in the files given.")

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

ends = []
for word in words:
    g = df[df.word == word].sort_values(year_col)
    colour = COLOURS.get(word)
    control = word in CONTROLS and not args.plain
    style = {"dashes": (5, 2.5), "lw": 1.4} if control else {"lw": 1.6}
    ax.plot(g[year_col], g[col], color=colour, label=word.capitalize(), **style)
    name = word.capitalize() + (" (control)" if control else "")
    ends.append([g[year_col].iloc[-1], g[col].iloc[-1], name, colour])

if not args.linear:
    ax.set_yscale("log")

# Direct labels at the line ends, nudged apart so they cannot collide
ends.sort(key=lambda e: e[1])
for j in range(1, len(ends)):
    if ends[j][1] < ends[j - 1][1] * 1.22:
        ends[j][1] = ends[j - 1][1] * 1.22
for x, y, name, colour in ends:
    ax.text(x + 1.5, y, name, fontsize=9.5, color=colour, va="center", ha="left")

lo, hi = df[year_col].min(), df[year_col].max()
# room on the right for the longest end label
ax.set_xlim(lo - 1, hi + 3 + 1.25 * max(len(e[2]) for e in ends))
ax.set_xticks(range((lo // 10) * 10, hi + 1, 10))

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Number of occurrences" + ("" if args.linear else " (log scale)"),
              fontsize=11)
drawn_controls = [w for w in words if w in CONTROLS]
title = "Occurrences per year, per term"
if drawn_controls and not args.plain:
    title += f" (dashed: {', '.join(drawn_controls)}, control)"
ax.set_title(title, fontsize=13, loc="left")
ax.grid(alpha=0.3)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(args.out, dpi=300, bbox_inches="tight")
print(f"Wrote {args.out}")

print(f"\n{'word':<9}{'years':>7}{'min':>9}{'max':>9}{'total':>12}"
      f"{'peak year':>11}")
for word in words:
    g = df[df.word == word]
    print(f"{word:<9}{len(g):>7}{g[col].min():>9,}{g[col].max():>9,}"
          f"{g[col].sum():>12,}{int(g.loc[g[col].idxmax(), year_col]):>11}")
