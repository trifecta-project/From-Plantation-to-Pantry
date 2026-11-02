#!/usr/bin/env python3
"""
plot_jsd_vs_frequency.py

For each target/control pair, plot the yearly JSD series above the yearly
frequency series, sharing an x-axis, so the two questions can be read off one
figure: does the control change less than its target, and does either measure
track frequency?

Layout: one column per pair, JSD on the top row, frequency on the bottom.
JSD and frequency are NEVER plotted on twinned y-axes -- a dual-scale chart
invites the reader to see a correspondence that the axes, not the data, put
there. Separate panels on a shared x let the eye make the comparison honestly.

INPUTS
------
--freq-csv   the yearly frequency table (word,year,count)
--jsd-csv    one or more CSVs holding yearly JSD. Column names are detected:
             the year comes from year / decade_start / decade / period, the
             value from js_divergence / jsd / divergence / js_distance, and
             the word from a 'word' column or, failing that, the filename.

NOTE ON THE JSD CONVENTION
--------------------------
If a file offers only js_distance, the script squares it to a divergence and
says so. Mixing a distance for one word with a divergence for another would
make the comparison meaningless.

USAGE

    python plot_jsd_vs_frequency.py \
        --freq-csv frequency_summary_yearly.csv \
        --jsd-csv jsd_coffee.csv jsd_sugar.csv jsd_presence.csv jsd_difference.csv \
        --pair coffee:presence --pair sugar:difference \
        --smooth 5 --out jsd_vs_frequency.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated two-slot categorical palette; blue = target, orange = control.
TARGET_C, CONTROL_C = "#5c3b9e", "#d6336c"
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2df", "#fcfcfb"

YEAR_COLS = ["year", "decade_start", "decade", "period", "year_start"]
JSD_COLS = ["js_divergence", "jsd", "divergence", "js_distance"]


def load_jsd(paths):
    """word -> DataFrame[year, jsd]. Column names are detected per file."""
    out = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"  [warn] not found, skipping: {path}")
            continue
        df = pd.read_csv(path)
        ycol = next((c for c in YEAR_COLS if c in df.columns), None)
        vcol = next((c for c in JSD_COLS if c in df.columns), None)
        if ycol is None or vcol is None:
            print(f"  [warn] {path}: no recognised year/JSD columns "
                  f"({list(df.columns)}), skipping")
            continue
        squared = False
        if vcol == "js_distance":
            df = df.assign(_v=df[vcol].astype(float) ** 2)
            squared = True
        else:
            df = df.assign(_v=df[vcol].astype(float))

        if "word" in df.columns:
            groups = df.groupby("word")
        else:
            stem = os.path.basename(path).rsplit(".", 1)[0]
            guess = stem.replace("jsd_", "").replace("_jsd", "")
            groups = [(guess, df)]
            print(f"  [note] {path}: no 'word' column; taking word from "
                  f"filename as '{guess}'")

        for word, g in groups:
            sub = (g[[ycol, "_v"]].rename(columns={ycol: "year", "_v": "jsd"})
                   .dropna().astype({"year": int}).sort_values("year"))
            out[str(word).strip().lower()] = sub
            note = "  (js_distance squared to divergence)" if squared else ""
            print(f"  loaded JSD for {word}: {len(sub)} points{note}")
    return out


def load_freq(path, words):
    want = {w.lower() for w in words}
    df = pd.read_csv(path)
    df["word"] = df["word"].astype(str).str.lower().str.strip()
    df = df[df["word"].isin(want)]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna().astype({"year": int})
    return {w: g[["year", "count"]].sort_values("year")
            for w, g in df.groupby("word")}


def smooth(s, window):
    if window <= 1:
        return np.asarray(s)
    return (pd.Series(s).rolling(window, center=True, min_periods=1)
            .mean().to_numpy())


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq-csv", required=True)
    ap.add_argument("--jsd-csv", nargs="+", required=True)
    ap.add_argument("--pair", action="append", required=True,
                    metavar="TARGET:CONTROL")
    ap.add_argument("--years", type=int, nargs=2, default=[1801, 1919])
    ap.add_argument("--smooth", type=int, default=5,
                    help="rolling-mean window for the plotted lines; the "
                         "printed correlations always use the raw series")
    ap.add_argument("--linear-freq", action="store_true",
                    help="linear instead of log frequency axis")
    ap.add_argument("--out", default="jsd_vs_frequency.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    pairs = []
    for spec in args.pair:
        if ":" not in spec:
            ap.error(f"--pair needs TARGET:CONTROL, got {spec!r}")
        t, c = spec.split(":", 1)
        pairs.append((t.strip().lower(), c.strip().lower()))

    words = [w for p in pairs for w in p]
    print("Loading JSD files...")
    jsd = load_jsd(args.jsd_csv)
    print("Loading frequencies...")
    freq = load_freq(args.freq_csv, words)

    missing = [w for w in words if w not in jsd]
    if missing:
        print(f"\nERROR: no JSD series for {missing}. "
              f"Available: {sorted(jsd)}", file=sys.stderr)
        sys.exit(1)
    missing = [w for w in words if w not in freq]
    if missing:
        print(f"\nERROR: no frequency rows for {missing}", file=sys.stderr)
        sys.exit(1)

    lo, hi = args.years
    ncol = len(pairs)
    fig, axes = plt.subplots(2, ncol, figsize=(6.4 * ncol, 6.4),
                             squeeze=False, sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for col, (target, control) in enumerate(pairs):
        ax_j, ax_f = axes[0][col], axes[1][col]
        for word, colour in ((target, TARGET_C), (control, CONTROL_C)):
            d = jsd[word]
            m = (d["year"] >= lo) & (d["year"] <= hi)
            ax_j.plot(d["year"][m], smooth(d["jsd"][m].to_numpy(), args.smooth),
                      "-", color=colour, linewidth=2.0, label=word,
                      solid_capstyle="round")
            f = freq[word]
            m = (f["year"] >= lo) & (f["year"] <= hi)
            ax_f.plot(f["year"][m], smooth(f["count"][m].to_numpy(), args.smooth),
                      "-", color=colour, linewidth=2.0, label=word,
                      solid_capstyle="round")

        ax_j.set_title(f"{target} vs {control}", fontsize=11.5, color=INK,
                       loc="left", pad=8)
        if not args.linear_freq:
            ax_f.set_yscale("log")
        for ax in (ax_j, ax_f):
            style(ax)
            ax.set_xlim(lo, hi)
        ax_f.set_xlabel("Year", fontsize=10, color=INK_SOFT)
        leg = ax_j.legend(loc="upper right", frameon=True, fontsize=9,
                          labelcolor=INK)
        leg.get_frame().set_facecolor(SURFACE)
        leg.get_frame().set_edgecolor(GRID)

    # Keep the axis labels short: two long labels in a two-row column collide.
    # The smoothing note belongs in the title, where it is stated once.
    axes[0][0].set_ylabel("JS divergence", fontsize=10, color=INK_SOFT)
    axes[1][0].set_ylabel("Occurrences per year", fontsize=10, color=INK_SOFT)
    sm = "" if args.smooth <= 1 else f", {args.smooth}-year rolling mean"
    fig.suptitle("Yearly semantic change and frequency, targets and controls"
                 + sm, fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    print(f"\nwrote {args.out}")

    # ---- the question the figure poses, answered numerically --------------
    from scipy.stats import spearmanr
    print("\nSpearman correlation, raw (unsmoothed) series:")
    print(f"  {'word':<14}{'JSD vs frequency':>20}{'JSD vs d(freq)':>18}"
          f"{'median JSD':>13}")
    print("  " + "-" * 63)
    for target, control in pairs:
        for word in (target, control):
            d = jsd[word].merge(freq[word], on="year", how="inner")
            d = d[(d["year"] >= lo) & (d["year"] <= hi)].sort_values("year")
            if len(d) < 4:
                print(f"  {word:<14}{'too few points':>20}")
                continue
            r1, p1 = spearmanr(d["jsd"], d["count"])
            dcount = d["count"].diff().abs()
            ok = dcount.notna()
            r2, p2 = spearmanr(d["jsd"][ok], dcount[ok])
            print(f"  {word:<14}{r1:>14.3f} (p={p1:.1g}){r2:>12.3f}"
                  f"{d['jsd'].median():>13.5f}")
    print("\nThe second column is the more pointed test: JSD against the "
          "year-on-year\nCHANGE in frequency, which is the confound a "
          "frequency-matched control exists\nto rule out.")


if __name__ == "__main__":
    main()
