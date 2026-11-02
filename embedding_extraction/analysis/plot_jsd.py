#!/usr/bin/env python3
"""
Plot the JSD CSVs written by cluster_jsd_by_period.py, at year or decade level.

  * The period step is read from the data (decade_end - decade_start), so
    transitions are labelled "1883->1884" at year level and "1870s->1880s" at
    decade level.
  * Above 20 transitions the x-axis is numeric, with ticks on round years.
  * If the clustering was run with --permute, the permutation noise floor
    (null mean) is drawn in grey, and transitions that are significant after
    Benjamini-Hochberg are drawn as filled markers.
  * --smooth N adds a centred rolling median over N periods.

The CSVs use the column names decade_start / decade_end whatever the
granularity.

Usage
-----
python plot_jsd.py --folder <output dir of cluster_jsd_by_period.py> \
    --words coffee tea sugar --combined --smooth 5
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORDS = ["coffee", "tea", "sugar", "cocoa"]

# Validated categorical order on a light surface; every adjacent pair clears the
# colour-vision separation floor.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#86857f", "#e6e5df"
C_NULL = "#b9b8b2"


def find_jsd_csv(folder, word):
    """
    Locate jsd_<word>.csv. The clustering script can be run with one output
    directory per word (<folder>/<word>/jsd_<word>.csv) or with a single shared
    one (<folder>/jsd_<word>.csv); accept either, and fall back to a recursive
    search so a deeper nesting still works.
    """
    for fp in (os.path.join(folder, word, f"jsd_{word}.csv"),
               os.path.join(folder, f"jsd_{word}.csv")):
        if os.path.exists(fp):
            return fp
    hits = sorted(glob.glob(os.path.join(folder, "**", f"jsd_{word}.csv"),
                            recursive=True))
    if len(hits) > 1:
        print(f"         [warn] {len(hits)} files named jsd_{word}.csv under "
              f"{folder}; using {hits[0]}")
    return hits[0] if hits else None


def load_jsd(folder, word, metric):
    """
    Returns a dict with the series and whatever extras the CSV carries, or None.

    The permutation columns are optional, so a CSV from a run without
    --permute still plots -- just without the noise floor.
    """
    fp = find_jsd_csv(folder, word)
    if fp is None:
        return None
    df = pd.read_csv(fp)
    if df.empty:
        return None
    df = df.sort_values("decade_start")

    col = "js_divergence" if metric == "divergence" else "js_distance"
    out = {
        "path": fp,
        "start": df["decade_start"].astype(int).to_numpy(),
        "end": df["decade_end"].astype(int).to_numpy(),
        "val": df[col].astype(float).to_numpy(),
        "null": None,
        "sig": None,
        "z": None,
    }

    if "null_mean_divergence" in df.columns:
        null = df["null_mean_divergence"].astype(float).to_numpy()
        # the null is computed on the divergence; convert if plotting distance
        out["null"] = np.sqrt(null) if metric == "distance" else null
    if "significant" in df.columns:
        out["sig"] = df["significant"].astype(str).str.lower().isin(
            ["true", "1", "yes"]).to_numpy()
    if "z" in df.columns:
        out["z"] = df["z"].astype(float).to_numpy()
    return out


def detect_granularity(series):
    """
    ('year'|'decade', step) read from the data rather than assumed.

    A decade run has every transition ten years wide and every boundary on a
    multiple of ten; anything else is treated as years.
    """
    steps, starts = [], []
    for s in series.values():
        steps += list(s["end"] - s["start"])
        starts += list(s["start"])
    if not steps:
        return "year", 1
    step = int(np.median(steps))
    if step == 10 and all(v % 10 == 0 for v in starts):
        return "decade", 10
    return "year", max(step, 1)


def rolling_median(xs, ys, window):
    """Centred rolling median in period units."""
    if window <= 1 or len(xs) < window:
        return [], []
    half = window // 2
    ox, oy = [], []
    for i in range(len(xs)):
        vals = [y for x, y in zip(xs, ys)
                if xs[i] - half <= x <= xs[i] + half and np.isfinite(y)]
        if len(vals) >= max(3, window // 2):
            ox.append(xs[i])
            oy.append(float(np.median(vals)))
    return ox, oy


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9, colors=TEXT2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#dedcd6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="last4_w50_yearly_balance")
    ap.add_argument("--words", nargs="+", default=WORDS)
    ap.add_argument("--metric", choices=["divergence", "distance"],
                    default="divergence")
    ap.add_argument("--combined", action="store_true",
                    help="also write one overlay figure with all words "
                         "(valid: JSD is k-independent)")
    ap.add_argument("--show-max", action="store_true",
                    help="draw the theoretical maximum (ln2 / sqrt(ln2)) as a "
                         "reference line")
    ap.add_argument("--no-null", dest="show_null", action="store_false",
                    help="do not draw the permutation noise floor even when the "
                         "CSV has it")
    ap.set_defaults(show_null=True)
    ap.add_argument("--smooth", type=int, default=0,
                    help="rolling-median window in period units (try 5 at year "
                         "level). 0 = off.")
    ap.add_argument("--ymax", type=float, default=None,
                    help="fix the y-axis upper limit")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or args.folder
    os.makedirs(out_dir, exist_ok=True)

    theo_max = np.log(2) if args.metric == "divergence" else np.sqrt(np.log(2))
    ylab = ("Jensen-Shannon divergence" if args.metric == "divergence"
            else "Jensen-Shannon distance")

    series = {}
    print(f"{'word':8s} {'transitions':>11s}   {args.metric} range")
    for word in args.words:
        res = load_jsd(args.folder, word, args.metric)
        if res is None:
            print(f"{word:8s} [skip] jsd_{word}.csv not found / empty")
            continue
        series[word] = res
        v = res["val"]
        extra = ""
        if res["sig"] is not None:
            extra = f"   {int(res['sig'].sum())}/{len(v)} above the noise floor"
        print(f"{word:8s} {len(v):>11d}   {v.min():.4f}-{v.max():.4f}{extra}")

    if not series:
        found = sorted(glob.glob(os.path.join(args.folder, "**", "jsd_*.csv"),
                                 recursive=True))
        msg = [f"Nothing to plot. No jsd_<word>.csv matched under {args.folder!r}."]
        if found:
            msg.append("Files that ARE present:")
            msg += [f"  {p}" for p in found[:20]]
            msg.append("Check --words against those filenames.")
        else:
            msg.append("No jsd_*.csv anywhere under that folder -- check "
                       "--folder, or whether the clustering run finished.")
        raise SystemExit("\n".join(msg))

    granularity, step = detect_granularity(series)
    n_trans = max(len(s["val"]) for s in series.values())
    numeric_x = n_trans > 20
    suffix = "s" if granularity == "decade" else ""
    print(f"\nGranularity: {granularity} (step {step}); "
          f"{n_trans} transitions, "
          f"{'numeric' if numeric_x else 'categorical'} x-axis")

    def tlabel(a, b):
        return f"{a}{suffix}→{b}{suffix}"

    # ---- per-word figures ------------------------------------------------
    for i, (word, s) in enumerate(series.items()):
        colour = PALETTE[i % len(PALETTE)]
        end, val = s["end"], s["val"]
        fig, ax = plt.subplots(figsize=(10 if numeric_x else 7.0, 4.2),
                               dpi=200, facecolor=SURFACE)
        _style(ax)

        x = end if numeric_x else np.arange(len(end))

        if s["null"] is not None and args.show_null:
            ax.plot(x, s["null"], lw=1.2, color=C_NULL, zorder=2,
                    label="noise floor (permutation null)")

        ax.plot(x, val, lw=1.0 if args.smooth else 2.0,
                alpha=0.35 if args.smooth else 1.0, color=colour,
                marker=None if numeric_x else "o", ms=5, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=ylab if not args.smooth else None)

        if args.smooth:
            sx, sy = rolling_median(x, val, args.smooth)
            if sx:
                ax.plot(sx, sy, lw=2.4, color=colour, zorder=4,
                        label=f"{args.smooth}-{granularity} rolling median")

        if s["sig"] is not None and s["sig"].any():
            ax.scatter(np.asarray(x)[s["sig"]], val[s["sig"]], s=22,
                       color=colour, edgecolors=SURFACE, linewidths=0.7,
                       zorder=5, label="above the floor (BH)")

        if not numeric_x:
            ax.set_xticks(x)
            ax.set_xticklabels([f"{d}{suffix}" for d in end],
                               rotation=45 if len(end) > 8 else 0, ha="right")
        ax.set_xlabel(f"{granularity} (change since the previous {granularity})",
                      fontsize=10, color=TEXT2)
        ax.set_ylabel(ylab, fontsize=10, color=TEXT2)
        top = args.ymax if args.ymax else max(val.max() * 1.15, 1e-3)
        ax.set_ylim(0, top)
        if args.show_max:
            ax.axhline(theo_max, ls="--", lw=1, color="grey", alpha=0.7,
                       label=f"max = {theo_max:.3f}")
        ax.set_title(f"'{word}': usage-distribution change between "
                     f"{granularity}s (JSD)", fontsize=12, color=TEXT, loc="left")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        out = os.path.join(out_dir, f"perword_jsd_{word}.png")
        fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # ---- combined overlay (valid because JSD is k-independent) ------------
    if args.combined:
        fig, ax = plt.subplots(figsize=(11 if numeric_x else 10, 4.6),
                               dpi=200, facecolor=SURFACE)
        _style(ax)

        all_end = sorted({int(d) for s in series.values() for d in s["end"]})
        end_to_idx = {d: i for i, d in enumerate(all_end)}

        ends = []
        for i, word in enumerate(args.words):
            if word not in series:
                continue
            s = series[word]
            colour = PALETTE[i % len(PALETTE)]
            x = s["end"] if numeric_x else [end_to_idx[d] for d in s["end"]]
            if args.smooth:
                ax.plot(x, s["val"], lw=0.8, color=colour, alpha=0.3, zorder=2)
                sx, sy = rolling_median(list(x), list(s["val"]), args.smooth)
                if sx:
                    ax.plot(sx, sy, lw=2.2, color=colour, zorder=4, label=word)
                    ends.append([sx[-1], sy[-1], word, colour])
                    continue
            ax.plot(x, s["val"], marker=None if numeric_x else "o", ms=5,
                    lw=2.0, color=colour, label=word, zorder=4)
            ends.append([list(x)[-1], s["val"][-1], word, colour])

        if not numeric_x:
            ax.set_xticks(list(range(len(all_end))))
            ax.set_xticklabels([tlabel(d - step, d) for d in all_end],
                               rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(granularity, fontsize=10, color=TEXT2)
        ax.set_ylabel(ylab, fontsize=10, color=TEXT2)
        gmax = max(s["val"].max() for s in series.values())
        ax.set_ylim(0, args.ymax if args.ymax else gmax * 1.15)
        if args.show_max:
            ax.axhline(theo_max, ls="--", lw=0.8, color="grey")

        if numeric_x:
            # direct end labels: a six-line legend on a crowded year-level plot
            # is a lookup table, the labels are read in place
            xlo, xhi = ax.get_xlim()
            ax.set_xlim(xlo, xhi + (xhi - xlo) * 0.12)
            ends.sort(key=lambda e: e[1])
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            for j in range(1, len(ends)):
                if ends[j][1] < ends[j - 1][1] + span * 0.05:
                    ends[j][1] = ends[j - 1][1] + span * 0.05
            for xx, yy, name, colour in ends:
                ax.text(xx, yy, f" {name}", fontsize=9, color=colour,
                        va="center", ha="left", zorder=6)
        else:
            ax.legend(frameon=False, fontsize=9, ncol=3)

        ax.set_title("Sense-distribution change across words (JSD)",
                     fontsize=12.5, color=TEXT, loc="left")
        fig.tight_layout()
        out = os.path.join(
            out_dir,
            f"jsd_all_words_{os.path.basename(args.folder.rstrip('/'))}.png")
        fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote combined overlay: {out}")

    print(f"Per-word figures in: {out_dir}")


if __name__ == "__main__":
    main()