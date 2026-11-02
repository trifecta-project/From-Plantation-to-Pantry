import argparse
import csv
import json
import math
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DECADES = [f"{y}s" for y in range(1800, 1920, 10)]

# One accent plus a context grey. Validated against the chart surface:
# normal-vision dE 18.0 and CVD dE 16.1 (protan) / 11.2 (tritan) between them,
# and the grey clears 3:1 contrast. Both survive greyscale printing.
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"
CONTEXT = "#d5d4ce"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#86857f"
GRID = "#ebeae4"


def load_counts(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def human(n):
    n = int(n)
    if n >= 1000:
        return f"~{n/1000:.0f}k"
    return str(n)


def axis_k(v):
    if v == 0:
        return "0"
    if v >= 1000:
        return f"{v/1000:.0f}k"
    return str(int(v))


def source_switch_index(aggregation_stats_path):
    """
    Index of the first decade where the dominant source collection changes.

    The per-FILE breakdown in the aggregation stats is authoritative, because
    reconstructing the split from the per-newspaper CSV is unreliable whenever
    a newspaper id appears under both source directories. Accepts the counts
    nested under "by_source" or at the top level, with either casing.
    """
    with open(aggregation_stats_path, encoding="utf-8") as fh:
        agg = json.load(fh)

    by_source = agg.get("by_source", agg)
    folded = {str(k).strip().lower(): v for k, v in by_source.items()
              if isinstance(v, dict)}

    def series(*aliases):
        for a in aliases:
            if a in folded:
                raw = folded[a]
                out = {}
                for d in DECADES:
                    bare = d.rstrip("s")
                    out[d] = raw.get(d, raw.get(bare, raw.get(int(bare), 0)))
                return out
        return None

    lwm = series("lwm", "lwm_articles", "living_with_machines")
    hmd = series("hmd", "hmd_articles", "heritage_made_digital")

    if lwm is None or hmd is None:
        raise SystemExit(
            f"Could not find the HMD and LwM per-decade counts in "
            f"{aggregation_stats_path}.\n"
            f"Top-level keys: {sorted(agg.keys())}\n"
            f"Nested keys: {sorted(folded.keys())}\n"
            f"Pass the file that holds the per-FILE source breakdown, or drop "
            f"--mark-source-transition.")

    was_hmd_leading = hmd[DECADES[0]] >= lwm[DECADES[0]]
    for i, d in enumerate(DECADES):
        is_hmd_leading = hmd[d] >= lwm[d]
        if is_hmd_leading != was_hmd_leading:
            return i
        was_hmd_leading = is_hmd_leading
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Articles per decade for the top-N newspapers, as small multiples.")
    ap.add_argument("--csv", required=True,
                    help="articles_by_newspaper_decade.csv from the counter script")
    ap.add_argument("--top-n", type=int, default=10,
                    help="number of newspapers to give a panel, ranked by total "
                         "articles (default 10)")
    ap.add_argument("--show-all-lines", action="store_true", default=True,
                    help="draw every OTHER newspaper as a light grey line behind "
                         "each panel, so the corpus shape stays visible. On by "
                         "default; --no-context turns it off.")  # all others, "
                         # not merely those outside --top-n
    ap.add_argument("--no-context", dest="show_all_lines", action="store_false",
                    help="omit the grey background lines")
    ap.add_argument("--free-y", action="store_true",
                    help="give each panel its own y-scale. Off by default: a "
                         "shared scale keeps the panels comparable, and the fact "
                         "that most titles are dwarfed by the largest is itself "
                         "part of the picture.")
    ap.add_argument("--ncols", type=int, default=None,
                    help="panels per row (default 5 for more than 8 panels, else 4)")
    ap.add_argument("--name-map", default=None,
                    help="optional CSV mapping BL ids to titles, with columns "
                         "newspaper_id,title or BL_ID,Title (e.g. "
                         "top30_newspapers_by_article_count.csv) -- if given, "
                         "panels use human-readable titles instead of raw BL ids")
    ap.add_argument("--title-fontsize", type=float, default=8.5,
                    help="panel heading size (default 8.5)")
    ap.add_argument("--max-title-lines", type=int, default=3,
                    help="how many lines a panel heading may wrap onto before "
                         "it is truncated (default 3). Names are wrapped to fit "
                         "the panel, not cut off.")
    ap.add_argument("--mark-source-transition", action="store_true",
                    help="draw a vertical dashed line at the decade where the "
                         "corpus's dominant source (LWM vs HMD) switches. "
                         "Requires --aggregation-stats for an accurate result.")
    ap.add_argument("--aggregation-stats", default=None,
                    help="path to the aggregation stats JSON from the ORIGINAL "
                         "aggregate_and_partition.py run, holding the per-FILE "
                         "source breakdown per decade. Required for "
                         "--mark-source-transition (see source_switch_index).")
    ap.add_argument("--figsize", default=None,
                    help="WIDTHxHEIGHT in inches; default scales with the grid")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default="articles_by_newspaper.png")
    ap.add_argument("--legend-fontsize", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.legend_fontsize is not None:
        print("NOTE: --legend-fontsize no longer applies -- each panel is headed "
              "by its newspaper name, so there is no legend. Ignoring it.")

    rows = load_counts(args.csv)
    rows.sort(key=lambda r: -int(r["total"]))
    top = rows[:args.top_n]
    rest = rows[args.top_n:]

    name_map = {}
    if args.name_map:
        with open(args.name_map, newline="", encoding="utf-8") as fh:
            # Accepts either newspaper_id,title or the BL_ID,Title columns of
            # top30_newspapers_by_article_count.csv.
            for row in csv.DictReader(fh):
                np_id = row.get("newspaper_id") or row.get("BL_ID")
                title = row.get("title") or row.get("Title")
                if np_id and title:
                    name_map[np_id] = title

    switch_idx = None
    if args.mark_source_transition:
        if not args.aggregation_stats:
            print("WARNING: --mark-source-transition requires "
                  "--aggregation-stats (the per-FILE source breakdown); "
                  "skipping the transition line. Reconstructing it from this "
                  "script's per-newspaper labels is unreliable (see --help).")
        else:
            switch_idx = source_switch_index(args.aggregation_stats)

    n = len(top)
    if n == 0:
        sys.exit("No newspapers to plot.")
    ncols = args.ncols or (5 if n > 8 else 4)
    nrows = math.ceil(n / ncols)

    if args.figsize:
        try:
            w, h = (float(v) for v in args.figsize.lower().split("x"))
        except ValueError:
            sys.exit(f"--figsize must look like 11x5, got {args.figsize!r}")
    else:
        w, h = 2.25 * ncols, 2.2 * nrows + 0.95

    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), dpi=args.dpi,
                             facecolor=SURFACE, squeeze=False,
                             sharex=True, sharey=not args.free_y)

    # Wrap the headings to the panel width rather than truncating them. The
    # tallest heading sets a single header height for every panel, so the grid
    # stays aligned instead of going ragged.
    panel_w_in = (w / ncols) * 0.80          # panel minus its margins
    title_fs = args.title_fontsize
    chars_per_line = max(10, int(panel_w_in * 72 / (title_fs * 0.52)))

    wrapped_titles = {}
    for r in top:
        raw = name_map.get(r["newspaper_id"], r["newspaper_id"]).strip().rstrip(".").strip()
        lines = textwrap.wrap(raw, chars_per_line) or [raw]
        if len(lines) > args.max_title_lines:
            lines = lines[:args.max_title_lines]
            lines[-1] = lines[-1][:chars_per_line - 1].rstrip() + "\u2026"
        wrapped_titles[r["newspaper_id"]] = "\n".join(lines)
    title_lines = max(t.count("\n") + 1 for t in wrapped_titles.values())

    # set_title anchors a heading by its BOTTOM, so a one-line name would sit
    # lower than a three-line one and the row of headings would come out
    # ragged. Padding each by the lines it is short of the tallest puts every
    # first line at the same height.
    line_pt = title_fs * 1.32
    title_pads = {k: 16 + (title_lines - (v.count("\n") + 1)) * line_pt
                  for k, v in wrapped_titles.items()}

    xs = range(len(DECADES))
    ymax = max(int(r[d]) for r in rows for d in DECADES)

    # Context is EVERY other title, not just those outside the top-N. In the
    # original single-panel chart the top-N were already drawn as coloured
    # lines in the same axes, so the grey layer correctly covered only the
    # remainder. Here each panel is its own axes, so restricting the grey to
    # the remainder would silently drop the nine largest titles from every
    # panel's backdrop -- the tallest grey line would top out around 200k and
    # the corpus's biggest contributor would never appear as context anywhere.
    all_series = {r["newspaper_id"]: [int(r[d]) for d in DECADES] for r in rows}

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor(SURFACE)

        if idx >= n:
            ax.axis("off")
            continue

        row = top[idx]
        focus_id = row["newspaper_id"]
        ys = all_series[focus_id]

        if args.show_all_lines:
            for other_id, cs in all_series.items():
                if other_id == focus_id:
                    continue
                ax.plot(xs, cs, color=CONTEXT, lw=0.6, zorder=1,
                        solid_capstyle="round")

        if switch_idx is not None:
            ax.axvline(switch_idx - 0.5, color=MUTED, ls=(0, (3, 3)), lw=0.9, zorder=2)

        ax.plot(xs, ys, color=ACCENT, lw=1.9, zorder=4, solid_capstyle="round")

        # Name and total on separate lines above the panel, so they cannot
        # collide however long the name is.
        ax.set_title(wrapped_titles[focus_id], fontsize=title_fs, color=TEXT,
                     loc="left", pad=title_pads[focus_id], linespacing=1.25)
        ax.text(0.0, 1.012, f"total {human(row['total'])}", transform=ax.transAxes,
                fontsize=7.3, color=MUTED, ha="left", va="bottom")

        ax.set_xlim(-0.4, len(DECADES) - 0.6)
        if args.free_y:
            # Scale to the panel's own series, not to the context behind it,
            # or every panel would be scaled by the corpus's largest title and
            # --free-y would do nothing. Context lines clip at the frame.
            ax.set_ylim(0, max(max(ys), 1) * 1.15)
        else:
            ax.set_ylim(0, ymax * 1.08)
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#dedcd6")
        ax.tick_params(length=0, labelsize=7.5, colors=TEXT2)

        is_bottom = idx >= n - ncols
        ax.set_xticks(list(xs))
        if is_bottom:
            ax.set_xticklabels([d[:4] for d in DECADES], rotation=90, fontsize=6.8)
        else:
            ax.set_xticklabels([])

        # With a shared y-axis, calling set_yticklabels([]) on one panel
        # propagates to every panel. Hide the labels per-axes instead.
        ax.yaxis.set_major_formatter(lambda v, _pos: axis_k(v))
        ax.tick_params(axis="y", labelleft=args.free_y or idx % ncols == 0)

    fig.text(0.008, 0.985, f"Number of articles per decade, top {n} newspapers",
             fontsize=12.5, fontweight="bold", color=TEXT, va="top")
    #sub = ("Each panel is one title; every other title is drawn behind it in grey."
           #if args.show_all_lines else "Each panel is one title.")
    #if args.free_y:
        #if args.show_all_lines:
            #sub += " Each panel scaled to its own title, so grey lines may clip."
    #else:
        #sub += " Shared vertical scale."
    #if switch_idx is not None:
        #sub += f"  Dashed line: HMD → LwM dominant source."
    #fig.text(0.008, 0.946, sub, fontsize=8.2, color=TEXT2, va="top")

    fig.supylabel("Published articles", fontsize=9, color=TEXT2, x=0.004)
    fig.tight_layout(rect=(0.012, 0.0, 1.0, 0.925))
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
