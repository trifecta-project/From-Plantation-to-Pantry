"""
Dependency profile of each target noun, per cluster or per period, with spaCy.

Reads cluster_all_snippets_<word>.csv written by cluster_jsd_by_period.py
(run with --all-snippets): one row per occurrence with its cluster, its period
and its context window, in which the target word is upper-cased. Each window is
parsed with spaCy, the target token is located, and its relations are counted:

  amod:<lemma>           adjective modifying the target   (fine coffee)
  compound:<lemma>       noun modifying the target         (breakfast coffee)
  compound-of:<lemma>    target modifying a noun           (coffee house)
  <dep>-of:<lemma>       target as a verbal argument       (dobj-of:roast)
  nummod:<lemma>, nmod-of:<lemma>   optional (--relations)

Lemmas must be alphabetic, at least two characters long and not the target
itself; modifiers and verbs that are stop words are excluded too. Relations seen fewer than --min-count times in a group are
dropped before shares are computed.

Grouping (--group-by):
  cluster          one group per cluster
  period           one group per time bin of --period-width years
                   (default 10; bins start on a multiple of the width,
                   e.g. 1800s, 1810s, ...; --anchor first starts them at the
                   first year present)
  cluster_period   one group per cluster and time bin
  decade, cluster_decade   aliases for period / cluster_period with width 10

Above --max-table-cols groups, the LaTeX table is written one row per group.

Usage
-----
# per decade (10-year bins of the yearly clusters)
python dep_profile_across_periods.py --words coffee tea sugar \
    --csv-dir <output dir of cluster_jsd_by_period.py> \
    --group-by period --period-width 10 --out-dir results/dependencies_by_period

# per cluster
python dep_profile_across_periods.py --words coffee tea sugar \
    --csv-dir <output dir of cluster_jsd_by_period.py> \
    --group-by cluster --out-dir results/dependencies_by_cluster
"""

import os
import re
import csv
import argparse
import random
from collections import Counter

try:
    import spacy
except ImportError:
    spacy = None

# spaCy (CLEAR-style) dependency labels we treat as verbal-argument relations
VERB_ARG_DEPS = {"nsubj", "nsubjpass", "dobj", "obj", "iobj", "obl", "agent"}

YEAR_RE = re.compile(r"(\d{4})")


def clean_snippet_and_locate(raw_snippet, word):
    """The cluster CSVs store a KWIC display snippet produced by kwic():
        "... " + <window of tokens, target UPPER-CASED> + " ..."
    There is no stored position integer here (unlike the HDF5 usage files),
    so we (1) strip the leading/trailing ellipsis tokens kwic() adds, and
    (2) locate the target by finding the single ALL-CAPS token -- kwic()
    only ever upper-cases the target itself.

    Returns (clean_text, position) or (None, None) if no confident match.
    """
    toks = raw_snippet.split(" ")
    while toks and toks[0] in ("\u2026", "..."):
        toks = toks[1:]
    while toks and toks[-1] in ("\u2026", "..."):
        toks = toks[:-1]

    wl = word.lower()
    cands = []
    for i, tok in enumerate(toks):
        cleaned = re.sub(r"[^A-Za-z]", "", tok)
        if cleaned and cleaned.isupper() and len(cleaned) >= 2:
            cands.append(i)

    idx = None
    if len(cands) == 1:
        idx = cands[0]
    elif len(cands) > 1:
        for i in cands:
            cleaned = re.sub(r"[^A-Za-z]", "", toks[i]).lower()
            if cleaned in (wl, wl + "s"):
                idx = i
                break
        if idx is None:
            idx = cands[0]

    if idx is None:
        return None, None
    return " ".join(toks), idx


def char_offset(snippet, position):
    """Character offset of the position-th space-delimited token in the snippet."""
    toks = snippet.split(" ")
    if not (0 <= position < len(toks)):
        return None
    return len(" ".join(toks[:position])) + (1 if position > 0 else 0)


def find_target_token(doc, snippet, position, word):
    """Locate the spaCy token for the intended occurrence.

    Uses the character offset of the position-th whitespace token (so the right
    occurrence is chosen when the window contains several), with a fallback to
    the word-matching token whose start char is nearest that offset.
    """
    off = char_offset(snippet, position)
    wl = word.lower()
    if off is not None:
        for t in doc:
            if t.idx <= off < t.idx + len(t.text):
                if t.text.lower() == wl or t.lemma_.lower() == wl or wl in t.text.lower():
                    return t
                break  # found the spanning token but it doesn't match -> fall back
    cands = [t for t in doc if t.text.lower() == wl or t.lemma_.lower() == wl]
    if not cands:
        return None
    if off is None:
        return cands[len(cands) // 2]
    return min(cands, key=lambda t: abs(t.idx - off))


def _ok_lemma(tok, wl):
    lm = tok.lemma_.lower().strip()
    return lm.isalpha() and len(lm) >= 2 and lm != wl and not tok.is_stop, lm


def extract_relations(tok, word, want):
    """Return the list of relation strings for one target-noun token."""
    rels = []
    wl = word.lower()
    # --- children of the target noun (modifiers) ---
    for c in tok.children:
        if c.is_punct or c.is_space:
            continue
        ok, lemma = _ok_lemma(c, wl)
        if not ok:
            continue
        if c.dep_ == "amod" and "amod" in want:
            rels.append(f"amod:{lemma}")
        elif c.dep_ == "compound" and "compound" in want:
            rels.append(f"compound:{lemma}")
        elif c.dep_ == "nummod" and "nummod" in want:
            rels.append(f"nummod:{lemma}")
    # --- the target's own head relation ---
    h = tok.head
    if h is not None and h is not tok:
        ok_h, hl = _ok_lemma(h, wl)
        if tok.dep_ == "compound" and h.pos_ in ("NOUN", "PROPN") and "compound-of" in want:
            if hl.isalpha() and len(hl) >= 2 and hl != wl:
                rels.append(f"compound-of:{hl}")
        elif tok.dep_ in VERB_ARG_DEPS and h.pos_ in ("VERB", "AUX") and "verb" in want:
            if ok_h:
                rels.append(f"{tok.dep_}-of:{hl}")
        elif tok.dep_ in ("pobj", "obj") and h.pos_ == "ADP" and "nmod-of" in want:
            g = h.head
            ok_g, gl = _ok_lemma(g, wl)
            if g.pos_ in ("NOUN", "PROPN") and ok_g:
                rels.append(f"nmod-of:{gl}")
    return rels


# ---------------------------------------------------------------------------
# Grouping -- the only part that differs from the per-cluster script
# ---------------------------------------------------------------------------
def parse_year(value):
    """Year out of a period cell: '1884', '1880s', '1880-1884' -> 1884/1880/1880."""
    m = YEAR_RE.search(str(value))
    return int(m.group(1)) if m else None


def period_label(start, width):
    if width == 1:
        return f"{start}"
    if width == 10 and start % 10 == 0:
        return f"{start}s"
    return f"{start}-{start + width - 1}"


def natural_key(key):
    """Sort cluster0..cluster10 and 1800..1910 in the order a reader expects."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", key)]


def read_cluster_rows(csv_path):
    """(cluster, year, snippet) for every usable row (word_visible == 1)."""
    out = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("word_visible", "")).strip() not in ("1", "True", "true"):
                continue
            snippet = row.get("snippet", "")
            if not snippet:
                continue
            out.append((row.get("cluster", ""),
                        parse_year(row.get("decade", "")),
                        snippet))
    return out


def group_rows(rows, group_by, width, anchor_mode="calendar"):
    """{group key: [snippets]}, with years binned into periods of `width`."""
    years = [y for _, y, _ in rows if y is not None]
    if not years:
        first = 0
    elif anchor_mode == "first":
        first = min(years)
    else:
        first = (min(years) // width) * width

    by_group = {}
    n_no_year = 0
    for cluster, year, snippet in rows:
        if group_by == "cluster":
            key = f"cluster{cluster}"
        else:
            if year is None:
                n_no_year += 1
                continue
            start = first + ((year - first) // width) * width
            plabel = period_label(start, width)
            key = plabel if group_by == "period" else f"cluster{cluster}_{plabel}"
        by_group.setdefault(key, []).append(snippet)
    return by_group, n_no_year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="+",
                    default=["coffee", "tea", "sugar", "cocoa"])
    ap.add_argument("--csv-dir", required=True,
                    help="dir containing cluster_all_snippets_<word>.csv "
                         "(or _clean.csv, via --csv-suffix)")
    ap.add_argument("--csv-suffix", default="",
                    help="e.g. '_clean' to read "
                         "cluster_all_snippets_<word><suffix>.csv instead of "
                         "the default cluster_all_snippets_<word>.csv")
    ap.add_argument("--group-by",
                    choices=["cluster", "period", "decade", "cluster_period",
                             "cluster_decade"],
                    default="cluster")
    ap.add_argument("--period-width", type=int, default=None,
                    help="years per time bin. Default 10, i.e. decades -- so a "
                         "yearly cluster CSV gives the familiar decade table. "
                         "Use 1 for year on year, 5 for the paper's middle slice.")
    ap.add_argument("--anchor", choices=["calendar", "first"], default="calendar",
                    help="'calendar' starts the bins on a multiple of the width "
                         "(1800s, 1810s, ...), which is what the decade tables "
                         "use. 'first' starts them at the first year present, "
                         "which gives 1801-1810 when 1800 is missing.")
    ap.add_argument("--max-per-group", type=int, default=0,
                    help="cap snippets parsed per group; 0 = all")
    ap.add_argument("--relations", nargs="+",
                    default=["amod", "compound", "compound-of", "verb"],
                    choices=["amod", "compound", "compound-of", "nummod", "verb",
                             "nmod-of"],
                    help="which relation types to count")
    ap.add_argument("--topn", type=int, default=10, help="rows in the per-group table")
    ap.add_argument("--min-count", type=int, default=3,
                    help="drop relations occurring fewer than this many times in a "
                         "group BEFORE computing shares; filters OCR-induced "
                         "singletons (set 1 to keep everything)")
    ap.add_argument("--max-table-cols", type=int, default=8,
                    help="above this many groups the LaTeX table is written "
                         "one row per group instead of one column per group")
    ap.add_argument("--model", default="en_core_web_sm",
                    help="spaCy model (en_core_web_sm fast; en_core_web_lg better)")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--n-process", type=int, default=1,
                    help="spaCy parallel processes (raise on a multi-core server)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    # Aliases, so the old commands keep working
    group_by = args.group_by
    width = args.period_width
    if group_by in ("decade", "cluster_decade"):
        group_by = "period" if group_by == "decade" else "cluster_period"
        width = 10
    if width is None:
        width = 10
    if width < 1:
        raise SystemExit("--period-width must be at least 1")

    if spacy is None:
        raise SystemExit("spaCy not installed. Run: pip install spacy && "
                         "python -m spacy download en_core_web_sm")
    try:
        nlp = spacy.load(args.model, disable=["ner", "textcat"])
    except OSError:
        raise SystemExit(f"spaCy model '{args.model}' not found. Run: "
                         f"python -m spacy download {args.model}")

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    want = set(args.relations)

    # The filenames carry the grouping, so a folder holding several groupings
    # stays readable
    tag = group_by if group_by == "cluster" else f"{group_by}{width}"

    for word in args.words:
        print(f"\n================  {word}  ================")
        csv_path = os.path.join(args.csv_dir,
                                f"cluster_all_snippets_{word}{args.csv_suffix}.csv")
        if not os.path.exists(csv_path):
            print(f"  Missing: {csv_path}")
            print(f"  (the clustering script writes it only with --all-snippets)")
            continue

        # bucket raw snippets by group first, so sampling is per-group and
        # unbiased regardless of how unevenly sized the clusters/periods are
        rows = read_cluster_rows(csv_path)
        by_group, n_no_year = group_rows(rows, group_by, width, args.anchor)
        if n_no_year:
            print(f"  {n_no_year} row(s) had no readable year and were skipped")
        print(f"  {len(rows):,} usable rows -> {len(by_group)} group(s)")

        per_group_counts = {}
        per_group_raw_total = {}
        for key in sorted(by_group, key=natural_key):
            snippets_raw = by_group[key]
            if args.max_per_group and len(snippets_raw) > args.max_per_group:
                snippets_raw = random.sample(snippets_raw, args.max_per_group)

            # locate targets BEFORE parsing (cheap), then batch-parse only the
            # usable ones through spaCy's nlp.pipe for speed
            clean_texts, positions = [], []
            n_unlocated = 0
            for raw_sn in snippets_raw:
                clean_text, pos = clean_snippet_and_locate(raw_sn, word)
                if clean_text is None:
                    n_unlocated += 1
                    continue
                clean_texts.append(clean_text)
                positions.append(pos)

            counts = Counter()
            for doc, pos, sn in zip(
                    nlp.pipe(clean_texts, batch_size=args.batch_size,
                             n_process=args.n_process),
                    positions, clean_texts):
                tok = find_target_token(doc, sn, pos, word)
                if tok is None:
                    continue
                for rel in extract_relations(tok, word, want):
                    counts[rel] += 1

            per_group_raw_total[key] = sum(counts.values())

            if args.min_count > 1:
                counts = Counter({r: n for r, n in counts.items()
                                  if n >= args.min_count})
            if counts:
                per_group_counts[key] = counts
                print(f"  [{key}] {len(snippets_raw)} snippets, {n_unlocated} unlocated, "
                      f"{sum(counts.values())} relations (>= {args.min_count}/group)",
                      flush=True)

        if not per_group_counts:
            print(f"  no relations for '{word}', skipping")
            continue

        keys = sorted(per_group_counts, key=natural_key)

        # A sample too small for --min-count produces a table whose entries are
        # nearly all AT the threshold, so which relations appear in the top-N is
        # close to arbitrary and the shares are quantised. Say so rather than
        # letting the numbers look authoritative.
        kept = sum(sum(per_group_counts[k].values()) for k in keys)
        raw = sum(per_group_raw_total.get(k, 0) for k in keys)
        at_floor = sum(1 for k in keys
                       for _, n in per_group_counts[k].most_common(args.topn)
                       if n == args.min_count)
        shown = sum(min(args.topn, len(per_group_counts[k])) for k in keys)
        if raw and (kept / raw < 0.5 or (shown and at_floor / shown > 0.3)):
            print(f"\n  WARNING: --min-count {args.min_count} dropped "
                  f"{raw - kept:,} of {raw:,} relations ({1 - kept / raw:.0%}), "
                  f"and {at_floor}/{shown} of the relations shown sit exactly at "
                  f"the threshold. The sample is too small for this filter: the "
                  f"top-N is mostly ties and the shares are quantised. Raise "
                  f"--max-per-group (or set 0 for all) before reading anything "
                  f"into these numbers.")

        cp = os.path.join(args.out_dir, f"deps_by_{tag}_{word}.csv")
        with open(cp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["word", "group", "relation", "count", "share"])
            for key in keys:
                tot = sum(per_group_counts[key].values())
                for rel, n in per_group_counts[key].most_common():
                    w.writerow([word, key, rel, n, round(n / tot, 4)])
        print(f"  wrote {cp}")

        tp = os.path.join(args.out_dir, f"deps_table_by_{tag}_{word}.txt")
        lines = [f"Top {args.topn} lemmatised dependencies of '{word}' per "
                 f"{group_by}" + ("" if group_by == "cluster"
                                  else f" ({width}-year bins)"),
                 "(share of all dependencies of the group in brackets)", ""]
        for key in keys:
            tot = sum(per_group_counts[key].values())
            lines.append(f"{key}  (total deps = {tot})")
            for rel, n in per_group_counts[key].most_common(args.topn):
                lines.append(f"    {rel}  ({n / tot:.2f})")
            lines.append("")
        table = "\n".join(lines)
        print("\n" + table)
        with open(tp, "w", encoding="utf-8") as fh:
            fh.write(table)
        print(f"  wrote {tp}")

        topn_per = {k: per_group_counts[k].most_common(args.topn) for k in keys}
        if len(keys) <= args.max_table_cols:
            lx = [r"\begin{tabular}{l" + "l" * len(keys) + "}", r"\toprule",
                  "rank & " + " & ".join(keys) + r" \\", r"\midrule"]
            for r in range(args.topn):
                cells = []
                for k in keys:
                    tot = sum(per_group_counts[k].values())
                    if r < len(topn_per[k]):
                        rel, n = topn_per[k][r]
                        cells.append(f"{rel.replace('_', chr(92)+'_')} ({n/tot:.2f})")
                    else:
                        cells.append("--")
                lx.append(f"{r+1} & " + " & ".join(cells) + r" \\")
            lx += [r"\bottomrule", r"\end{tabular}"]
        else:
            # one ROW per group: twelve periods will not fit across a page
            ncol = min(args.topn, 6)
            lx = [r"\begin{tabular}{l" + "l" * ncol + "}", r"\toprule",
                  group_by + " & " + " & ".join(str(i + 1) for i in range(ncol))
                  + r" \\", r"\midrule"]
            for k in keys:
                tot = sum(per_group_counts[k].values())
                cells = []
                for r in range(ncol):
                    if r < len(topn_per[k]):
                        rel, n = topn_per[k][r]
                        cells.append(f"{rel.replace('_', chr(92)+'_')} ({n/tot:.2f})")
                    else:
                        cells.append("--")
                lx.append(f"{k} & " + " & ".join(cells) + r" \\")
            lx += [r"\bottomrule", r"\end{tabular}"]
            print(f"  ({len(keys)} groups > --max-table-cols "
                  f"{args.max_table_cols}: LaTeX written one row per group, "
                  f"top {ncol} relations)")

        xp = os.path.join(args.out_dir, f"deps_table_by_{tag}_{word}.tex")
        with open(xp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lx))
        print(f"  wrote {xp}")


if __name__ == "__main__":
    main()
