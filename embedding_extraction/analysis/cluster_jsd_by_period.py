#!/usr/bin/env python3
"""
Shared sense clustering, cluster distributions and Jensen-Shannon divergence
(JSD) for one target word, per year or per decade.

All occurrence embeddings of the word, from every period, are pooled and
clustered once with k-means, so every period is described over the same k
clusters. Each period's cluster distribution (the share of its occurrences in
each cluster) is then compared with the next period's by JSD (base 2; both the
divergence and scipy's distance, its square root, are written).

  * Periods are discovered from the HDF5 filenames, so a yearly directory and
    a decade directory both work. --slice-width groups years into wider slices.
  * k is chosen by silhouette score over --k-range, or fixed with --k. The
    candidate fits and scores use --silhouette-sample pooled embeddings
    (default 2,000), or all of them when the value is at least the pooled
    count; the results in the paper use all of them. k is never chosen per
    period, which is what keeps the distributions comparable over time.
  * --balance-clustering weights each embedding by the inverse of its period's
    size, so that every period contributes equally to where the centroids lie,
    while all embeddings are used.
  * --permute r tests each JSD against sampling noise: the cluster labels of
    the two periods are pooled, shuffled and split again at the original sizes,
    r times. p = (number of permuted JSDs >= observed, observed included) /
    (r + 1); p-values are corrected across a word's transitions with
    Benjamini-Hochberg. The CSV also gives the null mean, the excess over it
    and z, which still discriminates when p reaches its floor.

For stable cluster ids across reruns, run with OMP_NUM_THREADS=1.

Usage
-----
OMP_NUM_THREADS=1 python cluster_jsd_by_period.py --word coffee \
    --emb-dir <yearly embeddings dir> --k 6 --silhouette-sample 100000000 \
    --balance-clustering --permute 999 --all-snippets --out-dir <output dir>

for w in coffee tea sugar; do
  OMP_NUM_THREADS=1 python cluster_jsd_by_period.py --word $w \
      --emb-dir <yearly embeddings dir> --silhouette-sample 100000000 \
      --balance-clustering --permute 999 --all-snippets --out-dir <output dir>
done
"""

import os
import csv
import glob
import re
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher

try:
    from scipy.stats import false_discovery_control
except ImportError:
    false_discovery_control = None

FILE_RE = re.compile(r"commodity_embeddings_(\d{4})\.h5$")

# ---- sense labels matched to clusters BY OVERALL SHARE, per word ----
# K-means may permute cluster ids between runs, so labels are matched to
# clusters by each cluster's share of ALL pooled usages, via optimal
# one-to-one assignment (smallest total share difference).
#
# The number of labels must equal the k actually selected for that word,
# otherwise the script falls back to plain cluster ids and prints a warning.
# Shares are read from cluster_snippets_<word>.txt, which prints
# "n=..., X% of all usages" for each cluster.
#
# NOTE: if two clusters have similar shares the matching is ambiguous and the
# script will warn. In that case set LABELS_BY_CLUSTER_ID below instead, which
# is exact (cluster ids are stable across reruns with a fixed seed and
# OMP_NUM_THREADS=1).

SENSE_LABELS_BY_SHARE = {
    "coffee": [
        (0.25, "cluster 0--coffee as a household provision"),
        (0.26, "cluster 1--coffee as a branded packaged product"),
        (0.10, "cluster 2--coffee as a social institution (coffee houses)"),
        (0.16, "cluster 3--coffee as a traded commodity"),
        (0.16, "cluster 4--coffee as material culture (services and pots)"),
        (0.06, "cluster 5--coffee as a wholesale trade range (plural form)"),
    ],
    "tea": [
        (0.12, "cluster 0--various types of tea being sold or advertised"),
        (0.34, "cluster 1--loose tea leaves in retail and trade"),
        (0.25, "cluster 2--tea as a drink/consumable"),
        (0.13, "cluster 3--tea services (cups, plates, trays, etc.)"),
        (0.17, "cluster 4--unrecognisable (OCR errors)"),
    ],
    "sugar": [
        (0.34, "cluster 0--sugar as an everyday consumable and household commodity"),
        (0.66, "cluster 1--sugar as a traded commodity"),
    ],
    "cocoa": [
        (0.29, "cluster 0--cocoa as a traded commodity"),
        (0.71, "cluster 1--cocoa as a branded commercial and consumer product"),
    ],
}

# Optional exact override: {word: [label for cluster 0, label for cluster 1, ...]}
LABELS_BY_CLUSTER_ID = {}


def sense_labels_for(word, labels_arr, k):
    """Return legend labels for a word's k clusters.

    Prefers an exact per-cluster-id list if one is defined; otherwise matches
    labels to cluster ids by overall pooled share. Falls back to 'cluster N'
    if no labels (or the wrong number of labels) are defined."""
    exact = LABELS_BY_CLUSTER_ID.get(word, [])
    if exact:
        if len(exact) == k:
            print(f"[sense labels] {word}: using exact per-cluster-id labels")
            return list(exact)
        print(f"[sense labels] {word}: {len(exact)} id-keyed labels defined "
              f"but k={k} -- ignoring them")

    entries = SENSE_LABELS_BY_SHARE.get(word, [])
    if len(entries) != k:
        if entries:
            print(f"[sense labels] {word}: {len(entries)} labels defined "
                  f"but k={k} -- falling back to cluster ids")
        return [f"cluster {c}" for c in range(k)]

    shares = np.array([(labels_arr == c).mean() for c in range(k)])
    stated = np.array([s for s, _ in entries])
    cost = np.abs(shares[:, None] - stated[None, :])
    rows, cols = linear_sum_assignment(cost)
    leg = [f"cluster {c}" for c in range(k)]
    print(f"[sense labels] share-based matching for '{word}':")
    for c, j in zip(rows, cols):
        diff = abs(shares[c] - stated[j])
        leg[c] = entries[j][1]
        print(f"  cluster {c}: observed {shares[c]:5.1%}  ->  "
              f"'{entries[j][1]}' (stated {stated[j]:.0%}, diff {diff:.1%})")
        if diff > 0.05:
            print("    WARNING: >5pp mismatch -- verify this label against "
                  "the cluster's snippets before trusting the figure")
        runner_up = np.partition(np.abs(stated - shares[c]), 1)[1]
        if runner_up - diff < 0.02:
            print("    WARNING: another stated share is within 2pp -- "
                  "matching is ambiguous for this cluster, verify via snippets "
                  "or use LABELS_BY_CLUSTER_ID")
    return leg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def discover_periods(emb_dir):
    """{period: path} from the HDF5 filenames."""
    out = {}
    for path in sorted(glob.glob(os.path.join(emb_dir, "commodity_embeddings_*.h5"))):
        m = FILE_RE.search(os.path.basename(path))
        if m:
            out[int(m.group(1))] = path
    return out


def load_word_period(path, word, period, rng, max_per_period,
                     exclude_forms=frozenset(), canonical_only=False,
                     exclude_next=frozenset()):
    """Load (and optionally subsample) one word's embeddings for one period.

    Matches the schema written by the extraction script:
      file  commodity_embeddings_<period>.h5
        group <word>            (attrs: count, forms_mode, context_window, ...)
          group usage_<i>       (attrs: snippet, position)
            dataset 'embedding' (768-d float vector)
    The snippet is the centred +/-context_window slice of the document and
    `position` is the target's index WITHIN that slice, so the matched surface
    form is snippet.split(' ')[position] and the following token is at
    position + 1.

    Returns (embeddings [n,d], snippets [n], positions [n]) or (None, None, None).
    """
    if not os.path.exists(path):
        print(f"  [{period}] file not found, skipping: {path}")
        return None, None, None
    with h5py.File(path, "r") as f:
        if word not in f:
            return None, None, None
        g = f[word]
        usage_keys = [k for k in g.keys() if k.startswith("usage_")]
        if not usage_keys:
            return None, None, None

        # Read the cheap attrs (snippet, position) for every usage first; the
        # 768-d embedding is read later only for the occurrences we keep.
        metas = []  # (key, snippet, position)
        for k in usage_keys:
            sn = g[k].attrs.get("snippet", "")
            if isinstance(sn, bytes):
                sn = sn.decode("utf-8", "replace")
            pos = int(g[k].attrs.get("position", -1))
            metas.append((k, sn, pos))

        # Drop excluded surface forms, and occurrences followed by an excluded
        # token (the 'cocoa nut' filter). Filtering BEFORE subsampling so that a
        # cap yields the requested number of *surviving* occurrences.
        if exclude_forms or canonical_only or exclude_next:
            target = word.lower()
            kept, n_dropped, n_next = [], 0, 0
            for k, sn, pos in metas:
                toks = sn.split(" ")
                form = toks[pos].lower() if 0 <= pos < len(toks) else None
                drop = (form in exclude_forms) if form is not None else False
                if canonical_only and form != target:
                    drop = True
                if drop:
                    n_dropped += 1
                    continue
                if exclude_next:
                    nxt = toks[pos + 1].lower() if 0 <= pos + 1 < len(toks) else None
                    if nxt is not None and nxt in exclude_next:
                        n_next += 1
                        continue
                kept.append((k, sn, pos))
            if n_dropped:
                what = ("non-canonical/unrecoverable" if canonical_only
                        else str(sorted(exclude_forms)))
                print(f"  [{period}] dropped {n_dropped} occurrence(s) ({what})")
            if n_next:
                print(f"  [{period}] dropped {n_next} occurrence(s) followed by "
                      f"{sorted(exclude_next)}")
            metas = kept
        if not metas:
            return None, None, None

        if max_per_period and len(metas) > max_per_period:
            idx = rng.choice(len(metas), size=max_per_period, replace=False)
            metas = [metas[i] for i in idx]

        embs, snips, poss = [], [], []
        for k, sn, pos in metas:
            embs.append(g[k]["embedding"][:])
            snips.append(sn)
            poss.append(pos)
    return (np.asarray(embs, dtype=np.float32),
            np.asarray(snips, dtype=object),
            np.asarray(poss, dtype=int))


def kwic(snippet, pos, window=12):
    """Return (display_string, word_visible), target centred and upper-cased."""
    toks = snippet.split(" ")
    if 0 <= pos < len(toks):
        lo, hi = max(0, pos - window), min(len(toks), pos + window + 1)
        seg = toks[lo:hi]
        ti = pos - lo
        if 0 <= ti < len(seg):
            seg[ti] = seg[ti].upper()
        pre = "… " if lo > 0 else ""
        suf = " …" if hi < len(toks) else ""
        return pre + " ".join(seg) + suf, True
    return snippet + "  [target word not recoverable from snippet]", False


def js_divergence(p, q, base=2):
    """Jensen-Shannon DIVERGENCE (scipy returns the distance = sqrt of it)."""
    dist = jensenshannon(p, q, base=base)
    return float(dist) ** 2


def dedup_exact_embeddings(X, S, P, period_of):
    """Remove exact-duplicate embeddings (byte-identical vectors), keeping the
    first occurrence of each -- e.g. verbatim market-report/wire-syndicated
    passages appearing in multiple articles."""
    seen, keep_idx = {}, []
    for i, emb in enumerate(X):
        key = emb.tobytes()
        if key not in seen:
            seen[key] = i
            keep_idx.append(i)
    keep_idx = np.array(keep_idx)
    n_before, n_after = len(X), len(keep_idx)
    report = {"n_before": n_before, "n_after": n_after,
              "n_removed": n_before - n_after,
              "pct_removed": (n_before - n_after) / n_before if n_before else 0.0}
    return X[keep_idx], S[keep_idx], P[keep_idx], period_of[keep_idx], report


def is_near_duplicate(text, chosen_texts, threshold=0.85, compare_last=200):
    """True if `text` is a close fuzzy match to any recently accepted string."""
    for t in chosen_texts[-compare_last:]:
        if SequenceMatcher(None, text, t).ratio() > threshold:
            return True
    return False


def benjamini_hochberg(ps):
    ps = np.asarray(ps, dtype=float)
    ok = np.isfinite(ps)
    adj = np.full(ps.shape, np.nan)
    if not ok.any():
        return adj
    if false_discovery_control is not None:
        adj[ok] = false_discovery_control(ps[ok], method="bh")
        return adj
    p = ps[ok]
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    adj[ok] = out
    return adj


def jsd_permutation(labels_a, labels_b, k, r, rng):
    """
    Null distribution of JSD between two periods, keeping both original sizes.

    The clustering is already fixed, so this only shuffles which period each
    occurrence's cluster id is attributed to -- cheap, and exactly the paper's
    permutation logic applied to the sense-based measure.

    Returns the distribution; element 0 is the observed value.
    """
    pooled = np.concatenate([labels_a, labels_b])
    n_a, N = len(labels_a), len(pooled)

    def props(lab):
        c = np.bincount(lab, minlength=k).astype(float)
        return c / c.sum()

    jsd = np.empty(r + 1)
    jsd[0] = js_divergence(props(labels_a), props(labels_b))
    for i in range(1, r + 1):
        perm = rng.permutation(pooled)
        jsd[i] = js_divergence(props(perm[:n_a]), props(perm[n_a:]))
    return jsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True)
    ap.add_argument("--emb-dir", required=True,
                    help="dir holding commodity_embeddings_<period>.h5 for ONE model")
    ap.add_argument("--periods", type=int, nargs="+", default=None,
                    help="subset of periods; default every period found")
    ap.add_argument("--decades", type=int, nargs="+", default=None,
                    help="alias for --periods")
    ap.add_argument("--slice-width", type=int, default=1,
                    help="years per slice, counted from the first period "
                         "present. 1 = one bin per file (year on year for a "
                         "yearly directory, decade on decade for a decade "
                         "directory). 3 and 5 match the paper's slice widths.")
    ap.add_argument("--k", type=int, default=None,
                    help="number of shared clusters; if omitted, silhouette-selected")
    ap.add_argument("--k-range", type=int, nargs=2, default=[2, 10],
                    help="inclusive range to search when --k is not given")
    ap.add_argument("--silhouette-sample", type=int, default=2000,
                    help="occurrences sampled when fitting/scoring candidate k "
                         "(silhouette is quadratic in the number of points). "
                         "A value at least the number of pooled embeddings uses "
                         "ALL of them, as for the results in the paper (slow). "
                         "The final clustering always uses the complete set.")
    ap.add_argument("--max-per-period", "--max-per-decade", dest="max_per_period",
                    type=int, default=0,
                    help="cap embeddings per period used for clustering+proportions; "
                         "0 = use ALL (default). NOTE this is now per YEAR on a "
                         "yearly directory, so the same number keeps ten times "
                         "as much data as it did per decade.")
    ap.add_argument("--exclude-forms", type=str, nargs="*", default=[],
                    help="surface forms to drop at load time, e.g. plurals")
    ap.add_argument("--canonical-only", action="store_true",
                    help="keep ONLY occurrences whose matched surface form is "
                         "exactly --word")
    ap.add_argument("--exclude-next-token", type=str, nargs="*", default=[],
                    help="drop occurrences whose FOLLOWING token is one of "
                         "these, e.g. nut nuts for 'cocoa nut'")
    ap.add_argument("--balance-clustering", action="store_true",
                    help="weight each embedding by the inverse of its period's "
                         "size when fitting, so every period contributes equally "
                         "to where the centroids land while ALL embeddings are used")
    ap.add_argument("--minibatch", action="store_true",
                    help="MiniBatchKMeans for the final fit. Worth it above "
                         "roughly half a million pooled vectors.")
    ap.add_argument("--permute", type=int, default=0,
                    help="permutations for the JSD test (e.g. 999). "
                         "0 = skip the test. Cheap: the clustering is already done.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-facets", type=int, default=16,
                    help="draw the per-period scatter grid only when there are "
                         "at most this many periods (default 16)")
    ap.add_argument("--facets", action="store_true",
                    help="force the scatter grid even above --max-facets")
    ap.add_argument("--plot-max-per-period", "--plot-max-per-decade",
                    dest="plot_max_per_period", type=int, default=3000,
                    help="cap points DRAWN per scatter panel, legibility only")
    ap.add_argument("--snippets", type=int, default=10,
                    help="print & save this many centroid-nearest snippets per "
                         "cluster; 0 = skip")
    ap.add_argument("--window", type=int, default=50,
                    help="tokens to show each side of the target word in snippets")
    ap.add_argument("--all-snippets", action="store_true",
                    help="also write cluster_all_snippets_<word>.csv, every "
                         "occurrence with its cluster and KWIC line. OFF by "
                         "default: it is over a million rows for tea and was "
                         "one of the slowest steps. cluster_snippets_<word>.txt "
                         "(the centroid-nearest examples you label from) is "
                         "always written.")
    ap.add_argument("--dedup-embeddings", action="store_true",
                    help="remove exact-duplicate embeddings before clustering")
    ap.add_argument("--near-dup-threshold", type=float, default=0.75,
                    help="fuzzy-match ratio above which two snippets in the "
                         "representative listing are collapsed. DISPLAY-ONLY. "
                         "1.0 disables.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    exclude_forms = {f.lower() for f in args.exclude_forms}
    exclude_next = {f.lower() for f in args.exclude_next_token}
    if args.canonical_only:
        print(f"Canonical-only: keeping exact '{args.word}' occurrences only.")
    elif exclude_forms:
        print(f"Excluding surface forms from analysis: {sorted(exclude_forms)}")
    if exclude_next:
        print(f"Dropping occurrences followed by: {sorted(exclude_next)}")

    emb_dir = os.path.expanduser(args.emb_dir)
    found = discover_periods(emb_dir)
    if not found:
        raise SystemExit(f"No commodity_embeddings_*.h5 in {emb_dir}")

    requested = args.periods or args.decades
    if requested:
        absent = [p for p in requested if p not in found]
        if absent:
            print(f"WARNING: no file for {absent}")
        file_periods = [p for p in requested if p in found]
    else:
        file_periods = sorted(found)
    if len(file_periods) < 2:
        raise SystemExit("Fewer than two periods selected.")

    granularity = "year" if len(found) > 20 else "decade"
    # How a period is written in output. A decade directory keeps the 1880s
    # style your existing figures and CSVs use.
    if granularity == "decade" and args.slice_width == 1:
        def plabel(p):
            return f"{p}s"
    elif args.slice_width == 1:
        def plabel(p):
            return f"{p}"
    else:
        def plabel(p):
            return f"{p}-{p + args.slice_width - 1}"

    print(f"Embeddings dir: {emb_dir}")
    print(f"Granularity:    {granularity}, slice width {args.slice_width} "
          f"({len(file_periods)} files)")

    # ---- load every period, remember which rows belong to which slice ----
    first = file_periods[0]
    X_parts, S_parts, P_parts, period_of = [], [], [], []
    for p in file_periods:
        embs, snips, poss = load_word_period(
            found[p], args.word, p, rng, args.max_per_period,
            exclude_forms, args.canonical_only, exclude_next)
        if embs is None or len(embs) == 0:
            continue
        slice_start = first + ((p - first) // args.slice_width) * args.slice_width
        X_parts.append(embs)
        S_parts.append(snips)
        P_parts.append(poss)
        period_of.append(np.full(len(embs), slice_start))
        print(f"  [{p}] loaded {len(embs)} embeddings")
    if not X_parts:
        raise SystemExit("No embeddings found for any period -- check --word / --emb-dir.")

    X = np.vstack(X_parts)
    S = np.concatenate(S_parts)
    P = np.concatenate(P_parts)
    period_of = np.concatenate(period_of)
    del X_parts, S_parts, P_parts
    present_periods = sorted(set(period_of.tolist()))
    print(f"Pooled: {X.shape[0]} embeddings x {X.shape[1]} dims across "
          f"{len(present_periods)} slices")

    if args.dedup_embeddings:
        X, S, P, period_of, dedup_report = dedup_exact_embeddings(X, S, P, period_of)
        print(f"Exact-duplicate dedup: {dedup_report['n_before']} -> "
              f"{dedup_report['n_after']} ({dedup_report['n_removed']} removed, "
              f"{dedup_report['pct_removed']:.2%})")
        dr = os.path.join(args.out_dir, f"dedup_report_{args.word}.csv")
        with open(dr, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["word", "n_before", "n_after", "n_removed", "pct_removed"])
            w.writerow([args.word, dedup_report["n_before"], dedup_report["n_after"],
                        dedup_report["n_removed"], f"{dedup_report['pct_removed']:.4f}"])
        print(f"Wrote {dr}")
        present_periods = sorted(set(period_of.tolist()))

    # ---- per-period embedding counts ----
    # POST-filter, post-dedup counts, so they differ from frequency_summary.csv.
    counts_by_period = {p: int((period_of == p).sum()) for p in present_periods}
    nc = os.path.join(args.out_dir, f"counts_{args.word}.csv")
    with open(nc, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["word", "decade", "n_embeddings"])   # column name kept
        w.writerows([(args.word, p, counts_by_period[p]) for p in present_periods])
    print(f"Wrote {nc}")
    lo_p = min(counts_by_period, key=counts_by_period.get)
    print(f"Sparsest slice: {plabel(lo_p)} with {counts_by_period[lo_p]:,} "
          f"occurrences; densest {max(counts_by_period.values()):,}")
    if counts_by_period[lo_p] < 50 and not args.permute:
        print("  NOTE: with slices this small the JSD between them is dominated "
              "by sampling noise. Use --permute 999 to test against it.")

    # ---- period-frequency weighting for the clustering (optional) ----
    if args.balance_clustering:
        mean_n = float(np.mean(list(counts_by_period.values())))
        sample_weight = np.array([mean_n / counts_by_period[p] for p in period_of],
                                 dtype=float)
        print("Balanced clustering: embeddings weighted by inverse period "
              f"frequency (all {len(X)} embeddings used).")
    else:
        sample_weight = None

    # ---- choose k (silhouette) and fit ONE shared clustering ----
    sil_rows = []
    n_samp = min(args.silhouette_sample, len(X))
    samp = rng.choice(len(X), size=n_samp, replace=False)
    sw_samp = None if sample_weight is None else sample_weight[samp]
    print(f"Silhouette sample: {n_samp} of {len(X)} occurrences")

    if args.k is None:
        best_k, best_s = None, -1.0
        for kk in range(args.k_range[0], args.k_range[1] + 1):
            lab = KMeans(n_clusters=kk, random_state=args.seed, n_init=10).fit_predict(
                X[samp], sample_weight=sw_samp)
            s = float(silhouette_score(X[samp], lab))
            sil_rows.append([args.word, kk, round(s, 4), 0])
            print(f"  k={kk}: silhouette={s:.3f}")
            if s > best_s:
                best_k, best_s = kk, s
        k = best_k
        print(f"Selected k={k} (silhouette={best_s:.3f})")
        for r_ in sil_rows:
            r_[3] = int(r_[1] == k)
    else:
        k = args.k
        lab = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(
            X[samp], sample_weight=sw_samp)
        s = float(silhouette_score(X[samp], lab))
        print(f"Using fixed k={k} (silhouette={s:.3f})")
        sil_rows.append([args.word, k, round(s, 4), 1])

    silp = os.path.join(args.out_dir, f"silhouette_{args.word}.csv")
    with open(silp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["word", "k", "silhouette", "selected"])
        w.writerows(sil_rows)
    print(f"Wrote {silp}")

    if args.minibatch:
        print(f"Fitting MiniBatchKMeans on {len(X):,} vectors...")
        km = MiniBatchKMeans(n_clusters=k, random_state=args.seed, n_init=10,
                             batch_size=4096).fit(X, sample_weight=sample_weight)
    else:
        print(f"Fitting KMeans on {len(X):,} vectors...")
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit(
            X, sample_weight=sample_weight)
    labels = km.predict(X)

    # ---- read off what each cluster IS: centroid-nearest snippets ----
    # Cosine of every point to its OWN centroid, without materialising a
    # normalised copy of X (that copy is 3.7 GB for tea, and on a machine that
    # then starts swapping it is slower than the clustering itself).
    cen_n = km.cluster_centers_ / (
        np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True) + 1e-9)
    row_norm = np.linalg.norm(X, axis=1) + 1e-9
    cos_to_cen = (np.einsum("ij,ij->i", X, cen_n[labels]) / row_norm)

    if args.snippets:
        lines = [f"Representative snippets per shared cluster for '{args.word}'",
                 "(target word UPPER-CASED and centred; ranked by closeness to the "
                 "cluster centroid; examples where the word is visible are preferred)", ""]
        for c in range(k):
            mem = np.where(labels == c)[0]
            order = mem[np.argsort(-cos_to_cen[mem])]
            vis, hidden, chosen_texts = [], [], []
            for j in order:
                disp, ok = kwic(S[j], P[j], args.window)
                if args.near_dup_threshold < 1.0 and is_near_duplicate(
                        disp, chosen_texts, threshold=args.near_dup_threshold):
                    continue
                chosen_texts.append(disp)
                (vis if ok else hidden).append((j, disp))
                if len(vis) >= args.snippets:
                    break
            chosen = (vis + hidden)[:args.snippets]
            lines.append(f"--- cluster {c}  (n={len(mem)}, "
                         f"{100 * len(mem) / len(X):.0f}% of all usages) ---")
            for j, disp in chosen:
                lines.append(f"  [{plabel(period_of[j])}] {disp}")
            lines.append("")
        text = "\n".join(lines)
        print("\n" + text)
        sp = os.path.join(args.out_dir, f"cluster_snippets_{args.word}.txt")
        with open(sp, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Wrote {sp}")

    # ---- optional: EVERY usage in every cluster, as a CSV ----
    if args.all_snippets:
        cp = os.path.join(args.out_dir, f"cluster_all_snippets_{args.word}.csv")
        with open(cp, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cluster", "decade", "cos_to_centroid",
                        "word_visible", "snippet"])
            for c in range(k):
                mem = np.where(labels == c)[0]
                for j in mem[np.argsort(-cos_to_cen[mem])]:
                    disp, ok = kwic(S[j], P[j], args.window)
                    w.writerow([c, plabel(period_of[j]), f"{cos_to_cen[j]:.6f}",
                                int(ok), disp])
        print(f"Wrote {cp}  ({len(X)} rows)")

    # ---- proportions per period over the shared cluster inventory ----
    props = np.zeros((len(present_periods), k))
    for i, p in enumerate(present_periods):
        counts = np.bincount(labels[period_of == p], minlength=k)
        props[i] = counts / counts.sum()

    pp = os.path.join(args.out_dir, f"props_{args.word}.csv")
    with open(pp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["word", "decade"] + [f"cluster_{c}" for c in range(k)])
        for i, p in enumerate(present_periods):
            w.writerow([args.word, plabel(p)] + [f"{v:.6f}" for v in props[i]])
    print(f"Wrote {pp}")

    # ---- JSD between consecutive periods, with the null ----
    print(f"\n=== Jensen-Shannon between consecutive slices for '{args.word}' "
          f"(shared k={k} clusters, base 2) ===")
    hdr = f"{'from':>10} {'to':>10} {'n_from':>9} {'n_to':>9} {'JS_dist':>9} {'JS_div':>9}"
    if args.permute:
        hdr += f" {'null_div':>9} {'z':>7} {'p':>8}"
    print(hdr)

    jsd_rows = []

    for i in range(len(present_periods) - 1):
        p0, p1 = present_periods[i], present_periods[i + 1]
        dist = float(jensenshannon(props[i], props[i + 1], base=2))
        div = js_divergence(props[i], props[i + 1], base=2)
        jrow = [args.word, p0, p1, round(dist, 4), round(div, 4)]

        line = (f"{plabel(p0):>10} {plabel(p1):>10} "
                f"{counts_by_period[p0]:>9,} {counts_by_period[p1]:>9,} "
                f"{dist:>9.4f} {div:>9.4f}")

        if args.permute:
            jd = jsd_permutation(labels[period_of == p0],
                                 labels[period_of == p1], k, args.permute, rng)
            p_jsd = float(np.mean(jd >= jd[0]))
            null = jd[1:]
            sd = float(null.std(ddof=1))
            z = (jd[0] - null.mean()) / sd if sd > 0 else float("nan")
            jrow += [round(float(null.mean()), 4), round(float(jd[0] - null.mean()), 4),
                     round(float(z), 2), round(p_jsd, 4)]
            line += f" {null.mean():>9.4f} {z:>7.1f} {p_jsd:>8.4f}"
        print(line)
        jsd_rows.append(jrow)

    jsd_header = ["word", "decade_start", "decade_end", "js_distance", "js_divergence"]
    if args.permute:
        # Benjamini-Hochberg across this word's slices, as in the paper
        bh_j = benjamini_hochberg([r_[8] for r_ in jsd_rows])
        for r_, b in zip(jsd_rows, bh_j):
            r_.append(round(float(b), 4))
            r_.append(bool(b < args.alpha))
        jsd_header += ["null_mean_divergence", "excess", "z", "p_value",
                       "p_value_bh", "significant"]

        n_sig = sum(r_[-1] for r_ in jsd_rows)
        print(f"\n  JSD above its own noise floor (BH p < {args.alpha}): "
              f"{n_sig}/{len(jsd_rows)} transitions")
        floor_p = 1.0 / (args.permute + 1)
        if floor_p > args.alpha / max(len(jsd_rows), 1):
            print(f"  NOTE: the smallest p available at --permute {args.permute} "
                  f"is {floor_p:.4f}, while Benjamini-Hochberg needs "
                  f"{args.alpha / len(jsd_rows):.5f} to reject a lone transition "
                  f"among {len(jsd_rows)}. Raise --permute if few come out "
                  f"significant.")

    jp = os.path.join(args.out_dir, f"jsd_{args.word}.csv")
    with open(jp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(jsd_header)
        w.writerows(jsd_rows)
    print(f"Wrote {jp}")

    # ---- figures -------------------------------------------------------
    pca = PCA(n_components=2, random_state=args.seed).fit(
        X[samp] if len(X) > 200000 else X)
    XY = pca.transform(X)
    cen_xy = pca.transform(km.cluster_centers_)
    var = pca.explained_variance_ratio_ * 100
    cmap = plt.get_cmap("tab10")
    colours = {c: cmap(c % 10) for c in range(k)}

    draw_facets = args.facets or len(present_periods) <= args.max_facets
    if not draw_facets:
        print(f"\nSkipping the scatter grid: {len(present_periods)} slices is "
              f"more than --max-facets {args.max_facets}. Use --slice-width 5 "
              f"or 10 for a readable grid, or --facets to force it.")
    else:
        pad = 0.5
        xlim = (XY[:, 0].min() - pad, XY[:, 0].max() + pad)
        ylim = (XY[:, 1].min() - pad, XY[:, 1].max() + pad)
        nd = len(present_periods)
        ncols = min(4, nd)
        nrows = int(np.ceil(nd / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax_i, p in enumerate(present_periods):
            ax = axes.flat[ax_i]
            ax.axis("on")
            idx = np.where(period_of == p)[0]
            n_total = len(idx)
            if args.plot_max_per_period and n_total > args.plot_max_per_period:
                idx = rng.choice(idx, size=args.plot_max_per_period, replace=False)
            lab_d, xy_d = labels[idx], XY[idx]
            for c in range(k):
                sel = lab_d == c
                if sel.any():
                    ax.scatter(xy_d[sel, 0], xy_d[sel, 1], s=10, alpha=0.45,
                               color=colours[c], linewidths=0)
            ax.scatter(cen_xy[:, 0], cen_xy[:, 1], marker="X", s=90,
                       c=[colours[c] for c in range(k)],
                       edgecolors="black", linewidths=1.5, zorder=5)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(f"{plabel(p)}  (n={n_total})", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
        legend = [Line2D([0], [0], marker="o", linestyle="", color=colours[c],
                         label=f"cluster {c}") for c in range(k)]
        fig.legend(handles=legend, loc="lower center", ncol=min(k, 6),
                   frameon=False, fontsize=10)
        fig.suptitle(f"'{args.word}': shared clusters across slices "
                     f"(shared PCA, PC1 {var[0]:.0f}% / PC2 {var[1]:.0f}%)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        f1 = os.path.join(args.out_dir, f"facets_{args.word}.png")
        fig.savefig(f1, dpi=150)
        plt.close(fig)
        print(f"Wrote {f1}")

    # ---- stacked composition: a real time series at year level ----
    leg_labels = sense_labels_for(args.word, labels, k)
    fig2, ax2 = plt.subplots(figsize=(10 if len(present_periods) > 20 else 8, 4.5))
    xs = np.array(present_periods, dtype=float)
    ax2.stackplot(xs, props.T, colors=[colours[c] for c in range(k)],
                  labels=leg_labels, alpha=0.85)
    if len(present_periods) > 20:
        # a numeric axis with decade ticks, instead of one tick per period
        step = 10 if args.slice_width <= 5 else 20
        first_tick = int(np.ceil(xs[0] / step) * step)
        ax2.set_xticks(np.arange(first_tick, xs[-1] + 1, step))
        ax2.set_xlim(xs[0], xs[-1])
    else:
        ax2.set_xticks(xs)
        ax2.set_xticklabels([plabel(p) for p in present_periods])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("share of usages")
    ax2.set_title(f"'{args.word}': usage-type composition over time "
                  f"(shared k={k} clusters)")
    long_labels = any(len(s) > 20 for s in leg_labels)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
               ncol=1 if long_labels else min(k, 6), frameon=False)
    fig2.tight_layout()
    f2 = os.path.join(args.out_dir, f"stacked_{args.word}.png")
    fig2.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {f2}")


if __name__ == "__main__":
    main()