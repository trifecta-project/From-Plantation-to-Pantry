import os
import csv
import h5py
import numpy as np
from collections import Counter
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

# =============================================================================
# Configuration
# =============================================================================

dir_in = "..." #change it to your path
dir_emb = os.path.join(dir_in, "output", "embeddings_macberth_finetuned")
dir_out = os.path.join(dir_in, "output", "analysis")

os.makedirs(os.path.join(dir_out, "csv"), exist_ok=True)

decades = [1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910]
TARGET_WORDS = ['coffee', 'tea', 'sugar', 'opium', 'cocoa', 'tobacco']

JSD_CONSEC_CSV = os.path.join(dir_out, "csv", "jsd_consecutive_decades.csv")
JSD_VS1910_CSV = os.path.join(dir_out, "csv", "jsd_vs_1910.csv")

# =============================================================================
# Data Loading
# =============================================================================

def load_commodity_embeddings(decade, dir_emb):
    """Load commodity embeddings from HDF5 file for a specific decade."""
    file_path = os.path.join(dir_emb, f"commodity_embeddings_{decade}.h5")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    print(f"Loading embeddings for {decade}...")
    word_data = {}

    with h5py.File(file_path, 'r') as f:
        for word in f.keys():
            count = f[word].attrs['count']
            embeddings = []

            for i in range(count):
                usage = f[word][f'usage_{i}']
                emb = usage['embedding'][:]
                embeddings.append(emb)

            word_data[word] = {
                'embeddings': embeddings,
                'count': count
            }
            print(f"  {word}: {count} usages")

    return word_data


def load_all_decades(decades, dir_emb):
    """Load embeddings for all decades."""
    all_data = {}
    for decade in decades:
        data = load_commodity_embeddings(decade, dir_emb)
        if data:
            all_data[decade] = data
    return all_data


# =============================================================================
# JSD Function (Giulianelli et al. 2020) - unchanged
# =============================================================================

def ap_jsd(embeddings1, embeddings2):
    """
    Compute Jensen-Shannon Divergence (distance) using Affinity Propagation
    clustering over the union of two embedding sets.
    """
    ap = AffinityPropagation(
        affinity='precomputed',
        damping=0.9,
        max_iter=200,
        convergence_iter=15,
        random_state=42
    )

    embeddings = np.concatenate([embeddings1, embeddings2], axis=0)
    sim = cosine_similarity(embeddings)
    ap.fit(sim)

    L = ap.labels_
    L1, L2 = L[:embeddings1.shape[0]], L[embeddings1.shape[0]:]
    labels = np.unique(np.concatenate([L1, L2]))

    c1 = Counter(L1)
    c2 = Counter(L2)
    L1_dist = np.array([c1[l] for l in labels], dtype=float)
    L2_dist = np.array([c2[l] for l in labels], dtype=float)
    L1_dist = L1_dist / L1_dist.sum()
    L2_dist = L2_dist / L2_dist.sum()

    return jensenshannon(L1_dist, L2_dist)


# =============================================================================
# Checkpointing helpers
# =============================================================================

def load_done_keys(csv_path, key_cols):
    """Read existing CSV and return a set of already-completed keys."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(tuple(str(row[c]) for c in key_cols))
    return done


def init_csv_if_missing(csv_path, header):
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_row(csv_path, row):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    print("Loading embeddings from all decades...")
    all_data = load_all_decades(decades, dir_emb)
    print(f"\nLoaded {len(all_data)} decades")

    # ----------------------------------------------------------------------
    # JSD between consecutive decades
    # ----------------------------------------------------------------------
    init_csv_if_missing(JSD_CONSEC_CSV, ["Word", "Decade_Start", "Decade_End", "JSD"])
    done_consec = load_done_keys(JSD_CONSEC_CSV, ["Word", "Decade_Start", "Decade_End"])
    print(f"\nAlready done (consecutive): {len(done_consec)} pairs")

    print("\n" + "=" * 60)
    print("JSD BETWEEN CONSECUTIVE DECADES")
    print("=" * 60)

    for word in TARGET_WORDS:
        print(f"\n{word}:")
        sorted_decades = sorted([d for d in all_data.keys() if word in all_data[d]])

        for i in range(len(sorted_decades) - 1):
            d1, d2 = sorted_decades[i], sorted_decades[i + 1]

            key = (word, str(d1), str(d2))
            if key in done_consec:
                print(f"  {d1}-{d2}: SKIP (already done)")
                continue

            emb1 = all_data[d1][word]['embeddings']
            emb2 = all_data[d2][word]['embeddings']

            if len(emb1) > 0 and len(emb2) > 0:
                try:
                    emb1_array = np.array(emb1)
                    emb2_array = np.array(emb2)
                    jsd = ap_jsd(emb1_array, emb2_array)
                    append_row(JSD_CONSEC_CSV, [word, d1, d2, jsd])
                    print(f"  {d1}-{d2}: JSD = {jsd:.4f}  [saved]")
                except Exception as e:
                    print(f"  {d1}-{d2}: Error computing JSD - {e}")

    print(f"\nResults appended to {JSD_CONSEC_CSV}")

    # ----------------------------------------------------------------------
    # JSD vs 1910 reference
    # ----------------------------------------------------------------------
    init_csv_if_missing(JSD_VS1910_CSV, ["Word", "Decade", "JSD_vs_1910"])
    done_vs = load_done_keys(JSD_VS1910_CSV, ["Word", "Decade"])
    print(f"\nAlready done (vs 1910): {len(done_vs)} pairs")

    print("\n" + "=" * 60)
    print("JSD VS 1910 REFERENCE")
    print("=" * 60)

    reference_decade = 1910

    for word in TARGET_WORDS:
        print(f"\n{word}:")
        if reference_decade not in all_data or word not in all_data[reference_decade]:
            print(f"  Reference decade {reference_decade} not available")
            continue

        ref_emb = np.array(all_data[reference_decade][word]['embeddings'])

        for decade in sorted(all_data.keys()):
            if decade == reference_decade:
                continue
            if word not in all_data[decade]:
                continue

            key = (word, str(decade))
            if key in done_vs:
                print(f"  {decade}: SKIP (already done)")
                continue

            try:
                decade_emb = np.array(all_data[decade][word]['embeddings'])
                jsd = ap_jsd(decade_emb, ref_emb)
                append_row(JSD_VS1910_CSV, [word, decade, jsd])
                print(f"  {decade}: JSD = {jsd:.4f}  [saved]")
            except Exception as e:
                print(f"  {decade}: Error computing JSD - {e}")

    print(f"\nResults appended to {JSD_VS1910_CSV}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
