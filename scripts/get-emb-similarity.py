"""
This script computes the cosine similarity of the top K clusters using MedCPT title/abstract embeddings.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.stats import ttest_ind


parser = argparse.ArgumentParser()
parser.add_argument("--weight_tag", type=str, default="")
args = parser.parse_args()

WEIGHT_TAG = args.weight_tag

# ==== CONFIGURATION ====
embedding_dir = "/scratch/donginn2/data/medcpt"

# --- set which edgelist ---
# WEIGHT_TAG = ""  # baseline
# WEIGHT_TAG = "_312"
# WEIGHT_TAG = "_213"

TOP_K = 100

membership_path_weighted = f"/scratch/donginn2/cs598-gck-project/scripts/results/membership/clusters_cpm_0.01_weighted{WEIGHT_TAG}.tsv"
membership_path_unweighted = "/scratch/donginn2/cs598-gck-project/scripts/results/membership/clusters_cpm_0.01_unweighted.tsv"
output_dir = "./results/similarity_res/"
os.makedirs(output_dir, exist_ok=True)

# load cluster membership
weighted_df = pd.read_csv(membership_path_weighted, sep='\t', dtype=str)
unweighted_df = pd.read_csv(membership_path_unweighted, sep='\t', dtype=str)

# load medcpt embiddings
print("Loading MedCPT embeddings...", flush=True)
all_pmids = []
all_embeddings = []

for fname in sorted(os.listdir(embedding_dir)):
    if fname.startswith("pmids") and fname.endswith(".json"):
        with open(os.path.join(embedding_dir, fname)) as f:
            chunk_pmids = json.load(f)
        all_pmids.extend(chunk_pmids)
    elif fname.startswith("embeds") and fname.endswith(".npy"):
        chunk_embeddings = np.load(os.path.join(embedding_dir, fname))
        all_embeddings.append(chunk_embeddings)

all_embeddings = np.vstack(all_embeddings)
pmid_to_index = {str(pmid): i for i, pmid in enumerate(all_pmids)}
print(f"Total PMIDs in MedCPT: {len(pmid_to_index):,}", flush=True)


def get_top_k_cluster_ids(df, k):
    return df['cluster_id'].value_counts().nlargest(k).index.tolist()


def compute_cluster_cohesion(pmid_list):
    indices = [pmid_to_index.get(str(p)) for p in pmid_list if str(p) in pmid_to_index]
    indices = [i for i in indices if i is not None]
    if len(indices) < 2:
        return None
    vecs = all_embeddings[indices]
    centroid = np.mean(vecs, axis=0, keepdims=True)
    sims = cosine_similarity(vecs, centroid)
    return float(np.mean(sims))


def compute_cluster_centroids(df, cluster_ids):
    centroids = {}
    for cid in cluster_ids:
        pmids = df[df['cluster_id'] == cid]['node_id'].tolist()
        indices = [pmid_to_index.get(str(p)) for p in pmids if str(p) in pmid_to_index]
        indices = [i for i in indices if i is not None]
        if len(indices) < 2:
            continue
        vecs = all_embeddings[indices]
        centroids[cid] = np.mean(vecs, axis=0)
    return centroids


def compute_centroid_distances(centroid_dict):
    cluster_ids = list(centroid_dict.keys())
    centroid_matrix = np.vstack([centroid_dict[cid] for cid in cluster_ids])
    dist_matrix = cosine_distances(centroid_matrix)
    results = []
    for i, cid in enumerate(cluster_ids):
        row = dist_matrix[i]
        row[i] = np.nan
        results.append({
            'cluster_id': cid,
            'min_centroid_dist': np.nanmin(row)
            # 'avg_centroid_dist': np.nanmean(row)
        })
    return results


def evaluate_clusters(df, top_ids, label):
    results = []
    for idx, cid in enumerate(top_ids):
        cluster_pmids = df[df['cluster_id'] == cid]['node_id'].tolist()
        cohesion = compute_cluster_cohesion(cluster_pmids)
        results.append({
            'type': label,
            'cluster_id': cid,
            'size': len(cluster_pmids),
            'cohesion': cohesion
        })
        print(f"[{idx+1}/{len(top_ids)}] Processed cluster {cid}", flush=True)
    return results


top_weighted_ids = get_top_k_cluster_ids(weighted_df, TOP_K)
top_unweighted_ids = get_top_k_cluster_ids(unweighted_df, TOP_K)

print("Evaluating weighted clusters...", flush=True)
weighted_results = evaluate_clusters(weighted_df, top_weighted_ids, "Weighted")

print("Evaluating unweighted clusters...", flush=True)
unweighted_results = evaluate_clusters(unweighted_df, top_unweighted_ids, "Unweighted")

results_df = pd.DataFrame(weighted_results + unweighted_results)
results_df.to_csv(os.path.join(output_dir, f"top{TOP_K}_topic_cluster_similarity_0.01{WEIGHT_TAG}.tsv"), sep='\t', index=False)

print("Computing inter-cluster centroid distances...", flush=True)
centroids_w = compute_cluster_centroids(weighted_df, top_weighted_ids)
centroids_u = compute_cluster_centroids(unweighted_df, top_unweighted_ids)

dists_w = compute_centroid_distances(centroids_w)
dists_u = compute_centroid_distances(centroids_u)

for row in dists_w: row['type'] = 'Weighted'
for row in dists_u: row['type'] = 'Unweighted'

centroid_dists_df = pd.DataFrame(dists_w + dists_u)
centroid_dists_df.to_csv(os.path.join(output_dir, f"top{TOP_K}_cluster_centroid_distances_0.01{WEIGHT_TAG}.tsv"), sep='\t', index=False)

# ============ T-TESTS + PLOTS (Cohesion & Centroid Distance) ==================

# Intra-cluster cohesion
coh_df = results_df.dropna(subset=['cohesion'])
w_coh = coh_df[coh_df['type'] == 'Weighted']['cohesion']
u_coh = coh_df[coh_df['type'] == 'Unweighted']['cohesion']

print("\nSummary Statistics (Cohesion):")
print(f"Weighted    → Mean: {w_coh.mean():.4f}, Std: {w_coh.std():.4f}")
print(f"Unweighted  → Mean: {u_coh.mean():.4f}, Std: {u_coh.std():.4f}")

t_stat, p_val = ttest_ind(w_coh, u_coh, equal_var=False)
print(f"T-test result (Cohesion): t = {t_stat:.4f}, p = {p_val:.4e}")

plt.figure(figsize=(8, 6))
sns.boxplot(x='type', y='cohesion', data=coh_df, showfliers=True)
plt.title("Topic Cohesion in Weighted vs. Unweighted Clusters")
plt.ylabel("Cohesion (Cosine Similarity)")
plt.xlabel("Network Type")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"top{TOP_K}_cohesion_comparison_boxplot{WEIGHT_TAG}.png"), dpi=300)
plt.close()

# Inter-cluster distance (Minimum Centroid Distance)
print("\nSummary Statistics (Minimum Inter-Cluster Distance):")
d_w = centroid_dists_df[centroid_dists_df['type'] == 'Weighted']['min_centroid_dist']
d_u = centroid_dists_df[centroid_dists_df['type'] == 'Unweighted']['min_centroid_dist']

print(f"Weighted    → Mean: {d_w.mean():.4f}, Std: {d_w.std():.4f}")
print(f"Unweighted  → Mean: {d_u.mean():.4f}, Std: {d_u.std():.4f}")

t_stat_min, p_val_min = ttest_ind(d_w, d_u, equal_var=False)
print(f"T-test result (Minimum Distance): t = {t_stat_min:.4f}, p = {p_val_min:.4e}")

plt.figure(figsize=(8, 6))
sns.boxplot(x='type', y='min_centroid_dist', data=centroid_dists_df, showfliers=True)
plt.title("Minimum Inter-Cluster Centroid Distance")
plt.ylabel("Cosine Distance to Nearest Cluster")
plt.xlabel("Network Type")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"top{TOP_K}_min_centroid_distance_boxplot{WEIGHT_TAG}.png"), dpi=300)
plt.close()

# # Inter-cluster distance (Average Centroid Distanc)
# print("\nSummary Statistics (Average Inter-Cluster Distance):")
# avg_d_w = centroid_dists_df[centroid_dists_df['type'] == 'Weighted']['avg_centroid_dist']
# avg_d_u = centroid_dists_df[centroid_dists_df['type'] == 'Unweighted']['avg_centroid_dist']

# print(f"Weighted    → Mean: {avg_d_w.mean():.4f}, Std: {avg_d_w.std():.4f}")
# print(f"Unweighted  → Mean: {avg_d_u.mean():.4f}, Std: {avg_d_u.std():.4f}")

# t_stat_avg, p_val_avg = ttest_ind(avg_d_w, avg_d_u, equal_var=False)
# print(f"T-test result (Average Distance): t = {t_stat_avg:.4f}, p = {p_val_avg:.4e}")

# plt.figure(figsize=(8, 6))
# sns.boxplot(x='type', y='avg_centroid_dist', data=centroid_dists_df, showfliers=True)
# plt.title("Average Inter-Cluster Centroid Distance")
# plt.ylabel("Mean Cosine Distance to All Other Clusters")
# plt.xlabel("Network Type")
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, f"top{TOP_K}_avg_centroid_distance_boxplot{WEIGHT_TAG}.png"), dpi=300)
# plt.close()

print("\nProcess finished!")
