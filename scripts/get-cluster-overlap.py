import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# CONFIGURATION
membership_files = [
    # "./results/membership/clusters_cpm_0.01_weighted_run1.tsv",
    # "./results/membership/clusters_cpm_0.01_weighted_run2.tsv",
    # "./results/membership/clusters_cpm_0.01_weighted_run3.tsv",
    # "./results/membership/clusters_cpm_0.01_weighted_run4.tsv",
    "./results/membership/clusters_cpm_0.01_weighted.tsv"
]
unweighted_path = "./results/membership/clusters_cpm_0.01_unweighted.tsv"
TOP_K = 500
JACCARD_THRESHOLD = 0.1
OUTPUT_DIR = "./results/overlap_res/jaccard_heatmaps"
PLOT_HEATMAP = False   # TOGGLE THIS FOR HEATMAP
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load data
print("Loading unweighted clustering...", flush=True)
unweighted_df = pd.read_csv(unweighted_path, sep="\t", dtype=str)
top_unweighted_ids = unweighted_df["cluster_id"].value_counts().nlargest(TOP_K).index.tolist()
unweighted_clusters = {
    cid: set(group["node_id"]) for cid, group in unweighted_df.groupby("cluster_id") if cid in top_unweighted_ids
}

# process jaccard similarity on membership files
for run_idx, weighted_path in enumerate(membership_files):
    print(f"\nProcessing Weighted Run {run_idx+1}...", flush=True)
    
    weighted_df = pd.read_csv(weighted_path, sep="\t", dtype=str)
    top_weighted_ids = weighted_df["cluster_id"].value_counts().nlargest(TOP_K).index.tolist()
    weighted_clusters = {
        cid: set(group["node_id"]) for cid, group in weighted_df.groupby("cluster_id") if cid in top_weighted_ids
    }
    
    print("Computing Jaccard overlap matrix...", flush=True)
    overlap_matrix = np.zeros((TOP_K, TOP_K))
    for i, w_id in enumerate(top_weighted_ids):
        w_nodes = weighted_clusters[w_id]
        for j, u_id in enumerate(top_unweighted_ids):
            u_nodes = unweighted_clusters[u_id]
            intersection = len(w_nodes & u_nodes)
            union = len(w_nodes | u_nodes)
            overlap_matrix[i, j] = intersection / union if union > 0 else 0.0

    
    # print(f"\n=== Weighted -> Unweighted (Run {run_idx+1}), Jaccard ≥ {JACCARD_THRESHOLD} ===", flush=True)
    # for i, w_id in enumerate(top_weighted_ids):
    #     matches = []
    #     for j, u_id in enumerate(top_unweighted_ids):
    #         jacc = overlap_matrix[i, j]
    #         if jacc >= JACCARD_THRESHOLD:
    #             matches.append(f"U{u_id} ({len(unweighted_clusters[u_id])}, J={jacc:.3f})")
    #     if len(matches) > 1:
    #         print(f"W{w_id} ({len(weighted_clusters[w_id])}) overlaps with: {', '.join(matches)}", flush=True)

    # print(f"\n=== Unweighted -> Weighted (Run {run_idx+1}), Jaccard ≥ {JACCARD_THRESHOLD} ===", flush=True)
    # for j, u_id in enumerate(top_unweighted_ids):
    #     matches = []
    #     for i, w_id in enumerate(top_weighted_ids):
    #         jacc = overlap_matrix[i, j]
    #         if jacc >= JACCARD_THRESHOLD:
    #             matches.append(f"W{w_id} ({len(weighted_clusters[w_id])}, J={jacc:.3f})")
    #     if len(matches) > 1:
    #         print(f"U{u_id} ({len(unweighted_clusters[u_id])}) is overlapped by: {', '.join(matches)}", flush=True)
    
    # plot heatmap
    if PLOT_HEATMAP:
        plt.figure(figsize=(18, 14))
        sns.heatmap(
            overlap_matrix,
            annot=False,
            cmap="Blues",
            xticklabels=[f"U{u_id}" for u_id in top_unweighted_ids],
            yticklabels=[f"W{w_id}" for w_id in top_weighted_ids],
            cbar_kws={"label": "Jaccard Similarity"}
        )
        plt.title(f"Jaccard Similarity: Weighted Run {run_idx+1} vs Unweighted", fontsize=16)
        plt.xlabel("Unweighted Clusters", fontsize=12)
        plt.ylabel("Weighted Clusters", fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, f"top{TOP_K}_overlap_heatmap_run{run_idx+1}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

print("\nProcess finished.")
