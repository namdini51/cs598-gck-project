import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
import os

# ==== CONFIGURATION ====
membership_files = [
    "./results/membership/clusters_cpm_0.01_weighted_run1.tsv",
    "./results/membership/clusters_cpm_0.01_weighted_run2.tsv",
    "./results/membership/clusters_cpm_0.01_weighted_run3.tsv",
    "./results/membership/clusters_cpm_0.01_weighted_run4.tsv",
    "./results/membership/clusters_cpm_0.01_weighted.tsv",  # 5th run
]
unweighted_path = "./results/membership/clusters_cpm_0.01_unweighted.tsv"

OUTPUT_DIR = "./results/overlap_res"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load uw clustering
print("Loading unweighted clustering...", flush=True)
unweighted_df = pd.read_csv(unweighted_path, sep="\t", dtype=str)

ari_scores = []

# get ARI
for run_idx, weighted_path in enumerate(membership_files):
    print(f"Processing Weighted Run {run_idx+1}...", flush=True)
    
    weighted_df = pd.read_csv(weighted_path, sep="\t", dtype=str)
    
    merged_df = weighted_df.merge(unweighted_df, on="node_id", suffixes=('_weighted', '_unweighted'))
    weighted_labels = merged_df['cluster_id_weighted']
    unweighted_labels = merged_df['cluster_id_unweighted']
    
    ari = adjusted_rand_score(weighted_labels, unweighted_labels)
    ari_scores.append(ari)
    print(f"Run {run_idx+1} ARI: {ari:.4f}")


mean_ari = np.mean(ari_scores)
std_ari = np.std(ari_scores)
print("\n=== ARI Summary Across Weighted Runs ===")
print(f"Mean ARI: {mean_ari:.4f}")
print(f"Standard Deviation of ARI: {std_ari:.4f}")

# save results
ari_output_path = os.path.join(OUTPUT_DIR, "ari_scores_across_runs.tsv")
ari_df = pd.DataFrame({
    "Run": [f"Run_{i+1}" for i in range(len(ari_scores))],
    "ARI": ari_scores
})
ari_df.to_csv(ari_output_path, sep='\t', index=False)

# plt.figure(figsize=(8, 6))
# sns.stripplot(data=ari_scores, size=8, color="blue", jitter=True)
# plt.title("ARI Scores Across Weighted Runs", fontsize=16)
# plt.ylabel("Adjusted Rand Index (ARI)")
# plt.ylim(0, 1.05)
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "ari_scores_plot.png"), dpi=300)
# plt.close()

print("\nProcess finished.")
