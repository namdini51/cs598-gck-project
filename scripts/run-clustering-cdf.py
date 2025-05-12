import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig

from scipy.stats import ks_2samp
import sys
import os

# === CONFIGURATION ===
edgelist_path = sys.argv[1]
output_plot_path = "./results/indegree/in_degree_cdf_comparison.png"
WEIGHT_FUNCTION = 'weight_frequency_log'  # try weight_frequency_log or weight_sqrt_freq_log

# load data
print("Loading edge list...", flush=True)
df = pd.read_csv(edgelist_path, sep="\t", dtype=str)
df[WEIGHT_FUNCTION] = df[WEIGHT_FUNCTION].astype(float)
edges = list(zip(df['pmid'], df['intxt_pmid']))
weights = df[WEIGHT_FUNCTION].tolist()

# build graph
print("Building iGraph graphs...", flush=True)
G_unweighted = ig.Graph.TupleList(edges, directed=True)
G_weighted = ig.Graph.TupleList(edges, directed=True)
G_weighted.es['weight'] = weights

# compute in-degree
print("Computing in-degree distributions...", flush=True)
deg_unweighted = np.array(G_unweighted.degree(mode="IN"))
deg_weighted = np.array(G_weighted.strength(mode="IN", weights='weight'))

# filter out zero in-degrees
deg_weighted = deg_weighted[deg_weighted > 0]
deg_unweighted = deg_unweighted[deg_unweighted > 0]

def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, cdf

x_unw, cdf_unw = compute_cdf(deg_unweighted)
x_w, cdf_w = compute_cdf(deg_weighted)

# KS Test
D, p_value = ks_2samp(deg_unweighted, deg_weighted)
print("\n[KS Test]", flush=True)
print(f"KS D-statistic: {D:.4f}", flush=True)
print(f"p-value: {p_value:.2e}", flush=True)

# plot cdf
plt.figure(figsize=(10, 6))
plt.plot(x_unw, cdf_unw, label="Unweighted", lw=2, linestyle="--")
plt.plot(x_w, cdf_w, label="Weighted", lw=2, linestyle="-")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("In-Degree (log scale)")
plt.ylabel("CDF (log scale)")
plt.title("CDF of In-Degree Distributions (Log-Log)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(output_plot_path, dpi=300)
plt.close()

# in-degree stats
def describe(label, values):
    print(f"\n[{label}]")
    print(f"Min:     {np.min(values):.4f}")
    print(f"Mean:    {np.mean(values):.4f}")
    print(f"Median:  {np.median(values):.4f}")
    print(f"Max:     {np.max(values):.4f}")
    print(f"Std Dev: {np.std(values):.4f}")

describe("Unweighted In-Degree", deg_unweighted)
describe("Weighted In-Degree", deg_weighted)
print("Process finished.", flush=True)
