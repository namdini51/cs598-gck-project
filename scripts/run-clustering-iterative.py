"""
Script for iterative clusterings runs for stability check
"""

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time

WEIGHT_FUNCTION = 'weight_frequency_log' 

def run_leiden_clustering(G, quality_function, use_weights, run_id):
    print(f"Checkpoint: Running Leiden with {quality_function} | edge_weights={use_weights} | run_id={run_id}.", flush=True)

    start_time = time.time()
    weights_arg = 'weight' if use_weights else None

    if quality_function == "cpm_0.1":
        resolution = 0.1
    elif quality_function == "cpm_0.01":
        resolution = 0.01
    elif quality_function == "cpm_0.001":
        resolution = 0.001
    else:
        print("Invalid Quality Function - Choose from: cpm_0.1, cpm_0.01, cpm_0.001")
        sys.exit(1)

    seed = 42 + run_id  # different seed per run
    clusters = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=resolution, weights=weights_arg, seed=seed)

    clustering_time = (time.time() - start_time) / 60
    print(f"Total Runtime: {clustering_time:.2f} minutes.", flush=True)

    return clusters

def compute_clustering_stats(clusters, total_nodes):
    nonsingleton_list = []
    nonsingleton_threshold = 10

    for cluster in clusters:
        if len(cluster) > nonsingleton_threshold:
            nonsingleton_list.append(len(cluster))

    total_cluster_count = len(clusters)
    nonsingleton_count = len(nonsingleton_list)
    singleton_count = total_cluster_count - nonsingleton_count
    nonsingleton_node_count = sum(nonsingleton_list)

    basic_stats = {
        "Total Cluster Count": total_cluster_count,
        "Singleton Cluster Count": singleton_count,
        "Non-Singleton Cluster Count": nonsingleton_count,
        "Percentage of Singleton Clusters": (singleton_count / total_cluster_count) * 100,
        "Node Coverage": (nonsingleton_node_count / total_nodes) * 100
    }

    return basic_stats

if __name__ == '__main__':
    edgelist_path = sys.argv[1]
    quality_function = sys.argv[2]
    run_id = int(sys.argv[3])

    use_weights = True
    mode = "weighted"

    basename = os.path.basename(edgelist_path)
    weight_tag = ""

    cluster_output_path = f"./results/membership/clusters_{quality_function}_{mode}_run{run_id}.tsv"
    graph_output_path = f"./results/graph/graph_{quality_function}_{mode}_run{run_id}.pkl"
    plot_path = f"./results/cluster_size_dist_{quality_function}_{mode}_run{run_id}.png"

    print(f"Loading edgelist...", flush=True)
    df = pd.read_csv(edgelist_path, sep="\t", dtype=str)
    df['pmid'] = df['pmid'].astype(str)
    df['intxt_pmid'] = df['intxt_pmid'].astype(str)

    edges = list(zip(df['pmid'], df['intxt_pmid']))

    if use_weights:
        df[WEIGHT_FUNCTION] = df[WEIGHT_FUNCTION].astype(float)
        weights = df[WEIGHT_FUNCTION].tolist()
        G = ig.Graph.TupleList(edges, directed=True)
        G.es['weight'] = weights
    else:
        G = ig.Graph.TupleList(edges, directed=True)

    print(f"Node Count: {G.vcount()} | Edge Count: {G.ecount()}", flush=True)

    clusters = run_leiden_clustering(G, quality_function, use_weights, run_id)

    membership = clusters.membership
    node_ids = G.vs['name']
    cluster_df = pd.DataFrame({'node_id': node_ids, 'cluster_id': membership})
    cluster_df.to_csv(cluster_output_path, sep="\t", index=False)
    print(f"Saved membership to: {cluster_output_path}", flush=True)

    G.write_pickle(graph_output_path)
    print(f"Saved graph object to: {graph_output_path}", flush=True)

    stats = compute_clustering_stats(clusters, G.vcount())
    print(f"Basic Clustering Stats: {stats}", flush=True)
