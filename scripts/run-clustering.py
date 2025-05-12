"""
This python module applies leiden algorithm (Traag et al., 2019) on iGraph networks.
"""

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import time

WEIGHT_FUNCTION = 'weight_frequency_log'  # try weight_frequency_log or weight_intent_only 

def run_leiden_clustering(G, quality_function, use_weights):
    """
    This function runs a Leiden clustering algorithm on a given network
    :param G: iGraph graph object
    :param quality_function: quality function for leiden (CPM or Modularity)
    :return: cluster
    """
    print(f"Checkpoint: Running Leiden with {quality_function} | edge_weights={use_weights}.", flush=True)

    start_time = time.time() # check clustering start time
    
    weights_arg = 'weight' if use_weights else None
    
    if quality_function == "cpm_0.05":
        clusters = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=0.1, weights=weights_arg, seed=42)
    elif quality_function == "cpm_0.01":
        clusters = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=0.01, weights=weights_arg, seed=42)
    elif quality_function == "cpm_0.001":
        clusters = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=0.001, weights=weights_arg, seed=42)
    else:
        print("Invalid Quality Function - Choose from 1) cpm_0.05, 2) cpm_0.01, 3) cpm_0.001")
        sys.exit(1)

    clustering_time = (time.time() - start_time) / 60 # get total clustering run time
    print(f"Total Runtime for Leiden {quality_function}: {clustering_time:.2f} minutes.", flush=True)
    
    return clusters


def compute_clustering_stats(clusters, total_nodes, plot_boxplot=False, plot_path="./cluster_size_dist.png"):
    """
    This function computes the followings:
    1. Basic stats - Count and percentage of singleton and non-singleton clusters & node coverage
    2. Size distribution of non-singleton clusters (plotting is also available using the plot_boxplot flag).
    :param clusters: list of clusters extracted from the run_leiden_clustering function
    :param total_nodes: total number of nodes in the original graph
    :param plot_boxplot: flag to enable plotting
    :param plot_path: default path and file name to save the plot
    :return: basic stats and size distribution
    """
    nonsingleton_list = []
    nonsingleton_threshold = 10 # non-singleton cluster size threshold (10 in WCC paper - min size of cluster was 11)

    for cluster in clusters:
        if len(cluster) > nonsingleton_threshold:
            nonsingleton_list.append(len(cluster))

    total_cluster_count = len(clusters)
    nonsingleton_count = len(nonsingleton_list)
    singleton_count = total_cluster_count - nonsingleton_count
    nonsingleton_node_count = sum(nonsingleton_list) # for node coverage

    basic_stats = {
        "Total Cluster Count": total_cluster_count,
        "Singleton Cluster Count": singleton_count,
        "Non-Singleton Cluster Count": nonsingleton_count,
        "Percentage of Singleton Clusters": (singleton_count/total_cluster_count) * 100,
        "Node Coverage": (nonsingleton_node_count/total_nodes) * 100
    }

    if nonsingleton_list:
        min_size = np.min(nonsingleton_list)
        q1, q2, q3 = np.percentile(nonsingleton_list, [25, 50, 75]) # median = q2
        max_size = np.max(nonsingleton_list)

        size_distribution = {
        "Minimum Size": min_size,
        "Quartile 1 (25%)": q1,
        "Quartile 2 (median)": q2,
        "Quartile 3 (75%)": q3,
        "Maximum Size": max_size
    }

    if plot_boxplot:
        plt.figure(figsize=(10,6))
        plt.boxplot(nonsingleton_list, vert=True)
        plt.yscale("log")
        plt.title("Non-Singleton Cluster Size Distribution")
        plt.ylabel("Cluster Size")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        plt.savefig(plot_path)
        plt.show()

    return basic_stats, size_distribution


if __name__ == '__main__':
    # set dataset directory & quality function to system arguments
    edgelist_path = sys.argv[1]
    quality_function = sys.argv[2]
    
    use_weights = True  # TOGGLE THIS FOR WEIGHTED OR UNWEIGHTED
    mode = "weighted" if use_weights else "unweighted"

    basename = os.path.basename(edgelist_path)
    if "111" in basename:
        weight_tag = "_111"
    elif "312" in basename:
        weight_tag = "_312"
    elif "213" in basename:
        weight_tag = "_213"
    else:
        weight_tag = ""  # baseline

    plot_path = f"./results/size/cluster_size_dist_{quality_function}_{mode}{weight_tag}.png"

    # convert edgelist to network
    # G = ig.Graph.Read_Edgelist(edgelist_path, directed=True)  # Read_Edgelist function seems to return wrong node counts (edge count is okay)
    # use Read_Ncol() instead of Read_Edgelist() -> referred to https://stackoverflow.com/questions/32513650/can-import-edgelist-to-igraph-python & https://igraph.discourse.group/t/the-difference-between-read-ncol-and-read-edgelist/1283/5 
    # G = ig.Graph.Read_Ncol(edgelist_path, directed=True)
    
    # Load edgelist with weights
    print(f"Loading edgelist...", flush=True)
    df = pd.read_csv(edgelist_path, sep="\t", dtype=str)
    
    df['pmid'] = df['pmid'].astype(str)
    df['intxt_pmid'] = df['intxt_pmid'].astype(str)
    
    edges = list(zip(df['pmid'], df['intxt_pmid']))

    # create graph from edgelist 
    if use_weights:
        df[WEIGHT_FUNCTION] = df[WEIGHT_FUNCTION].astype(float)
        weights = df[WEIGHT_FUNCTION].tolist()
        G = ig.Graph.TupleList(edges, directed=True)
        G.es['weight'] = weights
    else:
        G = ig.Graph.TupleList(edges, directed=True)

    # count node & edge count
    node_count = G.vcount()
    edge_count = G.ecount()
    print(f"Node Count: {node_count}\nEdge Count: {edge_count}", flush=True)

    # run leiden algorithm (https://leidenalg.readthedocs.io/en/stable/intro.html)
    clusters = run_leiden_clustering(G, quality_function, use_weights)
    
    membership = clusters.membership
    node_ids = G.vs['name']
    cluster_df = pd.DataFrame({
        'node_id': node_ids,
        'cluster_id': membership
    })
    
    cluster_output_path = f"./results/membership/clusters_{quality_function}_{mode}{weight_tag}.tsv"
    cluster_df.to_csv(cluster_output_path, sep="\t", index=False)
    print(f"Saved cluster assignments to: {cluster_output_path}")
    
    G.write_pickle(f"./results/graph/graph_{quality_function}_{mode}{weight_tag}.pkl")
    print(f"Saved graph object to: ./results/graph/graph_{quality_function}_{mode}{weight_tag}.pkl")

    # get basic stats and size distribution
    stats, distribution = compute_clustering_stats(clusters, node_count, plot_boxplot=True, plot_path=plot_path)
    print("Basic Stats: ", stats)
    print("Cluster Size Distribution: ", distribution)
    
    # Plot log-log in-degree distribution
    if use_weights:
        in_degrees = G.strength(mode="IN", weights='weight')
    else:
        in_degrees = G.degree(mode="IN")
        
    unique, counts = np.unique(in_degrees, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(unique, counts, alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("In-Degree (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title("In-Degree Distribution (Log-Log)")
    plt.grid(True, linestyle="--", alpha=0.5)

    in_degree_plot_path = f"./results/indegree/in_degree_dist_{quality_function}_{mode}{weight_tag}.png"
    plt.savefig(in_degree_plot_path)