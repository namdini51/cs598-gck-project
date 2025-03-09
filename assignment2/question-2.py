"""
This python module applies leiden algorithm (Traag et al., 2019) on iGraph networks.
"""

import igraph as ig
import leidenalg as la
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time


def run_leiden_clustering(G, quality_function):
    """
    This function runs a Leiden clustering algorithm on a given network
    :param G: iGraph graph object
    :param quality_function: quality function for leiden (CPM or Modularity)
    :return: cluster
    """
    print(f"Checkpoint: Running Leiden with {quality_function}.", flush=True)

    start_time = time.time() # check clustering start time
    
    if quality_function == "cpm_0.01":
        clusters = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=0.01)
    elif quality_function == "cpm_0.001":
        clusters = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=0.001)
    elif quality_function == "modularity":
        clusters = la.find_partition(G, la.ModularityVertexPartition)
    else:
        print("Invalid Quality Function - Choose from 1) cpm_0.01, 2) cpm_0.001, or 3) modularity.")
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

    dataset_name = os.path.basename(edgelist_path).split('.')[0] # referred to https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    plot_path = f"./cluster_size_dist_{dataset_name}_{quality_function}.png"

    # convert edgelist to network
    # G = ig.Graph.Read_Edgelist(edgelist_path, directed=True)  # Read_Edgelist function seems to return wrong node counts (edge count is okay)
    # use Read_Ncol() instead of Read_Edgelist() -> referred to https://stackoverflow.com/questions/32513650/can-import-edgelist-to-igraph-python & https://igraph.discourse.group/t/the-difference-between-read-ncol-and-read-edgelist/1283/5 
    G = ig.Graph.Read_Ncol(edgelist_path, directed=True)

    # count node & edge count
    node_count = G.vcount()
    edge_count = G.ecount()
    print(f"Checkpoint: iGraph Network created with {dataset_name}", flush=True)
    print(f"Node Count: {node_count}\nEdge Count: {edge_count}", flush=True)

    # run leiden algorithm (https://leidenalg.readthedocs.io/en/stable/intro.html)
    clusters = run_leiden_clustering(G, quality_function)

    # get basic stats and size distribution
    stats, distribution = compute_clustering_stats(clusters, node_count, plot_boxplot=True, plot_path=plot_path)
    print("Basic Stats: ", stats)
    print("Cluster Size Distribution: ", distribution)