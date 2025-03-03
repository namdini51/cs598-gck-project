"""
This python module applies leiden algorithm (Traag et al., 2019) on iGraph networks.
"""

import igraph as ig
import leidenalg as la
import numpy as np
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


def compute_clustering_stats(clusters):
    """
    This function computes the count and percentage of singleton and non-singleton clusters
    :param clusters: list of clusters extracted from the run_leiden_clustering function
    :return: count and percentage of singleton and non-singleton clusters
    """
    singleton_count = 0
    nonsingleton_count = 0

    for cluster in clusters:
        if len(cluster) == 1:    # only has one node
            singleton_count += 1
        else:
            nonsingleton_count += 1

    total_cluster_count = singleton_count + nonsingleton_count

    stats = {
        "Total Cluster Count": total_cluster_count,
        "Singleton Cluster Count": singleton_count,
        "Non-Singleton Cluster Count": nonsingleton_count,
        "Percentage of Singleton Clusters": (singleton_count/total_cluster_count) * 100
    }

    return stats


#TODO: need part that computes size distribution of non-singleton clusters


if __name__ == '__main__':
    # set dataset directory & quality function to system arguments
    edgelist_path = sys.argv[1]
    quality_function = sys.argv[2]

    # convert edgelist to network
    G = ig.Graph.Read_Edgelist(edgelist_path, directed=True)
    node_count = G.vcount()
    edge_count = G.ecount()
    print(f"Checkpoint: iGraph Network created with {edgelist_path}", flush=True)
    print(f"Node Count: {node_count}\nEdge Count: {edge_count}", flush=True)

    # run leiden algorithm (https://leidenalg.readthedocs.io/en/stable/intro.html)
    clusters = run_leiden_clustering(G, quality_function)
    stats = compute_clustering_stats(clusters)
    print(stats)