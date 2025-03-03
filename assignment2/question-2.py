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

    if quality_function == "cpm_0.01":
        cluster = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=0.01)
    elif quality_function == "cpm_0.001":
        cluster = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=0.001)
    elif quality_function == "modularity":
        cluster = la.find_partition(G, la.ModularityVertexPartition)
    else:
        print("Invalid Quality Function - Choose from 1) cpm_0.01, 2) cpm_0.001, or 3) modularity.")
        sys.exit(1)

    return cluster


def calculate_network_stats():
    pass


if __name__ == '__main__':
    # set dataset directory & quality function to system arguments
    edgelist_path = sys.argv[1]
    quality_function = sys.argv[2]

    start_time = time.time() # check program start time

    # convert edgelist to network
    # edgelist_path = "./data/cit_hepph_cleaned.tsv"
    G = ig.Graph.Read_Edgelist(edgelist_path, directed=True)
    print(f"Checkpoint: iGraph Network created with {edgelist_path}", flush=True)

    # run leiden algorithm (https://leidenalg.readthedocs.io/en/stable/intro.html)
    cluster = run_leiden_clustering(G, quality_function)
    # cluster = run_leiden_clustering(G, "cpm_0.01")

    end_time = time.time() - start_time # get total run time
    print(f"Total Runtime for Leiden {quality_function} on {edgelist_path}: {end_time:.2f} seconds.", flush=True)