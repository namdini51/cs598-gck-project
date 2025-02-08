"""
This python module conducts simple network tasks on citation network based on networkx package.
"""

import csv
import random
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def convert_data_to_graph(file, split_length=100000):
    """
    This function converts the input data into a networkx graph and prints edge/node counts
    :return: networkx graph object
    """

    G = nx.DiGraph()  # Directed graph (citing paper -> cited paper)

    # dtype_spec = {"citing": "str", "cited": "str"}
    #
    # # process by split data (handle OOM)
    # split_count = 0
    # for split in pd.read_csv(file, chunksize=split_length, dtype=dtype_spec):
    #     edge_list = zip(split["citing"], split["cited"])  # Process only this chunk
    #     G.add_edges_from(edge_list)
    #     split_count += 1

    with open(file, "r") as f:
        dataset = csv.reader(f)
        header = next(dataset)
        citing_idx = header.index("citing")
        cited_idx = header.index("cited")

        count = 0
        for row in dataset:
            G.add_edge(row[citing_idx], row[cited_idx])
            count += 1
            if count % 1000000 == 0:
                print(f"Checkpoint: Processed {count} edges...", flush=True)

    edge_count = G.number_of_edges()
    node_count = G.number_of_nodes()

    print("Number of edges: ", edge_count)
    print("Number of nodes: ", node_count)

    return G


def plot_in_degree_distribution(G, output_path="./in_degree_distribution.png"):
    """
    This function plots the in-degree distribution of every node
    :param G: networkx graph object
    :param output_path: directory path to save output
    :return: None
    """

    in_degree_list = []
    for node, degree in G.in_degree():
        in_degree_list.append(degree)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(in_degree_list, bins=30, edgecolor='black', color="green", log=True) # log scale

    ax.set_xlabel("In-Degree (i.e., number of received citation)")
    ax.set_ylabel("Frequency")
    ax.set_title("In-Degree Distribution")
    # plt.show()

    fig.savefig(output_path)


def extract_random_subgraph(G, num_subgraphs=5, min_node_count=100000):
    """
    This function randomly extract connected components from a given graph based on Weakly Connected Components and converts them into subgraphs
    :param G: networkx graph object
    :param num_subgraphs: number of needed subgraphs
    :param min_node_count: minimum number of nodes in a subgraph
    :return: list of subgraphs
    """

    weak_comp = nx.weakly_connected_components(G)

    weak_comp_list = []
    for comp in weak_comp:
        if len(comp) >= min_node_count:   # filter components that have more nodes than minimum node count
            weak_comp_list.append(comp)

    # randomly select
    selected_comp = random.sample(weak_comp_list, min(num_subgraphs, len(weak_comp_list)))

    # convert components to graphs
    subgraphs = []
    for comp in selected_comp:
        subgraphs.append(G.subgraph(comp))

    return subgraphs


def calculate_node_degree_stats(subgraphs):
    """
    This function calculates the following node degree statistics for the extracted subgraphs:
      1. minimum in-degree & total degree
      2. maximum in-degree & total degree
      3. median in degree & total degree
    :param subgraphs: list of subgraphs
    :return:
    """
    node_degree_stats = []

    for g in subgraphs:
        in_deg_list = []
        for node, degree in g.in_degree():  # extract in-degree from all nodes
            in_deg_list.append(degree)

        total_deg_list = []
        for node, degree in g.degree():  # extract total degree from all nodes
            total_deg_list.append(degree)

        # in-degree calculation
        in_deg_min = min(in_deg_list)
        in_deg_max = max(in_deg_list)
        in_deg_median = float(np.median(in_deg_list))

        # total degree calculation
        total_deg_min = min(total_deg_list)
        total_deg_max = max(total_deg_list)
        total_deg_median = float(np.median(total_deg_list))

        subgraph_node_deg_stats = {
            "Minimum In-Degree": in_deg_min,
            "Maximum In-Degree": in_deg_max,
            "Median In-Degree": in_deg_median,
            "Minimum Total Degree": total_deg_min,
            "Maximum Total Degree": total_deg_max,
            "Median Total Degree": total_deg_median
        }

        node_degree_stats.append(subgraph_node_deg_stats)

    return node_degree_stats


if __name__ == '__main__':
    file_path = "./data/sample_open_citations_curated.csv"
    citation_graph = convert_data_to_graph(file_path)
    # print(citation_graph)

    plot_in_degree_distribution(citation_graph)

    subgraphs = extract_random_subgraph(citation_graph, min_node_count=300, num_subgraphs=5)
    # print(subgraphs)

    node_degree_stats = calculate_node_degree_stats(subgraphs)
    for stats in node_degree_stats:
        print(stats)
