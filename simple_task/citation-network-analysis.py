"""
This python module conducts simple network tasks on citation network based on networkx package.
"""

import random
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def convert_data_to_graph(file):
    """
    This function converts the input data into a networkx graph and prints edge/node counts
    :return: networkx graph object
    """
    df = pd.read_csv(file)

    edge_list = df[['citing', 'cited']].values.tolist()

    G = nx.DiGraph() # directed graph (citing paper -> cited paper)
    G.add_edges_from(edge_list)
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
    in_degrees = dict(G.in_degree())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(in_degrees.values(), bins=30, edgecolor='black', color="green", log=True) # log scale

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

    weak_comp = list(nx.weakly_connected_components(G))

    weak_comp_list = []
    for wcc in weak_comp:
        if len(wcc) >= min_node_count:   # filter components that have more nodes than minimum node count
            weak_comp_list.append(wcc)

    # randomly select
    selected_comp = random.sample(weak_comp_list, min(num_subgraphs, len(weak_comp_list)))

    # convert components to graphs
    subgraphs = []
    for comp in selected_comp:
        subgraphs.append(G.subgraph(comp).copy())

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
    print(citation_graph)

    plot_in_degree_distribution(citation_graph)

    subgraphs = extract_random_subgraph(citation_graph, min_node_count=300, num_subgraphs=5)
    print(subgraphs)

    node_degree_stats = calculate_node_degree_stats(subgraphs)
    for stats in node_degree_stats:
        print(stats)
