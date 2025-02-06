"""
This python module conducts simple network tasks on citation network based on networkx package.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def convert_data_to_graph(file):
    """
    This function converts the input data into a networkx graph and prints edge/node counts
    :return: graph object
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
    :return:
    """
    in_degrees = dict(G.in_degree())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(in_degrees.values(), bins=30, edgecolor='black', color="green", log=True) # log scale

    ax.set_xlabel("In-Degree (i.e., number of received citation)")
    ax.set_ylabel("Frequency")
    ax.set_title("In-Degree Distribution")
    # plt.show()

    fig.savefig(output_path)


def extract_random_subgraph():
    """
    This function randomly extract connected subgraphs from a given graph
    :return:
    """
    pass


def calculate_node_stats():
    """
    This function calculates the following node statistics for the extracted subgraphs:
      1. minimum in-degree & total degree
      2. maximum in-degree & total degree
      3. median in degree & total degree
    :return:
    """
    pass


if __name__ == '__main__':
    file_path = "./data/sample_open_citations_curated.csv"
    citation_graph = convert_data_to_graph(file_path)
    print(citation_graph)

    plot_in_degree_distribution(citation_graph)