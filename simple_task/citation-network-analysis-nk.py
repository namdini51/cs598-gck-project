"""
This python module conducts simple network tasks on citation network based on networkit package.
"""

import csv
import random
import networkit as nk
import numpy as np
import matplotlib.pyplot as plt


def convert_data_to_graph(file):
    """
    This function converts the input data into a networkit graph and prints edge/node counts
    :return: networkit graph object and node dictionary for reference
    """

    G = nk.graph.Graph(directed=True)
    print("Checkpoint: Networkit Graph created!", flush=True)

    node_dict = {}
    next_node_idx = 0

    with open(file, "r") as f:
        dataset = csv.reader(f)
        header = next(dataset)
        citing_idx = header.index("citing")
        cited_idx = header.index("cited")

        count = 0
        for row in dataset:
            start = row[citing_idx]
            end = row[cited_idx]

            if start not in node_dict:
                node_dict[start] = next_node_idx
                G.addNode()
                next_node_idx += 1

            if end not in node_dict:
                node_dict[end] = next_node_idx
                G.addNode()
                next_node_idx += 1

            G.addEdge(node_dict[start], node_dict[end])

            count += 1
            if count % 100000000 == 0:
                print(f"Checkpoint: Processed {count} edges...", flush=True)

    edge_count = G.numberOfEdges()
    node_count = G.numberOfNodes()

    print("Number of edges: ", edge_count)
    print("Number of nodes: ", node_count)

    return G, node_dict


def plot_in_degree_distribution(G, output_path="./in_degree_distribution.png"):
    """
    This function plots the in-degree distribution of every node
    :param G: networkit graph object
    :param output_path: directory path to save output
    :return: None
    """

    in_degree = nk.centrality.DegreeCentrality(G, outDeg=False).run()
    in_degree_list = in_degree.scores()

    unique, counts = np.unique(in_degree_list, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(unique, counts, color="blue", alpha=0.7)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("In-Degree (Log Scale)")
    plt.ylabel("Frequency (Log Scale)")
    plt.title("In-Degree Distribution (Log-Log Scatter Plot)")
    plt.grid(True, linestyle="--", linewidth=0.5)

    plt.savefig(output_path)


def extract_random_subgraph(G, num_subgraphs=5, min_node_count=100000):
    """
    This function randomly extract subgraph by BFS method
    :param G: networkit graph object
    :param num_subgraphs: number of needed subgraphs
    :param min_node_count: minimum number of nodes in a subgraph
    :param radius: number of hops a node expands
    :return: list of subgraphs
    """

    components = nk.components.WeaklyConnectedComponents(G).run().getComponents()

    print(f"Checkpoint: Found {len(components)} total connected components.", flush=True)

    comp_list = []
    for comp in components:
        if len(comp) >= min_node_count:   # filter components that have more nodes than minimum node count
            comp_list.append(comp)

    print(f"Checkpoint: {len(comp_list)} components meet the size requirement.", flush=True)

    # randomly select
    selected_comp = random.sample(comp_list, min(num_subgraphs, len(comp_list)))

    # convert components to graphs
    subgraph_list = []
    for comp in selected_comp:
        subgraph = nk.graphtools.subgraphFromNodes(G, comp)
        subgraph_list.append(subgraph)

    print("Checkpoint: Subgraph extraction complete!", flush=True)
    return subgraph_list


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
        in_degree = nk.centrality.DegreeCentrality(g, outDeg=False).run()
        in_degree_list = np.array(in_degree.scores())

        out_degree = nk.centrality.DegreeCentrality(g, outDeg=True).run()
        out_degree_list = np.array(out_degree.scores())

        total_degree_list = in_degree_list + out_degree_list

        stats = {
            "Minimum In-Degree": np.min(in_degree_list),
            "Maximum In-Degree": np.max(in_degree_list),
            "Median In-Degree": np.median(in_degree_list),
            "Minimum Total Degree": np.min(total_degree_list),
            "Maximum Total Degree": np.max(total_degree_list),
            "Median Total Degree": np.median(total_degree_list)
        }

        node_degree_stats.append(stats)

    return node_degree_stats


if __name__ == '__main__':
    file_path = "./data/sample_open_citations_curated.csv"
    citation_graph, node_dict = convert_data_to_graph(file_path)
    # print(citation_graph)

    plot_in_degree_distribution(citation_graph)

    subgraphs = extract_random_subgraph(citation_graph, min_node_count=300, num_subgraphs=5)
    # print(subgraphs)

    node_degree_stats = calculate_node_degree_stats(subgraphs)
    for stats in node_degree_stats:
        print(stats)
