import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import community as community_louvain



def calculate_page_rank(matrix, beta=0.85, eps=1.0e-8, iterations=100000):
    n = matrix.shape[0]
    row_sums = matrix.sum(axis=1)
    for i in range(len(row_sums)):
        if row_sums[i] != 0:
            matrix[i] /= row_sums[i]
        else:
            row_sums[i] = 0

    ranks = np.ones(n) / n

    for _ in range(iterations):
        new_ranks = beta * matrix.T @ ranks + (1 - beta) / n
        new_ranks += beta * sum(ranks[np.where(row_sums == 0)]) / n
        if np.linalg.norm(new_ranks - ranks, 1) < eps:
            return new_ranks
        ranks = new_ranks
    return ranks


def calc_histogram(G, num_bins):
    adjacencyMatrix = nx.to_numpy_array(G, dtype=float)
    pagerank_list = calculate_page_rank(adjacencyMatrix)
    
    bin_edges = np.linspace(min(pagerank_list), max(pagerank_list), num_bins + 1)
    plt.hist(pagerank_list, bins=bin_edges,edgecolor='black')  
    plt.title('Histogram of PageRank Values')
    plt.xlabel('Bins')
    plt.ylabel('Frequency - PageRank')
    plt.show()


def Q_a():
    G = nx.Graph()
    # The Graph- Sub-Section a,1
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    edges = [(1, 2), (2, 3), (1, 3), (3, 6), (6, 7),
             (1, 4), (4, 5), (2, 8), (8, 9)]
    
    """"
        Adjacency Matrix
    [[0. 1. 1. 1. 0. 0. 0. 0. 0.]
    [1. 0. 1. 0. 0. 0. 0. 1. 0.]
    [1. 1. 0. 0. 0. 1. 0. 0. 0.]
    [1. 0. 0. 0. 1. 0. 0. 0. 0.]
    [0. 0. 0. 1. 0. 0. 0. 0. 0.]
    [0. 0. 1. 0. 0. 0. 1. 0. 0.]
    [0. 0. 0. 0. 0. 1. 0. 0. 0.]
    [0. 1. 0. 0. 0. 0. 0. 0. 1.]
    [0. 0. 0. 0. 0. 0. 0. 1. 0.]]
    """ 
    
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    calc_histogram(G, 3)
    
    return G


def Q_b(num_nodes, decrease_factor):
    # The Graph creation- Sub-Section b,1, the matrix in sparate file
    nodes = [i for i in range(1,num_nodes)] 
    G  = nx.DiGraph()
    G.add_nodes_from(nodes)

    for i in range(1,num_nodes):
        for j in range(1,int(i/decrease_factor)):
            G.add_edge(j,i)
    
    calc_histogram(G,30)
    return G


def aggregate_graph(G, partition):
    # This function was made by using the help of chatgpt 4o made the pr
    
    agg_graph = nx.Graph()
    communities = set(partition.values())
    community_pageranks = {}

    for community in communities:
        members = [node for node in partition.keys() if partition[node] == community]
        subgraph = G.subgraph(members)
        adjacencyMatrix = nx.to_numpy_array(subgraph, dtype=float)
        pagerank = calculate_page_rank(adjacencyMatrix)
        community_pageranks[community] = sum(pagerank) / len(pagerank)
        agg_graph.add_node(community, size=len(members), label=len(members))

    for node1, node2 in G.edges():
        comm1 = partition[node1]
        comm2 = partition[node2]
        if comm1 != comm2:
            weight = (community_pageranks[comm1] + community_pageranks[comm2]) / 2
            if agg_graph.has_edge(comm1, comm2):
                agg_graph[comm1][comm2]['weight'] += weight
            else:
                agg_graph.add_edge(comm1, comm2, weight=weight)

    sizes = [agg_graph.nodes[node]['size'] * 100 for node in agg_graph.nodes]
    pos = nx.circular_layout(agg_graph)
    edges = agg_graph.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]

    # Assign colors to communities
    community_colors = {community: plt.cm.tab20(i / len(communities)) for i, community in enumerate(communities)}
    node_colors = [community_colors[node] for node in agg_graph.nodes]

    labels = {node: agg_graph.nodes[node]['label'] for node in agg_graph.nodes}

    nx.draw(agg_graph, pos, labels=labels, node_size=sizes, font_size=10, node_color=node_colors, edge_color='gray')
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in agg_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(agg_graph, pos, edge_labels=edge_labels, font_color='red')
    plt.title('Aggregated Graph with PageRank Scores and Community Colors')
    plt.show()


def Q_c(G_a,G_b):
    partition_a = community_louvain.best_partition(G_a,resolution=1.3)
    partition_b = community_louvain.best_partition(G_b,resolution=1.055)

    aggregate_graph(G_a, partition_a)
    aggregate_graph(G_b, partition_b)


def main():
    G_a = Q_a()
    G_b = Q_b(100, 1.5).to_undirected()
    Q_c(G_a,G_b)


if __name__ == "__main__":
    main()