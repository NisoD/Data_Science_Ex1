import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pprint
import random


def calculate_page_rank(adjacencyMatrix,n, beta=0.85, max_iter=1000, eps=1.0e-8):

    row_sums = adjacencyMatrix.sum(axis=1)
    for i in range(len(row_sums)):
        if row_sums[i] != 0:
            adjacencyMatrix[i] /= row_sums[i]
        else:
            row_sums[i] = 0

    ranks = np.ones(n) / n

    for _ in range(max_iter):
        new_ranks = beta * adjacencyMatrix.T @ ranks + (1 - beta) / n
        new_ranks += beta * sum(ranks[np.where(row_sums == 0)]) / n

        if np.linalg.norm(new_ranks - ranks, 1) < eps:
            return  new_ranks
        ranks = new_ranks
    return ranks


def calc_histogram(G, num_bins):
    # Sub-Section (2) the calculation are the same for both section a and b
    n = G.number_of_nodes()
    adjacencyMatrix = nx.to_numpy_array(G, dtype=float)
    pagerank_list = calculate_page_rank(adjacencyMatrix,n)

    bin_edges = np.linspace(
        min(pagerank_list), max(pagerank_list), num_bins + 1)
    plt.hist(pagerank_list, bins=bin_edges,
             edgecolor='black')  # Equal-length bins
    plt.title('Histogram of PageRank Values')
    plt.xlabel('PageRank')
    plt.ylabel('Frequency')
    plt.show()


def Q_a():
    G = nx.Graph()

    # The Graph- Sub-Section a,1

    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    edges = [(1, 2), (2, 3), (1, 3), (3, 6), (6, 7),
             (1, 4), (4, 5), (2, 8), (8, 9)]

    # Sub-Section a,3
    """ Explanation: 
    We chose the edges such that for all  k the groups v_k={v:deg(v)=k} |v_k| is the same.
    The reason is that in such way the pagerank values will be the same for all vertices in the same group.
    """

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Vialuzation if needed :
    # plt.figure(figsize=(8, 6))
    # nx.draw(G, with_labels=True, node_color='lightblue',
    #         edge_color='gray', node_size=500, font_size=16)
    # plt.title("Custom Graph")
    # plt.show()

    calc_histogram(G, 3)
    # Sub-Sectopm a,4
    """an example of a dataset is that achevies uniform pagerank is dataset representing a
      social network of employees in a company where each department is structured
     such that all members have exactly the same number of connections within and outside their department. """


# def random_subset(seq, m, seed=42):
#     targets = set()
#     rng = random.Random(seed)
#     while len(targets) < m:
#         x = rng.choice(seq)
#         targets.add(x)
#     return targets


def Q_b():
    n = 200
    m = 8
    G = nx.Graph()
    nodes = list(range(m))
    G.add_nodes_from(nodes)

    for i in range(m):
        for j in range(i + 1, m):
            if i != j:
                G.add_edge(i, j)
    
    for i in range(n):
        new_node = m + i
        G.add_node(new_node)
        targets = np.random.choice(list(G.nodes()), m, replace=True)
        for target in targets:
            G.add_edge(new_node, target)



    plt.figure(figsize=(12, 8))
    nx.draw(G, with_labels=True, node_color='lightblue')
    plt.title("Randomly Generated Graph with Preferential Attachment")
    plt.show()

    calc_histogram(G, 10)

def main():
    Q_a()
    Q_b()


if __name__ == '__main__':
    main()
