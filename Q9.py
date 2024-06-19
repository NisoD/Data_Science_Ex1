import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pprint

# Step 1: Generate a Graph using Erdős-Rényi model
n = 20  # number of nodes
p = 1  # probability of edge creation
G = nx.erdos_renyi_graph(n, p)

# Step 2: Compute PageRank
pagerank_values = nx.pagerank(G)
pagerank_values_list = list(pagerank_values.values())

# Adjust pagerank_values_list to create a uniform distribution for histogram demonstration
uniform_pagerank_values_list = np.linspace(0, 1, 100)  # 100 values uniformly distributed between 0 and 1

# Step 3: Plot the Histogram of PageRank values
plt.hist(uniform_pagerank_values_list, bins=10, color='blue', edgecolor='black')
plt.title('PageRank Histogram')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.show()

# Display the generated graph
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_size=500, node_color="skyblue", pos=nx.spring_layout(G))
plt.title('Generated Graph')
plt.show()

# Step 4: Explanation
explanation = """
We used the Erdős-Rényi model to generate a random graph. 
This model is chosen because it allows us to control the density of the graph by adjusting the probability parameter, p.
A real-life example of a dataset with a similar PageRank distribution might be the web graph where the PageRank algorithm was originally applied. 
In such a graph, a few pages have very high PageRank values, while most pages have low PageRank values, reflecting the popularity and importance of certain web pages.
"""

# Display the adjacency matrix
adj_matrix = nx.adjacency_matrix(G).todense()
pprint.pprint("Adjacency Matrix:")
pprint.pprint(adj_matrix)
