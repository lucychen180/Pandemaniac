import sys
import json
import heapq
import networkx as nx
import numpy as np

if len(sys.argv) == 1:
    print("usage: pandy.py [graph filename]")
    sys.exit(1)

# filename format: num_players.num_seeds.unique_id
file_info = sys.argv[1].split('.')
num_seeds = int(file_info[1])

# Construct undirected graph
G = nx.Graph()
with open(sys.argv[1], 'r') as f:
    # load json file into a dictionary of form {node : [neighbors list]}
    adj_lists = json.load(f)
for node in adj_lists:
    for neighbor in adj_lists[node]:
        G.add_edge(node, neighbor)  # nodes are repr by strings 0 to V-1

print('number of nodes:', G.number_of_nodes())
print('number of edges:', G.number_of_edges())

# Find seed nodes based on degree
node_degs = G.degree()  # contains tuples (node, deg) of type (str, int)
seeds = heapq.nlargest(num_seeds, node_degs, key=(lambda pair : pair[1]))

# Output seed nodes
with open('seeds.txt', 'w') as f:
    for round in range(50):
        for node, deg in seeds:
            print(node, file=f)
