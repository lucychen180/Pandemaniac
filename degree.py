import sys
import json
import heapq
import networkx as nx
import numpy as np

if len(sys.argv) == 1:
    print("usage: pandy.py [graph filename] [num_seeds]")
    sys.exit(1)

# filename format: num_players.num_seeds.unique_id
file_info = sys.argv[1].split('.')
num_seeds = int(file_info[1])
# num_seeds = int(sys.argv[2])

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
seed_pool = heapq.nlargest(2*num_seeds, node_degs, key=(lambda pair : pair[1]))
seed_pool = np.array(seed_pool)

# Output seed nodes
out_file = '.'.join(file_info[:-1]) + '_seeds.txt'
with open(out_file, 'w') as f:
    for round in range(50):
        # guarantee best seeds
        top = min(3, num_seeds)
        top_pool = seed_pool[:top, 0]
        for node in top_pool:
            print(node, file=f)
        # randomly pick remaining seeds
        random_pool = np.random.choice(seed_pool[top:2*num_seeds-3, 0],
                                       size=(num_seeds-top), replace=False)
        for node in random_pool:
            print(node, file=f)
