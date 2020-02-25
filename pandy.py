import sys
from pandy_utils import *

if len(sys.argv) == 1:
    print("usage: pandy.py [graph filename]")
    sys.exit(1)

# filename format: num_players.num_seeds.unique_id
file_info = sys.argv[1].split('.') # sys.argv[1] = filename
num_players = int(file_info[0])
num_seeds = int(file_info[1])
id = int(file_info[2])

G = load_graph_from_json(sys.argv[1])

print('number of nodes:', G.number_of_nodes())
print('number of edges:', G.number_of_edges())

# Output seed nodes
num_lines = 0 # debugging
with open('{}.{}.{}'.format(num_players, num_seeds, id) + '_seeds.txt', 'w') as f:
    for round in range(50):
        seeds = seed_n_nodes_basic(G, num_seeds, num_players)
        for node in seeds:
            print(node, file=f)
            num_lines += 1

print(num_lines)
