import sys
import time
from pandy_utils import *
from discount_heuristic import *
from centrality_measures import *
import random

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
    start = time.time()
    discounters = [discount_neighbor, discount_degree]
    neighbor_c = neighbor_centrality(G)
    degree_c = nx.degree_centrality(G)

    # # seed by cluster?
    # clusters, seed_nums = partition_into_clusters(G, num_seeds, num_players)
        # for i in range(len(clusters)):
        #     cluster = clusters[i]
        #     seed_num = seed_nums[i]
        #     cluster_seeds = seed_by_discount(G.subgraph(cluster), seed_num, num_players, discounters, [neighbor_c, degree_c])
        #     seeds.extend(cluster_seeds)

    # final_seeds = select_best_discounts(G, num_seeds, num_players)
    # for seeds in final_seeds:
    for _ in range(50):
        seeds = seed_by_discount(G, num_seeds, num_players, discounters, [neighbor_c, degree_c])
        print('\rround {}/{}'.format(round, 50), end='')
        for node in seeds:
            print(node, file=f)
            num_lines += 1
    print('\n')
    end = time.time()

print('Time elapsed in seconds: {}'.format(end - start))
print('Number of seeds: {}'.format(num_lines))
