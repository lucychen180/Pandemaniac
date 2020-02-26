import sys
import time
from pandy_utils import *
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
    # if we're using clustering
    possible_seeds, seed_nums = possible_cluster_eigen_neighbor(G, num_seeds, num_players)
    # if we're not using clustering
    # possible_seeds = clusterless_neighbor(G, num_seeds, num_players)
    for round in range(50):
        # # if we're using clustering, uncomment
        seeds = []
        for i in range(len(possible_seeds)):
            possible_cluster_seeds = possible_seeds[i]
            cluster_seed_num = seed_nums[i]
            cluster_seeds = random.sample(possible_cluster_seeds, cluster_seed_num)
            seeds.extend(cluster_seeds)

        # # if we're not using clustering, uncomment
        # seeds = random.sample(possible_seeds, num_seeds)

        print('\rround {}/{}'.format(round, 50), end='')
        for node in seeds:
            print(node, file=f)
            num_lines += 1
    print('\n')
    end = time.time()

print('Time elapsed in seconds: {}'.format(end - start))
print('Number of seeds: {}'.format(num_lines))
