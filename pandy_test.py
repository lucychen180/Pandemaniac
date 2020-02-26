from pandy_utils import *
import sim
import pprint


graph_1 = load_graph_from_json('testgraph1.json')
graph_2 = load_graph_from_json('testgraph2.json')

# load graphs into dictionary
with open('testgraph1.json', 'r') as f:
    graph_1_json = json.load(f)

with open('testgraph2.json', 'r') as f:
    graph_2_json = json.load(f)

num_players = 2 # competing against TA; TA uses largest degree strategy for now

for num_seeds in range(5, 30, 5):
    print("Number of seeds: {}".format(num_seeds))

    # choose which nodes to seed
    strategy_dict_1 = {}
    strategy_dict_2 = {}
    strategy_dict_1['TA_strategy_graph1'] = seed_by_degree(graph_1, num_seeds)
    strategy_dict_2['TA_strategy_graph2'] = seed_by_degree(graph_2, num_seeds)
    # strategy_dict_1['basic_graph1'] = cluster_neighbor_centrality(graph_1, num_seeds, num_players)
    # strategy_dict_2['basic_graph2'] = cluster_neighbor_centrality(graph_2, num_seeds, num_players)

    strategy_dict_1['basic_graph1'] = clusterless_ksc_neighbor_eigen(graph_1, num_seeds, num_players)
    strategy_dict_2['basic_graph2'] = clusterless_ksc_neighbor_eigen(graph_2, num_seeds, num_players)
    print('Seeds generated!')

    print('Simulating graph 1...')
    results_1 = sim.run(graph_1_json, strategy_dict_1)
    print('Simulating graph 2...')
    results_2 = sim.run(graph_2_json, strategy_dict_2)

    print("Results for graph 1: ")
    pprint.pprint(results_1)
    print("Results for graph 2: ")
    pprint.pprint(results_2)
    print('\n')
