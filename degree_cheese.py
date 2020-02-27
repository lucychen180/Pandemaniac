import sys
import json
import sim
from centrality_measures import *
from pandy_utils import *
import itertools

def seeds_by_degree_cheese(G, G_json, num_cheese, total_seeds):
    '''
    Number of cheese nodes for destroying the TA_degree strategy
    Recommended to not use too many nodes (keep num_cheese as close to max)
    because runtime scales by choose function.
    @param G: graph
    @param num_cheese: number of nodes that we are cheesing
    @return nodes that win the best against TA_degree
    '''
    free = total_seeds - num_cheese
    graph_degrees = degree_centrality(G)
    graph_degrees_list = [(node, graph_degrees[node]) for node in graph_degrees]
    graph_degrees_list.sort(reverse=True, key=lambda x: x[1])

    seeds = [graph_degrees_list[i][0] for i in range(min(num_cheese, total_seeds))]
    win_by = 0

    sim_results = {}
    successful_seeds = []
    ta_seeds = [graph_degrees_list[total_seeds - i - 1][0] for i in range(max(free, 0))]
    combos = list(itertools.combinations(graph_degrees_list[num_cheese:], free))
    combos_length = len(combos)
    iteration = 0
    for combo in combos:
        print('\r combo {}/{}'.format(iteration, combos_length), end='')
        combo_nodes = tuple([node[0] for node in combo])
        strat = {'ta': ta_seeds, 'pandemonium': combo_nodes}
        sim_result = sim.run(G_json, strat)
        sim_results[combo_nodes] = sim_result
        if sim_result['pandemonium'] - sim_result['ta'] > win_by:
            win_by = sim_result['pandemonium'] - sim_result['ta']
            successful_seeds = combo_nodes
        iteration += 1

    print('Best win: ', win_by)

    return seeds + list(successful_seeds)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: degree_cheese.py [graph filename]")
        sys.exit(1)

    # filename format: num_players.num_seeds.unique_id
    file_info = sys.argv[1].split('.') # sys.argv[1] = graph filename
    num_players = int(file_info[0])
    num_seeds = int(file_info[1])
    id = int(file_info[2].split('-')[0])

    G = load_graph_from_json(sys.argv[1])

    with open(sys.argv[1], 'r') as f:
        G_json = json.load(f)

    print('number of nodes:', G.number_of_nodes())
    print('number of edges:', G.number_of_edges())

    print(seeds_by_degree_cheese(G, G_json, 9, 10))
