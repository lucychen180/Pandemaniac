import math
import random
from centrality_measures import *
import networkx as nx
import numpy
import sim

def make_dict_from_graph(G):
    dict = {}
    for node in G.nodes():
        dict[node] = list(nx.all_neighbors(G, node))

    return dict

def select_best_discounts(G, n, num_players, max_iter = 1000):
    '''
    Choose the best group of nodes based on simulations against a strategy that
    picks the best n out of the top 1-2n (random) degree or eigenvector centralities. After random walking,
    it returns the 50 best groups of nodes.
    '''
    round_score = {} # key is round number, value is
    round_seed = {} # key is round number, value is seed list
    neighbor_c = neighbor_centrality(G)
    degree_c = nx.degree_centrality(G)
    eigen_c = nx.eigenvector_centrality(G)
    discounters = [discount_neighbor, discount_degree]
    graph_dict = make_dict_from_graph(G)
    ta_possible_degree_nodes = list(G.nodes())
    ta_possible_eigen_nodes = list(G.nodes())
    ta_possible_degree_nodes.sort(reverse = True, key = lambda v: degree_c[v])
    ta_possible_eigen_nodes.sort(reverse = True, key = lambda v: eigen_c[v])
    node_lists = {} # pass into sim
    for i in range(max_iter):
        print('\rround {}/{}'.format(i, 1000), end='')
        # generate our nodes
        seeds = seed_by_discount(G, n, num_players, discounters, [neighbor_c, degree_c])
        node_lists['pandemonium'] = seeds

        # test it 5 times against random ta nodes
        # build ta nodes; score is how many total nodes it won
        score = 0
        for _ in range(3):
            for dummy_player in range(num_players):
                ta_nodes = random.choice([ta_possible_eigen_nodes, ta_possible_degree_nodes])
                rand_scaling_factor = random.uniform(1, 2)
                ta_nodes = random.sample(ta_nodes[:math.ceil(rand_scaling_factor * n)], n)
                node_lists['TA{}'.format(dummy_player)] = ta_nodes

            results = sim.run(graph_dict, node_lists)
            score += results['pandemonium']

        round_score[i] = score
        round_seed[i] = seeds

    # find best 50 scores by node
    indices = list(round_score.keys())
    indices.sort(reverse = True, key = lambda i: round_score[i])
    final_seeds = []
    for k in range(50):
        final_seeds.append(round_seed[indices[k]])

    return final_seeds

def seed_by_discount(G, n, num_players, discounters, centralities):
    # randomness that scales with number of players
    new_centralities = centralities.copy()
    scale_factor = max(math.sqrt(num_players - 1), 1)
    possible_nodes = list(G.nodes())
    seeds = []
    remaining_nodes = n

    # discount less if our node has a high chance of being canceled!
    # actually high chance of being canceled for 2 player, since we select for
    # best node each time
    if num_players == 1:
        discount_p = 1 # no chance of being canceled
    elif num_players == 2:
        discount_p = 0 # very high chance of being canceled
    else:
        discount_p = 1 # scale with low chance of getting canceled

    for _ in range(n):
        # normalize so sum isn't dominated by one centrality measure
        normalized_centralities = []
        for i in range(len(centralities)):
            normalized_centralities.append(normalize_dict(centralities[i]))

        possible_nodes.sort(reverse = True, key = lambda v: sum([c[v] for c in normalized_centralities]))
        # if num_players == 2:
        #     v = possible_nodes[0] # pick best possible node for 2 players
        # # pick a random node from the top scale factor * remaining_nodes left as a seed
        # else:
        v = random.choice(possible_nodes[:math.ceil(scale_factor * remaining_nodes)])
        for i in range(len(discounters)):
            discounter = discounters[i]
            centralities[i] = discounter(G, v, centralities[i], discount_p)
        seeds.append(v)
        possible_nodes.remove(v)
        remaining_nodes -= 1

    return seeds

def discount_neighbor(G, v, centrality_dict, discount_scale):
    '''
    Discount all neighbors of v if v is chosen into the seed set,
    given by the neighborhood centrality.
    '''
    updated_centralities = centrality_dict.copy()
    seen = set()
    seen.add(v)
    for neighbor in nx.all_neighbors(G, v):
        updated_centralities[neighbor] -= (discount_scale * 0.3 * centrality_dict[v])
        seen.add(neighbor)

    for neighbor in nx.all_neighbors(G, v):
        for deg_2_neighbor in nx.all_neighbors(G, neighbor):
            if deg_2_neighbor not in seen:
                updated_centralities[deg_2_neighbor] -= (discount_scale * 0.3 * 0.3 * centrality_dict[v])
                seen.add(deg_2_neighbor)

    return updated_centralities

def discount_degree(G, v, centrality_dict, discount_scale):
    '''
    Discount all neighbors of v if v is chosen into the seed set,
    given by the degree centrality.
    '''
    updated_centralities = centrality_dict.copy()
    for neighbor in nx.all_neighbors(G, v):
        updated_centralities[neighbor] -= discount_scale
    return updated_centralities
