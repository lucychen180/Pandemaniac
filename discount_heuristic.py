import math
import random
from centrality_measures import *
import networkx as nx
import numpy

def seed_by_discount(G, n, num_players, discounters, centralities):
    # randomness that scales with number of players
    new_centralities = centralities.copy()
    scale_factor = max(math.sqrt(num_players - 1), 1)
    possible_nodes = list(G.nodes())
    seeds = []
    remaining_nodes = n
    if num_players == 1:
        discount_p = 1 # no chance of being canceled
    elif num_players == 2:
        discount_p = 0 # very high chance of being canceled
    else:
        discount_p = num_players / 8 # scale with low chance of getting canceled

    for _ in range(n):
        # normalize so sum isn't dominated by one centrality measure
        for i in range(len(centralities)):
            centralities[i] = normalize_dict(centralities[i])

        possible_nodes.sort(reverse = True, key = lambda v: sum([c[v] for c in centralities]))
        if num_players == 2:
            v = possible_nodes[0] # pick best possible node for 2 players
        # pick a random node from the top scale factor * remaining_nodes left as a seed
        else:
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
        updated_centralities[neighbor] -= (discount_scale * 0.25 * centrality_dict[v])
        seen.add(neighbor)

    for neighbor in nx.all_neighbors(G, v):
        for deg_2_neighbor in nx.all_neighbors(G, neighbor):
            if deg_2_neighbor not in seen:
                updated_centralities[deg_2_neighbor] -= (discount_scale * 0.25 * 0.25 * centrality_dict[v])
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
