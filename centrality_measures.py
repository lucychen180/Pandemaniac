import networkx as nx
from networkx.algorithms.core import core_number
from networkx.algorithms import community
import itertools
import math
import pprint

def degree_centrality(G):
    '''
    Returns the degree centrality of the graph
    @param G: graph
    @return dict, keys are nodes, values are centralities
    '''
    return {node[0]: node[1] for node in G.degree()}

def normalize_dict(dictionary):
    d = dictionary.copy()
    max_v = max(d.values())
    for v in d.keys():
        d[v] = d[v] / max_v

    return d

def gravity_centrality(G):
    '''
    Returns the gravity centrality (global) of the graph.
    @param G: graph
    @return dict, keys are nodes, values are centralities
    '''
    res = {}
    k_index = nx.core_number(G)
    dist = dict(nx.shortest_path_length(G), cutoff = 3) # this is computation heavy
    for node in G.nodes():
        # find nodes with a distance of at most 3 from node
        neighbors = list(dist[node].keys())
        neighbors.remove(node)

        # compute gravity centrality of node
        grav_centrality = 0
        for neighbor in neighbors:
            grav_centrality += (k_index[node] * k_index[neighbor] / (dist[node][neighbor] ** 2))

        res[node] = grav_centrality

    return res

def neighbor_centrality(G, a = 0.3):
    '''
    Computes the neighbor centrality of a node based on its core centrality and the core centrality of neighbors and
    the core centrality of neigbors of neighbors. For some parameter a in [0,1], node v's ranking is given by
    (core centrality of v) + a*sum(core centrality of neighbors of v)
    + a^2*sum(core centrality of neighbors of neighbors of v)
    Paper: https://arxiv.org/pdf/1511.00441.pdf
    @param G: undirected networkx Graph
    @param a: scaling factor
    @return neighboorhood_centralities: dict with key as node, value as neighbor centrality value
    '''
    neighbor_centralities = {}
    core_numbers = core_number(G) #get core number of each vertex in the graph
    neighbors = {v: list(nx.all_neighbors(G, v)) for v in G}  #{v:neighbors of v} dictionary
    for v in G:
        n_centrality = core_numbers[v] #(core centrality of v)
        neighbors_1 = set(neighbors[v])
        neighbors_2 = set(itertools.chain.from_iterable([neighbors[w] for w in neighbors_1]))
        set_diff = neighbors_2.difference(neighbors_1.union(v)) #dist2 neighbors that are neither v nor dist1 neighbors
        for n in neighbors_1: #1st degree
            n_centrality += a*core_numbers[n]
        for n in set_diff: #2nd degree
            n_centrality += (a**2)*core_numbers[n]
        neighbor_centralities[v] = n_centrality

    return neighbor_centralities

def voterank(G):
    voteranks = nx.voterank(G)
    # convert ranking of nodes to an approximate value
    res = {}
    for i in range(len(voteranks)):
        node = voteranks[i]
        res[node] = 1 / math.log(i + 2)

    return res

def ksc_centrality(G, alpha = 0.5, beta = 0.5):
    '''
    Paper: https://www.researchgate.net/publication/262391998_A_New_Approach_to_Identify_Influential_Spreaders_in_Complex_Networks
    '''
    core_numbers = core_number(G)
    max_core_number = max(core_numbers.values())
    # internal influence
    f_internal = {}
    for v in G:
        f_internal[v] = core_numbers[v] / max_core_number

    # external influence
    f_external = {}
    curr_max_influence = 0
    comp = list(community.label_propagation_communities(G))
    comp.sort(reverse = True, key = len)
    for v in G:
        # count the influence (total number of neighbors in other communities)
        influence = 0
        for cluster in comp:
            cluster_set = set(cluster)
            num_neighbors = 0
            # count number of neighbors in the cluster
            for node in G.neighbors(v):
                if node in cluster_set:
                    num_neighbors += 1
            influence += (num_neighbors * len(cluster))
        f_external[v] = influence

    # normalize by maximum influence
    max_influence = max(f_external.values())
    for v in G:
        f_external[v] = f_external[v] / max_influence


    # total influence
    f_total = {}
    for v in G:
        f_total[v] = f_internal[v] * alpha + f_external[v] * beta

    return f_total
