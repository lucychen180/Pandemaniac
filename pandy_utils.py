import networkx as nx
import json
from networkx.algorithms import community
from networkx.algorithms.core import core_number
from collections import defaultdict
import numpy as np
import math
import itertools
import random
import heapq

# Construct undirected graph from json file
def load_graph_from_json(filename):
    G = nx.Graph()
    with open(filename, 'r') as f:
        # load json file into a dictionary of form {node : [neighbors list]}
        adj_lists = json.load(f)
    for node in adj_lists:
        for neighbor in adj_lists[node]:
            G.add_edge(node, neighbor)  # nodes are repr by strings 0 to V-1

    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def seed_n_nodes_degree(G, n):
    '''
    TA strategy: picks the n nodes with the largest degrees.
    @param G: networkx undirected Graph
    @param n: number of nodes to seed
    @return seeds: a list of nodes to seed in the graph.
    '''
    node_degrees = G.degree()
    seeds = heapq.nlargest(n, node_degrees, key = (lambda pair: pair[1]))
    return [tup[0] for tup in seeds]

def cluster_basic_seed(G, n, num_players):
    return seed_by_cluster(G, n, num_players, seed_n_nodes_basic)

def cluster_neighbor_centrality(G, n, num_players):
    return seed_by_cluster(G, n, num_players, neighbor_centrality)

def seed_by_cluster(G, n, num_players, seeder, threshold = 0.5):
    '''
    Passes a seeder function for a graph. Finds the clusters of the graph, then
    runs the seeder function on each cluster, and appends all the results together
    to form a final list of seeds for the whole graph.
    @param G: networkx undirected Graph
    @param n: total number of nodes to seed
    @param num_players: number of players
    @param seeder: Seeder function of the form seeder(G, n, num_players) that
    returns a list of seeds for G.
    @param threshold: we focus on only the top threshold fraction of nodes in clusters,
    since the best strategy is probably to dominate the large clusters,
    while ignoring the very small ones (idk?)
    '''
    seeds = []
    comp = list(community.label_propagation_communities(G)) # girvan newman too slow
    # comp.sort(reverse = True, key = len)

    # extract only the top clusters that form threshold fraction of nodes
    total_cluster_nodes = 0
    clusters = []
    for cluster in comp:
        clusters.append(cluster)
        total_cluster_nodes += len(cluster)
        if total_cluster_nodes >= threshold * len(G):
            break

    # partition our n seeds among the clusters, s.t. number of seeds given is
    # proportional to cluster size

    # ensure all n seeds get partitioned
    total_nodes_counted = 0
    total_seeds_given = 0
    seed_nums = []
    for i in range(len(clusters)):
        total_nodes_counted += len(comp[i])
        num_seeds = round(total_nodes_counted / total_cluster_nodes * n) - total_seeds_given
        if num_seeds == 1 and len(comp[i]) < total_cluster_nodes / n: # don't give seeds to very small clusters, picked arbitrary threshold
            seed_nums[0] += num_seeds
            seed_nums.append(0)
        else:
            seed_nums.append(num_seeds)
        total_seeds_given += num_seeds

    for i in range(len(seed_nums)):
        cluster = comp[i]
        num_seeds = seed_nums[i]
        cluster_seeds = seeder(G.subgraph(cluster), num_seeds, num_players)
        seeds.extend(cluster_seeds)

    return seeds

def seed_n_nodes_basic(G, n, num_players):
    '''
    Basic algorithm to choose n nodes from G to seed. First, find the clusters
    of the graph; then, divide the seeds among the graph s.t. number of seeds
    in each cluster is proportional to size of the cluster to take over. Then
    from each cluster randomly pick nodes from the top p fraction of eigenvector
    centrality to seed.
    @param G: networkx undirected Graph
    @param n: number of nodes to seed
    @param num_players: number of players in the graph (-1 = # competing against)
    @return seeds: a list of nodes to seed in the graph
    '''
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

    # we use a combination of eigenvector centrality and gravity centrality
    eigen_centralities, grav_centralities = nx.eigenvector_centrality(G), gravity_centrality(G)

    # construct total rank from eigenrank and gravity rank
    total_rank_dict = {}
    for node in G.nodes():
        total_rank_dict[node] = eigen_centralities[node] * grav_centralities[node] # avg of two ranks

    totalranks = list(total_rank_dict.keys())
    totalranks.sort(reverse = True, key = (lambda node: total_rank_dict[node]))
    # totalranks.sort(key = (lambda node: total_rank_dict[node])) # in order of increasing rank

    # seed_nums is an array st seed_nums[i] is the number of seeds community[i] gets
    possible_seeds = totalranks[:math.ceil(n * max(math.sqrt(num_players - 1), 1))]
    seeds = random.sample(possible_seeds, n)
    return seeds

def neighbor_centrality(G,num_seeds,num_players,a=0.2):
    """computes centrality of a node based on its core centrality and the core centrality of neighbors and
        the core centrality of neigbors of neighbors. For some parameter a in [0,1], node v's ranking is given by
        (core centrality of v) + a*sum(core centrality of neighbors of v)
        + a^2*sum(core centrality of neighbors of neighbors of v)
        Paper: https://arxiv.org/pdf/1511.00441.pdf"""

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
    player_scaling_factor = np.sqrt(num_players-1)

    strategy = heapq.nlargest(math.ceil(player_scaling_factor*num_seeds), neighbor_centralities, key = neighbor_centralities.get)
    strategy_random = random.sample(strategy,num_seeds)
    return strategy_random
