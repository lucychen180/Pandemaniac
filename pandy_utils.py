import networkx as nx
import json
from networkx.algorithms import community
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

    return G

def gravity_centrality(G):
    '''
    Returns the gravity centrality (global) of the graph.
    @param G: graph
    @return dict, keys are nodes, values are centralities
    '''
    res = {}
    k_index = nx.core_number(G)
    for node in G.nodes():
        # find nodes with a distance of at most 3 from node
        dist = nx.single_source_shortest_path_length(G, source = node, cutoff = 3)
        neighbors = list(dist.keys())
        neighbors.remove(node)

        # compute gravity centrality of node
        grav_centrality = 0
        for neighbor in neighbors:
            grav_centrality += (k_index[node] * k_index[neighbor] / (dist[neighbor] ** 2))

        res[node] = grav_centrality

    return res

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

def seed_n_nodes_basic(G, n, num_players, threshold = 0.75):
    '''
    Basic algorithm to choose n nodes from G to seed. First, find the clusters
    of the graph; then, divide the seeds among the graph s.t. number of seeds
    in each cluster is proportional to size of the cluster to take over. Then
    from each cluster randomly pick nodes from the top p fraction of eigenvector
    centrality to seed.
    @param G: networkx undirected Graph
    @param n: number of nodes to seed
    @param num_players: number of players in the graph (-1 = # competing against)
    @param threshold: we focus on only the top threshold fraction of nodes in clusters,
    since the best strategy is probably to dominate the large clusters,
    while ignoring the very small ones (idk?)
    @return seeds: a list of nodes to seed in the graph
    '''

    # we use a combination of eigenvector centrality and gravity centrality
    eigen_centralities, grav_centralities = nx.eigenvector_centrality(G), gravity_centrality(G)
    eigen_nodes = list(G.nodes())
    eigen_nodes.sort(reverse = True, key = (lambda node: eigen_centralities[node])) # nodes sorted in descending order of eigenvector centrality
    # voteranks = nx.voterank(G, max_iter = 2000) # this is kind of shit for the 1v1 test graph, but i feel like it might be better in real competition?

    gravity_nodes = list(G.nodes())
    gravity_nodes.sort(reverse = True, key = (lambda node: grav_centralities[node])) # nodes sorted in descending order of gravity centrality

    # construct total rank from eigenrank and gravity rank
    total_rank_dict = {}
    for node in G.nodes():
        total_rank_dict[node] = eigen_nodes.index(node) + gravity_nodes.index(node) # avg of two ranks

    totalranks = list(total_rank_dict.keys())
    totalranks.sort(key = (lambda node: total_rank_dict[node])) # in order of increasing rank

    seeds = []
    # comp = list(community.label_propagation_communities(G)) # girvan newman too slow
    comp = [list(G.nodes())]
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

    # seed_nums is an array st seed_nums[i] is the number of seeds community[i] gets
    for i in range(len(seed_nums)):
        cluster = comp[i] # list of nodes in the graph corresponding to cluster
        num_seeds = seed_nums[i]

        possible_seeds = totalranks[:math.ceil(num_seeds * max(math.sqrt(num_players - 1), 1))]
        cluster_seeds = random.sample(possible_seeds, num_seeds)
        seeds.extend(cluster_seeds)

    assert len(seeds) == n # idk lol
    # print(seeds)
    return seeds
