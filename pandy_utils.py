import networkx as nx
import json
from networkx.algorithms import community
from collections import defaultdict
from centrality_measures import *
import numpy as np
import math
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

def seed_by_degree(G, n):
    '''
    TA strategy: picks the n nodes with the largest degrees.
    @param G: networkx undirected Graph
    @param n: number of nodes to seed
    @return seeds: a list of nodes to seed in the graph.
    '''
    node_degrees = G.degree()
    seeds = heapq.nlargest(n, node_degrees, key = (lambda pair: pair[1]))
    return [tup[0] for tup in seeds]

'''
Usage: Write a function in centrality_measures.py that computes a centrality measure
for a graph (returns a dictionary mapping node to value). To make a seed generator
that compares centrality measures for the whole graph, use

seed_by_centrality_measures(G, n, num_players, (centrality_measures))

To make a seed generator that separates graph into clusters first, use

seed_by_cluster(G, n, num_players, seed_by_centrality_measures, (centrality_measures))

to generate a tuple cluster_seeds, seed_nums, then join the cluster seeds together.

See below for some examples.
'''

def cluster_eigen_gravity(G, n, num_players):
    cluster_seeds, seed_nums = seed_by_cluster(G, n, num_players, seed_by_centrality_measures, \
    nx.eigenvector_centrality, gravity_centrality)
    return sum(cluster_seeds, [])

def cluster_neighbor_centrality(G, n, num_players):
    cluster_seeds, seed_nums = seed_by_cluster(G, n, num_players, seed_by_centrality_measures, \
    neighbor_centrality)
    return sum(cluster_seeds, [])

def cluster_eigen_neighbor(G, n, num_players):
    cluster_seeds, seed_nums = seed_by_cluster(G, n, num_players, seed_by_centrality_measures, \
    nx.eigenvector_centrality, neighbor_centrality)
    return sum(cluster_seeds, [])

def possible_cluster_eigen_neighbor_degree(G, n, num_players):
    return seed_by_cluster(G, n, num_players, possible_seeds_by_centrality, \
    neighbor_centrality, nx.eigenvector_centrality, nx.degree_centrality)

def possible_cluster_neighbor(G, n, num_players):
    return seed_by_cluster(G, n, num_players, possible_seeds_by_centrality, \
    neighbor_centrality)

def clusterless_neighbor(G, n, num_players):
    return seed_by_centrality_measures(G, n, num_players, neighbor_centrality)

def possible_clusterless_ksc_neighbor_eigen(G, n, num_players):
    return possible_seeds_by_centrality(G, n, num_players, ksc_centrality, neighbor_centrality, \
    nx.eigenvector_centrality)

def seed_by_cluster(G, n, num_players, seeder, *argv):
    '''
    Passes a seeder function for a graph. Finds the clusters of the graph, then
    runs the seeder function on each cluster, and appends all the results together
    to form a final list of seeds for the whole graph.
    @param G: networkx undirected Graph
    @param n: total number of nodes to seed
    @param num_players: number of players
    @param seeder: Seeder function of the form seeder(G, n, num_players) that
    returns a list of seeds for G.
    @param argv: extra arguments seeder might need (ex seed_by_centrality_measures)
    @return seeds, seed_nums: seeds is a list where each element is the seeds assigned
    for an individual cluster. seed_nums is a list where each element is how many
    of the n seeds were partitioned to the corresponding cluster.
    '''
    seeds = []
    comp = list(community.label_propagation_communities(G)) # girvan newman too slow
    comp.sort(reverse = True, key = len)

    # we focus on only the top threshold fraction of nodes in clusters,
    # since the best strategy is probably to dominate the large clusters,
    # while ignoring the very small ones (idk?)

    threshold = 0.75 # change if needed

    # extract only the top clusters that form threshold fraction of nodes
    total_cluster_nodes = 0
    clusters = []
    for cluster in comp:
        clusters.append(cluster)
        total_cluster_nodes += len(cluster)
        if total_cluster_nodes >= threshold * len(G):
            break

    print(len(clusters))
    for cluster in clusters:
        print(len(cluster))

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
        cluster_seeds = seeder(G.subgraph(cluster), num_seeds, num_players, *argv)
        seeds.append(cluster_seeds)

    assert sum(seed_nums) == n
    # seeds is an array, each element is a list of seeds for that cluster
    # (if we passed in possible_..., then it is longer)
    # seed_nums is an array where each element corresponds to how many seeds
    # should be partitioned to that cluster
    return seeds, seed_nums

def seed_by_centrality_measures(G, n, num_players, *argv):
    '''
    Seeds a graph based on a combination of some number of centrality measures.
    Works by multiplying the centrality measures together, then ranking the nodes
    based on the composite centrality measure.
    @param G: graph
    @param n: number of seeds
    @param num_players: number of players
    @param argv: each extra argument is function computing the centrality measure,
    that returns a dictionary mapping node to its centrality value
    (ie nx.eigenvector_centrality is such a function)
    '''
    total_centrality = defaultdict(lambda: 1)
    lst_of_centralities = [measure(G) for measure in argv] # list of dictionaries for different centrality measures
    for centrality_dict in lst_of_centralities:
        for node in centrality_dict:
            assert centrality_dict[node] >= 0 # don't want to multiply by negative
            total_centrality[node] *= centrality_dict[node]

    totalranks = list(total_centrality.keys())
    totalranks.sort(reverse = True, key = (lambda node: total_centrality[node])) # in decreasing order of centrality
    player_scaling_factor = max(np.sqrt(num_players - 1), 1) # introduce scaling randomness for larger players
    possible_seeds = totalranks[:math.ceil(n * player_scaling_factor)]
    seeds = random.sample(possible_seeds, n) # take random n from top ~2n
    return seeds

def possible_seeds_by_centrality(G, n, num_players, *argv):
    '''
    Usage is same as above, but now it returns a list of possible seeds for the
    graph, instead of already sampled seeds.
    '''
    scaling_factor = max(np.sqrt(num_players - 1), 1)
    return seed_by_centrality_measures(G, math.ceil(n * scaling_factor), 2, *argv)
