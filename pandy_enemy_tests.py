import sys
import json
import sim
import pprint
from pandy_utils import *
from collections import defaultdict

if len(sys.argv) != 4:
    print("usage: pandy_enemy_tests.py [graph filename] [player seeds filename] [our seeds filename]")
    sys.exit(1)

# filename format: num_players.num_seeds.unique_id
file_info = sys.argv[1].split('.') # sys.argv[1] = graph filename
num_players = int(file_info[0])
num_seeds = int(file_info[1])
id = int(file_info[2].split('-')[0])

G = load_graph_from_json(sys.argv[1])

with open(sys.argv[1], 'r') as f:
    G_json = json.load(f)

# sys.argv[2] = seeds filename
with open(sys.argv[2], 'r') as f:
    player_seeds = json.load(f)

our_seeds = {}
with open(sys.argv[3], 'r') as f:
    for round in range(50):
        seeds = [f.readline().strip() for _ in range(num_seeds)]
        our_seeds[round] = seeds

print(our_seeds)

# player_seeds: dict with key team name, value is a list of lists of seeds;
# each nested list are the seeds given to the player for that round

print('number of nodes:', G.number_of_nodes())
print('number of edges:', G.number_of_edges())

final_scores = defaultdict(lambda: 0)
for round in range(50):
    print('round {}/{}'.format(round, 50))
    player_values = {} # seeds per round
    for player in player_seeds:
        if player != 'pandemonium': # don't add our own seeds from the round
            player_values[player] = player_seeds[player][round]

    # read our seeds: every num_seeds lines are the seeds for our round
    player_values['pandemonium'] = our_seeds[round]

    result = sim.run(G_json, player_values)
    print('Results for round {}'.format(round))
    pprint.pprint(result)

    # Store score of winner
    players = list(result.keys())
    winner = max(players, key=lambda player: result[player])
    final_scores[winner] += 1

print('Final Scores')
pprint.pprint(final_scores)
