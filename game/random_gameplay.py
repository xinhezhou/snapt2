
from game import Game
from utils import play_game_random

# chain * 
# isolated vertices
# fully connected
# star graph
num_device = 4
network = [[0, 1, 0, 0],
           [1, 0, 1, 0],
           [0, 1, 0, 1], 
           [0, 0, 1, 0]]

# location of vulnerability: center*, side
# number of vulnerable sites : 1
states = [1, 1, 0, 1]

# anything between 0 and 1
# distribution of rewards
rewards = [1, 1, 1, 1]

attack_probs = [1, 1, 1, 1]

influence_probs = [1, 1, 1, 1]


# right now: fix depth 
moves = 1

    
att_outcomes = []
for i in range(500):
    g = Game(network, states, rewards, attack_probs, influence_probs, moves)
    att_outcomes.append(play_game_random(g, 1))

print(sum(att_outcomes))

def_outcomes = []
for i in range(500):
    g = Game(network, states, rewards, attack_probs, influence_probs, moves)
    def_outcomes.append(play_game_random(g, -1))

print(sum(def_outcomes))
