
from game import Game
from random import randint
from random import random

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
depth = 4

def play_game(g, first_player):
    player = first_player
    for i in range(g.depth):
        if player == 1:
            g.attack(randint(0, g.n - 1))
        else:
            g.defend(randint(0, g.n - 1))
        player *= -1
    return g.compute_outcome()
    
att_outcomes = []
for i in range(500):
    g = Game(network, states, rewards, attack_probs, influence_probs, depth)
    att_outcomes.append(play_game(g, 1))

print(sum(att_outcomes))

def_outcomes = []
for i in range(500):
    g = Game(network, states, rewards, attack_probs, influence_probs, depth)
    def_outcomes.append(play_game(g, -1))

print(sum(def_outcomes))
