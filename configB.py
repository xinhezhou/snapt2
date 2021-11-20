from game.game import Game

# multiple configs w/ one vulnerable device
num_device = 4
network = [[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0], 
           [0, 0, 0, 0]]

# location of vulnerability: center*, side
# number of vulnerable sites : 1
states = [1, 1, 1, 1]

# anything between 0 and 1
# distribution of rewards
values = [1, 1, 1, 1]

attack_probs = [1, 1, 1, 1]

influence_probs = [1, 1, 1, 1]

moves = 1
games = []
for i in range(4):
    states_copy = states[:]
    states_copy[i] = 0
    games.append(Game(network, states_copy, values, attack_probs, influence_probs, moves))