from random import randint
import torch.optim as optim

def play_game_random(g, first_player):
    player = first_player
    for i in range(g.moves*2):
        if player == 1:
            g.attack(randint(0, g.n - 1))
        else:
            g.defend(randint(0, g.n - 1))
        player *= -1
    return g.compute_outcome()

def compute_attcker_reward(init_state, next_state, values):
    # delta compromised - delta secured
    reward = 0
    for i in range(len(init_state)):
        if next_state[i] == -1 and init_state[i] == 0:
            reward += values[i]
        elif next_state[i] == 1 and init_state[i] == 0:
            reward -= i
    return reward


def add_suplot(subplot, x, y, title):
    subplot.plot(x, y)
    subplot.title.set_text(title)
