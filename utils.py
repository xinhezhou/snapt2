from random import randint, uniform
import matplotlib.pyplot as plt
import random
import torch
import numpy as np

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
    # delta compromised + delta vulnerable
    reward = 0
    for i in range(len(init_state)):
        if init_state[i] == 1 and next_state[i] == 0:
            reward += 0.1 * values[i]
        if init_state[i] == 0 and next_state[i] == -1:
            reward += 0.9 * values[i]
    return reward / sum(values)

def compute_defender_reward(init_state, next_state, values):
    reward = 0
    for i in range(len(init_state)):
        if init_state[i] == 0 and next_state[i] == 1:
            reward += values[i]
    return reward / sum(values)

def select_action(policy_net, state, eps):
    # global steps_done
    sample = random.random()
    action_dist = policy_net(state)
    if sample > eps:
        with torch.no_grad():
            action = torch.argmax(action_dist)
    else:
        action = random.randrange(len(state))
    return action, action_dist

def test_att(model, game):
    game.reset()
    rewards = []
    for t in range(game.moves):
        state = game.get_states()
        action, action_dist = select_action(model, state, 0)
        game.attack(action)
        next_state = game.get_states()
        reward = compute_attcker_reward(state, next_state, game.get_values())
        rewards.append(reward)
    game.reset()
    return sum(rewards) / len(rewards)

def test_def(model, game):
    game.reset()
    rewards = []
    for t in range(game.moves):
        state = game.get_states()
        action, action_dist = select_action(model, state, 0)
        game.defend(action)
        next_state = game.get_states()
        reward = compute_defender_reward(state, next_state, game.get_values())
        rewards.append(reward)
    game.reset()
    return sum(rewards) / len(rewards)


def plot(rewards,losses, title, batch_size):
    average_rewards = []
    average_losses = []
    num_episodes = len(rewards)

    # x = range(num_episodes)
    x = range(num_episodes // batch_size)
    for i in x:
        average_rewards.append(np.mean(rewards[i*batch_size: (i+1)*batch_size]))
        average_losses.append(np.mean(losses[i*batch_size: (i+1)*batch_size]))

    # x = range(num_episodes // BATCH_SIZE)
    fig, ax = plt.subplots(2)
    ax[0].plot(x, average_rewards)
    ax[0].title.set_text("rewards")
    ax[1].plot(x, average_losses)
    ax[1].title.set_text("losses")

    plt.savefig(title)