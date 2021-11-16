import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim


from game import Game
from utils import compute_attcker_reward
from network import Network
from game_setup import *
from utils import *

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DQN Set Up
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
policy_net = Network(num_device, num_device, 10, device).to(device)
optimizer = optim.Adam(policy_net.parameters())
loss_fn = nn.SmoothL1Loss()
test_rewards = []
BATCH_SIZE = 10

def select_action(state, eps):
    # global steps_done
    sample = random.random()
    action_dist = policy_net(state)
    if sample > eps:
        with torch.no_grad():
            action = torch.argmax(action_dist)
    else:
        action = random.randrange(num_device)
    return action, action_dist


def test():
    rewards = []
    g = Game(network, states, values, attack_probs, influence_probs, moves)
    for t in range(g.moves):
        state = g.get_states()
        action, action_dist = select_action(state, 0)
        g.attack(action)
        next_state = g.get_states()
        reward = compute_attcker_reward(state, next_state, g.get_values())
        rewards.append(reward)
        # print(action, state, next_state)
    # print(rewards)
    return np.mean(rewards)

num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    g = Game(network, states, values, attack_probs, influence_probs, moves)
    state = g.get_states()
    for t in range(g.moves):
        # Select and perform an action
        action, action_dist = select_action(state, 0.2)
        g.attack(action)
        next_state = g.get_states()
        reward = compute_attcker_reward(state, next_state, g.get_values())
        reward = torch.tensor([reward], device=device)
        state = next_state

        # Perform one step of the optimization (on the policy network)
        # print(reward, action_dist[action])
        loss = loss_fn(reward, action_dist[action])
        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(reward.item())
        test_rewards.append(test())
        losses.append(loss.item())

# print(losses)
# print(rewards)
# print(test_rewards)
average_losses = []
average_rewards = []

x = range(num_episodes)
x = range(num_episodes // BATCH_SIZE)
for i in x:
    average_rewards.append(np.mean(test_rewards[i*BATCH_SIZE: (i+1)*BATCH_SIZE]))
    average_losses.append(np.mean(losses[i*BATCH_SIZE: (i+1)*BATCH_SIZE]))

# x = range(num_episodes // BATCH_SIZE)
fig, ax = plt.subplots(2)
# print(ax[0])
add_suplot(ax[0], x, average_losses, "losses")
add_suplot(ax[1], x, average_rewards, "rewards")
plt.savefig("dqn_simple.pdf")
