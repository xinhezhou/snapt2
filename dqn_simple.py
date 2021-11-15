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

def select_action(state):
    # global steps_done
    sample = random.random()
    eps_threshold = 0.5
    action_dist = policy_net(state)
    if sample > eps_threshold:
        with torch.no_grad():
            action = np.argmax(action_dist)
    else:
        action = random.randrange(num_device)
    return action, action_dist


num_episodes = 10
for i_episode in range(num_episodes):
    # Initialize the environment and state
    g = Game(network, states, values, attack_probs, influence_probs, moves)
    state = g.get_states()
    for t in range(g.moves):
        # Select and perform an action
        action, action_dist = select_action(state)
        g.attack(action)
        next_state = g.get_states()
        reward = compute_attcker_reward(state, next_state, g.get_values())
        reward = torch.tensor([reward], device=device)
        state = next_state

        # Perform one step of the optimization (on the policy network)
        print(reward, action_dist[action])
        loss = loss_fn(reward, action_dist[action])
        
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    rewards.append(reward.item())
    losses.append(loss.item())
    optimizer.step()

print(losses)
print(rewards)
# plt.savefig("dqn_progress_500000.pdf")
