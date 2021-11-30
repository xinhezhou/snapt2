import numpy as np
import torch
import random

import sys
sys.path.append('..')
from network import Network
from utils import compute_attcker_reward, select_action, test_att, plot
from configA import games 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network(4, 4, 10, device).to(device)
BATCH_SIZE = 10

train_rewards = []
test_rewards = []
losses = []


initial_weights  = {}
state_dict = model.state_dict()
# print("Model's state_dict:")
for param in state_dict:
    initial_weights[param] = torch.from_numpy(np.random.randn(*state_dict[param].size()))
    # print(initial_weights[param].size(), state_dict[param].size())
# print(initial_weights.keys(), state_dict.keys())

def update(weights, sigma, jitters):
    new_weights = {}
    for param in state_dict:
        jitter = torch.from_numpy(np.random.randn(*state_dict[param].size()))
        jitters[param].append(jitter)
        new_weights[param] = weights[param] + sigma * jitter
    return new_weights

def fitness(w, train_rewards, test_rewards):
    model.load_state_dict(w)
    g = random.choice(games)
    state = g.get_states()
    total_reward = 0

    for t in range(g.moves):
        # Select and perform an action
        action, action_dist = select_action(model, state, 0)
        g.attack(action)
        next_state = g.get_states()
        reward = compute_attcker_reward(state, next_state, g.get_values())
        state = next_state
        test_rewards.append(test_att(model, g))
        train_rewards.append(reward)
        total_reward += reward

    return total_reward

npop = 1     # population size
num_episodes = 1
sigma = 0.1    # noise standard deviation
alpha = 0.001  # learning rate
w = initial_weights
for i in range(num_episodes):
    R = np.zeros(npop)
    jitters = {}
    for param in initial_weights:
        jitters[param] = []
    for j in range(npop):
        w_try = update(w, sigma, jitters)
        R[j] = fitness(w_try,  train_rewards, test_rewards)
    if np.sum(R) != 0:
        A = (R - np.mean(R)) / np.std(R)
        for param in w:
            N = torch.stack(jitters[param])
            w[param] = w[param] + alpha/(npop*sigma) * np.dot(N.T, A)

 