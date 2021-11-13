# import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T


from game import Game
from utils import compute_attcker_reward
from network import DQN

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

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# Game Init
num_device = 4
network = [[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0], 
           [0, 0, 0, 0]]

# location of vulnerability: center*, side
# number of vulnerable sites : 1
states = [1, 1, 0, 1]

# anything between 0 and 1
# distribution of rewards
values = [1, 1, 1, 1]

attack_probs = [1, 1, 1, 1]

influence_probs = [1, 1, 1, 1]

moves = 1

rewards = []
losses = []

# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy())
# plt.title('Example extracted screen')
# plt.show(block=True)

BATCH_SIZE = 10
GAMMA = 1
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Attacker initial network setting
policy_net = DQN(num_device, num_device, 10, device).to(device)
target_net = DQN(num_device, num_device, 10, device).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    # global steps_done
    sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    eps_threshold = 0.5
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print(policy_net(state).max(0)[1])

            return policy_net(state).max(0)[1]
    else:
        # print(torch.tensor([random.randrange(num_device)], device=device, dtype=torch.long))
        return torch.tensor([random.randrange(num_device)], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = next_state_values + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    rewards.append(batch.reward[-1])
    losses.append(loss.item())
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 10000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    g = Game(network, states, values, attack_probs, influence_probs, moves)
    state = g.get_states()
    for t in range(g.moves):
        # Select and perform an action
        action = select_action(state)
        g.attack(action)
        next_state = g.get_states()
        reward = compute_attcker_reward(state, next_state, g.get_values())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        # last_screen = current_screen
        # current_screen = get_screen()

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # print(reward)
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        # print(t)
        optimize_model()
        episode_durations.append(reward)
        # plot_durations()
        break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

# print(losses)
# print(rewards)
print('Complete')

average_rewards = []
average_losses = []

x = range(num_episodes // BATCH_SIZE - 1)
for i in x:
    average_rewards.append(np.mean(rewards[i*BATCH_SIZE: (i+1)*BATCH_SIZE]))
    average_losses.append(np.mean(losses[i*BATCH_SIZE: (i+1)*BATCH_SIZE]))


data = [average_losses, average_rewards]
title = ["losses", "rewards"]

fig, ax = plt.subplots(nrows=2, ncols=1)

i = 0
for row in ax:
    row.plot(x, data[i])
    row.title(title[i])
    i += 1



plt.savefig("dqn_progress_10000.pdf")
