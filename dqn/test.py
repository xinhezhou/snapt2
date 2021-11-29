import torch
import torch.optim as optim
import torch.nn as nn
from dqn_def import train

import sys
sys.path.append('..')
from network import Network
from utils import plot
from configD import games 
# from configB import games
# from configC import games


# DQN Set Up
num_device = 4
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = Network(num_device, num_device, 10, device).to(device)
optimizer = optim.Adam(policy_net.parameters())
loss_fn = nn.SmoothL1Loss()
num_episodes = 1000
BATCH_SIZE = 10

train_rewards, test_rewards, losses = train(games, policy_net, optimizer, loss_fn, num_episodes, device)
print(losses)
print(train_rewards)
print(test_rewards)

plot(test_rewards, losses, "configD_dqn_def.pdf", BATCH_SIZE)
# plot(losses, test_rewards, "configB_dqn.pdf", BATCH_SIZE)
# plot(losses, test_rewards, "configC_dqn.pdf", BATCH_SIZE)