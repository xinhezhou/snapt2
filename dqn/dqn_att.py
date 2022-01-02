import random
import numpy as np
import torch
import sys
sys.path.append('..')
from utils import compute_attcker_reward, select_action, test_att



def train(games, model, optimizer, loss_fn, num_episodes, device):
    train_rewards = []
    test_rewards = []
    losses = []
    for i in range(num_episodes):
        # Initialize the environment and state
        g = random.choice(games)
        state = g.get_states()
        for t in range(g.moves):
            # Select and perform an action
            action, action_dist = select_action(model, g, 0.2)
            g.attack(action)
            next_state = g.get_states()
            reward = torch.tensor([compute_attcker_reward(state, next_state, g.get_values())], device=device)
            state = next_state

            # Perform one step of the optimization (on the policy network)
            # print(reward, action_dist[action])
            loss = loss_fn(reward, action_dist[action])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            g.reset()

            # record test results 
            train_rewards.append(reward.item())
            test_rewards.append(test_att(model, g))
            losses.append(loss.item())
    return train_rewards, test_rewards, losses


