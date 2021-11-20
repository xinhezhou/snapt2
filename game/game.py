from random import random 
import numpy as np
import torch
from .device import Device


class Game(object):
    def __init__(self, network, states, values, attack_probs, influence_probs, moves) -> None:
        super().__init__()
        self.devices = []
        self.network = network
        self.states = states
        self.values = values
        self.attack_probs = attack_probs
        self.influence_probs = influence_probs
        self.moves = moves
        self.reset()
    
    def reset(self):
        self.devices = []
        for i in range(len(self.states)):
            d = Device(self.states[i], self.values[i], self.attack_probs[i], self.influence_probs[i])
            self.devices.append(d)
  

    def get_states(self):
        state = [d.state for d in self.devices]
        state = np.ascontiguousarray(state, dtype=np.float32)
        return torch.from_numpy(state)

    def get_values(self):
        return [d.value for d in self.devices]
    
    def get_attack_probs(self):
        return [d.attack_prob for d in self.devices]

    def get_influence_probs(self):
        return [d.influence_prob for d in self.devices]
    
    def defend(self, idx):
        self.devices[idx].state = min(self.devices[idx].state+1, 1)
        
    def attack(self, idx):
        # print(idx)
        if self.devices[idx].state == 0:
            if random() < self.devices[idx].attack_prob:
                self.devices[idx].state = -1
            for v in range(len(self.states)):
                if random() < self.devices[v].influence_prob and self.network[idx][v] == 1 and self.devices[idx].state ==  1:
                    self.devices[idx].state = 0





    


        
