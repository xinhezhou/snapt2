from device import Device
from random import Random, random

class Game(object):
    def __init__(self, network, states, rewards, attack_probs, influence_probs, depth) -> None:
        super().__init__()
        self.states_init = states[:]
        self.network = network[:]
        self.devices = []
        self.attack_probs = influence_probs[:]
        self.influence_probs = attack_probs[:]
        self.n = len(network)
        self.depth = depth
        for i in range(self.n):
            self.devices.append(Device(states[i], rewards[i]))
    
    
    def defend(self, idx):
        self.devices[idx].state = min(self.devices[idx].state+1, 1)
        
    def attack(self, idx):
        if self.devices[idx].state == 0:
            if random() < self.attack_probs[idx]:
                self.devices[idx].state = -1
            for v in range(self.n):
                if random() < self.influence_probs[v] and self.network[idx][v] == 1 and self.devices[v].state ==  1:
                    self.devices[v].state = 0

    def compute_outcome(self):
        # if at least one device is compromised, the attacker wins. (otherwise the defender wins)
        for i in range(self.n):
            device = self.devices[i]
            if device.state == -1:
                return 1
        return -1





    


        
