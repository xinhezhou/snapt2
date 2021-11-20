class Device(object):
    def __init__(self, state, value, attack_prob, influence_prob) -> None:
        super().__init__()
        self.state = state 
        self.value = value
        self.attack_prob = attack_prob
        self.influence_prob = influence_prob