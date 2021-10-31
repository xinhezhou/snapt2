class Device(object):
    def __init__(self, state, reward) -> None:
        super().__init__()
        self.state = state 
        self.reward = reward

    def __repr__(self):
        return "(" + str(self.state) + ", " + str(self.reward) + ")"

