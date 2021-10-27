class Device(Object):
    def __init__(self, state, reward) -> None:
        super().__init__()
        self.state = state 
        self.reward = reward

