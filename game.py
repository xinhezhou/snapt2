from device import Device

class Game(Object):
    def __init__(self, n) -> None:
        super().__init__()
        self.adj_matrix = []
        self.devices = []
        for i in range(n):
            self.devices.append(Device("S", 0))
            self.adj_matrix.append([0]*n)
    
    def add_edge(self, u, v):
        self.adj_matrix[u][v] = 1

    def change_device_state(self, idx, state):
        self.devices[idx] = state
    
    def apply_action(self, idx, f):
        f(idx, self.adj_matrix, self.devices)



    


        
