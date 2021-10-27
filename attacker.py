
class Attacker(Object):
    def __init__(self) -> None:
        super().__init__()

    def attack(self, idx, adj_matrix, devices):
        if devices[idx].state == "V":
            devices[idx].state = "C"
        
        for v in range(len(adj_matrix)):
            if adj_matrix[idx][v] == 1 and devices[v] == "S":
                devices[v] = "V"
