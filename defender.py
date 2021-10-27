
class Defender(Object):
    def __init__(self) -> None:
        super().__init__()

    def defend(self, idx, adj_matrix, devices):
        if devices[idx].state == "V":
            devices[idx].state = "S"