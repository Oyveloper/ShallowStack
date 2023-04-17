from shallowstack.player.hybrid_player import HybridPlayer


class ResolvePlayer(HybridPlayer):
    def __init__(self, name: str, chips: int = 1000):
        super().__init__(name, chips, 1.0)
