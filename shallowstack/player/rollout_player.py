from shallowstack.player.hybrid_player import HybridPlayer


class RolloutPlayer(HybridPlayer):
    def __init__(self, name: str, chips: int = 1000):
        super().__init__(name, chips, 0.0)
