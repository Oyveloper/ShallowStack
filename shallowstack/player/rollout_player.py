from shallowstack.player.hybrid_player import HybridPlayer


class RolloutPlayer(HybridPlayer):
    def __init__(self, name: str):
        super().__init__(name, 0.0)
