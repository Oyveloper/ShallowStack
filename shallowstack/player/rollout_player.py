from shallowstack.player.hybrid_player import HybridPlayer


class RolloutPlayer(HybridPlayer):
    def __init__(self, name: str, range_size: int = 1326):
        super().__init__(name, 0.0, range_size=range_size)
