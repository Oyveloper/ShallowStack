from shallowstack.player.hybrid_player import HybridPlayer


class ResolvePlayer(HybridPlayer):
    def __init__(self, name: str, chips: int = 1000, show_internals: bool = False):
        super().__init__(name, chips, 1.0, show_internals)
