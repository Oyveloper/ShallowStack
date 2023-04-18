from shallowstack.player.hybrid_player import HybridPlayer


class ResolvePlayer(HybridPlayer):
    def __init__(self, name: str, show_internals: bool = False):
        super().__init__(name, 1.0, show_internals)
