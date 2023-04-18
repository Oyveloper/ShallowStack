from shallowstack.player.hybrid_player import HybridPlayer


class ResolvePlayer(HybridPlayer):
    def __init__(self, name: str, range_size: int = 1326, show_internals: bool = False):
        super().__init__(
            name, 1.0, range_size=range_size, show_internals=show_internals
        )
